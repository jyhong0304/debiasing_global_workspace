import torch
import torch.nn as nn
import torch.nn.functional as F


# JYH: By referring -
# https://openaccess.thecvf.com/content/WACV2024/html/Hong_Concept-Centric_Transformers_Enhancing_Model_Interpretability_Through_Object-Centric_Concept_Learning_Within_WACV_2024_paper.html
class MiniCCTQSA(nn.Module):
    def __init__(
            self,
            in_feature=7,
            embedding_dim=768,
            latent_dim=768,
            num_heads=2,
            attention_dropout=0.1,
            n_spatial_concepts=10,
            num_iterations=3,
            *args,
            **kwargs,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.n_spatial_concepts = n_spatial_concepts
        self.spatial_concept_slots_init = nn.Embedding(self.n_spatial_concepts, latent_dim)
        nn.init.xavier_uniform_(self.spatial_concept_slots_init.weight)

        # Encoder
        self.in_feature = in_feature
        self.encoder = nn.Sequential(
            nn.Linear(embedding_dim, in_feature * in_feature * latent_dim),
            nn.LayerNorm(in_feature * in_feature * latent_dim)

        )

        # Workspace
        self.spatial_concept_slot_attention = ConceptQuerySlotAttention(num_iterations=num_iterations,
                                                                        slot_size=latent_dim,
                                                                        mlp_hidden_size=latent_dim)
        self.spatial_concept_slot_pos = nn.Parameter(torch.zeros(1, 1, n_spatial_concepts * latent_dim),
                                                     requires_grad=True)
        self.spatial_concept_tranformer = CrossAttentionEmbedding(
            dim=latent_dim,
            num_heads=num_heads,
            attention_dropout=attention_dropout,
        )

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(in_feature * in_feature * latent_dim, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )

    def forward(self, x, sigma=0):
        if x.dim() < 3:  # this module requires the tensor with the dim_size = 3.
            x = torch.unsqueeze(x, 1)

        # Encoding
        x = self.encoder(x).reshape(x.shape[0], self.in_feature * self.in_feature, -1)  # [B, in_feature*in_feature, D]
        # Global workspace
        mu = self.spatial_concept_slots_init.weight.expand(x.size(0), -1, -1)
        z = torch.randn_like(mu).type_as(x)
        spatial_concept_slots_init = mu + z * sigma * mu.detach()
        spatial_concepts, _ = self.spatial_concept_slot_attention(x,
                                                                  spatial_concept_slots_init)  # [B, num_concepts, embedding_dim]
        x, spatial_concept_attn = self.spatial_concept_tranformer(x, spatial_concepts)
        spatial_concept_attn = spatial_concept_attn.mean(1)  # average over heads
        # x: [B, in_feature*in_feature, D]
        # attn: [B, in_feature*in_feature, n_concepts]
        # Decoding
        x = torch.flatten(x, start_dim=1)
        x = self.decoder(x)

        return x, spatial_concept_attn


class CrossAttentionEmbedding(nn.Module):
    def __init__(
            self, dim, num_heads=8, attention_dropout=0.1
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // self.num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=False)
        self.kv = nn.Linear(dim, dim * 2, bias=False)
        self.attn_drop = nn.Dropout(attention_dropout)

    def forward(self, x, y):
        B, Nx, C = x.shape
        By, Ny, Cy = y.shape

        assert C == Cy, "Feature size of x and y must be the same"

        q = self.q(x).reshape(B, Nx, 1, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        kv = (
            self.kv(y)
            .reshape(By, Ny, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )

        q = q[0]
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, Nx, C)

        return x, attn


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """
    Obtained from: github.com:rwightman/pytorch-image-models
    Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class ConceptQuerySlotAttention(nn.Module):
    """
    BO-QSA module for extracting concepts.
    We refer to "Improving Object-centric Learning with Query Optimization" to implement it.
    [https://arxiv.org/abs/2210.08990]
    We simplify this by removing the last LayerNorm and MLP.
    """

    def __init__(
            self,
            num_iterations,
            slot_size,
            mlp_hidden_size,
            truncate='bi-level',
            epsilon=1e-8,
            drop_path=0.2,
    ):
        super().__init__()
        self.slot_size = slot_size
        self.epsilon = epsilon
        self.truncate = truncate
        self.num_iterations = num_iterations

        self.norm_feature = nn.LayerNorm(slot_size)
        self.norm_slots = nn.LayerNorm(slot_size)

        self.project_k = linear(slot_size, slot_size, bias=False)
        self.project_v = linear(slot_size, slot_size, bias=False)
        self.project_q = linear(slot_size, slot_size, bias=False)

        self.gru = gru_cell(slot_size, slot_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, features, slots_init):
        # `feature` has shape [batch_size, num_feature, inputs_size].
        features = self.norm_feature(features)
        k = self.project_k(features)  # Shape: [B, num_features, slot_size]
        v = self.project_v(features)  # Shape: [B, num_features, slot_size]

        B, N, D = v.shape
        slots = slots_init
        # Multiple rounds of attention.
        for i in range(self.num_iterations):
            if i == self.num_iterations - 1:
                slots = slots.detach() + slots_init - slots_init.detach()
            slots_prev = slots
            slots = self.norm_slots(slots)
            q = self.project_q(slots)
            # Attention
            scale = D ** -0.5
            attn_logits = torch.einsum('bid,bjd->bij', q, k) * scale
            attn = F.softmax(attn_logits, dim=1)

            # Weighted mean
            attn_sum = torch.sum(attn, dim=-1, keepdim=True) + self.epsilon
            attn_wm = attn / attn_sum
            updates = torch.einsum('bij, bjd->bid', attn_wm, v)

            # Update slots
            slots = self.gru(
                updates.reshape(-1, D),
                slots_prev.reshape(-1, D)
            )
            slots = slots.reshape(B, -1, D)

        return slots, attn


def linear(in_features, out_features, bias=True, weight_init='xavier', gain=1.):
    m = nn.Linear(in_features, out_features, bias)

    if weight_init == 'kaiming':
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
    else:
        nn.init.xavier_uniform_(m.weight, gain)

    if bias:
        nn.init.zeros_(m.bias)

    return m


def gru_cell(input_size, hidden_size, bias=True):
    m = nn.GRUCell(input_size, hidden_size, bias)

    nn.init.xavier_uniform_(m.weight_ih)
    nn.init.orthogonal_(m.weight_hh)

    if bias:
        nn.init.zeros_(m.bias_ih)
        nn.init.zeros_(m.bias_hh)

    return m


class MLP_generator(nn.Module):
    def __init__(self, in_feature):
        super(MLP_generator, self).__init__()
        self.decoder = nn.Sequential(
            nn.Linear(in_feature, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 3 * 28 * 28),
            nn.ReLU()
        )
        self.final_activation = nn.Tanh()

    def forward(self, x):
        x = self.decoder(x)
        x = self.final_activation(x)
        # Optionally, you can reshape the output to (Batch, Channels, Height, Width)
        # x = x.view(-1, 3, 28, 28)
        return x
