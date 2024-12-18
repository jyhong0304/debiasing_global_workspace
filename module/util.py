''' Modified from https://github.com/alinlab/LfF/blob/master/module/util.py '''

from torch import nn
from module.resnet import resnet20
from module.mlp import *
from torchvision.models import resnet18
from .model_util import GlobalWorkspaceModel


def get_model(model_tag, num_classes, in_feature=7, embedding_dim=32, latent_dim=32, n_concepts=20, num_iterations=3):
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet20_OURS":
        model = resnet20(num_classes)
        model.fc = nn.Linear(128, num_classes)
        return model
    elif model_tag == "ResNet18":
        print('bringing no pretrained resnet18 ...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(512, num_classes)
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "mlp_DISENTANGLE":
        return MLP_DISENTANGLE(num_classes=num_classes)
    elif model_tag == 'resnet_DISENTANGLE':
        print('bringing no pretrained resnet18 disentangle...')
        model = resnet18(pretrained=False)
        model.fc = nn.Linear(1024, num_classes)
        return model
    # Add global workspace model
    elif model_tag == 'global_workspace':
        return GlobalWorkspaceModel(embedding_dim=embedding_dim, n_spatial_concepts=n_concepts, num_iterations=num_iterations, in_feature=in_feature, latent_dim=latent_dim)
    else:
        raise NotImplementedError
