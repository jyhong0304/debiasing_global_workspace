# Learning Decomposable and Debiased Representations via Attribute-Centric Information Bottlenecks

This is the official implementation of ***Debiasing Global Workspace (DGW)***.

## Abstract

Biased attributes, spuriously correlated with target labels in a dataset, can problematically lead to neural networks
that learn improper shortcuts for classifications and limit their capabilities for out-of-distribution (OOD)
generalization. Although many debiasing approaches have been proposed to ensure correct predictions from biased
datasets, few studies have considered learning latent embedding consisting of intrinsic and biased attributes that
contribute to improved performance and explain how the model pays attention to attributes. In this paper, we propose a
novel debiasing framework, ***Debiasing Global Workspace***, introducing attention-based information bottlenecks for
learning compositional representations of attributes without defining specific bias types. Based on our observation that
learning shape-centric representation helps robust performance on OOD datasets, we adopt those abilities to learn robust
and generalizable representations of decomposable latent embeddings corresponding to intrinsic and biasing attributes.
We conduct comprehensive evaluations on biased datasets, along with both quantitative and qualitative analyses, to
showcase our approach's efficacy in attribute-centric representation learning and its ability to differentiate between
intrinsic and bias-related features.

## Requirments

```python
conda create --name py38DGW python=3.8
conda activate py38DGW
pip install -r requirements.txt
```

## Datasets

Please check the repo
of [Learning Debiased Represntations via Disentangled Feature Augmentation (LFA)](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).
You can download all dataests via the link.

## Usage

- For ```[vanilla, lfa, dgw]```, check script files in the folder ```scripts``` to execute models.
You can execute ```[vanilla, lfa, dgw]```.
- For ```[ReBias, LfF]```, we re-implemented them based on their repos. Please check ```dev``` branch in our repo to see our implementations of the two baselines.

## Our Pretrained Models

You can download our pretrained models of DGW via the following links to check their test accuracies in our paper.

- [Colored MNIST](https://www.dropbox.com/scl/fo/facnuc58ird1tjwpf89u9/h?rlkey=tcr7v7ceuofqjhdeplutg4j7u&dl=0)
- [Corrupted CIFAR-10](https://www.dropbox.com/scl/fo/8md77wo1dpa1olrnqiwp8/h?rlkey=t4qugp13dtfty2yzbcnc3v0qa&dl=0)
- [BFFHQ](https://www.dropbox.com/scl/fo/psapn37flcy8e37gtn0hk/h?rlkey=im8gtvhqt1adux017l68h77oq&dl=0)

## Acknowledgement
Our source codes are based on:
- [Concept-Centric Transformers [WACV2023]](https://github.com/jyhong0304/concept_centric_transformers): We implemented our method referring to this repo.
- [Learning Debiased Represntations via Disentangled Feature Augmentation (LFA) [NeurIPS2021]](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled): We implemented our method based on this repo and compare our model with this as one of our baselines. 
- [Learning De-biased Representations with Biased Representations (ReBias) [ICML2020]](https://github.com/clovaai/rebias): This is one of our baselines.
- [Learning from Failure: De-biasing Classifier from Biased Classifier (LfF) [NeurIPS2020]](https://github.com/alinlab/LfF): This is one of our baselines.
