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
You can download all dataests.

## Usage

Check script files in the folder ```scripts``` to execute models.
You can execute ```[vanilla, lfa, dgw]```.

## Acknowledgement
Our source codes are based on:
- [Learning Debiased Represntations via Disentangled Feature Augmentation (LFA) [NeurIPS21]](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled): We implemented our method based on this repo and compare our model with this as one of our baselines. 
- [Concept-Centric Transformers [WACV2023]](https://github.com/jyhong0304/concept_centric_transformers): We implemented our method referring to this repo.
