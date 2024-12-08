# Debiasing Global Workspace: A Cognitive Neural Framework for Learning Debiased and Interpretable Representations

This is the official implementation of ***Debiasing Global Workspace (DGW)***.
This work has been accepted to:
- (Short paper) [NeurIPS 2024 Workshop on Behavioral Machine Learning](https://openreview.net/forum?id=SoqFoRAofo)
- (Full paper) [NeurIPS 2024 Workshop on UniReps: 2nd Edition of the Workshop on Unifying Representations in Neural Models](https://openreview.net/forum?id=obiwUsWlki#discussion)


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
- [Concept-Centric Transformers [WACV2023]](https://github.com/jyhong0304/concept_centric_transformers): We implemented our method referring to this repo and used this as one of our baselines.
- [Learning Debiased Represntations via Disentangled Feature Augmentation (LFA) [NeurIPS2021]](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled): We implemented our method based on this repo and compare our model with this as one of our baselines. 
- [Learning De-biased Representations with Biased Representations (ReBias) [ICML2020]](https://github.com/clovaai/rebias): This is one of our baselines.
- [Learning from Failure: De-biasing Classifier from Biased Classifier (LfF) [NeurIPS2020]](https://github.com/alinlab/LfF): This is one of our baselines.
