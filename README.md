# Learning Decomposable Debiased Representations within Debiasing Global Workspace

This is the official implementation of Debiasing Global Workspace.

## Requirments
```python
pip install -r requirements.txt
```

## Datasets
Please check the repo of [Learning Debiased Represntations (LDD)](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).
You can download all dataests.

## Usage

Best test accuracy (%) comparsion between LDD and DGW (ours).

### Colored MNIST

- 0.5pct: 68.2 (LDD) VS ***70.7*** (DGW)
```python
python train.py
--dataset
cmnist
--exp=cmnist_0.5_DuoCCT
--lr=0.01
--percent=0.5pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=10
--lambda_swap_align=10
--lr_decay_step=10000
--lr_gamma=0.8
--lr_cct=0.001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 1.0pct: 77.1 (LDD) VS ***80.8*** (DGW)
```python
python train.py
--dataset
cmnist
--exp=cmnist_1.0_DuoCCT
--lr=0.01
--percent=1pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=10
--lambda_swap_align=10
--lr_decay_step=10000
--lr_gamma=0.8
--lr_cct=0.001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 2.0pct: 85.2 (LDD) VS ***86.8*** (DGW)
```python
python train.py
--dataset
cmnist
--exp=cmnist_2.0_DuoCCT
--lr=0.01
--percent=2pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=10
--lambda_swap_align=10
--lr_decay_step=10000
--lr_gamma=0.8
--lr_cct=0.01
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 5.0pct: 87.7 (LDD) VS ***88.7*** (DGW)
```python
python train.py
--dataset
cmnist
--exp=cmnist_5.0_DuoCCT
--lr=0.02
--percent=5pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=10
--lambda_swap_align=10
--lr_decay_step=10000
--lr_gamma=0.8
--lr_cct=0.02
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```


### Corrupted CIFAR10

- 0.5pct: 27.7 (LDD) VS ***33.1*** (DGW)
```python
python train.py 
--dataset
cifar10c
--exp=cifar10c_0.5_DuoCCT
--lr=0.001
--percent=0.5pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=1
--lambda_swap_align=1
--lr_decay_step=10000
--lr_gamma=0.5
--lr_cct=0.0001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 1.0pct: 31.5 (LDD) VS ***33.9*** (DGW)
```python
python train.py 
--dataset
cifar10c
--exp=cifar10c_1.0_DuoCCT
--lr=0.001
--percent=1pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=1
--lambda_swap_align=1
--lr_decay_step=10000
--lr_gamma=0.5
--lr_cct=0.0001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 2.0pct: 41.7 (LDD) VS ***44.1*** (DGW) 
```python
python train.py
--dataset
cifar10c
--exp=cifar10c_2.0_DuoCCT
--lr=0.001
--percent=2pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=5
--lambda_swap_align=5
--lr_decay_step=10000
--lr_gamma=0.5
--lr_cct=0.0001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--n_iters=2
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```

- 5.0pct: 50.7 (LDD) VS ***51.4*** (DGW)
```python
python train.py
--dataset
cifar10c
--exp=cifar10c_5.0_DuoCCT
--lr=0.001
--percent=5pct
--curr_step=10000
--lambda_swap=1.
--lambda_dis_align=1
--lambda_swap_align=1
--lr_decay_step=10000
--lr_gamma=0.5
--lr_cct=0.0001
--num_steps=50000
--use_lr_decay
--rep_alpha=0.5
--seed=0
--n_concepts=10
--n_iters=2
--lambda_ent=0.01
--train_dcct
--tensorboard
--wandb
--data_dir
DATA_PATH
```


### BFFHQ

- 0.5pct: 60.8 (LDD) VS ***63.4*** (DGW)
```python
python train.py 
--dataset
bffhq
--exp=bffhq_0.5_DGW
--lr = 0.0002
--percent = 0.5pct
--lambda_swap = 0.1
--curr_step = 10000
--use_lr_decay
--lr_decay_step = 10000
--lambda_dis_align
2.
--lambda_swap_align
2.
--dataset
bffhq
--lr_cct = 0.0002
--rep_alpha = 0.5
--n_concepts = 20
--lambda_ent = 0.01
--train_dcct
--tensorboard
--wandb
--seed = 0
--data_dir
DATA_PATH
```
