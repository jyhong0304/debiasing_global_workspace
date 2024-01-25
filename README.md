# Learning Decomposable Debiased Representations within Debiasing Global Workspace

This is the official implementation of Debiasing Global Workspace (DGW).

## Requirments
```python
pip install -r requirements.txt
```

## Datasets
Please check the repo of [Learning Debiased Represntations via Disentangled Feature Augmentation (LFA)](https://github.com/kakaoenterprise/Learning-Debiased-Disentangled).
You can download all dataests.

## Usage

Best test accuracy (%) comparsion between LFA and DGW (ours).

___


### Colored MNIST - 0.5pct
- Performance: 65.7 (LFA) VS ***70.1*** (DGW)
- LFA Training Command:

```python
python train.py --dataset cmnist --exp=cmnist_0.5_LFA --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --seed=0 --train_lfa --wandb --data_dir [YOUR_DATA_PATH] 
```

- LFA Test Command:

```python
python test.py --pretrained_path [LFA_MODEL_PATH] --test_lfa --dataset cmnist --exp=cmnist_0.5_LFA --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --seed=0 --data_dir [YOUR_DATA_PATH]
```


- DGW Training Command:

```python
python train.py --dataset cmnist --exp=cmnist_0.5_DGW --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=10 --lambda_swap_align=10 --lr_decay_step=10000 --lr_gamma=0.8 --lr_cct=0.001 --num_steps=50000 --use_lr_decay --rep_alpha=0.5 --seed=0 --n_concepts=10 --n_iters=2 --lambda_ent=0.01 --train_dgw --wandb --data_dir [YOUR_DATA_PATH]
```

- DGW Test Command:

```python
python test.py --pretrained_path [DGW_MODEL_PATH] --test_dgw --dataset cmnist --exp=cmnist_0.5_DGW --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=10 --lambda_swap_align=10 --lr_decay_step=10000 --lr_gamma=0.8 --lr_cct=0.001 --num_steps=50000 --use_lr_decay --rep_alpha=0.5 --seed=0 --n_concepts=10 --n_iters=2 --lambda_ent=0.01 --data_dir [YOUR_DATA_PATH]
```

[//]: # ()
[//]: # (- 1.0pct: 77.1 &#40;LDD&#41; VS ***80.8*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py)

[//]: # (--dataset)

[//]: # (cmnist)

[//]: # (--exp=cmnist_1.0_DuoCCT)

[//]: # (--lr=0.01)

[//]: # (--percent=1pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=10)

[//]: # (--lambda_swap_align=10)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.8)

[//]: # (--lr_cct=0.001)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # (- 2.0pct: 85.2 &#40;LDD&#41; VS ***86.8*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py)

[//]: # (--dataset)

[//]: # (cmnist)

[//]: # (--exp=cmnist_2.0_DuoCCT)

[//]: # (--lr=0.01)

[//]: # (--percent=2pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=10)

[//]: # (--lambda_swap_align=10)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.8)

[//]: # (--lr_cct=0.01)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # (- 5.0pct: 87.7 &#40;LDD&#41; VS ***88.7*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py)

[//]: # (--dataset)

[//]: # (cmnist)

[//]: # (--exp=cmnist_5.0_DuoCCT)

[//]: # (--lr=0.02)

[//]: # (--percent=5pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=10)

[//]: # (--lambda_swap_align=10)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.8)

[//]: # (--lr_cct=0.02)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### Corrupted CIFAR10)

[//]: # ()
[//]: # (- 0.5pct: 27.7 &#40;LDD&#41; VS ***33.1*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py )

[//]: # (--dataset)

[//]: # (cifar10c)

[//]: # (--exp=cifar10c_0.5_DuoCCT)

[//]: # (--lr=0.001)

[//]: # (--percent=0.5pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=1)

[//]: # (--lambda_swap_align=1)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.5)

[//]: # (--lr_cct=0.0001)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # (- 1.0pct: 31.5 &#40;LDD&#41; VS ***33.9*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py )

[//]: # (--dataset)

[//]: # (cifar10c)

[//]: # (--exp=cifar10c_1.0_DuoCCT)

[//]: # (--lr=0.001)

[//]: # (--percent=1pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=1)

[//]: # (--lambda_swap_align=1)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.5)

[//]: # (--lr_cct=0.0001)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # (- 2.0pct: 41.7 &#40;LDD&#41; VS ***44.1*** &#40;DGW&#41; )

[//]: # (```python)

[//]: # (python train.py)

[//]: # (--dataset)

[//]: # (cifar10c)

[//]: # (--exp=cifar10c_2.0_DuoCCT)

[//]: # (--lr=0.001)

[//]: # (--percent=2pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=5)

[//]: # (--lambda_swap_align=5)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.5)

[//]: # (--lr_cct=0.0001)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--n_iters=2)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # (- 5.0pct: 50.7 &#40;LDD&#41; VS ***51.4*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py)

[//]: # (--dataset)

[//]: # (cifar10c)

[//]: # (--exp=cifar10c_5.0_DuoCCT)

[//]: # (--lr=0.001)

[//]: # (--percent=5pct)

[//]: # (--curr_step=10000)

[//]: # (--lambda_swap=1.)

[//]: # (--lambda_dis_align=1)

[//]: # (--lambda_swap_align=1)

[//]: # (--lr_decay_step=10000)

[//]: # (--lr_gamma=0.5)

[//]: # (--lr_cct=0.0001)

[//]: # (--num_steps=50000)

[//]: # (--use_lr_decay)

[//]: # (--rep_alpha=0.5)

[//]: # (--seed=0)

[//]: # (--n_concepts=10)

[//]: # (--n_iters=2)

[//]: # (--lambda_ent=0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)

[//]: # ()
[//]: # ()
[//]: # (### BFFHQ)

[//]: # ()
[//]: # (- 0.5pct: 60.8 &#40;LDD&#41; VS ***63.4*** &#40;DGW&#41;)

[//]: # (```python)

[//]: # (python train.py )

[//]: # (--dataset)

[//]: # (bffhq)

[//]: # (--exp=bffhq_0.5_DGW)

[//]: # (--lr = 0.0002)

[//]: # (--percent = 0.5pct)

[//]: # (--lambda_swap = 0.1)

[//]: # (--curr_step = 10000)

[//]: # (--use_lr_decay)

[//]: # (--lr_decay_step = 10000)

[//]: # (--lambda_dis_align)

[//]: # (2.)

[//]: # (--lambda_swap_align)

[//]: # (2.)

[//]: # (--dataset)

[//]: # (bffhq)

[//]: # (--lr_cct = 0.0002)

[//]: # (--rep_alpha = 0.5)

[//]: # (--n_concepts = 20)

[//]: # (--lambda_ent = 0.01)

[//]: # (--train_dcct)

[//]: # (--tensorboard)

[//]: # (--wandb)

[//]: # (--seed = 0)

[//]: # (--data_dir)

[//]: # (DATA_PATH)

[//]: # (```)
