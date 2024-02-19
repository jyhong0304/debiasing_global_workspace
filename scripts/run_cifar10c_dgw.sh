# 0.5pct
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar10c --exp=cifar10c_0.5_DGW-seed:0 --lr=0.001 --percent=0.5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=0 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_0.5_DGW-seed:1 --lr=0.001 --percent=0.5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=1 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_0.5_DGW-seed:2 --lr=0.001 --percent=0.5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=2 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
#
## 1pct
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar10c --exp=cifar10c_1_DGW-seed:0 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=0 --n_concepts=6 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_1_DGW-seed:1 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=1 --n_concepts=6 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_1_DGW-seed:2 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=2 --n_concepts=6 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/

## 2pct
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar10c --exp=cifar10c_2_DGW-seed:0 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=0 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_2_DGW-seed:1 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=1 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_2_DGW-seed:2 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=2 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/

## 5pct
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar10c --exp=cifar10c_5_DGW-seed:0 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=0 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_5_DGW-seed:1 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=1 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_5_DGW-seed:2 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1. --lambda_dis_align=1 --lambda_swap_align=1 --lr_decay_step=10000 --lr_gamma=0.5 --lr_cct=0.0001 --num_steps=50000 --use_lr_decay --rep_alpha=0.2 --seed=2 --n_concepts=5 --n_iters=3 --dim_slots=16 --lambda_ent=0.01 --train_dgw --wandb --data_dir /data/jhong53/datasets/
