# Cifar10c
# 0.5pct
python train.py --dataset cifar10c --exp=cifar10c_0.5_lfa-seed:0 --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_0.5_lfa-seed:1 --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_0.5_lfa-seed:2 --lr=0.0005 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/


# 1pct
CUDA_VISIBLE_DEVICES=1 python train.py --dataset cifar10c --exp=cifar10c_1_lfa-seed:0 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=2 python train.py --dataset cifar10c --exp=cifar10c_1_lfa-seed:1 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
CUDA_VISIBLE_DEVICES=3 python train.py --dataset cifar10c --exp=cifar10c_1_lfa-seed:2 --lr=0.001 --percent=1pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/

# 2pct
python train.py --dataset cifar10c --exp=cifar10c_2_lfa-seed:0 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_2_lfa-seed:1 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_2_lfa-seed:2 --lr=0.001 --percent=2pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=5 --lambda_swap_align=5 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/

# 5pct
python train.py --dataset cifar10c --exp=cifar10c_5_lfa-seed:0 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_5_lfa-seed:1 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cifar10c --exp=cifar10c_5_lfa-seed:2 --lr=0.001 --percent=5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=1 --lambda_swap_align=1 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/
