# CMNIST
# 0.5pct
python train.py --dataset cmnist --exp=cmnist_0.5_LFA-seed:0 --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_0.5_LFA-seed:1 --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_0.5_LFA-seed:2 --lr=0.01 --percent=0.5pct --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/

# 1pct
python train.py --dataset cmnist --exp=cmnist_1_LFA-seed:0 --lr=0.01 --percent=1pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_1_LFA-seed:1 --lr=0.01 --percent=1pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_1_LFA-seed:2 --lr=0.01 --percent=1pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/

# 2pct
python train.py --dataset cmnist --exp=cmnist_2_LFA-seed:0 --lr=0.01 --percent=2pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_2_LFA-seed:1 --lr=0.01 --percent=2pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_2_LFA-seed:2 --lr=0.01 --percent=2pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/

# 5pct
python train.py --dataset cmnist --exp=cmnist_5_LFA-seed:0 --lr=0.01 --percent=5pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=0 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_5_LFA-seed:1 --lr=0.01 --percent=5pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=1 --data_dir /data/jhong53/datasets/
python train.py --dataset cmnist --exp=cmnist_5_LFA-seed:2 --lr=0.01 --percent=5pct  --curr_step=10000 --lambda_swap=1 --lambda_dis_align=10 --lambda_swap_align=10 --use_lr_decay --lr_decay_step=10000 --lr_gamma=0.5 --train_lfa --wandb --seed=2 --data_dir /data/jhong53/datasets/