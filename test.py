import numpy as np
import torch
import random
from learner import Learner
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Learning Decomposable Representation within a Debiasing Global Workspace')

    # training
    parser.add_argument("--batch_size", help="batch_size", default=256, type=int)
    parser.add_argument("--lr", help='learning rate', default=1e-3, type=float)
    parser.add_argument("--weight_decay", help='weight_decay', default=0.0, type=float)
    parser.add_argument("--momentum", help='momentum', default=0.9, type=float)
    parser.add_argument("--num_workers", help="workers number", default=16, type=int)
    parser.add_argument("--exp", help='experiment name', default='Test', type=str)
    parser.add_argument("--device", help="cuda or cpu", default='cuda', type=str)
    parser.add_argument("--num_steps", help="# of iterations", default=500 * 100, type=int)
    parser.add_argument("--target_attr_idx", help="target_attr_idx", default=0, type=int)
    parser.add_argument("--bias_attr_idx", help="bias_attr_idx", default=1, type=int)
    parser.add_argument("--dataset", help="data to train, [cmnist, cifar10, bffhq]", default='cmnist', type=str)
    parser.add_argument("--percent", help="percentage of conflict", default="1pct", type=str)
    parser.add_argument("--use_lr_decay", action='store_true', help="whether to use learning rate decay")
    parser.add_argument("--lr_decay_step", help="learning rate decay steps", type=int, default=10000)
    parser.add_argument("--q", help="GCE parameter q", type=float, default=0.7)
    parser.add_argument("--lr_gamma", help="lr gamma", type=float, default=0.1)
    parser.add_argument("--lambda_dis_align", help="lambda_dis in Eq.2", type=float, default=1.0)
    parser.add_argument("--lambda_swap_align", help="lambda_swap_b in Eq.3", type=float, default=1.0)
    parser.add_argument("--lambda_swap", help="lambda swap (lambda_swap in Eq.4)", type=float, default=1.0)
    parser.add_argument("--ema_alpha", help="use weight mul", type=float, default=0.7)
    parser.add_argument("--curr_step", help="curriculum steps", type=int, default=0)
    parser.add_argument("--use_type0", action='store_true', help="whether to use type 0 CIFAR10C")
    parser.add_argument("--use_type1", action='store_true', help="whether to use type 1 CIFAR10C")
    parser.add_argument("--use_resnet20", help="Use Resnet20",
                        action="store_true")  # ResNet 20 was used in Learning From Failure CifarC10 (We used ResNet18 in our paper)
    parser.add_argument("--model", help="which network, [MLP, ResNet18, ResNet20, ResNet50]", default='MLP', type=str)

    # logging
    parser.add_argument("--log_dir", help='path for loading data', default='./log', type=str)
    parser.add_argument("--data_dir", help='path for saving models & logs', default='dataset', type=str)
    parser.add_argument("--valid_freq", help='frequency to evaluate on valid/test set', default=500, type=int)
    parser.add_argument("--log_freq", help='frequency to log on tensorboard', default=500, type=int)
    parser.add_argument("--save_freq", help='frequency to save model checkpoint', default=1000, type=int)
    parser.add_argument("--wandb", action="store_true", help="whether to use wandb")
    parser.add_argument("--tensorboard", action="store_true", help="whether to use tensorboard")

    # experiment
    parser.add_argument("--pretrained_path", help="path for pretrained model", type=str)
    parser.add_argument("--test_lfa", action="store_true", help="whether to test LFA method (NeurIPS21)")

    # JYH: Add new arguments
    parser.add_argument("--test_dgw", action="store_true", help="whether to test Debiasing Global Workspace (Ours)")
    parser.add_argument("--rep_alpha", help="the ratio of representations using GWS", type=float, default=0.7)
    parser.add_argument('--seed', default=1, type=int, help='random seed')
    parser.add_argument("--n_concepts", help='number of concepts', default=10, type=int)
    parser.add_argument("--n_iters", help='number of iterations for slots', default=3, type=int)
    parser.add_argument("--lr_cct", help="learning rate for CCT", type=float, default=1e-3)
    parser.add_argument("--lambda_ent", help="hyperparam for entropy", type=float, default=0.1)

    args = parser.parse_args()

    # init learner
    learner = Learner(args)

    # actual training
    print('Official Pytorch Code of "Learning Decomposable Representation within a Debiasing Global Workspace"')
    print('Test starts ...')

    if args.test_lfa:
        learner.test_lfa(args)
    elif args.test_dgw:
        learner.test_dgw(args)
    else:
        print('choose one of the two options ... (LFA, DGW)')
        import sys

        sys.exit(0)
