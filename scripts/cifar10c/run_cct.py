from copy import deepcopy
from multiprocessing import Process, Queue
from itertools import product
import sys, os
import numpy as np
import time
import argparse

sys.path.append(os.path.abspath("."))


def kwargs_to_cmd(kwargs, gpu_num):
    cmd = f"CUDA_VISIBLE_DEVICES={gpu_num} /data/jhong53/anaconda3/envs/py38KD/bin/python3 train.py "
    for flag, val in kwargs.items():
        cmd += f"--{flag}={val} "

    cmd += "--use_lr_decay "
    cmd += "--train_dcct "
    cmd += "--tensorboard "
    cmd += "--wandb "

    return cmd


def run_exp(gpu_num, in_queue):
    while not in_queue.empty():
        try:
            experiment = in_queue.get(timeout=3)
        except:
            return

        before = time.time()

        print(f"==> Starting experiment {kwargs_to_cmd(experiment, gpu_num)}")
        os.system(kwargs_to_cmd(experiment, gpu_num))

        with open("output.txt", "a+") as f:
            f.write(
                f"Finished experiment {experiment} in {str((time.time() - before) / 60.0)}."
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu-sets', default=0, type=lambda x: [a for a in x.split("|") if a])
    parser.add_argument('--data_dir', default='~/data', type=str)
    args = parser.parse_args()

    gpus = args.gpu_sets
    # seeds = list(range(args.seeds))
    seed = 0
    data_dir = args.data_dir
    experiments = []

    # '0.5pct'
    experiments.append({
        "exp": f"cifar10c_0.5_DuoCCT",
        "lr": 0.0005,
        "percent": '0.5pct',
        "curr_step": 10000,
        "lambda_swap": 1,
        "lambda_dis_align": 1,
        "lambda_swap_align": 1,
        "lr_decay_step": 10000,
        "lr_gamma": 0.5,
        "lr_cct": 0.0005,
        "rep_alpha": 0.5,
        "n_concepts": 20,
        "lambda_ent": 0.01,
        "data_dir": data_dir,
        "seed": seed
    })
    # '1pct'
    experiments.append({
        "exp": f"cifar10c_1_DuoCCT",
        "lr": 0.001,
        "percent": '1pct',
        "curr_step": 10000,
        "lambda_swap": 1,
        "lambda_dis_align": 5,
        "lambda_swap_align": 5,
        "lr_decay_step": 10000,
        "lr_gamma": 0.5,
        "lr_cct": 0.001,
        "rep_alpha": 0.5,
        "n_concepts": 20,
        "lambda_ent": 0.01,
        "data_dir": data_dir,
        "seed": seed
    })
    # '2pct'
    experiments.append({
        "exp": f"cifar10c_2_DuoCCT",
        "lr": 0.001,
        "percent": '2pct',
        "curr_step": 10000,
        "lambda_swap": 1,
        "lambda_dis_align": 5,
        "lambda_swap_align": 5,
        "lr_decay_step": 10000,
        "lr_gamma": 0.5,
        "lr_cct": 0.001,
        "rep_alpha": 0.5,
        "n_concepts": 20,
        "lambda_ent": 0.01,
        "data_dir": data_dir,
        "seed": seed
    })
    # '5pct'
    experiments.append({
        "exp": f"cifar10c_5_DuoCCT",
        "lr": 0.001,
        "percent": '5pct',
        "curr_step": 10000,
        "lambda_swap": 1,
        "lambda_dis_align": 1,
        "lambda_swap_align": 1,
        "lr_decay_step": 10000,
        "lr_gamma": 0.5,
        "lr_cct": 0.001,
        "rep_alpha": 0.5,
        "n_concepts": 20,
        "lambda_ent": 0.01,
        "data_dir": data_dir,
        "seed": seed
    })


    print(experiments)
    # input("Press any key to continue...")
    queue = Queue()

    for e in experiments:
        queue.put(e)

    processes = []
    for gpu in gpus:
        p = Process(target=run_exp, args=(gpu, queue))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()


if __name__ == "__main__":
    main()
