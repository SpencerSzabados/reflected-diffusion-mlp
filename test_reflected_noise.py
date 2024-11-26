"""
    File contains script for testing the boundary condition constraines
    used in generating fowards reflected noise for the anulus_checkerboard_density
    dataset.

    Script is based on sample_mlp.py

    Example command:
    CUDA_VISIBLE_DEVICES=2 NCCL_P2P_LEVEL=NVL mpiexec -n 1 python test_reflected_noise.py --exps test_ref_noise --data_dir /home/sszabados/datasets/checkerboard/anulus_checkerboard_density_dataset.npz --diff_type ref --pred_type s --num_timesteps 100 --beta_start 0.0001 --beta_end 0.02
"""


import os
import argparse
from model.utils import distribute_util
import torch.distributed as dist

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from model.utils.point_dataset_loader import load_data
from model.mlp_diffusion import NoiseScheduler

from tqdm import tqdm
from model.utils import logger
from datetime import datetime


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():
    defaults = dict(
        exps="mlp",
        working_dir="",
        data_dir="",
        diff_type='ddpm', # ddpm, ref
        pred_type='eps',   # epx, x
        num_timesteps=50,
        step_size=-1.0,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        global_batch_size=10000,
        global_sample_size=100000,
        batch_size=-1,
        user_id='dummy',
        slurm_id='-1',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    """
    Initilization and sampling loop.

    Params:
        Args (Dict): Launch options are configured using create_argparser object which
            returns a dict of all configurable command line launch options.
    """
    args = create_argparser().parse_args()

    # print(args.user_id, args.slurm_id)
    if args.user_id != '-1':
        os.environ["SLURM_JOB_ID"] = args.slurm_id
        os.environ['USER'] = args.user_id

    if args.working_dir == "":
        args.working_dir = os.getcwd()

    outdir = os.path.join(args.working_dir, f"exps/{args.exps}")
    os.makedirs(outdir, exist_ok=True)
    os.makedirs(f"{outdir}/images/", exist_ok=True)

    distribute_util.setup_dist()
    logger.configure(dir=outdir, log_suffix="sampling")

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nJob started.")
    logger.log(f"Experiment: {args.exps}\n")
    
    # Log all args to the log file
    for arg in vars(args):
        value = getattr(args, arg)
        logger.log(f"{arg}: {value}")

    logger.log(f"Setting and loading constants...")
    global_step = 0

    logger.log("Creating data loader...")
    if args.batch_size == -1:
        batch_size = args.global_batch_size // dist.get_world_size()
        sample_size = args.global_sample_size // dist.get_world_size()
        if args.global_batch_size % dist.get_world_size() != 0:
            logger.log(
                f"warning, using smaller global_batch_size of {dist.get_world_size()*batch_size} instead of {args.global_batch_size}"
            )
    else:
        batch_size = args.batch_size

    train_loader, test_loader = load_data(data_dir=args.data_dir, train_batch_size=batch_size, test_batch_size=sample_size)

    logger.log("Creating noise scheduler...")  
    noise_scheduler = NoiseScheduler(diff_type=args.diff_type,
                                     pred_type=args.pred_type,
                                     beta_start=args.beta_start,
                                     beta_end=args.beta_end,
                                     step_size=args.step_size,
                                     num_timesteps=args.num_timesteps,
                                     beta_schedule=args.beta_schedule)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nSampling from model...\n")

    with th.no_grad():
        sample = next(iter(test_loader))[0]
        timesteps = list(range(len(noise_scheduler)))[::-1]
        
        for t in tqdm(range(args.num_timesteps)):
            noise = th.randn_like(sample)
            sample = noise_scheduler.add_noise(sample, noise, 1)

            # Save incremental frames
            frame = sample.detach().cpu().numpy()
            plt.figure(figsize=(8, 8))
            plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
            plt.axis('off')
            plt.savefig(f"{outdir}/images/{args.exps}_debug_ref_{args.num_timesteps}steps_xnoisy_sample_{t}.png", transparent=True)
            plt.close()

        
if __name__ == "__main__":
    main()
