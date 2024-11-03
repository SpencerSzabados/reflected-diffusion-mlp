"""
    Script for sampling from a mlp diffusion model on point data.

    Example launch command:
    CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL mpiexec -n 1 python sample_mlp.py --exps mlp_anulus_ddpm_dsm_x --data_dir /home/sszabados/datasets/checkerboard/anulus_checkerboard_density_dataset.npz --g_equiv False --diff_type ddpm --loss dsm --pred_type x --resume_checkpoint /home/sszabados/models/reflected-diffusion-mlp/exps/mlp_anulus_ddpm_dsm_x/checkpoint_500000.pth
"""

import os
import argparse
from model.utils import distribute_util
import torch.distributed as dist

import matplotlib.pyplot as plt
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import blobfile as bf

from model.utils.point_dataset_loader import load_data
from model.utils.checkpoint_util import load_and_sync_parameters, save_checkpoint
from model.mlp import MLP
from model.mlp_diffusion import NoiseScheduler
from model.model_modules.mlp_layers import rot_fn, inv_rot_fn

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
        g_equiv=False,     # False, True
        g_input=None,      # None, C5
        eqv_reg=False,     # False, True 
        loss="dsm",        # dsm, ism
        diff_type='pfode', # pfode, ddpm, ref
        pred_type='eps',   # epx, x
        hidden_layers=3,
        hidden_size=128,
        emb_size=128,
        time_emb="sinusoidal",
        input_emb="sinusoidal",
        num_timesteps=50,
        step_size=None,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        ema=0.994,
        lr_anneal_steps=0,
        global_batch_size=10000,
        global_sample_size=100000,
        batch_size=-1,
        log_interval=2000,
        sample_interval=20000,
        save_interval=100000,
        training_steps=1000000,
        resume_training=True,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    """
    Model initilization and sampling loop.

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
    logger.log(f"Experiment: {args.exps}")
    logger.log(f"diff_type: {args.diff_type}")
    logger.log(f"pred_type: {args.pred_type}")
    logger.log(f"loss: {args.loss}")
    logger.log(f"Number of timesteps: {args.num_timesteps}\n")

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

    logger.log("Creating model and noise scheduler...")  
    model = MLP(hidden_layers=args.hidden_layers, 
                hidden_size=args.hidden_size,
                emb_size=args.emb_size,
                time_emb=args.time_emb,
                input_emb=args.input_emb,
                diff_type=args.diff_type,
                pred_type=args.pred_type)

    
    noise_scheduler = NoiseScheduler(diff_type=args.diff_type,
                                     pred_type=args.pred_type,
                                     beta_start=args.beta_start,
                                     beta_end=args.beta_end,
                                     step_size=args.step_size,
                                     num_timesteps=args.num_timesteps,
                                     beta_schedule=args.beta_schedule)

    global_step, model = load_and_sync_parameters(args, model)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nSampling from model...\n")

    with th.no_grad():
        if args.diff_type == "ddpm":
            sample = th.randn(sample_size, 2)
            timesteps = list(range(len(noise_scheduler)))[::-1]
            for i, t in enumerate(tqdm(timesteps)):
                sample = sample.to(distribute_util.dev())
                t = th.from_numpy(np.repeat(t, sample_size)).long().to(distribute_util.dev())
                residual = model(sample, t).to(distribute_util.dev())
                sample = noise_scheduler.step(residual, t[0], sample)

        elif args.diff_type == "ref":
                sample_batch = next(iter(test_loader))[0]
                sample, noise = noise_scheduler._sample_forwards_reflected_noise(sample_batch)
                timesteps = list(range(len(noise_scheduler)))[::-1]
                
                for i, t in enumerate(tqdm(timesteps)):
                    sample = sample.to(distribute_util.dev())

                    # TODO: debug - remove. This was added to plot denosing sequence 
                    frame = sample.detach().cpu().numpy()
                    plt.figure(figsize=(8, 8))
                    plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
                    plt.axis('off')
                    plt.savefig(f"{outdir}/images/{args.exps}_debug_sample_{t}.png", transparent=True)
                    plt.close()
     
                    t = th.from_numpy(np.repeat(t, sample.shape[0])).long().to(distribute_util.dev())
                    residual = model(sample, t).to(distribute_util.dev())
                    sample = noise_scheduler.step(residual, t[0], sample) # TODO: Debug - maybe the time index is misaligned between the forwards and backwards pass due to passing t as the loop range when it should be t+1?

        else:
            raise NotImplementedError(f"Invalid value for diff_type.")

        frame = sample.detach().cpu().numpy()

        logger.log("Saving plot...")
        plt.figure(figsize=(8, 8))
        plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
        plt.axis('off')
        plt.savefig(f"{outdir}/images/{args.exps}_sample.png", transparent=True)
        plt.close()
        logger.log(f"Saved plot to {outdir}/images/{args.exps}_sample.png")

if __name__ == "__main__":
    main()
