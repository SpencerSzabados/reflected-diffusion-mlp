"""
    Script for visualizing the score vector filed of a trained dsm diffusion
    model over point data. Returns a sequence of plots of the score as it
    transports points from p_T to p_0.

    Example launch command:
    CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL mpiexec -n 1 python visulize_score_field.py --exps cmlp_anulus_ref_ism_s_w_1000N --data_dir /home/sszabados/datasets/checkerboard/anulus_checkerboard_density_dataset.npz --g_equiv False --loss ism --resume_checkpoint /home/sszabados/models/reflected-diffusion-mlp/exps/cmlp_anulus_ref_ism_s_w_1000N/checkpoint_100000.pth
"""

import os
import argparse
from model.utils import distribute_util
import torch.distributed as dist

import matplotlib.pyplot as plt
import numpy as np
import torch as th

from model.utils.point_dataset_loader import load_data
from model.utils.checkpoint_util import load_and_sync_parameters
from model.cmlp import CMLP
from model.cmlp_diffusion import NoiseScheduler

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
        loss="ism",        # ism
        hidden_layers=3,
        hidden_size=256,
        emb_size=128,
        div_num_samples=50,            # Number of samples used to estimate ism divergence term
        div_distribution="Rademacher", # Guassian, Rademacher
        eps=1e-5,
        num_steps=1000,  
        sigma_min=0.001,
        sigma_max=5.0,
        lr=1e-4,
        weight_decay=0.0,
        weight_lambda=False,
        ema=0.994,
        ema_interval=1000,
        lr_anneal_steps=0,
        scale_output=False,
        score_scale=False,
        score_scale_interval=1000,
        global_batch_size=10000,
        global_sample_size=100000,
        batch_size=-1,
        log_interval=2000,
        sample_interval=20000,
        save_interval=100000,
        training_steps=1000000,
        resume_training=True,
        resume_checkpoint="",
        user_id='dummy',
        slurm_id='-1',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


def main():
    """
    Model initilization and training loop.

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
    logger.configure(dir=outdir, log_suffix="visulize")

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

    logger.log("Creating model and noise scheduler...")  

    model = CMLP(hidden_layers=args.hidden_layers, 
                hidden_size=args.hidden_size,
                emb_size=args.emb_size,
                scale_output=args.scale_output)
    
    noise_scheduler = NoiseScheduler(eps=args.eps,
                                     sigma_min=args.sigma_min,
                                     sigma_max=args.sigma_max,
                                     num_steps=args.num_steps)

    global_step, model = load_and_sync_parameters(args, model)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nSampling from model...\n")

    model.eval()
    with th.no_grad():
        sample_batch = next(iter(test_loader))[0]
        sample, noise = noise_scheduler._forward_reflected_noise(sample_batch, 1.0)
        timesteps = th.linspace(1.0, args.eps, args.num_steps+1)[:-1]
        
        for i, t in enumerate(tqdm(timesteps)):
            sample = sample.to(distribute_util.dev())
            sigma = noise_scheduler.sigma(t).to(distribute_util.dev())
            score = model(sample, sigma).to(distribute_util.dev())

            drift = noise_scheduler.get_drift(model, sample, sigma)

            sample = noise_scheduler.step(drift, t, sample)
        
            # Plot the score field every 20th step.
            if i%10 == 0:
                # # Create quiver plot of score vectors
                # sample_cpu = sample.detach().cpu().numpy()
                # score_cpu = 0.1*score.detach().cpu().numpy()
                # plt.figure(figsize=(8, 8))
                # plt.quiver(
                #     sample_cpu[:, 0], sample_cpu[:, 1], # Positions
                #     score_cpu[:, 0], score_cpu[:, 1],   # Score vectors
                #     angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.6
                # )
                # plt.axis('equal')
                # plt.grid(True)

                # # Create flow plot of score vectors
                # Define a grid over the plot area
                x_min, x_max = -1.5, 1.5
                y_min, y_max = -1.5, 1.5
                xx, yy = np.meshgrid(
                    np.linspace(x_min, x_max, 200),
                    np.linspace(y_min, y_max, 200)
                )

                # Flatten grid for model input
                grid_points = np.vstack([xx.ravel(), yy.ravel()]).T
                grid_points_tensor = th.tensor(grid_points, dtype=th.float32).to(distribute_util.dev())

                # Compute scores at grid points
                drift_grid = noise_scheduler.get_drift(model, grid_points_tensor, sigma).cpu().numpy()
                scores_grid = model(grid_points_tensor, sigma).cpu().numpy()

                # Reshape scores for plotting
                u = drift_grid[:, 0].reshape(xx.shape)
                v = drift_grid[:, 1].reshape(yy.shape)

                # Plot streamlines
                plt.figure(figsize=(8, 8))
                plt.streamplot(xx, yy, u, v, density=2.0, color='darkblue', linewidth=1)
                plt.axis('equal')
                plt.grid(True)
                
                plt.savefig(f"tmp/cmlp_score_sample_{i}.png", transparent=True)
                plt.close()

        else:
            raise NotImplementedError(f"Invalid value for diff_type.")


if __name__ == "__main__":
    main()