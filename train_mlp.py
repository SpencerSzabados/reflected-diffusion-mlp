"""
    Script for training a mlp diffusion model on point data.

    Example launch command:
    CUDA_VISIBLE_DEVICES=1 NCCL_P2P_LEVEL=NVL mpiexec -n 1 python train_mlp.py --exps mlp_ism --data_dir /home/sszabados/datasets/checkerboard/radial_checkerboard_density_dataset.npz --g_equiv False --loss ism
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


def obtain_slurm_ckpt_dir():
    SLURM_JOB_ID = os.environ['SLURM_JOB_ID']
    USER = os.environ['USER']
    ckpt_dir=f'/checkpoint/{USER}/{SLURM_JOB_ID}'
    return ckpt_dir


def _parse_resume_step_from_filename(filename):
        """
        Parse filenames of the form path/to/checkpoint_NNNNNN.pt, where NNNNNN is the
        checkpoint's number of steps.
        """
        split = filename.split("checkpoint_")
        if len(split) == 2:
            split1 = split[-1].split(".")[0]
            try:
                return int(split1)
            except ValueError:
                return 0
        else:
            return 0


def find_resume_checkpoint_aux(ckpt_dir):
    # search the lastest checkpoint in ckpt_dir
    if not os.path.exists(ckpt_dir):
        logger.warn(f'CANNOT FIND: {ckpt_dir}')
        return None

    last_resume_step = -1
    for filename in os.listdir(ckpt_dir):
        if ('checkpoint' in filename) and ('.pth' in filename):
            resume_step = _parse_resume_step_from_filename(filename)
            if resume_step > last_resume_step:
                last_resume_step = resume_step

    if last_resume_step > -1:
        return os.path.join(ckpt_dir, f'checkpoint_{last_resume_step}.pth')
    else:
        return None


def find_resume_checkpoint(args):
    try:
        ckpt_dir = os.path.join(args.working_dir, f"exps/{args.exps}/")
        ckpt_file_path = find_resume_checkpoint_aux(ckpt_dir)
        logger.info(f'ckpt_file_path is set to {ckpt_file_path}')
        return ckpt_file_path

    except KeyError:
        logger.info('Cannot detect checkpoint dir.')
        return None


def load_and_sync_parameters(args, model, ema_model, optimizer):
    """
    Loads model, ema, and optimizer states from checkpoints.
    """
    resume_step = 0
    if args.resume_training:
        resume_checkpoint = find_resume_checkpoint(args) or find_resume_checkpoint_aux(args.resume_checkpoint)
        if resume_checkpoint is not None:
            logger.log(f"Found checkpoint: {resume_checkpoint}")
            resume_step = _parse_resume_step_from_filename(resume_checkpoint)
            resume_checkpoint = th.load(resume_checkpoint, map_location=distribute_util.dev())
            if dist.get_rank() == 0:
                logger.log(f"Loading model state...")
                model.load_state_dict(resume_checkpoint["model"])
                logger.log(f"Loading ema_model state...")
                ema_model.load_state_dict(resume_checkpoint["ema_model"])
                logger.log(f"Loading optimizer state...")
                optimizer.load_state_dict(resume_checkpoint["optimizer"])
            
    model = model.to(device=distribute_util.dev())
    ema_model = ema_model.to(device=distribute_util.dev())

    return resume_step, model, ema_model, optimizer


def update_ema(model, ema_model, ema):
    """
    Updates model weights using exponential moving average.

    Paramters:
        model (nn.Module): Current model being trained
        ema_model (nn.Module): No gradient copy of model
        ema (th.Tensor): EMA decay value
    """
    with th.no_grad():
        model_params = list(model.parameters())
        ema_params = list(ema_model.parameters())
        
        for model_param, ema_param in zip(model_params, ema_params):
            ema_param.data.mul_(ema).add_(model_param.data, alpha=(1 - ema))


def hutchinson_divergence(model, noisy, timesteps, noise_scheduler, num_samples=10):
    """
    Estimates the divergence term using Hutchinson's estimator.
    
    Params:
        model (Class): The model that predicts noise (which is then converted to score).
        noisy (th.Tensor): The noisy inputs at timestep t.
        timesteps (th.Tensor): The time value associated with the noisy inputs.
        noise_scheduler (Class): The noise scheduler used to convert predicted noise to score.
        num_samples (Int): Number of Gussian samples to used in estimating divergence.
    Returns:
        divergence_term: The estimated divergence term.
    """
    divergence = 0
    for _ in range(num_samples):
        # Sample random noise for Hutchinson's estimator
        z = th.randn_like(noisy)
        # Predict the noise (epsilon) using the model
        pred_noise = model(noisy, timesteps)
        # Convert predicted noise to score using the noise scheduler
        pred_score = noise_scheduler.get_score_from_noise(pred_noise, timesteps[0])
        # Compute the gradient of the predicted score w.r.t. the noisy input
        grad_pred_score = th.autograd.grad(
                                outputs=pred_score, 
                                inputs=noisy, 
                                grad_outputs=z, 
                                create_graph=True, 
                                retain_graph=True
                            )[0]
        
        # Compute the Hutchinson divergence estimate
        divergence += (grad_pred_score * z).sum(dim=1).mean()
    
    return divergence/num_samples


# Function to return either mse_loss or the implicit score matching loss
def get_loss_fn(args):
    """
    Retruns an anonymous loss function based on selected args.

    Params:
        args (dict): Dictionary list of launch paramters .
        model (Class): Model to be trained.
        noise_scheduler (Class): The noise scheduler used to convert predicted noise to score.
    Returns:
        loss_fn (Function): Function that computes the forwards loss.
    """
    if args.loss == 'ism':
        if args.pred_type == "eps":
            def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch):
                # Ensure noisy requires gradient for autograd to compute the divergence
                noisy.requires_grad_(True)
                pred_noise = model(noisy, timesteps)
                pred_score = noise_scheduler.get_score_from_noise(pred_noise, timesteps[0])
                # 1/2 ||s_theta(x)||_2^2 (regularization term)
                norm_term = 0.5 * th.sum(pred_score**2, dim=1).mean()
                # div(s_theta(x)) (Hutchinson divergence estimator)
                divergence_term = hutchinson_divergence(model, noisy, timesteps, noise_scheduler)
                # Total ISM loss
                return norm_term + divergence_term
            return loss_fn

    elif args.loss == "dsm":
        # Define the loss function based on g_equiv and pred_type
        if args.pred_type == 'eps':
            if args.g_equiv and args.g_input == "C5":
                # Precompute the logic for rotational MSE loss for 'eps'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch):
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2 * th.pi / 5.0, k)
                        noise_pred = inv_rot_fn(model(noisy_rot, timesteps), 2 * th.pi / 5.0, k)
                        loss += F.mse_loss(noise_pred, noise)
                    return loss / 5.0  # Average over the 5 rotations
            else:
                # Standard MSE loss for 'eps'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch):
                    noise_pred = model(noisy, timesteps)
                    return F.mse_loss(noise_pred, noise)

        elif args.pred_type == 'x':
            if args.g_equiv and args.g_input == "C5":
                # Precompute the logic for rotational MSE loss for 'x'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch):
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2 * th.pi / 5.0, k)
                        x_pred = inv_rot_fn(model(noisy_rot, timesteps), 2 * th.pi / 5.0, k)
                        loss += F.mse_loss(x_pred, batch)
                    return loss / 5.0  # Average over the 5 rotations
            else:
                # Standard MSE loss for 'x'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch):
                    x_pred = model(noisy, timesteps)
                    return F.mse_loss(x_pred, batch)
                
        else:
            raise NotImplementedError(f"The option {args.loss} is not supported.")

        return loss_fn
    

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
    logger.configure(dir=outdir)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nJob started.")
    logger.log(f"Experiment: {args.exps}\n")

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

    dataloader, dataset = load_data(data_dir=args.data_dir, batch_size=batch_size)

    logger.log("Creating model and noise scheduler...")  
    model = MLP(hidden_layers=args.hidden_layers, 
                hidden_size=args.hidden_size,
                emb_size=args.emb_size,
                time_emb=args.time_emb,
                input_emb=args.input_emb)

    
    noise_scheduler = NoiseScheduler(num_timesteps=args.num_timesteps,
                                     beta_schedule=args.beta_schedule)

    
    # Get the loss function (ISM or MSE) based on args.loss
    loss_fn = get_loss_fn(args)

    # Set up the optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)

    # Create a deepcopy of the model to store the EMA weights
    ema_model = deepcopy(model)
    # Disable gradient tracking for the EMA model
    for param in ema_model.parameters():
        param.requires_grad = False

    global_step, model, ema_model, optimizer = load_and_sync_parameters(args, model, ema_model, optimizer)

    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nTraining model...\n")
    while global_step < args.training_steps:
        model.train()
        for step, batch in enumerate(dataloader):
            batch = batch[0]
            noise = th.randn(batch.shape)
            timesteps = th.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            # put data on gpu
            batch = batch.to(distribute_util.dev())
            noise = noise.to(distribute_util.dev())
            noisy = noisy.to(distribute_util.dev())
            timesteps = timesteps.to(distribute_util.dev())

            # Compute loss
            optimizer.zero_grad()
            # Compute the loss using the selected loss function
            loss = loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch)
            loss.backward()
            # nn.utils.clip_grad_norm_(model.parameters(), 1.0) # TODO: removed this line to speed up convergence.
            optimizer.step()
            
            if args.ema > 0:
                # Update the EMA model after the optimizer step
                update_ema(model, ema_model, args.ema)

            global_step += 1

            if global_step % args.log_interval == 0 and global_step > 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.log(f"[{time}]"+"-"*20)
                logger.log(f"Step: {global_step}")
                logger.log(f"Loss: {loss.detach().item()}")

            if global_step % args.sample_interval == 0 and global_step > 0:
                logger.log("Logging (saving) sample plot...")
                # generate data with the model to later visualize the learning process
                model.eval()
                with th.no_grad():
                    sample = th.randn(sample_size, 2)
                    timesteps = list(range(len(noise_scheduler)))[::-1]

                    for i, t in enumerate(tqdm(timesteps)):
                        sample = sample.to(distribute_util.dev())
                        t = th.from_numpy(np.repeat(t, sample_size)).long().to(distribute_util.dev())
                        residual = model(sample, t).to(distribute_util.dev())
                        sample = noise_scheduler.step(residual, t[0], sample)

                    frame = sample.detach().cpu().numpy()

                    logger.log("Saving plot...")
                    plt.figure(figsize=(8, 8))
                    plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
                    plt.axis('off')
                    plt.savefig(f"{outdir}/images/sample_{global_step}.png", transparent=True)
                    plt.close()
                model.train()

            if global_step % args.save_interval == 0 and global_step > 0:
                logger.log("Saving model...")
                checkpoint = {"global_step": global_step,
                              "model": model.state_dict(),
                              "ema_model": ema_model.state_dict(),
                              "optimizer": optimizer.state_dict()}
                th.save(checkpoint, f"{outdir}/checkpoint_{global_step}.pth")


if __name__ == "__main__":
    main()
