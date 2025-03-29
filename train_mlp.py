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
        g_equiv=False,      # False, True
        g_input=None,       # None, C5
        eqv_reg=False,      # False, True 
        loss="dsm",         # dsm, ism
        diff_type='pfode',  # pfode, ddpm, ref
        pred_type='eps',    # eps, x, s
        div_num_samples=50, # Number of samples used to estimate ism divergence term
        div_distribution="Guassian", # Guassian, Rademacher
        hidden_layers=3,
        hidden_size=128,
        emb_size=128,
        time_emb="sinusoidal",
        input_emb="sinusoidal",
        num_timesteps=50,
        step_size=-1.0,     # -1, fixed int
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="linear",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        weight_lambda=False,
        clip_grad=1.0,
        ema=0.994,
        ema_interval=1,
        lr_anneal_steps=0,
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
        use_fp16=False,
        fp16_scale_growth=1e-3,
        user_id='dummy',
        slurm_id='-1',
    )
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


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


def hutchinson_divergence(args, model, noisy, timesteps, noise_scheduler, num_samples=50, type="Guassian"):
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
    noisy.requires_grad_(True)
    divergence = th.zeros(noisy.shape[0]).to(device=noisy.device)

    for _ in range(num_samples):
        # Sample random noise for Hutchinson's estimator
        if type == "Guassian":
            z = th.randn_like(noisy)
        elif type == "Rademacher":
            z = th.randint_like(noisy, low=0, high=2).float()*2 - 1.
        else:
            raise NotImplementedError(f"Only Guassian and Rademacher Type distributions are currently supported.")

        if args.pred_type == "eps":
            # Predict the noise (epsilon) using the model
            pred_noise = model(noisy, timesteps)
            # Convert predicted noise to score using the noise scheduler
            pred_score = noise_scheduler.get_score_from_noise(pred_noise, timesteps[0])

        elif args.pred_type == "s":
            pred_score = model(noisy, timesteps)

        else:
            raise ValueError(f"Invalid paramter value for {args.pred_type}.")
        
        sum_score = th.sum(pred_score*z)
        # Compute the gradient of the predicted score w.r.t. the noisy input
        grad_pred_score = th.autograd.grad(outputs=sum_score,
                                           inputs=noisy,
                                           create_graph=True,
                                           retain_graph=True  # Set to True if you need to compute higher-order derivatives
                                        )[0]
        grad_pred_score.detach()
        # Compute the Hutchinson divergence estimate for this sample
        divergence_sample = th.einsum('bi,bi->b', grad_pred_score, z)
        divergence += divergence_sample
    
    # Compute the average divergence estimate
    divergence_mean = (divergence/num_samples).mean()
    
    return divergence_mean


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
            def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                # Ensure noisy requires gradient for autograd to compute the divergence
                noisy.requires_grad_(True)
                pred_noise = model(noisy, timesteps)
                pred_score = noise_scheduler.get_score_from_noise(pred_noise, timesteps[0])
                # 1/2 ||s_theta(x)||_2^2 (regularization term)
                norm_term = 0.5*th.sum(pred_score**2, dim=1).mean()
                # div(s_theta(x)) (Hutchinson divergence estimator)
                divergence_term = hutchinson_divergence(args, model, noisy, timesteps, noise_scheduler, num_samples=args.div_num_samples, type=args.div_distribution)
                noisy.requires_grad_(False)
                # Total ISM loss
                return norm_term + divergence_term
            return loss_fn

        elif args.pred_type == "s": 
            if args.g_equiv and args.g_input == "C5":
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    # Ensure noisy requires gradient for autograd to compute the divergence
                    noisy.requires_grad_(True)
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2*th.pi/5.0, k)
                        pred_score = inv_rot_fn(model(noisy_rot, timesteps), 2*th.pi/5.0, k)
                        # 1/2 ||s_theta(x)||_2^2 (regularization term)
                        norm_term = 0.5*th.sum(score_w*pred_score**2, dim=1).mean()
                        # div(s_theta(x)) (Hutchinson divergence estimator)
                        divergence_term = score_w*hutchinson_divergence(args, model, noisy, timesteps, noise_scheduler, num_samples=args.div_num_samples, type=args.div_distribution)
                        # Total ISM loss
                        loss += norm_term + divergence_term
                    noisy.requires_grad_(False)
                    loss = loss_w*loss/5.0
                    return loss
                return loss_fn
            else:
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    # Ensure noisy requires gradient for autograd to compute the divergence
                    noisy.requires_grad_(True)
                    pred_score = model(noisy, timesteps)
                    # 1/2 ||s_theta(x)||_2^2 (regularization term)
                    norm_term = 0.5*th.sum(score_w*(pred_score**2), dim=1).mean()
                    # div(s_theta(x)) (Hutchinson divergence estimator)
                    divergence_term = score_w*hutchinson_divergence(args, model, noisy, timesteps, noise_scheduler, num_samples=args.div_num_samples, type=args.div_distribution)
                    noisy.requires_grad_(False)
                    # Total ISM loss
                    loss = loss_w*(norm_term + divergence_term) 
                    return loss 
                return loss_fn

    elif args.loss == "dsm":
        # Define the loss function based on g_equiv and pred_type
        if args.pred_type == 'eps': # Predicting epsilon added to x_0 to get x_t
            if args.g_equiv and args.g_input == "C5":
                # Precompute the logic for rotational MSE loss for 'eps'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2 * th.pi / 5.0, k)
                        noise_pred = inv_rot_fn(model(noisy_rot, timesteps), 2 * th.pi / 5.0, k)
                        loss += F.mse_loss(noise_pred, noise)
                    loss = loss/5.0
                    return loss  # Average over the 5 rotations
                return loss_fn
            
            else:
                # Standard MSE loss for 'eps'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    noise_pred = model(noisy, timesteps)
                    return F.mse_loss(noise_pred, noise)
                return loss_fn

        elif args.pred_type == 'x': # Predicting x_0 from x_t
            if args.g_equiv and args.g_input == "C5":
                # Precompute the logic for rotational MSE loss for 'x'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2*th.pi/5.0, k)
                        x_pred = inv_rot_fn(model(noisy_rot, timesteps), 2*th.pi/5.0, k)
                        loss += F.mse_loss(x_pred, batch)
                    loss = loss/5.0
                    return loss  # Average over the 5 rotations
                return loss_fn
            
            else:
                # Standard MSE loss for 'x'
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    x_pred = model(noisy, timesteps)
                    return F.mse_loss(x_pred, batch)
                return loss_fn
                
        elif args.pred_type == "s": # Predicting the score at time t
            if args.g_equiv and args.g_input == "C5":
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    loss = 0
                    for k in range(5):
                        noisy_rot = rot_fn(noisy, 2*th.pi/5.0, k)
                        pred_score = inv_rot_fn(model(noisy_rot, timesteps), 2*th.pi/5.0, k)
                        # Compute analytical score estimate 
                        score = noise_scheduler.get_score_from_pred(pred_score, noisy, batch, timesteps[0])
                        # Total loss
                        loss += F.mse_loss(pred_score, score)
                    loss = loss/5.0
                    return loss
                return loss_fn
            else:
                def loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=1.0, score_w=1.0):
                    pred_score = model(noisy, timesteps)
                    # Compute analytical score estimate 
                    score = noise_scheduler.get_score_from_pred(pred_score, noisy, batch, timesteps[0])
                    # Total loss
                    return F.mse_loss(pred_score, score)
                return loss_fn
                
        else:
            raise NotImplementedError(f"The option {args.loss} is not supported.")

    

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
    model = MLP(hidden_layers=args.hidden_layers, 
                hidden_size=args.hidden_size,
                emb_size=args.emb_size,
                time_emb=args.time_emb,
                input_emb=args.input_emb,
                diff_type=args.diff_type,
                pred_type=args.pred_type).to(distribute_util.dev())
    
    noise_scheduler = NoiseScheduler(diff_type=args.diff_type,
                                     pred_type=args.pred_type,
                                     beta_start=args.beta_start,
                                     beta_end=args.beta_end,
                                     step_size=args.step_size,
                                     num_timesteps=args.num_timesteps,
                                     beta_schedule=args.beta_schedule)
    
    logger.log("Creating optimizer...")
    # Get the loss function (ISM or MSE) based on args.loss
    loss_w = 1.0
    score_w = 1.0
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
        for step, batch in enumerate(train_loader):
            batch = batch[0]
            noise = th.randn(batch.shape)
            timesteps = th.randint(0, noise_scheduler.num_timesteps, (batch.shape[0],)).long()
            noisy = noise_scheduler.add_noise(batch, noise, timesteps)

            # put data on gpu
            batch = batch.to(distribute_util.dev())
            noise = noise.to(distribute_util.dev())
            noisy = noisy.to(distribute_util.dev())
            # timesteps = timesteps.to(distribute_util.dev())

            # Estimate score scale to normalize the scale between norm and div of score estimates.
            if args.score_scale and \
               args.pred_type == "ism" and \
               global_step % args.score_scale_interval == 0:
                with th.no_grad():
                    # Ensure noisy requires gradient for autograd to compute the divergence
                    noisy.requires_grad_(True)
                    pred_score = model(noisy, timesteps)
                    # 1/2 ||s_theta(x)||_2^2 (regularization term)
                    norm = 0.5*th.sum(pred_score**2, dim=1).mean()
                    # div(s_theta(x)) (Hutchinson divergence estimator)
                    div = hutchinson_divergence(args, model, noisy, timesteps, noise_scheduler, num_samples=args.div_num_samples, type=args.div_distribution)
                    # estimate score of score
                    score_w = -div/norm

            # Add loss weighting function loss_w=(t+1)
            if args.weight_lambda:
                with th.no_grad():
                    loss_w = (timesteps[0]+1.0).to(distribute_util.dev())

            # Compute loss
            optimizer.zero_grad()
            # Compute the loss using the selected loss function
            loss = loss_fn(noisy, model, noise_scheduler, timesteps, noise, batch, loss_w=loss_w, score_w=score_w)
            # Update gradients
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad) 
            optimizer.step()
            
            if args.ema > 0 and global_step % args.ema_interval == 0:
                # Update the EMA model after the optimizer step
                update_ema(model, ema_model, args.ema)

            global_step += 1

            # If logging interval is reached log time, step number, and current batch loss.
            if global_step % args.log_interval == 0 and global_step > 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.log(f"[{time}]"+"-"*20)
                logger.log(f"Time: {timesteps[0]}")
                logger.log(f"Step: {global_step}")
                logger.log(f"Loss: {loss.detach().item()}")

            # If sampling interval is reached generate and save incremental sample.
            if global_step % args.sample_interval == 0 and global_step > 0:
                logger.log("Logging (saving) sample plot...")
                # Generate data with the model to later visualize the learning process.
                # Sampling process changes based on the chosen parameterization.
                model.eval()
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
                                t = th.from_numpy(np.repeat(t, sample.shape[0])).long().to(distribute_util.dev())
                                residual = model(sample, t).to(distribute_util.dev())
                                sample = noise_scheduler.step(residual, t[0], sample)
                    
                    else:
                        raise NotImplementedError(f"Invalid value for diff_type.")

                    frame = sample.detach().cpu().numpy()

                    logger.log("Saving plot...")
                    plt.figure(figsize=(8, 8))
                    plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
                    plt.axis('off')
                    plt.savefig(f"{outdir}/images/sample_{global_step}.png", transparent=True)
                    plt.close()
                model.train()

            if global_step % args.save_interval == 0 and global_step > 0:
                save_checkpoint(outdir, model, ema_model, optimizer, global_step)
                

if __name__ == "__main__":
    main()
