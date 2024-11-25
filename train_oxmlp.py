"""
Script for training a mlp diffusion model on point data.

    Example launch command:
    CUDA_VISIBLE_DEVICES=2 NCCL_P2P_LEVEL=NVL mpiexec -n 1 python train_oxmlp.py --exps oxmlp_radial_cdif_ism_w_1000N_t --data_dir /home/sszabados/datasets/checkerboard/radial_checkerboard_density_dataset.npz --g_equiv False --weight_lambda True 
"""

import os
import argparse
from model.utils import distribute_util
import torch.distributed as dist

from model.utils.point_dataset_loader import load_data

import numpy as np
import torch as th
from copy import deepcopy
import torch.optim as optim
from model.oxmlp import CEmbedMLP, CMLP
from model.oxmlp_diffusion import SDE, Sigma
from model.manifolds.euclidean import Euclidean

import matplotlib.pyplot as plt
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
        loss="ism",        # ism
        input_dim=2,
        embedding_dim=128,
        hidden_dims=[512, 512, 512],
        output_dim=2,
        activation='gelu',
        manifold='disc',
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
        ema_interval=1,
        lr_anneal_steps=0,
        scale_output=False,
        global_batch_size=10000,
        global_sample_size=100000,
        batch_size=-1,
        log_interval=2000,
        sample_interval=20000,
        save_interval=100000,
        training_steps=1000000,
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


def _ism_loss_fn(model, sde, batch, weight_lambda=True, hutchinson_type="Rademacher", eps=1e-3):
    batch_size = batch.shape[0]
    device = batch.device

    # Sample t uniformly between sde.t0 + eps and sde.tf
    t = th.rand(batch_size, device=device) * (sde.tf - sde.t0 - eps) + sde.t0 + eps  # [batch_size]

    # Sample x_t from marginal distribution
    x_t = sde.marginal_sample(batch, t)

    # Compute score
    x_t.requires_grad_(True)
    score_fn = sde.reparametrize_score_fn(model)
    score = score_fn(x_t, t) 

    if hutchinson_type == "Rademacher":
        z = th.randint(0, 2, x_t.shape, device=device).float() * 2 - 1
    else:  
        z = th.randn_like(x_t)

    # Compute divergence estimate
    y = (score * z).sum()
    grad_y = th.autograd.grad(outputs=y, inputs=x_t, create_graph=True)[0]  
    divergence_estimate = (grad_y * z).sum(dim=1)  

    # Compute ||score||^2
    norm_score = (score**2).sum(dim=1)  

    # Compute losses
    loss = norm_score + 2.0*divergence_estimate 

    if weight_lambda:
        loss = (sde.diffusion(t)**2)*loss 

    loss = loss.mean()

    return loss


def hutchinson_divergence(score_fn, sde, noisy, t, num_samples=30, type="Guassian"):
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
    divergence = th.zeros(noisy.shape[0]).to(device=noisy.device)

    for _ in range(num_samples):
        # Sample random noise for Hutchinson's estimator
        if type == "Guassian":
            z = th.randn_like(noisy)
        elif type == "Rademacher":
            z = th.randint_like(noisy, low=0, high=2).float()*2 - 1.
        else:
            raise NotImplementedError(f"Only Guassian and Rademacher Type distributions are currently supported.")

        pred_score = score_fn(noisy, t)
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


def get_ism_loss_fn(score_fn, sde, weight_lambda=True, eps=1e-4, distribution="Rademacher", num_samples=20):
    def ism_loss(noisy, t):
        noisy.requires_grad_(True)

        pred_score = score_fn(noisy, t)
        # ||s_theta(x)||_2^2 (regularization term)
        norm_term = th.sum((pred_score**2), dim=1).mean()
        # div(s_theta(x)) (Hutchinson divergence estimator)
        divergence_term = hutchinson_divergence(score_fn, sde, noisy, t, num_samples=num_samples, type=distribution)
        # Total ISM loss
        if weight_lambda:
            loss_w = sde.diffusion(t)[0]**2
            loss = loss_w*(norm_term + 2.0*divergence_term) 
        else:
            loss = (norm_term + 2.0*divergence_term) 

        return loss
    return ism_loss


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
    running_avg_loss = 0.0
    loss_count = 0

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

    # Initialize model and manifold
    # manifold = DiscManifold(radius=1.0)
    manifold = Euclidean() # TODO: this was removed to test the function of the model for normal diffusion.
    
    # model = CEmbedMLP(
    #             input_dim=args.input_dim,
    #             embedding_dim=args.embedding_dim,
    #             hidden_dims=args.hidden_dims,
    #             output_dim=args.output_dim,
    #             activation=args.activation,
    #             manifold=manifold
    #         ).to(device=distribute_util.dev())
    
    model = CMLP(input_dim=args.input_dim,
                hidden_dims=args.hidden_dims,
                output_dim=args.output_dim,
                activation=args.activation,
                manifold=manifold
            ).to(device=distribute_util.dev())

    # Initialize SDE
    sigma = Sigma(eps=args.eps,
                  sigma_min=args.sigma_min, 
                  sigma_max=args.sigma_max, 
                  T=1.0)
    sde = SDE(sigma)
    # Get score function
    score_fn = sde.reparametrize_score_fn(model)
    # Get loss function
    loss_fn = get_ism_loss_fn(
                score_fn=score_fn,
                sde=sde,
                weight_lambda=args.weight_lambda,
                distribution=args.div_distribution,
                eps=args.eps,
            )

    logger.log("Creating optimizer...")
    # Set up the optimizer
    optimizer = th.optim.AdamW(model.parameters(), lr=args.lr)
    # Create a deepcopy of the model to store the EMA weights
    ema_model = deepcopy(model)
    # Disable gradient tracking for the EMA model
    for param in ema_model.parameters():
        param.requires_grad = False

    # Training loop
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.log(f"[{time}]"+"="*20+"\nTraining model...\n")
    while global_step < args.training_steps:
        model.train()
        for step, batch in enumerate(train_loader):
            batch = batch[0].to(distribute_util.dev())
            # Simulate forwards sde
            t = sde.rand_t(batch_size, device=distribute_util.dev())
            noisy = sde.marginal_sample(batch, t)
        
            # Compute loss using ISM loss function
            optimizer.zero_grad()
            loss = loss_fn(noisy, t)
            loss.backward()
            optimizer.step()

            if args.ema > 0 and global_step % args.ema_interval == 0:
                # Update the EMA model after the optimizer step
                update_ema(model, ema_model, args.ema)

            global_step += 1
            loss_count += 1
            running_avg_loss += (loss.detach().item() - running_avg_loss) / loss_count

            # If logging interval is reached log time, step number, and current batch loss.
            if global_step % args.log_interval == 0 and global_step > 0:
                time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                logger.log(f"[{time}]"+"-"*20)
                logger.log(f"Step: {global_step}")
                logger.log(f"Step loss: {loss.detach().item()}")
                logger.log(f"Avg loss: {running_avg_loss}")
                loss_count = 0
                running_avg_loss = 0.0

            # If sampling interval is reached generate and save incremental sample.
            if global_step % args.sample_interval == 0 and global_step > 0:
                logger.log("Logging (saving) sample plot...")
                # Generate data with the model to later visualize the learning process.
                model.eval()
                with th.no_grad():
                    score_fn = sde.reparametrize_score_fn(model)
                    # sample = manifold.sample(sample_size).to(distribute_util.dev())
                    sample = sde.prior_sample(sample_size).to(distribute_util.dev())
                    timesteps = th.linspace(1.0, args.eps, args.num_steps+1)[:-1]
                    for i, t in enumerate(tqdm(timesteps)):
                        t = t*th.ones((sample_size,)).to(distribute_util.dev())
                        score = score_fn(sample, t)
                        sample = sample + (args.sigma_max-args.sigma_min)/args.num_steps * score
                    
                        frame = sample.detach().cpu().numpy()

                        logger.log("Saving plot...")
                        plt.figure(figsize=(8, 8))
                        plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
                        plt.axis('off')
                        plt.savefig(f"{outdir}/images/sample_{global_step}.png", transparent=True)
                        plt.close()
                model.train()

if __name__ == "__main__":
    main()
