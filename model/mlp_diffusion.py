"""
    File contains noise scheduler for mlp diffusion model.

    Implementation based on (https://github.com/tanelp/tiny-diffusion/blob/master/ddpm.py)
"""

import torch as th
import torch.nn.functional as F


class NoiseScheduler():
    """
    Noise scheduler for MLP diffusion model. Constructs noise cdf based on given paramters 
    for the ddpm discrete diffusion setting. 
    """
    def __init__(self,
                 diff_type="ddpm",
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.num_timesteps = num_timesteps

        if beta_schedule == "linear":
            self.betas = th.linspace(
                beta_start, beta_end, num_timesteps, dtype=th.float32)
        elif beta_schedule == "quadratic":
            self.betas = th.linspace(
                beta_start ** 0.5, beta_end ** 0.5, num_timesteps, dtype=th.float32) ** 2

        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = th.cumprod(self.alphas, axis=0)
        self.alphas_cumprod_prev = F.pad(
            self.alphas_cumprod[:-1], (1, 0), value=1.)

        # required for self.add_noise
        self.sqrt_alphas_cumprod = self.alphas_cumprod ** 0.5
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5

        # required for reconstruct_x0
        self.sqrt_inv_alphas_cumprod = th.sqrt(1 / self.alphas_cumprod)
        self.sqrt_inv_alphas_cumprod_minus_one = th.sqrt(
            1 / self.alphas_cumprod - 1)

        # required for q_posterior
        self.posterior_mean_coef1 = self.betas * th.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * th.sqrt(self.alphas) / (1. - self.alphas_cumprod)

    def reconstruct_x0(self, x_t, t, noise):
        assert(x_t.device == noise.device)

        s1 = self.sqrt_inv_alphas_cumprod[t]
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t]
        s1 = s1.reshape(-1, 1).to(device=x_t.device)
        s2 = s2.reshape(-1, 1).to(device=x_t.device)

        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        assert(x_0.device == x_t.device)

        s1 = self.posterior_mean_coef1[t]
        s2 = self.posterior_mean_coef2[t]
        s1 = s1.reshape(-1, 1).to(device=x_t.device)
        s2 = s2.reshape(-1, 1).to(device=x_t.device)
        mu = s1 * x_0 + s2 * x_t

        return mu

    def get_variance(self, t):
        if t == 0:
            variance = th.tensor(1e-20, dtype=th.float32)
        else:
            variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)

        return variance
    
    def get_score_from_noise(self, pred_noise, t):
        padding = th.ones_like(pred_noise).to(device=pred_noise.device)
        sigmas = (self.get_variance(t)**0.5)*padding
        pred_score = -pred_noise/sigmas

        return pred_score

    def step(self, model_output, t, sample):
        assert(model_output.device == sample.device)

        pred_original_sample = self.reconstruct_x0(sample, t, model_output)
        pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)

        variance = 0
        if t > 0:
            noise = th.randn_like(model_output)
            variance = (self.get_variance(t) ** 0.5) * noise

        pred_prev_sample = pred_prev_sample + variance

        return pred_prev_sample

    def add_noise(self, x_start, x_noise, t):
        assert(x_start.device == x_noise.device)

        s1 = self.sqrt_alphas_cumprod[t]
        s2 = self.sqrt_one_minus_alphas_cumprod[t]

        s1 = s1.reshape(-1, 1)
        s2 = s2.reshape(-1, 1)

        return s1 * x_start + s2 * x_noise

    def __len__(self):
        return self.num_timesteps