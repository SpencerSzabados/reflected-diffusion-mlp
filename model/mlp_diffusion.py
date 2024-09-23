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
                 pred_type="eps",
                 num_timesteps=1000,
                 beta_start=0.0001,
                 beta_end=0.02,
                 beta_schedule="linear"):

        self.diff_type = diff_type
        self.pred_type = pred_type
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

        s1 = self.sqrt_inv_alphas_cumprod[t].reshape(-1, 1).to(device=x_t.device)
        s2 = self.sqrt_inv_alphas_cumprod_minus_one[t].reshape(-1, 1).to(device=x_t.device)

        return s1 * x_t - s2 * noise

    def q_posterior(self, x_0, x_t, t):
        assert(x_0.device == x_t.device)

        s1 = self.posterior_mean_coef1[t].reshape(-1, 1).to(device=x_t.device)
        s2 = self.posterior_mean_coef2[t].reshape(-1, 1).to(device=x_t.device)
    
        mu = s1 * x_0 + s2 * x_t

        return mu
    
    def get_variance(self, t):
        if t == 0:
            variance = th.tensor(1e-20, dtype=th.float32)
        else:
            variance = self.betas[t] * (1. - self.alphas_cumprod_prev[t]) / (1. - self.alphas_cumprod[t])
        variance = variance.clip(1e-20)

        return variance
    
    def get_score_from_pred(self, model_output, x_t, t):
        """
        This function converts a noise or x_0 prediction to an estimate of the score at time t.
        This estimate is only valid  for diff_type=="ddpm". (or approximatly valid if for 
        sufficient time steps taken with small step size for diff_type=="ref")

        Params:
            pred_pred (Torch.tensor): Model output. Predicted noise that was added to x_0 to 
                get x_t in the forwards process.

        Returns:
            pred_score (Torch.tensor): Estimate of the score function at time t for input x_t.
        """
        padding = th.ones_like(model_output).to(device=model_output.device)
        pred_score = th.zeros_like(model_output)

        if self.diff_type == "ddpm":
            if self.pred_type == "eps":
                sigmas = self.sqrt_one_minus_alphas_cumprod[t]*padding
                pred_score = -model_output/(sigmas+1e-20)

            elif self.pred_type == "x":
                pred_score = -(x_t - self.alphas_cumprod[t]*model_output)/(self.sqrt_one_minus_alphas_cumprod**2)

            elif self.pred_type == "s":
                sigmas = self.sqrt_one_minus_alphas_cumprod[t]*padding
                pred_score = -model_output/(sigmas+1e-20)

            else:
                raise NotImplementedError
            
        elif self.diff_type == "ref":
            if self.pred_type == "x":
                if isinstance(t, th.Tensor):
                    t = t[0] 

                if t == 0:
                    step_size = self.betas[1]-self.betas[0]
                else:
                    step_size = self.betas[t]-self.betas[t-1]

                variance = (step_size*t + 1e-20)*padding
                pred_score = -(x_t-model_output)/variance
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError

        return pred_score


    # TODO: The current _compute_collisions function uses hard coded boundaries.
    #       Implement one that loads the boundaries from a config file or dataloder.
    @th.no_grad()
    def _compute_collisions(self, x_t):
        """
        Determines whether each point in x_t lies within the annulus defined by r_in and r_out.

        Args:
            x_t (torch.Tensor): A tensor of shape [batch_size, 2] representing 2D points.

        Returns:
            torch.Tensor: A boolean tensor of shape [batch_size], where True indicates the point
                        is within the annulus, and False otherwise.
        """
        r_in = 0.25
        r_out = 1.0

        distances = th.linalg.norm(x_t, ord=2, dim=1)
        
        # Determine if distances are within [r_in, r_out]
        free_idx = (distances >= r_in) & (distances <= r_out)
        coll_idx = ~free_idx

        return coll_idx, free_idx

    @th.no_grad()
    def _forward_reflected_noise(self, x_start, t):
        # Perform Metropolis-Hastings random walk 

        # TODO: change the way t is passed to this function.
        if isinstance(t, th.Tensor):
            t = t[0] 

        if t == 0:
            step_size = self.betas[1]-self.betas[0]
        else:
            step_size = self.betas[t]-self.betas[t-1]
        step_size = th.sqrt(step_size) # TODO: DEBUG using linear (fixed) step size requires a larger number of sampling steps

        noise = th.randn_like(x_start)
        x_noisy = x_start
        previous_noise = noise.clone()
        previous_x_noisy = x_noisy.clone()

        # Perform accept-reject based on local geometry
        for k in range(t):
            coll_idx, free_idx = self._compute_collisions(x_noisy)  
            
            # If there are collisions, revert the changes for collided indices
            if len(coll_idx) > 0:
                noise[coll_idx] = previous_noise[coll_idx]
                x_noisy[coll_idx] = previous_x_noisy[coll_idx]

            # Update the previous_x_noisy for non-colliding indices
            if len(free_idx) > 0:
                previous_noise[free_idx] = noise[free_idx]
                previous_x_noisy[free_idx] = x_noisy[free_idx]

            # Resample noise for all trajectories
            noise_sample = th.randn_like(x_start)
            noise = noise+step_size*noise_sample
            x_noisy = x_noisy+step_size*noise_sample

        # TODO: Debug - incremental savinging of generated x_noisy
        #     import matplotlib.pyplot as plt
        #     frame = x_noisy.detach().cpu().numpy()
        #     plt.figure(figsize=(4, 4))
        #     plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
        #     plt.axis('off')
        #     plt.savefig(f"tmp/sample_{k}.png", transparent=True)
        #     plt.close()
        # exit()

        return x_noisy, noise
    
    @th.no_grad()
    def _sample_forwards_reflected_noise(self, x_start):
        # Perform Metropolis-Hastings random walk 
        step_size = self.betas[1]-self.betas[0]
        step_size = th.sqrt(step_size) # TODO: DEBUG using linear (fixed) step size requires a larger number of sampling steps

        noise = th.randn_like(x_start)
        x_noisy = x_start

        previous_noise = noise.clone()
        previous_x_noisy = x_noisy.clone()

        # Perform accept-reject based on local geometry
        for k in range(self.num_timesteps):
            coll_idx, free_idx = self._compute_collisions(x_noisy)  
            
            # If there are collisions, revert the changes for collided indices
            if len(coll_idx) > 0:
                noise[coll_idx] = previous_noise[coll_idx]
                x_noisy[coll_idx] = previous_x_noisy[coll_idx]

            # Update the previous_x_noisy for non-colliding indices
            if len(free_idx) > 0:
                previous_noise[free_idx] = noise[free_idx]
                previous_x_noisy[free_idx] = x_noisy[free_idx]

            # Resample noise for all trajectories
            noise_sample = th.randn_like(x_start)
            noise = noise+step_size*noise_sample
            x_noisy = x_noisy+step_size*noise_sample

        # TODO: Debug - incremental savinging of generated x_noisy
        #     import matplotlib.pyplot as plt
        #     frame = x_noisy.detach().cpu().numpy()
        #     plt.figure(figsize=(4, 4))
        #     plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
        #     plt.axis('off')
        #     plt.savefig(f"tmp/x_noise_sample_{k}.png", transparent=True)
        #     plt.close()
        # exit()
     
        return x_noisy, noise
    
    def step(self, model_output, t, sample):
        assert(model_output.device == sample.device)

        if self.diff_type == "ddpm":
            if self.pred_type == "eps":
                pred_original_sample = self.reconstruct_x0(sample, t, model_output)
                pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
            elif self.pred_type == "x":
                pred_prev_sample =  self.q_posterior(model_output, sample, t)
            elif self.pred_type == "s":
                pred_prev_sample = sample + 0.5*self.get_variance(t)*model_output
            else:
                raise NotImplementedError(f"Must select valid self.pred_type.")
            
            variance = 0
            if t > 0:
                noise = th.randn_like(model_output)
                variance = (self.get_variance(t) ** 0.5) * noise
            pred_prev_sample = pred_prev_sample + variance

        elif self.diff_type == "ref":
            if self.pred_type == "eps":
                pred_original_sample = self.reconstruct_x0(sample, t, model_output)
                pred_prev_sample = self.q_posterior(pred_original_sample, sample, t)
            elif self.pred_type == "x":
                pred_prev_sample = model_output
            else:
                raise NotImplementedError(f"Must select valid self.pred_type.")
            
            noise = 0
            if t > 0:
                noisy, noise = self._forward_reflected_noise(pred_prev_sample, t)
            pred_prev_sample = pred_prev_sample + noise

        # TODO: Debug - incremental savinging of generated x_noisy
        # import matplotlib.pyplot as plt
        # frame = pred_prev_sample.detach().cpu().numpy()
        # plt.figure(figsize=(4, 4))
        # plt.scatter(frame[:, 0], frame[:, 1], alpha=0.5, s=1)
        # plt.axis('off')
        # plt.savefig(f"tmp/x_denoise_sample_{t}.png", transparent=True)
        # plt.close()
        # exit()

        return pred_prev_sample

    def add_noise(self, x_start, noise, t):
        assert(x_start.device == noise.device)

        if self.diff_type == "ddpm":
            s1 = self.sqrt_alphas_cumprod[t].reshape(-1, 1)
            s2 = self.sqrt_one_minus_alphas_cumprod[t].reshape(-1, 1)
            x_noisy =  s1 * x_start + s2 * noise
        elif self.diff_type == "ref":
            x_noisy, noise = self._forward_reflected_noise(x_start, t)
        else:
            raise NotImplementedError(f"Must select valid self.diff_type.")

        return x_noisy

    def __len__(self):
        return self.num_timesteps