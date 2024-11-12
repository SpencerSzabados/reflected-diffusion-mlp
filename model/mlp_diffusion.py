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
                 eps=1e-5,
                 sigma_min=0.001,
                 sigma_max=0.6,
                 num_steps=1000):

        self.eps = eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.dt = (sigma_max-sigma_min)/num_steps

        # Diffusion schedule 
    
    def sigma(self, t):
        t = th.rand((1,))*(1 - self.eps) + self.eps
        sigma = self.sigma_min*(self.sigma_max/self.sigma_min)**t

        return sigma

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
        margin = 0.05

        distances = th.linalg.norm(x_t, ord=2, dim=1)
        
        # Determine if distances are within [r_in, r_out]
        free_idx = (distances >= r_in-margin) & (distances <= r_out+margin)
        coll_idx = ~free_idx

        return coll_idx, free_idx

    @th.no_grad()
    def _forward_reflected_noise(self, x_start, t):
        # Perform Metropolis-Hastings random walk 

        # Get time paramterization
        sigma = self.sigma(t)
        # Compute number of time steps to run given sigma
        N = th.round((sigma/self.sigma_max)*self.num_steps).long()
        # Init paramters
        noise = th.zeros_like(x_start)
        x_noisy = x_start
        previous_noise = noise.clone()
        previous_x_noisy = x_noisy.clone()

        # Perform accept-reject based on local geometry
        for _ in range(N):
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
            noise = noise + th.sqrt(th.tensor(self.dt))*noise_sample
            x_noisy = x_noisy + th.sqrt(th.tensor(self.dt))*noise_sample

        return x_noisy, noise
    
    def step(self, model_output, t, sample):
        assert(model_output.device == sample.device)

        pred_prev_sample = sample + self.dt*model_output
        
        noise = 0
        noisy = 0
        if t > 0:
            noisy, noise = self._forward_reflected_noise(pred_prev_sample, t)
        pred_prev_sample = pred_prev_sample 

        return pred_prev_sample

    def add_noise(self, x_start, t):
        x_noisy, noise = self._forward_reflected_noise(x_start, t)

        return x_noisy

    def __len__(self):
        return self.num_timesteps