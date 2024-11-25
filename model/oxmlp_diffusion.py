# diffusion.py

import numpy as np
import torch as th
import torch.nn.functional as F


class Sigma:
    """
        Diffusion coefficient function
            sigma(t) = sigma_min + t*(sigma_max-sigma_min)/T
    """
    def __init__(self, 
                 eps=1e-4,
                 sigma_min=0.001, 
                 sigma_max=5.0, 
                 T=1.0):
        self.eps = eps
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T

    def __call__(self, t):
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
    

class SDE:
    """
        SDE class for simulating the forwards heat kernel using VE-SDE schedule.
            sigma(t) = sigma_min*(sigma_max/sigma_min)^t
            dX_t = sqrt(2*sigma(t))dW_t
            var(X_t|X_0) = 2int_0^t sigma(t)ds
    """
    def __init__(self, sigma):
        self.sigma = sigma
        self.t0 = 0.0
        self.tf = sigma.T

    def rand_t(self, batch_size, device):
        # Sample t uniformly between sde.t0 + eps and sde.tf
        t = th.rand((1,), device=device) * (self.tf - self.t0 - self.sigma.eps) + self.t0 + self.sigma.eps  
        t = t*th.ones((batch_size,)).to(device)

        return t
        
    def drift(self, x, t):
        return th.zeros_like(x)

    def diffusion(self, t):
        sigma_t = self.sigma(t)
        diffusion_t = sigma_t * th.sqrt(th.tensor(2*(np.log(self.sigma.sigma_max) - np.log(self.sigma.sigma_min))))
                                                
        return diffusion_t
    
    def variance(self, t):
        sigma_t = self.sigma(t)

        return sigma_t

    def marginal_prob(self, x, t):
        mean = x
        std = self.sigma(t).view(-1,1)

        return mean, std

    def marginal_sample(self, x, t):
        mean, std = self.marginal_prob(x, t)
        z = th.randn_like(x)
        x_t = mean + std * z

        return x_t
    
    def prior_sample(self, batch_size):
        return th.randn((batch_size,2))*self.sigma.sigma_max

    def reparametrize_score_fn(self, model, std_trick=True, residual_trick=False):
        def score_fn(x, t):

            score = model(x, t)

            if std_trick:
                _, std = self.marginal_prob(th.zeros_like(x), t)
                score = score / std

            if residual_trick:
                fwd_drift = self.drift(x, t)
                sigma_t = self.sigma(t).view(-1, 1)
                residual = 2 * fwd_drift / sigma_t
                score = score + residual

            return score
        
        return score_fn

