import torch as th
import torch.nn.functional as F
import numpy as np


class Disc:
    def __init__(self, 
                 radius=1.0, 
                 dim=2
            ):
        
        self.radius = radius
        self.dim = dim

        assert dim == 2 

    def projv(self, v, x):
        """
            Project v onto tangent plane T_x of disc.
        """
        return v

    def reflect(self, x):
        """
            Reflects points along the inwards normal direction back into the interior
            of the disc.
        """
        norm = x.norm(dim=1, keepdim=True)
        mask = norm > self.radius

        x_reflected = x.clone()
        x_reflected[mask] = x_reflected[mask] / norm[mask] * (2 * self.radius - norm[mask])
        
        return x_reflected
    
    def sample(self, batch_size):
        """
            Uniformly samples points on the disc. 
            This samples from the limiting distribution of the disc under the heat kerenl.
        """
        angles = th.rand(batch_size) * 2 * np.pi
        radii = th.sqrt(self.radius*th.rand(batch_size))
        x = th.stack([radii * th.cos(angles), 
                      radii * th.sin(angles)], dim=1)
        
        return x

    