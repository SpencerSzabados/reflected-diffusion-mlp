import torch as th
import torch.nn.functional as F
import numpy as np


class Euclidean:
    def __init__(self, 
                dim=2,
            ):
        
        self.dim = dim

    def projv(self, v, x):
        """
            Project v onto tangent plane T_x.
        """
        return v

    def reflect(self, x):
        """
            Reflects points along the inwards normal direction back into the interior
            of the disc.
        """
        return x
    
    def sample(self, batch_size):
        """
            Samples ponits from Guassian distribution in Euclidean space. 
            This samples from the limiting distribution of the disc under the heat kerenl.
        """
        x = th.randn((batch_size, self.dim))
        
        return x

    