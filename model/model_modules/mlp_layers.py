"""
    File contains layer definitions used in constructing mlp diffusion model.

    Implementation based on (https://github.com/tanelp/tiny-diffusion/blob/master/ddpm.py)
"""


import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def rot_fn(points, angle, k):
    """
    Rotates a batch of [x, y] points around (0, 0) by k*angle. This is used to perform frame 
    averaging for the C_{k*angle} group during training to condition the model to be invariant. 

    Parameters:
        points (torch.Tensor): A tensor of shape [batch_size, 2] containing [x, y] points.
        angle (torch.float): The base angle in radians that everything will be rotated by.
        k (Int): Rotation multiplier.

    Retruns: 
        (torch.Tensor): batch of points [batch_size, 2] rotated around (0,0) by k*angle.
    """
    # Compute the rotation matrix
    cos_angle = np.cos(k*angle)
    sin_angle = np.sin(k*angle)
    rotation_matrix = th.tensor([[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=points.dtype, device=points.device)
    
    # Apply the rotation matrix to the batch of points
    return th.matmul(points, rotation_matrix)


def inv_rot_fn(points, angle, k):
    """
    Rotates a batch of [x, y] points around (0, 0) by -k*angle (inverse of the rotation).
    This is used to rever the operation of rot_fn during the frame averaging computation 
    for the C_{k*angle} group during training to condition the model to be invariant. 

    Parameters:
        points (torch.Tensor): A tensor of shape [batch_size, 2] containing [x, y] points.
        angle (torch.float): The base angle in radians that everything will be rotated by.
        k (Int): Rotation multiplier.

    Retruns: 
        (torch.Tensor): batch of points [batch_size, 2] rotated around (0,0) by -k*angle.
    """
    # Inverse rotation is equivalent to rotating in the opposite direction by (4 - k) * 90 degrees
    return rot_fn(points, angle, -k)
    


class SinusoidalEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: th.Tensor):
        x = x * self.scale
        half_size = self.size // 2
        emb = th.log(th.Tensor([10000.0])) / (half_size - 1)
        emb = th.exp(-emb * th.arange(half_size))
        emb = x.unsqueeze(-1) * emb.unsqueeze(0).to(device=x.device)
        emb = th.cat((th.sin(emb), th.cos(emb)), dim=-1)
        return emb

    def __len__(self):
        return self.size


class LinearEmbedding(nn.Module):
    def __init__(self, size: int, scale: float = 1.0):
        super().__init__()
        self.size = size
        self.scale = scale

    def forward(self, x: th.Tensor):
        x = x / self.size * self.scale
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class LearnableEmbedding(nn.Module):
    def __init__(self, size: int):
        super().__init__()
        self.size = size
        self.linear = nn.Linear(1, size)

    def forward(self, x: th.Tensor):
        return self.linear(x.unsqueeze(-1).float() / self.size)

    def __len__(self):
        return self.size


class IdentityEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor):
        return x.unsqueeze(-1)

    def __len__(self):
        return 1


class ZeroEmbedding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: th.Tensor):
        return x.unsqueeze(-1) * 0

    def __len__(self):
        return 1


class PositionalEmbedding(nn.Module):
    def __init__(self, size: int, type: str, **kwargs):
        super().__init__()

        if type == "sinusoidal":
            self.layer = SinusoidalEmbedding(size, **kwargs)
        elif type == "linear":
            self.layer = LinearEmbedding(size, **kwargs)
        elif type == "learnable":
            self.layer = LearnableEmbedding(size)
        elif type == "zero":
            self.layer = ZeroEmbedding()
        elif type == "identity":
            self.layer = IdentityEmbedding()
        else:
            raise ValueError(f"Unknown positional embedding type: {type}")

    def forward(self, x: th.Tensor):
        return self.layer(x)


class MLPBlock(nn.Module):
    def __init__(self, size: int):
        super().__init__()

        self.ff = nn.Linear(size, size)
        self.act = nn.GELU()

    def forward(self, x: th.Tensor):
        return x + self.act(self.ff(x))
