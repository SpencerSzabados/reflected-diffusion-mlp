import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SinActivation(nn.Module):
    def __init__(self):
        super(SinActivation, self).__init__()
        return
    def forward(self, x):
        return th.sin(x)


def get_timestep_embedding(timesteps, embedding_dim, dtype=th.float32):
    assert len(timesteps.shape) == 1
    timesteps *= 1000.0

    half_dim = embedding_dim // 2
    emb = np.log(10000) / (half_dim - 1)
    emb = (th.arange(half_dim, dtype=dtype, device=timesteps.device) * -emb).exp()
    emb = timesteps.to(dtype)[:, None] * emb[None, :]
    emb = th.cat([emb.sin(), emb.cos()], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, (0, 1))
    assert emb.shape == (timesteps.shape[0], embedding_dim)
    return emb


def create_network(in_dim, hidden_dim, out_dim, n_hidden, act):
    if act.lower() == "sin":
        act = SinActivation
    elif act.lower() == "silu":
        act = nn.SiLU
    elif act.lower() == "relu":
        act = nn.ReLU
    else:
        raise NotImplementedError

    if n_hidden == 0:
        return nn.Linear(in_dim, out_dim)
    network = [nn.Linear(in_dim, hidden_dim)]
    for _ in range(n_hidden - 1):
        network += [act(), nn.Linear(hidden_dim, hidden_dim)]
    network += [act(), nn.Linear(hidden_dim, out_dim)]
    return nn.Sequential(*network)


class CMLP(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 256, 
            emb_size: int = 128,
            hidden_layers: int = 5, 
            boundary_tol: float = 0.005,
            scale_output: bool = False,
        ):

        super().__init__()

        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.boundary_tol = boundary_tol
        self.scale_output = scale_output
        self.func = create_network(2 + 1 + emb_size, hidden_size, 2, hidden_layers, "relu")
        self.fn = nn.ReLU()

    def _compute_boundary_distance(self, x):
        """
        Determines the minimum distance from each point in x_t to the closer of 
        the two annulus boundaries.

        Args:
            x_t (torch.Tensor): A tensor of shape [batch_size, 2] representing 2D points.

        Returns:
            torch.Tensor: A tensor of shape [batch_size], containing the minimum distance
                        from each point to the closest boundary.
        """
        r_in = 0.25
        r_out = 1.0

        # Compute the Euclidean distance of each point from the origin
        distances = th.linalg.norm(x, ord=2, dim=1)

        # Compute the absolute distances to the inner and outer boundaries
        distance_to_inner = th.abs(distances - r_in)
        distance_to_outer = th.abs(distances - r_out)

        # Determine the minimum distance to the closest boundary
        min_distance = th.min(distance_to_inner, distance_to_outer)

        return min_distance

    def forward(self, x, sigma):
        # Ensure sigma is a tensor with shape [B, 1]
        batch_size = x.shape[0]
        sigma = sigma.view(1, 1).expand(batch_size, 1)

        # Optionally include sigma directly
        inp = [x, sigma]

        # Generate time embeddings (all identical in the batch)
        if self.emb_size > 0:
            sigma_flat = sigma.view(batch_size).clone()  # Shape [B]
            t_harmonics = get_timestep_embedding(sigma_flat, self.emb_size)
            inp.append(t_harmonics)

        # Concatenate inputs
        input = th.cat(inp, dim=-1)  # Shape [B, input_dim]

        # Pass through the network
        pred_score = self.func(input)

        boundary_dist = self._compute_boundary_distance(x)
        pred_score = th.min(th.ones_like(boundary_dist), self.fn(boundary_dist-self.boundary_tol)).reshape(-1, 1)*pred_score

        if self.scale_output:
            pred_score = pred_score / sigma
        
        return pred_score

