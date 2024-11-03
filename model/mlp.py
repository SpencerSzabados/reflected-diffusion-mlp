"""
    A simple multi-layer perceptron model used for learning toy datasets and testing. 

    File implements group invariant/equivariant mlp architecture using the equivariant layers and blocks 
    defined in the model_modeules folder.

   Implementation is based on (https://github.com/tanelp/tiny-diffusion)
"""


import torch as th
import torch.nn as nn

from .model_modules.mlp_layers import MLPBlock, PositionalEmbedding

class MLP(nn.Module):
    def __init__(
            self, 
            hidden_size: int = 128, 
            hidden_layers: int = 3, 
            emb_size: int = 128,
            time_emb: str = "sinusoidal", 
            input_emb: str = "sinusoidal",
            diff_type: str = "ddpm",
            pred_type: str = "eps",
            boundary_tol: float = 0.005,
        ):

        super().__init__()

        self.diff_type = diff_type
        self.pred_type = pred_type
        self.boundary_tol = boundary_tol

        self.time_mlp = PositionalEmbedding(emb_size, time_emb)
        self.input_mlp1 = PositionalEmbedding(emb_size, input_emb, scale=25.0)
        self.input_mlp2 = PositionalEmbedding(emb_size, input_emb, scale=25.0)

        concat_size = len(self.time_mlp.layer) + len(self.input_mlp1.layer) + len(self.input_mlp2.layer)
        layers = [nn.Linear(concat_size, hidden_size), nn.GELU()]
        self.fn = nn.ReLU()

        for _ in range(hidden_layers):
            layers.append(MLPBlock(hidden_size))
        layers.append(nn.Linear(hidden_size, 2))

        self.joint_mlp = nn.Sequential(*layers)


    def _compute_boundary_distance(self, x):
        """
        Determines the minimum distance from each point in x_t to the closer of the two annulus boundaries.

        Args:
            x_t (torch.Tensor): A tensor of shape [batch_size, 2] representing 2D points.

        Returns:
            torch.Tensor: A tensor of shape [batch_size], containing the minimum distance from each point to the closest boundary.
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


    def forward(self, x_t, t):
        t = t.to(device=x_t.device)

        x1_emb = self.input_mlp1(x_t[:, 0])
        x2_emb = self.input_mlp2(x_t[:, 1])
        t_emb = self.time_mlp(t)

        x_t_emb = th.cat((x1_emb, x2_emb, t_emb), dim=-1)
        x = self.joint_mlp(x_t_emb)

        if self.diff_type == "ref":
            if self.pred_type == "s":
                boundary_dist = self._compute_boundary_distance(x_t)
                x = th.min(th.ones_like(boundary_dist), self.fn(boundary_dist-self.boundary_tol)).reshape(-1, 1)*x

        return x