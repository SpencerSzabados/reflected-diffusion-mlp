"""
    File contains script for testing the score boundary constraint on output of 
    MLP model for reflected diffusion score estimation task.
"""

import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt


def _compute_boundary_distance(x):
        """
        Determines the minimum distance from each point in x_t to the closer of the two annulus boundaries.

        Args:
            x_t (torch.Tensor): A tensor of shape [batch_size, 2] representing 2D points.

        Returns:
            torch.Tensor: A tensor of shape [batch_size], containing the minimum distance from each point to the closest boundary.

        TODO: Boundary values are currently hard coded. Should make this configurable.
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


def main():
    fn = nn.ReLU()

    x_t = th.randn(size=(10000,2))

    boundary_dist = _compute_boundary_distance(x_t)
    min_dist = th.min(th.ones_like(boundary_dist), fn(boundary_dist-0.05)).reshape(-1, 1)

    plt.figure(figsize=(8, 8))
    plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=1)
    plt.axis('off')
    plt.savefig(f"tmp/test_boundary_constraint_sample.png", transparent=True)
    plt.close()

    x_t = min_dist*x_t
    plt.figure(figsize=(8, 8))
    plt.scatter(x_t[:, 0], x_t[:, 1], alpha=0.5, s=1)
    plt.axis('off')
    plt.savefig(f"tmp/test_boundary_constraint_min_dist_sample.png", transparent=True)
    plt.close()


    print(boundary_dist)
    print(min_dist)


if __name__ == "__main__":
    main()