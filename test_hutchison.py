"""
    This file contains scripts for testing the accuracy of the implemented Hutchison divergence
    estiamted code used in training reflected mlp diffusion.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt


class TestFunction(torch.nn.Module):
    def forward(self, x):
        # x is of shape [batch_size, 2]
        x1 = x[:, 0]
        x2 = x[:, 1]
        f1 = x1 ** 2
        f2 = x2 ** 2
        # Return shape should be [batch_size, 2]
        return torch.stack([f1, f2], dim=1)


def analytical_divergence(x):
    # x is of shape [batch_size, 2]
    x1 = x[:, 0]
    x2 = x[:, 1]
    divergence = 2 * x1 + 2 * x2
    return divergence  # shape [batch_size]


def hutchinson_divergence(f, x, num_samples=1000):
    """
    Estimates the divergence of function f at points x using Hutchinson's estimator.

    Args:
        f (callable): The function from R^2 to R^2.
        x (torch.Tensor): Input tensor of shape [batch_size, 2] with requires_grad=True.
        num_samples (int): Number of samples to use in the estimator.

    Returns:
        divergence_estimate (torch.Tensor): Estimated divergence, shape [batch_size]
    """
    batch_size = x.shape[0]
    divergence_estimates = torch.zeros(batch_size, device=x.device)

    for _ in range(num_samples):
        z = torch.randn_like(x)  # Sample z ~ N(0, I)
        f_x = f(x)  # Compute f(x)
        # Compute the dot product z^T (∇_x f(x))
        grad_f_x = torch.autograd.grad(
            outputs=f_x,
            inputs=x,
            grad_outputs=z,
            create_graph=False,
            retain_graph=True,
        )[0]  # Shape [batch_size, 2]
        # divergence_estimates.append(divergence_sample)
        divergence_sample = torch.einsum('bi,bi->b', grad_f_x, z)  # Element-wise product and sum over features
        divergence_estimates += divergence_sample

    divergence_estimates /= num_samples  # Average over samples
    return divergence_estimates  # Shape [batch_size]


def main():
    # Create a grid of points
    x_vals = np.linspace(-1, 1, 50)
    y_vals = np.linspace(-1, 1, 50)
    X, Y = np.meshgrid(x_vals, y_vals)
    grid_points = np.stack([X.ravel(), Y.ravel()], axis=1)  # Shape [2500, 2]
    grid_points = torch.tensor(grid_points, dtype=torch.float32, requires_grad=True)

    # Initialize the function
    test_function = TestFunction()

    # Compute analytical divergence
    analytical_div = analytical_divergence(grid_points).detach().numpy()  # Shape [2500]

    # Compute estimated divergence
    estimated_div = hutchinson_divergence(test_function, grid_points, num_samples=100).detach().numpy()

    # Compute absolute error
    error = np.abs(estimated_div - analytical_div)

    # Reshape the results for plotting
    error_grid = error.reshape(X.shape)
    analytical_div_grid = analytical_div.reshape(X.shape)
    estimated_div_grid = estimated_div.reshape(X.shape)

    # Plot the error as a heatmap
    plt.figure(figsize=(8, 6))
    plt.contourf(X, Y, error_grid, levels=50, cmap=plt.cm.YlGnBu_r)
    plt.colorbar(label='Absolute Error')
    plt.title('Error between Analytical and Estimated Divergence')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.savefig(f"tmp/test_hutchison.png", transparent=True)
    plt.close()


if __name__=="__main__":
    main()