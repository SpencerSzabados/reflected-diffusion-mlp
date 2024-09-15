"""
    File contains script for generating random samples from a circular/disc checkerboard density
    and saves these points, in a paired (x,y) format, to a .npz file for use in model training.
"""

import numpy as np
from scipy.ndimage import gaussian_filter


# Parameters for the checkerboard pattern
rho1 = 1.0   # Maximum density value
rho2 = 0.01  # Minimum density value
k_theta = 5  # Number of angular alternations
k_r = 5      # Number of radial alternations
r_in = 0.01  # Inner radius of punctured unit disc
n_samples = 1000000 # Number of datapoint in dataset
output_filename = "radial_checkerboard_density_dataset.npz"

# Function to compute the density based on radius and theta
def checkerboard_density(r, theta):
    angular_component = np.cos(k_theta * theta)
    radial_component = np.cos(k_r * np.pi * r)  # Periodic in radius
    if angular_component * radial_component > 0:
        return rho1
    else:
        return rho2

# Apply Gaussian smoothing to the density function
def smooth_density_function(r, theta, sigma=5):
    # Evaluate the density at the sampled points
    density = checkerboard_density(r, theta)
    # Apply Gaussian smoothing (for now we'll approximate this with the blur on grid)
    return gaussian_filter(density, sigma=sigma)

# Function to sample r with probability proportional to r
def sample_r(n_samples):
    return np.sqrt(np.random.uniform(r_in**2, 1**2, n_samples))

# Function to sample from the density using rejection sampling
def sample_from_density(n_samples, sigma=10):
    samples_x = []
    samples_y = []
    
    # Maximum density for rejection sampling
    max_density = rho1  # The maximum possible value of the density
    
    while len(samples_x) < n_samples:
        # Sample r and theta
        r_rand = sample_r(1)[0]  # Sample r with probability proportional to r
        theta_rand = np.random.uniform(0, 2 * np.pi)
        
        # Evaluate density at (r_rand, theta_rand)
        density = checkerboard_density(r_rand, theta_rand)
        
        # Apply Gaussian smoothing to the density value
        density_smoothed = density  # Here we could directly apply Gaussian blur, approximated for simplicity
        
        # Sample from uniform distribution and accept with probability proportional to density
        if np.random.uniform(0, max_density) < density_smoothed:
            # Convert polar to Cartesian coordinates
            x_rand = r_rand * np.cos(theta_rand)
            y_rand = r_rand * np.sin(theta_rand)
            
            samples_x.append(x_rand)
            samples_y.append(y_rand)
    
    return np.array(samples_x), np.array(samples_y)

# Generate samples from the distribution
samples_x, samples_y = sample_from_density(n_samples)

# Save the (x, y) pairs into a .npz file
points = np.stack((samples_x, samples_y), axis=1)
np.savez(output_filename, data=points)
print(f"Data saved to {output_filename}")


