import numpy as np

def metropolis_hastings(n_iterations, initial_params, target_density, proposal_widths, data):
    """
    Implements the Metropolis-Hastings algorithm as described in the image.
    
    Parameters:
    - n_iterations: Number of samples to generate
    - initial_params: Tuple of (xi_0, sigma_0)
    - target_density: A function that calculates the posterior pi(xi, sigma | data)
    - proposal_widths: Tuple of (v_xi, v_sigma) for the normal proposal distributions
    - data: The observations x_u
    """
    
    # Initialize arrays to store the chain
    xi = np.zeros(n_iterations)
    sigma = np.zeros(n_iterations)
    
    # Set starting parameters (Step: j=1, start parameters xi_0, sigma_0)
    xi[0], sigma[0] = initial_params
    v_xi, v_sigma = proposal_widths

    for j in range(1, n_iterations):
        # 1. Generate candidate parameters from Normal distributions
        # xi* ~ N(xi_j-1, v_xi), sigma* ~ N(sigma_j-1, v_sigma)
        xi_star = np.random.normal(xi[j-1], v_xi)
        sigma_star = np.random.normal(sigma[j-1], v_sigma)
        
        # 2. Evaluate the probability of acceptance
        # Note: For symmetric proposals (like Normal), the proposal ratio cancels out.
        # We calculate the ratio of the target density (posterior).
        numerator = target_density(xi_star, sigma_star, data)
        denominator = target_density(xi[j-1], sigma[j-1], data)
        
        # Avoid division by zero and handle log-space if necessary 
        # (Standard ratio shown in image)
        r = numerator / denominator if denominator > 0 else 0
        tau = min(1, r)
        
        # 3. Generate u ~ U(0, 1)
        u = np.random.uniform(0, 1)
        
        # 4. Accept or Reject
        if u < tau:
            xi[j] = xi_star
            sigma[j] = sigma_star
        else:
            xi[j] = xi[j-1]
            sigma[j] = sigma[j-1]
            
    return xi, sigma

# --- Example Usage (Dummy Posterior) ---
def dummy_posterior(xi, sigma, data):
    # Sigma must be positive for most distributions (e.g., GPD)
    if sigma <= 0:
        return 0
    # This is a placeholder; replace with your actual Likelihood * Prior
    return np.exp(-0.5 * (xi**2 + (sigma-2)**2)) 

# Run the sampler
xi_samples, sigma_samples = metropolis_hastings(
    n_iterations=10000, 
    initial_params=(0.1, 1.0), 
    target_density=dummy_posterior, 
    proposal_widths=(0.05, 0.05),
    data=None
)
# ... [Include the metropolis_hastings function from above] ...

# 1. Run the sampler
xi_samples, sigma_samples = metropolis_hastings(
    n_iterations=5000, 
    initial_params=(0.5, 2.0), 
    target_density=dummy_posterior, 
    proposal_widths=(0.1, 0.1),
    data=None
)

# 2. Print the results (Discarding the first 1000 samples as "burn-in")
burn_in = 1000
print(f"Mean Xi: {np.mean(xi_samples[burn_in:]):.4f}")
print(f"Mean Sigma: {np.mean(sigma_samples[burn_in:]):.4f}")

# 3. Visualize the results
import matplotlib.pyplot as plt

plt.figure(figsize=(12, 5))

# Plot the Trace of Xi
plt.subplot(1, 2, 1)
plt.plot(xi_samples)
plt.title("Trace Plot for $\\xi$")
plt.xlabel("Iteration")

# Plot the Trace of Sigma
plt.subplot(1, 2, 2)
plt.plot(sigma_samples, color='orange')
plt.title("Trace Plot for $\\sigma$")
plt.xlabel("Iteration")

plt.tight_layout()
plt.show()
