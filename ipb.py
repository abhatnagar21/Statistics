import numpy as np
from scipy.stats import genpareto
import matplotlib.pyplot as plt

# -----------------------------
# 1. Generate fake extreme loss data
# -----------------------------
np.random.seed(42)

true_xi = 0.2      # true tail parameter
true_sigma = 1.5  # true scale

data = genpareto.rvs(c=true_xi, scale=true_sigma, size=500)


# -----------------------------
# 2. Likelihood function
# -----------------------------
def log_likelihood(xi, sigma, data):
    # parameters must be positive
    if sigma <= 0:
        return -np.inf
    
    return np.sum(genpareto.logpdf(data, c=xi, scale=sigma))


# -----------------------------
# 3. Prior (informative prior)
# assume xi ~ N(0.3, 0.3)
# assume sigma ~ N(2, 1)
# -----------------------------
def log_prior(xi, sigma):
    if sigma <= 0:
        return -np.inf
    
    xi_prior = -0.5 * ((xi - 0.3) / 0.3)**2
    sigma_prior = -0.5 * ((sigma - 2) / 1)**2
    
    return xi_prior + sigma_prior


# -----------------------------
# 4. Posterior = likelihood + prior
# -----------------------------
def log_posterior(xi, sigma, data):
    return log_likelihood(xi, sigma, data) + log_prior(xi, sigma)


# -----------------------------
# 5. Metropolis-Hastings
# -----------------------------
iterations = 10000

xi_chain = []
sigma_chain = []

# starting guess
current_xi = 0.5
current_sigma = 2.5

for i in range(iterations):
    
    # propose new candidate
    proposal_xi = np.random.normal(current_xi, 0.05)
    proposal_sigma = np.random.normal(current_sigma, 0.1)
    
    # compute acceptance ratio
    current_post = log_posterior(current_xi, current_sigma, data)
    proposal_post = log_posterior(proposal_xi, proposal_sigma, data)
    
    acceptance_ratio = np.exp(proposal_post - current_post)
    
    # accept/reject
    if np.random.rand() < acceptance_ratio:
        current_xi = proposal_xi
        current_sigma = proposal_sigma
    
    xi_chain.append(current_xi)
    sigma_chain.append(current_sigma)


# -----------------------------
# 6. Results
# -----------------------------
print("Estimated xi:", np.mean(xi_chain[2000:]))      # burn-in removed
print("Estimated sigma:", np.mean(sigma_chain[2000:]))

# Plot chains
plt.figure(figsize=(12,5))

plt.subplot(1,2,1)
plt.plot(xi_chain)
plt.title("Xi Chain")

plt.subplot(1,2,2)
plt.plot(sigma_chain)
plt.title("Sigma Chain")

plt.show()
