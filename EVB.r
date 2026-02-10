rm(list = ls())
set.seed(123)

# -------------------------------
# 1. Target Posterior
# -------------------------------
dummy_posterior <- function(xi, sigma, data = NULL) {
  if (sigma <= 0) return(0)
  exp(-0.5 * (xi^2 + (sigma - 2)^2))
}

# -------------------------------
# 2. Metropolis-Hastings Algorithm
# -------------------------------
metropolis_hastings <- function(n_iterations, initial_params, target_density, proposal_widths, data = NULL) {
  
  xi <- numeric(n_iterations)
  sigma <- numeric(n_iterations)
  
  xi[1] <- initial_params[1]
  sigma[1] <- initial_params[2]
  
  v_xi <- proposal_widths[1]
  v_sigma <- proposal_widths[2]
  
  for (j in 2:n_iterations) {
    
    xi_star <- rnorm(1, mean = xi[j-1], sd = v_xi)
    sigma_star <- rnorm(1, mean = sigma[j-1], sd = v_sigma)
    
    num <- target_density(xi_star, sigma_star, data)
    den <- target_density(xi[j-1], sigma[j-1], data)
    
    r <- ifelse(den > 0, num / den, 0)
    tau <- min(1, r)
    
    u <- runif(1)
    if (u < tau) {
      xi[j] <- xi_star
      sigma[j] <- sigma_star
    } else {
      xi[j] <- xi[j-1]
      sigma[j] <- sigma[j-1]
    }
  }
  
  return(list(xi = xi, sigma = sigma))
}

# -------------------------------
# 3. Run Sampler
# -------------------------------
samples <- metropolis_hastings(
  n_iterations = 5000,
  initial_params = c(0.5, 2.0),
  target_density = dummy_posterior,
  proposal_widths = c(0.1, 0.1),
  data = NULL
)

xi_samples <- samples$xi
sigma_samples <- samples$sigma

# -------------------------------
# 4. Burn-in and Results
# -------------------------------
burn_in <- 1000

cat("Mean Xi:", mean(xi_samples[(burn_in+1):length(xi_samples)]), "\n")
cat("Mean Sigma:", mean(sigma_samples[(burn_in+1):length(sigma_samples)]), "\n")
