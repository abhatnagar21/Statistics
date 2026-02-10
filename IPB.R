set.seed(42)

# -----------------------------
# 1. Generate GPD data manually
# -----------------------------
rgpd_manual <- function(n, xi, sigma) {
  u <- runif(n)
  
  if (abs(xi) > 1e-6) {
    return(sigma * ((u^(-xi) - 1) / xi))
  } else {
    return(-sigma * log(u))  # exponential case
  }
}

true_xi <- 0.2
true_sigma <- 1.5

data <- rgpd_manual(500, true_xi, true_sigma)


# -----------------------------
# 2. Log-likelihood (manual)
# -----------------------------
log_likelihood <- function(xi, sigma, data) {
  
  if (sigma <= 0) return(-Inf)
  
  # constraint: 1 + xi*x/sigma > 0
  if (any(1 + xi * data / sigma <= 0)) return(-Inf)
  
  n <- length(data)
  
  if (abs(xi) > 1e-6) {
    ll <- -n * log(sigma) - (1/xi + 1) *
      sum(log(1 + xi * data / sigma))
  } else {
    ll <- -n * log(sigma) - sum(data) / sigma
  }
  
  return(ll)
}


# -----------------------------
# 3. Informative priors
# -----------------------------
log_prior <- function(xi, sigma) {
  
  if (sigma <= 0) return(-Inf)
  
  xi_prior <- -0.5 * ((xi - 0.3) / 0.3)^2
  sigma_prior <- -0.5 * ((sigma - 2) / 1)^2
  
  return(xi_prior + sigma_prior)
}


# -----------------------------
# 4. Posterior
# -----------------------------
log_posterior <- function(xi, sigma, data) {
  log_likelihood(xi, sigma, data) + log_prior(xi, sigma)
}


# -----------------------------
# 5. Metropolis-Hastings
# -----------------------------
iterations <- 10000

xi_chain <- numeric(iterations)
sigma_chain <- numeric(iterations)

xi_chain[1] <- 0.5
sigma_chain[1] <- 2.5

for (j in 2:iterations) {
  
  xi_star <- rnorm(1, xi_chain[j-1], 0.05)
  sigma_star <- rnorm(1, sigma_chain[j-1], 0.1)
  
  log_r <- log_posterior(xi_star, sigma_star, data) -
           log_posterior(xi_chain[j-1], sigma_chain[j-1], data)
  
  if (log(runif(1)) < log_r) {
    xi_chain[j] <- xi_star
    sigma_chain[j] <- sigma_star
  } else {
    xi_chain[j] <- xi_chain[j-1]
    sigma_chain[j] <- sigma_chain[j-1]
  }
}


# -----------------------------
# 6. Estimates
# -----------------------------
burn_in <- 2000

xi_estimate <- mean(xi_chain[burn_in:iterations])
sigma_estimate <- mean(sigma_chain[burn_in:iterations])

cat("Estimated xi:", xi_estimate, "\n")
cat("Estimated sigma:", sigma_estimate, "\n")
