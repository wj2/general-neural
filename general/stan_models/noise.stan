data {
  int<lower=0> N; // number of samples
  vector[N] y; // outcome vector

  real<lower=0> mu_std_prior;
  real<lower=0> sigma_std_prior;
}

parameters {
  real alpha; // intercept
  real<lower=0> sigma; // error scale
}

model {
  alpha ~ normal(0, mu_std_prior);
  sigma ~ normal(0, sigma_std_prior);
  
  y ~ normal(alpha, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | alpha, sigma);
    err_hat[i] = normal_rng(alpha, sigma);
  }
}
