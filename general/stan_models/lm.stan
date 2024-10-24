data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  matrix[N, K] x; // predictor matrix
  vector[N] y; // outcome vector

  real<lower=0> mu_std_prior;
  real<lower=0> sigma_std_prior;
}

parameters {
  real alpha; // intercept
  vector[K] beta; // coefficients on Q_ast
  real<lower=0> sigma; // error scale
}

model {
  alpha ~ normal(0, mu_std_prior);
  beta ~ normal(0, mu_std_prior);
  sigma ~ normal(0, sigma_std_prior);
  
  y ~ normal(x * beta + alpha, sigma);
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | x[i] * beta + alpha, sigma);
    err_hat[i] = normal_rng(x[i] * beta + alpha, sigma);
  }
}
