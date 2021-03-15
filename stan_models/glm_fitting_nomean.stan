
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  matrix[N, K] x; // predictor matrix
  vector[N] y; // outcome vector
  real<lower=0> beta_var;
  real<lower=0> sigma_var;
}

parameters {
  vector[K] beta; // coefficients on Q_ast
  real<lower=0> sigma; // error scale
}

model {
  beta ~ normal(0, beta_var);
  sigma ~ normal(0, sigma_var);
  y ~ normal(x * beta, sigma); // likelihood
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | x[i] * beta, sigma);
    err_hat[i] = normal_rng(x[i] * beta, sigma);
  }
}
