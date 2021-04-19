
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  matrix[N, K] x; // predictor matrix
  vector<lower=0, upper=2*pi()>[N] y; // outcome vector
  real<lower=0> beta_var;
  real<lower=0> sigma_mean;
  real<lower=0> sigma_var;
}

parameters {
  vector[K] beta; // coefficients on Q_ast
  real b;
  real<lower=.0001> sigma; // error scale
}

model {
  beta ~ normal(0, beta_var);
  b ~ normal(0, beta_var);
  sigma ~ normal(sigma_mean, sigma_var);
  if (sigma > 100) {
    y ~ normal(b + x * beta, sqrt(1/sigma));
  } else {
    y ~ von_mises(b + x * beta, sigma);
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = von_mises_lpdf(y[i] | b + x[i] * beta, sigma);
    err_hat[i] = von_mises_rng(b + x[i] * beta, sigma);
  }
}
