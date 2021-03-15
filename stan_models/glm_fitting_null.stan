
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  vector[N] y; // outcome vector
  real<lower=0> sigma_var;
}

parameters {
  real<lower=0> sigma; // error scale
}

model {
  sigma ~ normal(0, sigma_var);

  y ~ normal(0, sigma); // likelihood
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = normal_lpdf(y[i] | 0, sigma);
    err_hat[i] = normal_rng(0, sigma);
  }
}
