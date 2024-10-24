
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  int<lower=0> L; // number of neurons
  matrix[N, K] x; // predictor matrix
  vector[N] y; // outcome vector
  vector[N] context; // context vector
  int neuron_num[N]; // indicates which neuron
  real<lower=0> beta_var;
  real<lower=0> sigma_var;
}

parameters {
  matrix[L, K] beta; // coefficients on Q_ast
  vector<lower=0>[L] sigma; // error scale
}

model {
  int ni;
  for (i in 1:L) {
    beta[i] ~ normal(0, beta_var);
  }
  sigma ~ normal(0, sigma_var);

  for (i in 1:N) {
    ni = neuron_num[i];
    y[i] ~ normal(x[i] * beta[ni]', sigma[ni]); // likelihood
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    int ni = neuron_num[i];
    log_lik[i] = normal_lpdf(y[i] | x[i] * beta[ni]',
			     sigma[ni]);
    err_hat[i] = normal_rng(x[i] * beta[ni]', sigma[ni]);
  }
}
