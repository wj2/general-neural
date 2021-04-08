data {
  int<lower=0> T; // number of samples
  int<lower=0> K; // number of predictors
  int<lower=0> N; // number of neurons
  matrix[T, K] x; // predictor matrix
  vector[T] y; // outcome vector
  vector[T] context; // context vector
  int<lower=1, upper=N> neur_inds[T]; // neuron index

  real<lower=0> beta_mean_var;
  real<lower=0> beta_var_var;

  real<lower=0> sigma_mean_mean;
  real<lower=0> sigma_mean_var;
  real<lower=0> sigma_var_var;
  real<lower=0> sigma_var_mean;
}

parameters {
  matrix[N, K] beta_raw; // coefficients on Q_ast
  matrix<lower=0>[N, 2] sigma_raw; // error scales

  vector[K] beta_mean;
  vector[K] beta_var;
  
  vector<lower=0>[2] sigma_mean;
  vector<lower=0>[2] sigma_var;
}

transformed parameters {
  matrix[N, K] beta; // coefficients on Q_ast
  matrix<lower=0>[N, 2] sigma; // error scales

  for (i in 1:N) {
    beta[i] = (beta_mean + beta_var .* beta_raw[i]')';
    sigma[i] = (sigma_mean + sigma_var .* sigma_raw[i]')';
  }
}

model {
  int neur;

  for (i in 1:N) {
    beta_raw[i] ~ normal(0, 1);
    sigma_raw[i] ~ normal(0, 1);
  }

  beta_mean ~ normal(0, beta_var_var);
  beta_var ~ normal(beta_mean_var, beta_var_var);
  sigma_mean ~ normal(sigma_mean_mean, sigma_mean_var);
  sigma_var ~ normal(sigma_var_mean, sigma_var_var);

  for (i in 1:T) {
    neur = neur_inds[i];
    y[i] ~ normal(x[i] * beta[neur]',
		  sigma[neur, 1]*context[i] + sigma[neur, 2]*(1 - context[i]));
  }
}

generated quantities {
  vector[T] log_lik;
  vector[T] err_hat;

  for (i in 1:T) {
    int neur = neur_inds[i];
    log_lik[i] = normal_lpdf(y[i] | x[i] * beta[neur]',
			     sigma[neur, 1]*context[i]
			     + sigma[neur, 2]*(1 - context[i]));
    err_hat[i] = normal_rng(x[i] * beta[neur]',
			    sigma[neur, 1]*context[i]
			    + sigma[neur, 2]*(1 - context[i]));
  }
}
