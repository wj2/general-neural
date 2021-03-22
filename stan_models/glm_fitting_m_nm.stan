
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  matrix[N, K] x; // predictor matrix
  vector[N] y; // outcome vector
  vector[N] context; // context vector
  real<lower=0> beta_var;
  real<lower=0> sigma_var;
  real<lower=0> modul_var;
}

parameters {
  vector[K] beta; // coefficients on Q_ast
  vector[2] tm;
  real<lower=0> sigma; // error scale
  real<lower=-1> modulator; // modulation constant
}

model {
  vector[N] context_mod;
  beta ~ normal(0, beta_var);
  tm ~ normal(0, beta_var);
  sigma ~ normal(0, sigma_var);
  modulator ~ normal(0, modul_var);

  context_mod = modulator*context + 1;
  y ~ normal(tm[1]*context + tm[2]*(1 - context)
	     + context_mod .* (x * beta), sigma); 
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    real mod = modulator*context[i] + 1;
    log_lik[i] = normal_lpdf(y[i] | tm[1]*context[i] + tm[2]*(1 - context[i])
			     + mod * x[i] * beta, sigma);
    err_hat[i] = normal_rng(tm[1]*context[i] + tm[2]*(1 - context[i])
			    + mod * x[i] * beta, sigma);
  }
}
