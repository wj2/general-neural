
data {
  int<lower=0> N; // number of samples
  int<lower=0> K; // number of predictors
  int<lower=0> T;
  matrix[N, K] x; // predictor matrix
  int<lower=1, upper=T> time[N];
  vector<lower=0, upper=2*pi()>[N] y; // outcome vector

  real<lower=0> beta_mean_var;
  real<lower=0> beta_var_mean;
  real<lower=0> beta_var_var;

  real<lower=0> sigma_mean_mean;
  real<lower=0> sigma_mean_var;
  real<lower=0> sigma_var_mean;
  real<lower=0> sigma_var_var;
}

parameters {
  vector[K] beta_mean;
  vector<lower=0>[K] beta_var;

  real b_mean;
  real<lower=0> b_var;

  real<lower=0> sigma_mean;
  real<lower=0.0001> sigma_var;
    
  matrix[T, K] beta_raw; // coefficients on Q_ast
  vector[T] b_raw;
  vector[T] sigma_raw; // error scale
}

transformed parameters {  
  vector<lower=0.0001>[T] sigma;
  vector[T] b;
  matrix[T, K] beta;

  sigma = sigma_mean + sigma_var*sigma_raw;
  b = b_mean + b_var*b_raw;
  for (i in 1:T) {
    beta[i] = (beta_mean + beta_raw[i]*beta_var)';
  }
}

model {
  int t;

  for (j in 1:T) {
    beta_raw[j] ~ normal(0, 1);
  }
  b_raw ~ normal(0, 1);
  sigma_raw ~ normal(0, 1);
  
  beta_mean ~ normal(0, beta_mean_var);
  beta_var ~ normal(beta_var_mean, beta_var_var);

  b_mean ~ normal(0, beta_mean_var);
  b_var ~ normal(beta_var_mean, beta_var_var);

  sigma_mean ~ normal(sigma_mean_mean, sigma_mean_var);
  sigma_var ~ normal(sigma_var_mean, sigma_var_var);
    
  for (i in 1:N) {
    t = time[i];
    if (sigma[t] > 100) {
      y[i] ~ normal(b[t] + x[i] * beta[t]', sqrt(1/sigma[t]));
    } else {
      y[i] ~ von_mises(b[t] + x[i] * beta[t]', sigma[t]);
    }
  }
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    int t = time[i];
    if (sigma[t] > 100) {
      log_lik[i] = normal_lpdf(y[i] | b[t] + x[i] * beta[t]', sqrt(1/sigma[t]));
      err_hat[i] = normal_rng(b[t] + x[i] * beta[t]', sqrt(1/sigma[t]));
    } else {
      log_lik[i] = von_mises_lpdf(y[i] | b[t] + x[i] * beta[t]', sigma[t]);
      err_hat[i] = von_mises_rng(b[t] + x[i] * beta[t]', sigma[t]);
    }
  }
}
