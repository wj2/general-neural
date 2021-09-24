data {
  int<lower=0> N;
  int<lower=1> K;
  matrix[N, K] x;
  int<lower=0,upper=1> y[N];
  real<lower=0> prior_width;
}

parameters {
  real alpha;
  vector[K] beta;
}

model {
  alpha ~ normal(0, prior_width);
  beta ~ normal(0, prior_width);
  
  y ~ bernoulli_logit(alpha + x * beta);
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | alpha + x[i] * beta);
    err_hat[i] = bernoulli_logit_rng(alpha + x[i] * beta);
  }
}

