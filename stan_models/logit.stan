data {
  int<lower=0> N;
  vector[N] x;
  int<lower=0,upper=1> y[N];
  real<lower=0> prior_width;
}

parameters {
  real alpha;
  real beta;
}

model {
  alpha ~ normal(0, prior_width);
  beta ~ normal(0, prior_width);
  
  y ~ bernoulli_logit(alpha + beta * x);
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | alpha + beta*x[i]);
    err_hat[i] = bernoulli_logit_rng(alpha + beta*x[i]);
  }
}

