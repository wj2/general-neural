data {
  int<lower=0> N;
  int<lower=0,upper=1> y[N];
  real<lower=0> prior_width;
}

parameters {
  real alpha;
}

model {
  alpha ~ normal(0, prior_width);
  
  y ~ bernoulli_logit(alpha);
}

generated quantities {
  vector[N] log_lik;
  vector[N] err_hat;

  for (i in 1:N) {
    log_lik[i] = bernoulli_logit_lpmf(y[i] | alpha);
    err_hat[i] = bernoulli_logit_rng(alpha);
  }
}

