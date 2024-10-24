data {
  int<lower=0> N; // trials
  vector<lower=0>[N] err;
  real<lower=0> fi_mu;
}

parameters {
  real<lower=0, upper=1> p;
  real<lower=0> sigma; 
}

model {
  real local_p;
  real thr_p;
  
  p ~ uniform(0, 1);
  sigma ~ normal(0, 1/fi_mu);

  for (i in 1:N) {
    local_p = normal_lpdf(err[i] | 1/fi_mu, sigma);
    thr_p = log(2*(1 - sqrt(err[i])));
    target += log_sum_exp(log(p) + thr_p, log(1 - p) + local_p);
  }
}

generated quantities {
  vector[N] err_hat;
  for (i in 1:N) {
    int type;
    type = bernoulli_rng(p);
    if (type == 1) {
      err_hat[i] = fabs(uniform_rng(0, 1) - uniform_rng(0, 1));
    } else {
      err_hat[i] = normal_rng(1/fi_mu, sigma);    
    }
  }
}
