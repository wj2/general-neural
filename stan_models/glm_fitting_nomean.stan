
data {
int<lower=0> N; // number of samples
int<lower=0> K; // number of predictors
matrix[N, K] x; // predictor matrix
vector[N] y; // outcome vector
}

parameters {
vector[K] beta; // coefficients on Q_ast
real<lower=0> sigma; // error scale
}

model {
beta ~ normal(0, 5);
sigma ~ inv_gamma(1, 1);
y ~ normal(x * beta, sigma); // likelihood
}
