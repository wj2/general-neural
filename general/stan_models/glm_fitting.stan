
data {
int<lower=0> N; // number of samples
int<lower=0> K; // number of predictors
matrix[N, K] x; // predictor matrix
vector[N] y; // outcome vector
}

parameters {
real<lower=0, upper=1000> alpha; // intercept
vector[K] beta; // coefficients on Q_ast
real<lower=0> sigma; // error scale
}

model {
y ~ normal(x * beta + alpha, sigma); // likelihood
}
