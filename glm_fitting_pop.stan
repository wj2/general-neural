
data {
int<lower=0> N; // number of samples
int<lower=0> K; // number of predictors
int<lower=0> C; // number of cells
matrix[C, N, K] x; // predictor matrix
matrix[C, N] y; // outcome vector
}

parameters {
vector[K]<lower=0, upper=1000> alpha; // intercept
matrix[C, K] beta; // coefficients on Q_ast
vector[K]<lower=0> sigma; // error scale
}

model {
for (c in 1:C)
    y[c] ~ normal(x[c] * beta[c] + alpha[c], sigma[c]); // likelihood
}
