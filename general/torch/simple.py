import numpy as np
import pyro
import torch
import logging

import pyro.contrib.gp as gp


class SparseGPWrapper:
    def __init__(self, kernel=None, input_dim=1, ls=1, noise=1, var=0.2):
        if kernel is None:
            kernel = gp.kernels.RBF(
                input_dim=input_dim,
                variance=torch.tensor(var),
                lengthscale=torch.tensor(ls),
            )
        self.kernel = kernel
        self.noise = torch.tensor(noise)
        self.rng = np.random.default_rng()
        self.gpr = None

    def fit(
        self,
        X,
        y,
        inducing=None,
        lr=0.005,
        smoke_test=False,
        num_steps=2000,
        show_logging=False,
        fix_params=(),
        max_points=50,
    ):
        pyro.clear_param_store()
        if len(X) > max_points or inducing is None:
            inducing = self.rng.choice(X, size=max_points)
        X = torch.tensor(X)
        y = torch.tensor(y)
        inducing = torch.tensor(inducing)
        gpr = gp.models.SparseGPRegression(
            X,
            y,
            self.kernel,
            inducing,
            noise=self.noise,
        )
        param_list = list(
            p for name, p in gpr.named_parameters() if name not in fix_params
        )
        optimizer = torch.optim.Adam(param_list, lr=lr)
        loss_fn = pyro.infer.Trace_ELBO().differentiable_loss
        use_steps = num_steps if not smoke_test else 2
        noises = np.zeros(use_steps)
        variances = np.zeros_like(noises)
        lengthscales = np.zeros_like(noises)
        losses = np.zeros_like(noises)
        for i in range(use_steps):
            noises[i] = gpr.noise.item()
            variances[i] = gpr.kernel.variance.item()
            lengthscales[i] = gpr.kernel.lengthscale.item()
            optimizer.zero_grad()
            loss = loss_fn(gpr.model, gpr.guide)
            loss.backward()
            optimizer.step()
            losses[i] = loss.item()
            if i % 100 == 0 and show_logging:
                logging.info("{}. Elbo loss: {}".format(i, loss))
        self.gpr = gpr
        
        return self

    def predict(self, X, noiseless=True, full_cov=True, **kwargs):
        if self.gpr is None:
            raise IOError("model has not been fit")
        X = torch.tensor(X)
        mu, cov = self.gpr(X, noiseless=noiseless, full_cov=full_cov, **kwargs)
        mu = mu.detach().numpy()
        cov = cov.detach().numpy()
        return mu, cov
