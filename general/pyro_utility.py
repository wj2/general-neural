
import numpy as np
import pyro
import torch
import logging

import pyro.distributions.constraints as constraints
import pyro.distributions as dist
from pyro import poutine


def fit_model(
        features,
        targets,
        model,
        guide=None,
        approach=None,
        loss=None,
        optim=None,
        lr=.01,
        n_steps=1000,
        smoke_test=False,
        show_logging=False,
        n_samps=500,
        block_vars=None,
):
    pyro.clear_param_store()
    if approach is None:
        approach = pyro.infer.SVI
    if loss is None:
        loss = pyro.infer.Trace_ELBO()
    if guide is None:
        guide = pyro.infer.autoguide.AutoNormal
    if optim is None:
        optim = pyro.optim.Adam({"lr": lr})
    if block_vars:
        blocked_model = poutine.block(model, hide=block_vars)
    else:
        blocked_model = model
    use_guide = guide(blocked_model)
    all_inp = features + targets
    model_render = pyro.render_model(
        model,
        model_args=all_inp,
        render_distributions=True,
        render_params=True,
    )
    guide_render = pyro.render_model(
        use_guide,
        model_args=all_inp,
        render_distributions=True,
        render_params=True,
    )
    optimizer = approach(model, use_guide, optim, loss)

    losses = []
    for step in range(n_steps if not smoke_test else 2):
        loss = optimizer.step(*all_inp)
        losses.append(loss)
        if step % 100 == 0 and show_logging:
            logging.info("Elbo loss: {}".format(loss))

    with pyro.plate("samples", n_samps, dim=-1):
        samples = use_guide(*features)
    new_samples = {}
    for k, v in samples.items():
        new_samples[k] = v.detach().numpy()
    out_dict = {
        "samples": new_samples,
        "model": model,
        "guide": use_guide,
        "model_render": model_render,
        "guide_render": guide_render,
        "losses": losses,
    }
    return out_dict


def sample_fit_model(
        features, model, use_guide, n_samps=500,
):
    predictive = pyro.infer.Predictive(model, guide=use_guide, num_samples=n_samps)
    pred_samples = predictive(*features)
    return pred_samples["obs"]
