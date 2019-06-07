
import numpy as np
import os
import pickle
import pystan as ps

def recompile_model(mp):
    p, ext = os.path.splitext(mp)
    stan_path = p + '.stan'
    sm = ps.StanModel(file=stan_path)
    pickle.dump(sm, open(mp, 'wb'))
    return mp

def store_models(model_collection):
    new_collection = {}
    for k, (fit, params, diags) in model_collection.items():
        new_fit = ModelFitContainer(fit)
        new_collection[k] = (new_fit, params, diags)
    return new_collection

def get_stan_params(mf, param, mask=None, skip_end=1):
    names = mf.flatnames
    means = mf.get_posterior_mean()[:-skip_end]
    param = '\A' + param
    par_mask = np.array(list([re.match(param, x) is not None for x in names]))
    par_means = means[par_mask]
    if mask is not None:
        par_means = par_means[mask]
    return par_means

class ModelFitContainer(object):

    def __init__(self, fit):
        self.flatnames = fit.flatnames
        self._posterior_means = fit.get_posterior_mean()
        self.samples = fit.extract()
        self._summary = fit.stansummary()

    def get_posterior_mean(self):
        return self._posterior_means

    def stansummary(self):
        return self._summary

    def __repr__(self):
        return self._summary
