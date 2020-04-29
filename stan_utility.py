
import numpy as np
import os
import pickle
import pystan as ps
import re
import arviz as az

def recompile_model(mp):
    p, ext = os.path.splitext(mp)
    stan_path = p + '.stan'
    sm = ps.StanModel(file=stan_path)
    pickle.dump(sm, open(mp, 'wb'))
    return mp

def store_models(model_collection, store_arviz=False):
    new_collection = {}
    for k, (fit, params, diags) in model_collection.items():
        if store_arviz:
            arviz_manifest = params['arviz_manifest']
        else:
            arviz_manifest = None
        new_fit = ModelFitContainer(fit, arviz_manifest=arviz_manifest)
        new_collection[k] = (new_fit, params, diags)
    return new_collection

def get_stan_params_ind(mf, param_name, ind_set, mask=None, skip_end=1,
                        stan_dict=False, mean=False):
    ind_set = (str(i) for i in ind_set)
    param = param_name + '[' + ','.join(ind_set) +']'
    if stan_dict:
        out = mf[param]
    else:
        out = get_stan_params(mf, param, mask=mask, skip_end=skip_end)
    if mean:
        out = np.mean(out)
    return out

def get_stan_params(mf, param, mask=None, skip_end=1):
    names = mf.flatnames
    means = mf.get_posterior_mean()[:-skip_end]
    param = '\A' + param
    par_mask = np.array(list([re.match(param, x) is not None for x in names]))
    par_means = means[par_mask]
    if mask is not None:
        par_means = par_means[mask]
    return par_means

def make_stan_model_dict(mf, samples=False):
    d = {}
    if samples:
        d = mf.samples
    else:
        for i, n in enumerate(mf.flatnames):
            d[n] = mf.get_posterior_mean()[i]
    return d

class ModelFitContainer(object):

    def __init__(self, fit, arviz_manifest=None):
        self.flatnames = fit.flatnames
        self._posterior_means = fit.get_posterior_mean()
        self.samples = fit.extract()
        self._summary = fit.stansummary()
        if arviz_manifest is not None:
            self.arviz = az.from_pystan(posterior=fit, **arviz_manifest)

    def get_posterior_mean(self):
        return self._posterior_means

    def stansummary(self):
        return self._summary

    def __repr__(self):
        return self._summary
