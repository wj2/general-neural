
import numpy as np
import os
import pickle
import pystan as ps
import re
import arviz as az

def recompile_model(mp, **kwargs):
    p, ext = os.path.splitext(mp)
    stan_path = p + '.stan'
    pkl_path = p + '.pkl'
    sm = ps.StanModel(file=stan_path, **kwargs)
    pickle.dump(sm, open(pkl_path, 'wb'))
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

generic_manifest = {'observed_data':'y',
                    'log_likelihood':{'y':'log_lik'},
                    'posterior_predictive':'err_hat'}
manifest_dict = {
    'general/stan_models/logit.pkl':generic_manifest,
    'general/stan_models/unif_resp.pkl':generic_manifest,
    'r1r2r3/stan_models/sum_od.pkl':generic_manifest,
    'general/stan_models/lm.pkl':generic_manifest,
}

def fit_model(data_dict, model_path, max_treedepth=10, adapt_delta=.8,
              manifest=None, default_manifest=generic_manifest,
              arviz_convert=True, fixed_param=False, **kwargs):
    if manifest is None:
        manifest = manifest_dict.get(model_path, default_manifest)
    if fixed_param:
        algorithm = 'Fixed_param'
    else:
        algorithm = None
    sm = pickle.load(open(model_path, 'rb'))
    control = dict(max_treedepth=max_treedepth, adapt_delta=adapt_delta)
    fit = sm.sampling(data=data_dict, control=control, algorithm=algorithm,
                      **kwargs)
    diag = ps.check_hmc_diagnostics(fit)
    if arviz_convert:
        fit_az = az.from_pystan(posterior=fit, **manifest)
    else:
        fit_az = None
    return fit, fit_az, diag

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
