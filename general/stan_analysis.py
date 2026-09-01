import os
import pickle

import arviz as az
import numpy as np
import sklearn.decomposition as skd
import sklearn.impute as skimp
import sklearn.pipeline as sklpipe
import sklearn.preprocessing as skp

stan_file_trunk = "general/stan_models/"
stan_file_glm_mean = os.path.join(stan_file_trunk, "glm_fitting.pkl")
stan_file_glm_nomean = os.path.join(stan_file_trunk, "glm_fitting_nomean.pkl")
stan_file_glm_modu_nomean = os.path.join(stan_file_trunk, "glm_fitting_m_nm.pkl")
stan_file_glm_nomean_cv = os.path.join(stan_file_trunk, "glm_fitting_nm_mvar.pkl")
stan_file_glm_modu_nomean_cv = os.path.join(
    stan_file_trunk, "glm_fitting_m_nm_mvar.pkl"
)

stan_file_glm_nomean_pop = os.path.join(stan_file_trunk, "glm_fitting_nm_pop.pkl")
stan_file_glm_modu_nomean_pop = os.path.join(
    stan_file_trunk, "glm_fitting_m_nm_pop.pkl"
)
stan_file_glm_nomean_cv_pop = os.path.join(
    stan_file_trunk, "glm_fitting_nm_mvar_pop.pkl"
)
stan_file_glm_modu_nomean_cv_pop = os.path.join(
    stan_file_trunk, "glm_fitting_m_nm_mvar_pop.pkl"
)
stan_logit_path = os.path.join(stan_file_trunk, "logit.pkl")
stan_unif_resp_path = os.path.join(stan_file_trunk, "unif_resp.pkl")
glm_arviz = {
    "observed_data": "y",
    "log_likelihood": {"y": "log_lik"},
    "posterior_predictive": "err_hat",
}
glm_pop_arviz = {
    "observed_data": "y",
    "log_likelihood": {"y": "log_lik"},
    "posterior_predictive": "err_hat",
    "dims": {"beta": ["neur_inds"], "sigma": ["neur_inds"]},
}

def pop_regression_timestan(
    pop,
    reg_vals,
    model=None,
    norm=True,
    pre_pca=None,
    impute_missing=False,
    pre_rescale=False,
    **model_params,
):
    x_len = pop.shape[-1]
    steps = []
    if norm:
        steps.append(skp.StandardScaler())
    if impute_missing:
        steps.append(skimp.SimpleImputer())
    if pre_pca is not None:
        steps.append(skd.PCA(n_components=pre_pca))
    pipe = sklpipe.make_pipeline(*steps)
    pop_flat = np.concatenate(tuple(pop[:, i] for i in range(pop.shape[1])), axis=1)
    shuff_inds = np.random.choice(reg_vals.shape[0], reg_vals.shape[0], replace=False)
    reg_shuff = np.array(reg_vals)[shuff_inds]
    pop_list = []
    reg_list = []
    reg_shuff_list = []
    time_list = []
    for j in range(x_len):
        if pre_rescale:
            skss = skp.StandardScaler()
            pfj = skss.fit_transform(pop_flat[..., j].T).T
            pop_list.append(pfj)
        else:
            pop_list.append(pop_flat[..., j])
        reg_list.append(reg_vals)
        reg_shuff_list.append(reg_shuff)
        time_list.append(np.ones(len(reg_vals), dtype=int) * (j + 1))
    pop_full = np.concatenate(pop_list, axis=1)
    reg_full = np.concatenate(reg_list, axis=0)
    reg_shuff_full = np.concatenate(reg_shuff_list, axis=0)
    time_full = np.concatenate(time_list, axis=0)

    pop_proc = pipe.fit_transform(pop_full.T)
    m1 = model(**model_params)
    m1.fit(pop_proc, reg_full, time_full)
    m2 = model(**model_params)
    m2.fit(pop_proc, reg_shuff_full, time_full)
    comp = az.compare(dict(m=m1.get_arviz(), m_shuff=m2.get_arviz()))
    for j in range(x_len):
        pop_t_proc = pipe.transform(pop_flat[..., j].T)
        scores = m1.score(pop_t_proc, reg_vals, time_list[j])
        scores_shuff = m2.score(pop_t_proc, reg_vals, time_list[j])
        if j == 0:
            tcs = np.zeros((scores.shape[0], x_len))
            tcs_shuff = np.zeros((scores.shape[0], x_len))
        tcs[:, j] = 1 - scores
        tcs_shuff[:, j] = 1 - scores_shuff
    return tcs, tcs_shuff, (m1, m2), comp


def pop_regression_stan(
    pop,
    reg_vals,
    model=None,
    norm=True,
    pre_pca=0.99,
    impute_missing=False,
    do_arviz=False,
    **model_params,
):
    x_len = pop.shape[-1]
    comps = []
    steps = []
    if norm:
        steps.append(skp.StandardScaler())
    if impute_missing:
        steps.append(skimp.SimpleImputer())
    if pre_pca is not None:
        steps.append(skd.PCA(n_components=pre_pca))
    pipe = sklpipe.make_pipeline(*steps)
    pop_flat = np.concatenate(tuple(pop[:, i] for i in range(pop.shape[1])), axis=1)
    shuff_inds = np.random.choice(reg_vals.shape[0], reg_vals.shape[0], replace=False)
    reg_shuff = np.array(reg_vals)[shuff_inds]
    m1s = []
    m2s = []
    for j in range(x_len):
        pop_proc = pipe.fit_transform(pop_flat[..., j].T)
        m1 = model(**model_params)
        m1.fit(pop_proc, reg_vals)
        m2 = model(**model_params)
        m2.fit(pop_proc, reg_shuff)
        if do_arviz:
            comp = az.compare(dict(m=m1.get_arviz(), m_shuff=m2.get_arviz()))
        else:
            comp = np.nan
        scores = m1.score(pop_proc, reg_vals)
        scores_shuff = m2.score(pop_proc, reg_vals)
        if j == 0:
            tcs = np.zeros((scores.shape[0], x_len))
            tcs_shuff = np.zeros((scores.shape[0], x_len))
        tcs[:, j] = 1 - scores
        tcs_shuff[:, j] = 1 - scores_shuff
        m1s.append(m1)
        m2s.append(m2)
        comps.append(comp)
    return tcs, tcs_shuff, (m1s, m2s), comps


def fit_logit(
    measured,
    outcome,
    manifest=glm_arviz,
    model_path=stan_logit_path,
    null_model_path=stan_unif_resp_path,
    stan_iters=500,
    stan_chains=4,
    prior_width=5,
    norm=True,
):
    if norm:
        measured_m = np.mean(measured)
        measured_v = np.std(measured - measured_m)
        measured = (measured - measured_m) / measured_v
    measured = np.expand_dims(measured, 1)
    stan_data = {
        "N": len(measured),
        "K": 1,
        "y": outcome,
        "x": measured,
        "prior_width": prior_width,
    }
    sm_logit = pickle.load(open(model_path, "rb"))
    m_logit = sm_logit.sampling(data=stan_data, iter=stan_iters, chains=stan_chains)
    m_logit_az = az.from_pystan(posterior=m_logit, **manifest)
    sm_null = pickle.load(open(null_model_path, "rb"))
    m_null = sm_null.sampling(data=stan_data, iter=stan_iters, chains=stan_chains)
    m_null_az = az.from_pystan(posterior=m_null, **manifest)
    comp = az.compare(dict(logit=m_logit_az, null=m_null_az), scale="log")
    return m_logit, m_null, comp

