
import numpy as np
import scipy.stats as sts
import general.utility as u
from sklearn import svm, linear_model
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
import sklearn.exceptions as ske
import sklearn.model_selection as skms 
from dPCA.dPCA import dPCA
from hmmlearn import hmm
import warnings
import itertools as it
import string
import os
import pickle

def apply_function_on_runs(func, args, data_ind=0, drunfield='datanum',
                           ret_index=False):
    data = args[data_ind]
    runs = np.unique(data[drunfield])
    outs = []
    store_inds = []
    for run in runs:
        run_data = data[data[drunfield] == run]
        args[data_ind] = run_data
        out = func(*args)
        outs.append(out)
        store_inds.append(run)
    if ret_index:
        out = (outs, store_inds)
    else:
        out = outs
    return out

### ORGANIZE SPIKES ###

def format_raw_glm(data, constraint_funcs, shape, labels, marker_func,
                   start_time, end_time, binsize, binstep, cond_labels=None,
                   min_trials=15, min_spks=5, zscore=False,
                   full_interactions=True, double_factors=None,
                   collapse_time_zscore=False):
    marker_funcs = (marker_func,)*len(constraint_funcs)
    out = organize_spiking_data(data, constraint_funcs, marker_funcs,
                                start_time, end_time, binsize, binstep,
                                min_trials=min_trials, min_spks=min_spks,
                                zscore=zscore,
                                collapse_time_zscore=collapse_time_zscore)
    spks, xs = out
    spks = np.array(spks, dtype=object)
    spks = np.reshape(spks, shape)
    full_arr = array_format(spks, min_trials)
    if cond_labels is None:
        cond_labels = range(len(shape))
    if double_factors is None:
        double_factors = (False,)*len(shape)
    out = condition_mask(full_arr, cond_labels=cond_labels,
                         factor_labels=labels,
                         full_interactions=full_interactions,
                         double_factors=double_factors)
    dat_full, conds_full, labels_full = out 
    return dat_full, conds_full, labels_full, xs

def bin_spiketimes(spts, binsize, bounds, binstep=None, accumulate=False):
    binedges = make_binedges(binsize, bounds, binstep)
    bspks, _ = np.histogram(spts, bins=binedges)
    if accumulate:
        aspks = np.zeros_like(bspks)
        for i in range(len(bspks)):
            aspks[i] = np.sum(bspks[:i+1])
        bspks = aspks
    if binstep is not None and binstep < binsize:
        filt = np.ones(int(binsize/binstep))
        bspks = np.convolve(bspks, filt, mode='valid')
    bspks = bspks*(1000./binsize)
    return bspks   
 
def make_binedges(binsize, bounds, binstep=None):
    if binstep is not None and binstep < binsize:
        usebin = binstep
    else:
        usebin = binsize
    binedges = np.arange(bounds[0], bounds[1] + binsize + 1, usebin)
    return binedges

def collect_ISIs(spts, binsize, bounds, binstep=None, accumulate=False):
    binedges = make_binedges(binsize, bounds, binstep)
    if binstep is not None and binstep < binsize:
        n_binsper = int(np.floor(binsize / binstep)) - 1
    else:
        n_binsper = 0
    isis = np.zeros(len(binedges) - n_binsper, dtype=object)
    for i, beg in enumerate(binedges[:-n_binsper]):
        end = beg + binsize
        rel_spks = spts[np.logical_and(spts >= beg, spts < end)]
        isis[i] = np.diff(rel_spks)
    return isis

def organize_spiking_data(data, discrim_funcs, marker_funcs, pretime, posttime, 
                          binsize, binstep=None, cumulative=False, 
                          drunfield='datanum', spikefield='spike_times',
                          func_out=(bin_spiketimes, None), min_trials=None,
                          modulated_dict=None, min_spks=None, zscore=False,
                          collapse_time_zscore=False, bhv_extract_func=None,
                          causal_timing=False, remove_low_spiking=None):
    """
    Converts a sequence of MonkeyLogic trials over multiple days (with 
    associated neurons) into PSTHs oriented to markerfunc

    data - structured array, N, with fields 
    discrims - list<function> K [[structured array N]->[array<boolean> N]]
      to indicate which trials to include
    markerfuncs - list<function> K [[structured array N]->[array<float> N]]
      functions that choose start time for PSTHs
    pretime - float, time before marker to start PSTH at
    posttime - float, time after marker to end PSTH at
    binsize - float, size of PSTH bins (ms)
    binstep - float or None, size of step (ms), binstep < binsize,
      no step/smoothing if left None
    cumulative - boolean, accumulate spike counts or use windows
    """
    spk_func, out_type = func_out
    druns = np.unique(data[drunfield])
    all_discs = []
    if bhv_extract_func is not None:
        all_bhvs = []
    if binstep is not None and binstep < binsize:
        xs = np.arange(pretime + binsize/2., posttime + binsize/2. + binstep,
                       binstep)
    else:
        xs = np.arange(pretime + binsize/2., posttime + binsize/2. + 1, binsize)
    for i, df in enumerate(discrim_funcs):
        mf = marker_funcs[i]
        d_trials = data[df(data)]
        neurs = {}
        if bhv_extract_func is not None:
            bhv_vals = {}
        for j, run in enumerate(druns):
            all_dictnames = data[data[drunfield] == run][spikefield]
            nr_names = np.concatenate([list(x.keys()) for x in all_dictnames])
            n_names = np.unique(nr_names)
            udt = d_trials[d_trials[drunfield] == run]
            n_kvs = [((run, name), u.nan_array((udt.shape[0], xs.shape[0]),
                                               dtype=out_type)) 
                     for name in n_names]
            neurs.update(n_kvs)
            if bhv_extract_func is not None:
                bhv_kvs = [((run, name),
                            u.nan_array((udt.shape[0],), dtype=object))
                           for name in n_names]
                bhv_vals.update(bhv_kvs)
            marks = mf(udt)
            for k, trial in enumerate(udt):
                if bhv_extract_func is not None:
                    val = bhv_extract_func(trial)
                for l, neurname in enumerate(trial[spikefield].keys()):
                    key = (run, neurname)
                    spikes = trial[spikefield][neurname]
                    if min_spks is not None:
                        enough_spks = len(spikes) >= min_spks
                    else:
                        enough_spks = True
                    if not np.isnan(marks[k]) and enough_spks:
                        spikes = spikes - marks[k]
                        psth = spk_func(spikes, binsize, 
                                        (pretime, posttime), binstep,
                                        accumulate=cumulative)
                    else:
                        psth = np.ones_like(xs)
                        psth[:] = np.nan
                    neurs[key][k, :] = psth
                    if bhv_extract_func is not None:
                        bhv_vals[key][k] = val
            for key in neurs.keys():
                all_psth = neurs[key]
                mask = np.logical_not(np.all(np.isnan(all_psth), axis=1))
                keep_psth = all_psth[mask]
                neurs[key] = keep_psth
                if bhv_extract_func is not None:
                    bhv_vals[key] = bhv_vals[key][mask]
        all_discs.append(neurs)
        if bhv_extract_func is not None:
            all_bhvs.append(bhv_vals)
    if modulated_dict is not None:
        all_discs = filter_modulated(all_discs, modulated_dict)
    if min_trials is not None:
        all_discs = filter_min_trials(all_discs, min_trials)
    if remove_low_spiking is not None:
        all_discs = remove_low_spiking_neurons(all_discs, remove_low_spiking)
    if zscore:
        if collapse_time_zscore:
            axis = None
        else:
            axis = 0
        all_discs = zscore_organized_data(all_discs, axis=axis)
    if causal_timing:
        xs = xs + binsize/2
    out = all_discs, xs
    if bhv_extract_func is not None:
        out = out + (all_bhvs,)
    return out

def remove_low_spiking_neurons(all_discs, spiking_level, percent_trials=.9):
    ks = all_discs[0].keys()
    pop_keys = []
    for k in ks:
        keep = True
        for d in all_discs:
            firing_mask = np.mean(d[k], axis=1) >= spiking_level
            pt = np.sum(firing_mask)/len(firing_mask)
            keep = keep*(pt >= percent_trials)
        if not keep:
            pop_keys.append(k)
    for k in pop_keys:
        for d in all_discs:
            d.pop(k)
    return all_discs

def organize_spiking_data_pop(data, discrim_funcs, marker_funcs, pretime,
                              posttime, binsize, binstep=None,
                              bhv_extract_func=None, **kwargs):
    out = organize_spiking_data(data, discrim_funcs, marker_funcs, pretime,
                                posttime, binsize, binstep=binstep, 
                                bhv_extract_func=bhv_extract_func,
                                **kwargs)
    if bhv_extract_func is not None:
        spks_conds, xs, bhv = out
    else:
        spks_conds, xs = out
    dnums_conds = []
    bhvs_conds = []
    for i, spks in enumerate(spks_conds):
        dnums = {x[0]:() for x in spks.keys()}
        bhvs = {x[0]:() for x in spks.keys()}
        for (dn, s), v in spks.items():
            dnums[dn] = dnums[dn] + (v,)
            if bhv_extract_func is not None:
                bhvs[dn] = bhv[i][(dn, s)]
        for dn in dnums.keys():
            dnums[dn] = np.array(dnums[dn])
        dnums_conds.append(dnums)
        bhvs_conds.append(bhvs)
    if bhv_extract_func is None:
        out = dnums_conds, xs
    else:
        out = dnums_conds, xs, bhvs_conds
    return out

def zscore_organized_data(all_discs, axis=0):
    for k in all_discs[0].keys():
        spks_list = [d[k] for d in all_discs]
        all_trials = np.concatenate(spks_list, axis=0)
        m = np.nanmean(all_trials, axis=axis)
        v = np.nanstd(all_trials, axis=axis)
        if axis is not None:
            v[v == 0] = 1
        for d in all_discs:
            d[k] = (d[k] - m)/v
    return all_discs    

def filter_trial_selective(data, pre_marker, pre_offset, post_marker,
                           post_offset=0, binsize=500, sig_thr=.05, boots=1000,
                           drunfield='datanum', spike_field='spike_times',
                           errorfield='TrialError',
                           noerr_code=0):
    trl_select = u.make_trial_constraint_func((errorfield,),
                                              (noerr_code,), 
                                              (np.equal,),
                                              combfunc=np.logical_and)  
    pre_timer = u.make_time_field_func(pre_marker)
    ns_pre, xs_pre = organize_spiking_data(data, (trl_select,), (pre_timer,),
                                           pre_offset, binsize + pre_offset,
                                           binsize + 1)

    post_timer = u.make_time_field_func(post_marker)
    ns_post, xs_post = organize_spiking_data(data, (trl_select,), (post_timer,),
                                             post_offset, binsize + post_offset,
                                             binsize + 1)
    pre = ns_pre[0]
    post = ns_post[0]
    modulated = {}
    for k in pre.keys():
        num_trls = pre[k].shape[0]
        pre_k = pre[k]
        post_k = post[k]
        diffs = np.zeros(boots)
        for i in range(boots):
            pre_samp = u.resample_on_axis(pre[k], num_trls, axis=0)
            post_samp = u.resample_on_axis(post[k], num_trls, axis=0)
            diffs[i] = np.nanmean(pre_samp) - np.nanmean(post_samp)
        high_p = np.sum(diffs > 0)/boots
        low_p = 1 - high_p
        p = min(high_p, low_p)
        modulated[k] = p < sig_thr/2
    return modulated

def filter_modulated(all_discs, md):
    for k in list(all_discs[0].keys()):
        if not md[k]:
            for i in range(len(all_discs)):
                all_discs[i].pop(k)
                all_discs[i] = all_discs[i]
    return all_discs

def filter_min_trials(all_discs, min_trials):
    for k in list(all_discs[0].keys()):
        include = np.zeros(len(all_discs))
        for i in range(len(all_discs)):
            n_trls = all_discs[i][k].shape[0]
            include[i] = n_trls >= min_trials
        final_include = np.product(include)
        if not final_include:
            for j in range(len(all_discs)):
                all_discs[j].pop(k)
                all_discs[j] = all_discs[j]
    return all_discs

def fano_factor_tc(spks_tc, boots=1000, spks_per_s=False, window_size=None):
    ffs = np.zeros((boots, spks_tc.shape[-1]))
    ms = np.zeros_like(ffs)
    vs = np.zeros_like(ffs)
    if spks_per_s:
        spks_tc = spks_tc*(window_size/1000)
    for i in range(boots):
        samp = u.resample_on_axis(spks_tc, spks_tc.shape[0], axis=0)
        ms[i, :] = np.mean(samp, axis=0)
        vs[i, :] = np.var(samp, axis=0)
        ffs[i, :] = vs[i, :]/ms[i, :]
    return ffs, ms, vs

def _compute_snr(samps, ct=np.median, var=np.var, min_samps=2):
    m_samps = list(filter(lambda x: len(x) >= min_samps, samps))
    list_cents = list(ct(x, axis=0) for x in m_samps)
    list_vars = list(var(x, axis=0) for x in m_samps)
    snrs = var(list_cents, axis=0) / ct(list_vars, axis=0)
    return snrs

def _null_snr(samps, ct=np.median, var=np.var, min_samps=2):
    filt_samps = list(filter(lambda x: len(x) >= min_samps, samps))
    ns = (len(x) for x in filt_samps)
    flat_samps = np.concatenate(filt_samps, axis=0)
    np.random.shuffle(flat_samps)
    m_samps = []
    accum = 0
    for i, n in enumerate(ns):
        m_samps.append(flat_samps[accum:accum+n])
        accum = accum + n
    list_cents = list(ct(x, axis=0) for x in m_samps)
    list_vars = list(var(x, axis=0) for x in m_samps)
    snrs = var(list_cents, axis=0) / ct(list_vars, axis=0)
    return snrs

def snr_tc(ns, central_tend=np.median, variance=np.var, boots=1000):
    n_neurs = len(ns[0])
    ns = np.array(ns, dtype=object)
    snr_func = lambda x: _compute_snr(x, central_tend, variance)
    null_snr_func = lambda x: _null_snr(x, central_tend, variance)
    for i, k in enumerate(ns[0].keys()):
        if i == 0:
            t = ns[0][k].shape[1]
            snrs = np.zeros((n_neurs, boots, t))
            null_snrs = np.zeros_like(snrs)
        samps = np.array(list(n[k] for n in ns), dtype=object)
        snrs[i] = u.bootstrap_list(samps, snr_func, n=boots, out_shape=(t,))
        null_snrs[i] = u.bootstrap_list(samps, null_snr_func, n=boots,
                                        out_shape=(t,))
    return snrs, null_snrs

### ROC ###
                    
def roc_tc(s1_tc, s2_tc, boots=1000):
    aucs = np.zeros((boots, s1_tc.shape[1])) + .5
    if s1_tc.shape[0] > 0 and s2_tc.shape[0] > 0:
        for j in range(boots):
            s1_indsamp = np.random.choice(s1_tc.shape[0], s1_tc.shape[0])
            s2_indsamp = np.random.choice(s2_tc.shape[0], s2_tc.shape[0])
            s1_tcsamp = s1_tc[s1_indsamp, :]
            s2_tcsamp = s2_tc[s2_indsamp, :]
            for i in range(s1_tc.shape[1]):
                aucs[j, i] = roc(s1_tcsamp[:, i], s2_tcsamp[:, i])
    return aucs

def mwu_tc(s1_tc, s2_tc, alternative='two-sided'):
    us = np.zeros(s2_tc.shape[1])
    ps = np.ones_like(us)
    if s1_tc.shape[0] > 0 and s2_tc.shape[0] > 0:
        for i in range(s1_tc.shape[1]):
            try:
                us[i], ps[i] = sts.mannwhitneyu(s1_tc[:, i], s2_tc[:, i],
                                                alternative=alternative)
            except:
                us[i], ps[i] = np.nan, np.nan
    return us, ps

def roc(samples1, samples2):
    all_samples = np.concatenate((samples1, samples2))
    crit_min = np.min(all_samples)
    crit_max = np.max(all_samples)
    crits = np.arange(crit_min - 1, crit_max + 1, 1)
    p_hit = np.zeros_like(crits)
    p_fa = np.zeros_like(crits)
    for i, c in enumerate(crits):
        p_hit[i] = np.sum(samples1 > c)/len(samples1)
        p_fa[i] = np.sum(samples2 > c)/len(samples2)
    auc = -np.trapz(p_hit, p_fa)
    return auc

"""
chapter sof thesis and their status
-- what is the status of assignment problem work
-- spend most of time sketching experimental work

"""

### HMM ###
def _hmm_pop_format(pop):
    """ 
    a typical pop is neurons x trials x timepoints 
    for hmmlearn, need trials*timepoints x neurons and a list of
    all timepoint lengths
    """
    spop = np.swapaxes(pop, 0, 1)
    pop_flat = np.concatenate(spop, axis=1).T
    len_arr = np.ones(pop.shape[1], dtype=int)*pop.shape[2]
    return pop_flat, len_arr

def _hmm_trial_format(states, lens):
    n_trls = int(states.shape[0]/lens[0])
    trls = np.reshape(states, (n_trls, lens[0]))
    return trls

def fit_hmm_pops(pops, n_components, n_fits=1, min_size=2, print_pop=False):
    models = {}
    for k, pop in pops.items():
        if pop.shape[0] > min_size:
            pop_flat, len_arr = _hmm_pop_format(pop)
            if print_pop:
                print('pop {}'.format(k))
            pop_fits = []
            scores = []
            all_states = []
            for i in range(n_fits):
                m = hmm.GaussianHMM(n_components=n_components)
                fit_obj = m.fit(pop_flat, len_arr)
                pop_fits.append(fit_obj)
                score = fit_obj.score(pop_flat)
                scores.append(score)
                states = fit_obj.predict(pop_flat)
                trl_states = _hmm_trial_format(states, len_arr)
                all_states.append(trl_states)
            models[k] = (pop_fits, scores, all_states)
    return models
        
### SVM ###

def sample_trials_svm(dims, n, with_replace=False):
    n_trls = int(n*dims.shape[1])
    trls = np.zeros((len(dims), n_trls, dims[0,0].shape[1]))
    for i, d in enumerate(dims):
        for j, c in enumerate(d):
            trl_inds = np.random.choice(c.shape[0], int(n),
                                        replace=with_replace)
            trls[i, j*n:(j+1)*n, :] =  c[trl_inds, :]
    return trls

def _fold_model(cat1, cat2, leave_out=1, model=svm.SVC, norm=True, eps=.00001,
                shuff_labels=False, stability=False, params=None, 
                collapse_time=False, equal_fold=False):
    alltr = np.concatenate((cat1, cat2), axis=1)
    l1 = np.zeros(cat1.shape[1], dtype=int)
    l2 = np.ones(cat2.shape[1], dtype=int)
    alllabels = np.concatenate((l1, l2))
    inds = np.arange(alltr.shape[1])
    np.random.shuffle(inds)
    alltr = alltr[:, inds, :]
    if shuff_labels:
        inds = np.arange(alltr.shape[1])
        np.random.shuffle(inds)
    alllabels = alllabels[inds]
    if norm:
        mu = np.expand_dims(alltr.mean(1), 1)
        sig = np.expand_dims(alltr.std(1), 1)
        sig[sig < eps] = 1.
        alltr = (alltr - mu)/sig
    if equal_fold and norm:
        cat1 = (cat1 - mu)/sig
        cat2 = (cat2 - mu)/sig
    folds_n, leave_out = _compute_folds_n(cat1.shape[1], leave_out, equal_fold)
    if stability:
        results = np.zeros((folds_n, cat1.shape[2], cat1.shape[2]))
    else:
        results = np.zeros((folds_n, cat1.shape[2]))
    sup_vecs = np.zeros((folds_n, cat1.shape[2], cat1.shape[0]))
    inter = np.zeros((folds_n, cat1.shape[2]))
    for i in range(folds_n):
        if not equal_fold:
            train_tr = np.concatenate((alltr[:, (i+1)*leave_out:], 
                                       alltr[:, :i*leave_out]),
                                      axis=1)
            train_l = np.concatenate((alllabels[(i+1)*leave_out:], 
                                      alllabels[:i*leave_out]))
            test_tr = alltr[:, i*leave_out:(i+1)*leave_out]
            test_l = alllabels[i*leave_out:(i+1)*leave_out]
        else:
            train_tr = np.concatenate((cat1[:, (i+1)*leave_out:],
                                       cat1[:, :i*leave_out],
                                       cat2[:, (i+1)*leave_out:],
                                       cat2[:, :i*leave_out]),
                                      axis=1)
            train_l = np.concatenate((l1[(i+1)*leave_out:], 
                                      l1[:i*leave_out],
                                      l2[(i+1)*leave_out:], 
                                      l2[:i*leave_out]))
            test_tr = np.concatenate((cat1[:, i*leave_out:(i+1)*leave_out],
                                      cat2[:, i*leave_out:(i+1)*leave_out]),
                                     axis=1)
            test_l = np.concatenate((l1[i*leave_out:(i+1)*leave_out],
                                     l2[i*leave_out:(i+1)*leave_out]))
        out = model_decode_tc(train_tr, train_l, test_tr, test_l, model=model, 
                              stability=stability, params=params, 
                              collapse_time=collapse_time)
        results[i], sup_vecs[i], inter[i] = out
    mr = np.mean(results, axis=0)
    return mr, results, alltr, sup_vecs, inter

def model_decode_tc(train, trainlabels, test, testlabels, model=svm.SVC, 
                    stability=False, params=None, collapse_time=False):
    n_labels = float(len(testlabels))
    if params is None:
        params = {}
    if stability:
        pc_shape = (test.shape[2], test.shape[2])
    else:
        pc_shape = test.shape[2]
    percent_corr = np.zeros(pc_shape)
    svs = np.zeros((test.shape[2], test.shape[0]))
    inter = np.zeros((test.shape[2]))
    if collapse_time:
        ct_train = u.collapse_array_dim(train, 2, 1)
        ct_labels = np.tile(trainlabels, train.shape[2])
        s = model(**params)
        s.fit(ct_train.T, ct_labels)
    for i in range(train.shape[2]):
        if not collapse_time:
            s = model(**params)
            s.fit(train[:, :, i].T, trainlabels)
        if stability:
            for j in range(train.shape[2]):
                preds = s.predict(test[:, :, j].T)
                percent_corr[i,j] = np.sum(preds == testlabels) / n_labels
        else:
            preds = s.predict(test[:, :, i].T)
            percent_corr[i] = np.sum(preds == testlabels) / n_labels
        if s.kernel == 'linear':
            svs[i] = s.coef_[0]
            inter[i] = s.intercept_[0]
    return percent_corr, svs, inter

def svm_decoding(cat1, cat2, leave_out=1, require_trials=15, resample=100,
                 with_replace=False, shuff_labels=False, stability=False,
                 penalty=1, format_=True, model=svm.SVC, kernel='linear',
                 collapse_time=False, max_iter=1000, gamma='scale', pop=False,
                 min_population=1, multi_cond=False, **kwargs):
    spec_params = {'C':penalty, 'max_iter':max_iter, 'gamma':gamma,
                   'kernel':kernel, 'class_weight':'balanced'}
    if kernel != 'linear':
        spec_params.update(('kernel', kernel),)
        spec_params.pop('penalty')
        spec_params.pop('dual')
        spec_params.pop('loss')
    if pop:
        out = decoding_pop(cat1, cat2, leave_out=leave_out, 
                           require_trials=require_trials, resample=resample,
                           with_replace=with_replace, shuff_labels=shuff_labels,
                           stability=stability, params=spec_params,
                           model=model,
                           min_population=min_population,
                           collapse_time=collapse_time, **kwargs)
    else:
        out = decoding(cat1, cat2, leave_out=leave_out, multi_cond=multi_cond,
                       require_trials=require_trials, resample=resample,
                       with_replace=with_replace, shuff_labels=shuff_labels,
                       stability=stability, params=spec_params, format_=format_,
                       model=model, collapse_time=collapse_time, **kwargs)
    return out

def decoding_pop(cat1, cat2, model=svm.SVC, leave_out=1, require_trials=15, 
                 resample=100, with_replace=False, shuff_labels=False, 
                 stability=False, params=None, collapse_time=False,
                 min_population=1, use_avail_trials=True, equal_fold=False,
                 **kwargs):
    n_pops = len(cat1.keys())
    pop_shape = list(cat1.values())[0].shape
    n_times = pop_shape[2]
    if stability:
        tcs_shape = (resample, n_times, n_times)
    else:
        tcs_shape = (resample, n_times)
    tcs_pops = {}
    ms_pops = {}
    for k, c1_pop in cat1.items():
        c2_pop = cat2[k]
        n_neurs = c1_pop.shape[0]
        n1_trials = c1_pop.shape[1]
        n2_trials = c2_pop.shape[1]
        if (n_neurs >= min_population and n1_trials >= require_trials
            and n2_trials >= require_trials):
            if use_avail_trials:
                use_trials = min(n1_trials, n2_trials)
            else:
                use_trials = require_trials
            folds_n, leave_out = _compute_folds_n(use_trials, leave_out,
                                                  equal_fold)
            ms_shape = (resample, folds_n, n_times, n_neurs)
            tcs = np.zeros(tcs_shape)
            ms = np.zeros(ms_shape)
            for i in range(resample):
                c1_samp = u.resample_on_axis(c1_pop, use_trials, 1,
                                             with_replace=with_replace)
                c2_samp = u.resample_on_axis(c2_pop, use_trials, 1,
                                             with_replace=with_replace)
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    out = _fold_model(c1_samp, c2_samp, leave_out,
                                      model=model,
                                      shuff_labels=shuff_labels,
                                      stability=stability, params=params,
                                      collapse_time=collapse_time,
                                      equal_fold=equal_fold,
                                      **kwargs)
                    tcs[i], _, _, ms[i], _ = out
            tcs_pops[k] = tcs
            ms_pops[k] = ms
    return tcs_pops, ms

def _svm_organize(c1, c2, require_trials=20, use_avail_trials=True):
    lens = np.ones(len(c1[0]))*np.inf
    c1_arr = np.zeros((lens.shape[0], len(c1)), dtype=object)
    c2_arr = np.zeros((lens.shape[0], len(c2)), dtype=object)
    for i, k in enumerate(c1[0].keys()):
        for j, c1_i in enumerate(c1):
            c1_i_k = len(c1_i[k])
            lens[i] = min(lens[i], c1_i_k)
            c1_arr[i, j] = c1_i[k]
        for j, c2_i in enumerate(c2):
            c2_i_k = len(c2_i[k])
            lens[i] = min(lens[i], c2_i_k)
            c2_arr[i, j] = c2_i[k]
    mask = lens >= require_trials
    lens = lens[mask]
    c1_arr = c1_arr[mask]
    c2_arr = c2_arr[mask]
    if use_avail_trials:
        require_trials = min(lens)
    x_len = c1_arr[0, 0].shape[1]
    return c1_arr, c2_arr, int(require_trials), x_len
    

def decoding(cat1, cat2, model=svm.SVC, leave_out=1, require_trials=15, 
             resample=100, with_replace=False, shuff_labels=False, 
             stability=False, params=None, collapse_time=False,
             format_=True, equal_fold=False, multi_cond=False,
             use_avail_trials=True,
             **kwargs):
    if format_:
        if not multi_cond:
            cat1 = (cat1,)
            cat2 = (cat2,)
        out = _svm_organize(cat1, cat2, require_trials=require_trials,
                            use_avail_trials=use_avail_trials)
        cat1_f, cat2_f, require_trials, x_len = out
    else:
        if not multi_cond:
            cat1_f = np.array(cat1, dtype=object)
            cat1_f = np.expand_dims(cat1_f, 1)
            cat2_f = np.array(cat2, dtype=object)
            cat2_f = np.expand_dims(cat2_f, 1)
        else:
            cat1_f = cat1
            cat2_f = cat2
        x_len = cat2_f[0, 0].shape[1]
        if use_avail_trials:
            c1_min = min(len(c1_i) for c1_i in cat1_f)
            c2_min = min(len(c2_i) for c2_i in cat2_f)
            require_trials = min(c1_min, c2_min)
    if stability:
        tcs_shape = (resample, x_len, x_len)
    else:
        tcs_shape = (resample, x_len)
    folds_n, leave_out = _compute_folds_n(require_trials*cat1_f.shape[1],
                                          leave_out, equal_fold)
    ms = np.zeros((resample, folds_n, x_len, cat1_f.shape[0]))
    inter = np.zeros((resample, folds_n, x_len))
    tcs = np.zeros(tcs_shape)
    for i in range(resample):
        cat1_samp = sample_trials_svm(cat1_f, require_trials, with_replace)
        cat2_samp = sample_trials_svm(cat2_f, require_trials, with_replace)
        out = _fold_model(cat1_samp, cat2_samp, leave_out, model=model,
                          shuff_labels=shuff_labels, stability=stability, 
                          params=params, collapse_time=collapse_time,
                          equal_fold=equal_fold, **kwargs)
        tcs[i], _, _, ms[i], inter[i] = out
    return tcs, cat1_f, cat2_f, ms, inter

def _compute_folds_n(use_trials, leave_out, equal_fold=False):
    if equal_fold:
        fact = 1
    else:
        fact = 2
    if leave_out < 1:
        leave_out = int(np.ceil(fact*use_trials*leave_out))
    folds_n = int(np.floor(use_trials*fact/leave_out))
    return folds_n, leave_out

def svm_cross_decoding(c1_train, c1_test, c2_train, c2_test, require_trials=15,
                       model=svm.SVC, stability=False, shuff_labels=False,
                       params=None, collapse_time=False, format_=True,
                       with_replace=False, max_iter=1000, gamma='scale',
                       penalty=1, kernel='linear',
                       resample=100, **kwargs):
    params = {'C':penalty, 'max_iter':max_iter, 'gamma':gamma,
                   'kernel':kernel}
    if kernel != 'linear':
        params.update(('kernel', kernel),)
        params.pop('penalty')
        params.pop('dual')
        params.pop('loss')
    if format_:
        cat1 = np.array(list(c1_train.values()))
        cat2 = np.array(list(c2_train.values()))
        bool1 = [x.shape[0] < require_trials for x in cat1]
        bool2 = [x.shape[0] < require_trials for x in cat2]
        combool = np.logical_not(np.logical_or(bool1, bool2))
        c1_train_f = cat1[combool]
        c2_train_f = cat2[combool]
        cat1 = np.array(list(c1_test.values()))
        cat2 = np.array(list(c2_test.values()))
        bool1 = [x.shape[0] < require_trials for x in cat1]
        bool2 = [x.shape[0] < require_trials for x in cat2]
        combool = np.logical_not(np.logical_or(bool1, bool2))
        c1_test_f = cat1[combool]
        c2_test_f = cat2[combool]
    else:
        c1_train_f = cat1
        c2_train_f = cat2
        c1_test_f = c1_test
        c2_test_f = c2_test
    if stability:
        tcs_shape = (resample, c1_train_f[0].shape[1], c1_train_f[0].shape[1])
    else:
        tcs_shape = (resample, c1_train_f[0].shape[1])
    folds_n = 1
    ms = np.zeros((resample, folds_n, c1_train_f[0].shape[1],
                   c1_train_f.shape[0]))
    inter = np.zeros((resample, folds_n, c1_train_f[0].shape[1]))
    tcs = np.zeros(tcs_shape)

    for i in range(resample):
        c1_train_samp = sample_trials_svm(c1_train_f, require_trials,
                                          with_replace)
        c2_train_samp = sample_trials_svm(c2_train_f, require_trials,
                                          with_replace)
        c1_test_samp = sample_trials_svm(c1_test_f, require_trials,
                                          with_replace)
        c2_test_samp = sample_trials_svm(c2_test_f, require_trials,
                                          with_replace)
        train_samp = np.concatenate((c1_train_samp, c2_train_samp),
                                    axis=1)
        test_samp = np.concatenate((c1_test_samp, c2_test_samp),
                                   axis=1)
        trainlabels = np.concatenate((np.zeros(c1_train_samp.shape[1],
                                               dtype=int), 
                                      np.ones(c2_train_samp.shape[1],
                                              dtype=int)))
        testlabels = np.concatenate((np.zeros(c1_test_samp.shape[1],
                                              dtype=int), 
                                     np.ones(c2_test_samp.shape[1],
                                             dtype=int)))
        
        out = model_decode_tc(train_samp, trainlabels, test_samp, testlabels,
                              model=model, stability=stability, params=params,
                              collapse_time=collapse_time)
        tcs[i], ms[i], inter[i] = out
    return tcs, ms, inter


def svm_multi_decoding(data, leave_out=1, require_trials=15, resample=50,
                       with_replace=False, shuff_labels=False, stability=False,
                       kernel='linear', penalty=1, collapse_time=False,
                       norm=True, regularizer='l1', dual=False,
                       loss='squared_hinge', max_iter=1000):
    # spec_params = {'C':penalty, 'kernel':kernel, 'penalty':'l1'}
    spec_params = {'C':penalty, 'penalty':regularizer, 'dual':dual, 'loss':loss,
                   'max_iter':max_iter}
    model = svm.LinearSVC
    out = multi_decoding(data, model=model, leave_out=leave_out, 
                         require_trials=require_trials,
                         resample=resample, with_replace=with_replace, 
                         shuff_labels=shuff_labels, stability=stability,
                         params=spec_params, collapse_time=collapse_time,
                         norm=norm)
    return out

def lda_multi_decoding(data, leave_out=1, require_trials=15, resample=50,
                       with_replace=False, shuff_labels=False, stability=False,
                       shrinkage=None, collapse_time=False):
    spec_params = {'shrinkage':shrinkage}
    model = da.LinearDiscriminantAnalysis
    out = multi_decoding(data, model=model, leave_out=leave_out, 
                         require_trials=require_trials,
                         resample=resample, with_replace=with_replace, 
                         shuff_labels=shuff_labels, stability=stability,
                         params=spec_params, collapse_time=collapse_time)
    return out
                         
def multi_decoding(data, model=svm.SVC, leave_out=1, require_trials=15, 
                   resample=50, with_replace=False, shuff_labels=False, 
                   stability=False, params=None, collapse_time=False,
                   norm=True):
    arr = array_format(data, require_trials)
    arr = np.swapaxes(arr, 0, 1)
    if stability:
        tcs_shape = (resample, len(arr.shape[3:]), arr.shape[2], arr.shape[2])
    else:
        tcs_shape = (resample, len(arr.shape[3:]), arr.shape[2])
    folds_n = int(np.floor((require_trials*2*len(arr.shape[3:]))/leave_out))
    ms = np.zeros((resample, len(arr.shape[3:]), folds_n,
                   arr.shape[2], arr.shape[0]))
    tcs = np.zeros(tcs_shape)
    used_arrs = np.zeros((resample,)+arr.shape)
    for i in range(resample):
        arr = array_format(data, require_trials, with_replace)
        arr = np.swapaxes(arr, 0, 1)
        for j in range(len(arr.shape[3:])):
            col_arr = u.collapse_array_dim(arr, 3 + j, stack_dim=1)
            out = _fold_model(col_arr[..., 0], col_arr[..., 1], leave_out,
                              model=model, shuff_labels=shuff_labels, 
                              stability=stability, params=params, 
                              collapse_time=collapse_time, norm=norm)
            tcs[i, j], _, _, ms[i, j], _ = out
        used_arrs[i] = arr
    return tcs, ms, used_arrs

def glm_fit_full(data, ind_structure, constr_funcs, marker_func, start_time,
                 end_time, binsize, binstep, req_trials=15, cond_labels=None,
                 double_factors=None, perms=5000, demean=True, zscore=True,
                 alpha=1, min_spks=1, interactions=None, with_replace=False,
                 use_all=True):
    marker_funcs = (marker_func,)*len(constr_funcs)
    out = organize_spiking_data(data, constr_funcs, marker_funcs,
                                start_time, end_time, binsize, binstep,
                                min_spks=min_spks)
    spks, xs = out
    out = glm_fitting_diff_trials(spks, ind_structure, req_trials=req_trials,
                                  with_replace=with_replace,
                                  cond_labels=cond_labels,
                                  interactions=interactions, use_all=use_all,
                                  double_factors=double_factors, perms=perms,
                                  demean=demean, zscore=zscore, alpha=alpha)
    return out, xs

def glm_asymm_trials_format(neurs, key, ind_structure, cond_labels=None,
                            single_conds=(), interactions=(),
                            double_factors=None, full_interactions=False,
                            factor_labels=None):
    inds_arr = np.array(ind_structure)
    n_vals = np.max(inds_arr, axis=0) + 1
    dum_arr = np.zeros((1, 1, 1,) + tuple(n_vals))
    out = condition_mask(dum_arr, cond_labels=cond_labels,
                         single_conds=single_conds,
                         interactions=interactions,
                         double_factors=double_factors,
                         full_interactions=full_interactions,
                         factor_labels=factor_labels)
    _, conds, labels = out
    for i, ind in enumerate(ind_structure):
        curr_dat = neurs[ind][key].T
        curr_cond = conds[0, i]
        if i == 0:
            full_dat = curr_dat
            full_cond = (curr_cond,)*curr_dat.shape[1]
        else:
            full_dat = np.concatenate((full_dat, curr_dat), axis=1)
            full_cond = full_cond + (curr_cond,)*curr_dat.shape[1]
    full_dat = np.expand_dims(full_dat, 1)
    full_cond = np.expand_dims(np.array(full_cond), 0)
    return full_dat, full_cond, labels

def glm_fitting_diff_trials(dat, ind_structure, req_trials=15, 
                            with_replace=False, cond_labels=None,
                            interactions=None, double_factors=None,
                            perms=5000, demean=True, zscore=True, alpha=1,
                            xs_mask=None, use_all=True):
    neur_form_shape = np.max(ind_structure, axis=0) + 1
    glm_coeffs = []
    glm_ps = []
    glms = []
    if zscore:
        alpha = alpha*.1
    for k in dat[0].keys():
        neur_form = np.zeros(neur_form_shape, dtype=dict)
        trls = np.inf
        for j, d_j in enumerate(dat):
            trls = int(np.min((trls, len(d_j[k]))))
            neur_form[ind_structure[j]] = {k:d_j[k]}
        if trls >= req_trials:
            if use_all:
                out = glm_asymm_trials_format(neur_form, k, ind_structure,
                                              cond_labels=cond_labels,
                                              interactions=interactions,
                                              double_factors=double_factors)
            else:
                arr_form = array_format(neur_form, trls,
                                        with_replace=with_replace)
                out = condition_mask(arr_form, cond_labels=cond_labels,
                                     interactions=interactions,
                                     double_factors=double_factors)
            glm_dat, glm_conds, _ = out
            
            if xs_mask is None:
                xs_mask = np.ones(glm_dat.shape[0], dtype=bool)
            out = fit_glms_with_perm(glm_dat[xs_mask], glm_conds, alpha=alpha,
                                     perms=perms, demean=demean, z_score=zscore)
            models, coeffs, ps_coeff, null_coeffs = out
            if not np.any(np.isnan(coeffs)):
                glm_coeffs.append(coeffs)
                glm_ps.append(ps_coeff)
                glms.append((k, models))

    full_coeffs = np.concatenate(glm_coeffs, axis=0)
    full_ps = np.concatenate(glm_ps, axis=0)
    return full_coeffs, full_ps, glms

def array_format(data, require_trials, with_replace=True, normalize=None,
                 norm_func=None):
    """
    Format PSTH data into array for use in some analyses.

    Parameters
    ----------
    data : list<list<...<dict>...>>
        An arbitrary depth list, terminating in a dictionary, each level within
        the list will be considered a separate dimension for decomposition, all
        dictionaries must have the same keys
    require_trials : int
        The number of trials required per neuron (per condition) for inclusion 
        of that neuron in the population
    with_replace : boolean
        sample trials for inclusion in the pseudopopulation with replacement or
        not

    Returns
    -------
    out : ndarray
        Array of shape (require_trials, N, T, ...) where N is the number of 
        neurons with trials >= require_trials for each condition and T is the
        number of timepoints in the original data. The rest of the dimensions
        come from the structure of the input. 
    """
    dat, bools = _array_format_helper(data, require_trials, 
                                      with_replace=with_replace)
    out = dat[:, bools]
    if normalize is not None:
        out = norm_func(out, axis=normalize)
    return out

def _array_format_helper(data, require_trials, shape=(), inds=(), 
                       with_replace=True):
    try:
        ks = list(data.keys())
        ts = data[ks[0]].shape[1]
        mkb = np.array([data[k].shape[0] >= require_trials for k in ks])
        datarr = np.zeros((require_trials, len(ks), ts) + shape)
        for j, k in enumerate(ks):
            if mkb[j]:
                ref = (slice(0, require_trials), j, slice(0, ts)) + inds
                data_samp = u.resample_on_axis(data[k], require_trials, axis=0,
                                               with_replace=with_replace)
                datarr[ref] = data_samp
    except AttributeError:
        shape = shape + (len(data),)
        for i, ent in enumerate(data):
            ent_inds = inds + (i,)
            dat_slice, keybool = _array_format_helper(ent, require_trials, 
                                                      shape=shape, 
                                                      inds=ent_inds,
                                                      with_replace=with_replace)
            if i == 0:
                mkb = keybool
                datarr = dat_slice
            else:
                mkb = mkb*keybool
                datarr = datarr + dat_slice
    return datarr, mkb

def angle_distribution(vecs, axis=0, degrees=True):
    vecs = np.swapaxes(vecs, axis, 0)
    n_vecs = vecs.shape[0]
    pairs = list(it.combinations(range(n_vecs), 2))
    angs = np.zeros(len(pairs))
    for i, p in enumerate(pairs):
        angs[i] = u.vector_angle(vecs[p[0]], vecs[p[1]], degrees=degrees)
    return angs    

def angle_deviation_tc(vecs, ref_t, degrees=True, within_samp=False):
    """
    Produce a timecourse of angle deviation with respect to some reference
    time point. This will quantify how stable the separating hyperplane is
    across time.

    Parameters
    ----------
    vecs : array
        Array of shape (K, T, N), where K are samples, T is time, N is 
        dimensionality.
    ref_t : int
        Index to reference in the time dimension, all measurements of angle
        difference will be taken with respect the vectors at this index.

    Returns
    -------
    out : array
        Array of shape (Kchoose2, T), which are the angle differences at each
        time point.    
    """
    comp_vecs = np.ones_like(vecs)*vecs[:, ref_t:ref_t + 1, :]
    out = compare_angles_tc(vecs, comp_vecs, degrees=degrees, 
                            within_samp=within_samp)
    return out

def angle_nulldistrib_tc(vs1, vs2, samples, degrees=True, within_samp=False,
                         central_tendency=np.nanmedian):
    distrib = np.zeros((samples, vs1.shape[1]))
    for i in range(samples):
        samp = compare_angles_tc(vs1, vs2, degrees=degrees, 
                                 within_samp=within_samp, shuff=True)
        distrib[i] = central_tendency(samp, axis=0)
    return distrib

def compare_angles_tc(vs1, vs2, degrees=True, within_samp=False, shuff=False):
    if shuff:
        l1, l2 = vs1.shape[0], vs2.shape[0]
        comb = np.concatenate((vs1, vs2), axis=0)
        r = list(range(comb.shape[0]))
        np.random.shuffle(r)
        vs1 = comb[r[:l1]]
        vs2 = comb[r[l1:]]
    if within_samp:
        assert(vs1.shape[0] == vs2.shape[0])
        pairs = list(zip(range(vs1.shape[0]), range(vs2.shape[0])))
    else:
        pairs = list(it.product(range(vs1.shape[0]), 
                                       range(vs2.shape[0])))
    n_pairs = len(pairs)
    comps = np.zeros((n_pairs, vs1.shape[1]))
    for t in range(vs1.shape[1]):
        for j, p in enumerate(pairs):
            comps[j, t] = u.vector_angle(vs1[p[0], t], vs2[p[1], t], 
                                         degrees=degrees)
    return comps

def basis_trajectories(data, basis_vecs):
    """
    Find trajectory of data relative to some basis vectors.

    Parameters
    ----------
    data : array
        Array of shape (K, N, T) where K is samples, N is dimensionality, 
        T is timepoints.
    basis_vecs : list of array-like
        A list of C vectors to use as the basis set for constructing the 
        trajectories. Each vector should be length N.
    
    Returns
    -------
    out : array
        Array of shape (K, C, T), which are the same K trials in the new 
        coordinate space. 
    """
    trans_mat = np.stack(basis_vecs, axis=0)
    out = np.zeros((data.shape[0], trans_mat.shape[0], data.shape[2]))
    for k in range(data.shape[0]):
        out[k, :, :] = np.dot(trans_mat, data[k, :, :])
    return out

### PCA and dPCA ###        

def dpca_wrapper(data, labels, regularizer='auto', signif=False, protect='t',
                 n_shuffles=10, n_splits=10, n_consecutive=10):
    for i, char in enumerate(protect):
        orig_ind = labels.index(char)
        orig_arrind = orig_ind + (len(data.shape) - len(labels))
        data = np.swapaxes(data, orig_arrind, -(i + 1))
        labels[orig_ind] = labels[-(i+1)]
        labels[-(i+1)] = char
    labels = ''.join(labels)
    data_mean = np.nanmean(data, axis=0)
    dpca = dPCA(labels=labels, regularizer=regularizer)
    dpca.protect = protect
    fd = dpca.fit_transform(data_mean, trialX=data)
    ret = fd
    if signif: 
        s_mask = dpca.significance_analysis(data_mean, data, axis='t', 
                                            n_shuffle=n_shuffles,
                                            n_splits=n_splits,
                                            n_consecutive=n_consecutive)
        ret = (fd, s_mask)
    return ret, dpca

def estimate_pca(data, require_trials=15, normalize=None, norm_func=None,
                 ests=1000):
    pcas = []
    
    for i in range(ests):
        arr = array_format(data, require_trials=require_trials,
                           normalize=normalize, norm_func=norm_func)
        if i == 0:
            coeffs = np.zeros((ests, arr.shape[1], arr.shape[1]))
            exp_var = np.zeros((ests, arr.shape[1]))
            trans_pts = np.zeros((ests,)+arr.shape)
        pca = pca_wrapper(arr)
        coeffs[i] = pca.components_
        exp_var[i] = pca.explained_variance_ratio_
        for j in range(arr.shape[-1]):
            out = pca_trajectories(arr[..., j], pca,
                                   n_comp=arr.shape[1])
            trans_pts[i, ..., j] = out[1]
        pcas.append(pca)
    return coeffs, exp_var, trans_pts, pcas
        
def pca_wrapper(data):
    sh = data.shape
    flat_data = np.reshape(data, (-1, sh[1]))
    pca = PCA()
    pca.fit(flat_data)
    return pca

def pca_trajectories(data, pca, n_comp=None, comps=(0,1)):
    if n_comp is not None:
        comps = tuple(range(n_comp))
    all_traj = np.zeros((data.shape[0], len(comps), data.shape[2]))
    mean_traj = pca.transform(np.nanmean(data, axis=0).T).T[comps, :]
    for i, d in enumerate(data):
        all_traj[i] = pca.transform(data[i].T).T[comps, :]
    return mean_traj, all_traj

def function_on_dim(data, func, dims=None, norm_dims=True,
                    boots=1000):
    """
    Bootstrap a function on some collection of dimensions across some
    timecourse.

    Parameters
    ----------
    data : array
        Array organized as by the array_format function, shape (K, N, T)
        where K is the number of trials, N is the number of neurons, and T
        is the number of time points
    func : function
        Function to apply to the data, should take an axis parameter.
    dims : list[array]
        List of arrays of shape (N,)
    norm_dims : boolean
        Normalize the vector to length 1 (default True)
    boots : int
        Number of times to resample the trials for constructing estimate of
        function value (default 1000)    
    """
    if dims is None:
        dims = [np.ones(data.shape[1])]
    n_dim = [dim / np.sqrt(np.sum(dim**2)) for dim in dims]
    data = basis_trajectories(data, dims)
    boot_data = np.zeros((boots,) + data.shape[1:])
    for i in range(boots):
        samp = u.resample_on_axis(data, data.shape[0], axis=0)
        boot_data[i] = func(samp, axis=0)
    return boot_data        

### (Generalized) Linear Models ###

def _generate_label_list(n_factors, ref_string=string.ascii_lowercase):
    labels = []
    for i in range(n_factors):
        string_ind = i % len(ref_string)
        string_mult = int(i/len(ref_string) + 1)
        labels.append(ref_string[string_ind]*string_mult)
    return labels

def _generate_interaction_terms(inter, labels, with_replace=True):
    x = list(filter(lambda x: x[0] in inter, labels))
    if with_replace:
        combos = list(it.combinations_with_replacement(x, 
                                                              len(inter)))
    else:
        combos = list(it.combinations(x, len(inter)))
    return combos

def _generate_factor_labels(factors, labels=None, interactions=(),
                            double_factors=None, factor_labels=None):
    if labels is None:
        labels = _generate_label_list(len(factors))
    if double_factors is None:
        double_factors = len(factors)*(True,)
    if factor_labels is None:
        factor_labels = [list(range(f)) for f in factors]
    factor_singles = [list(it.product((l,), factor_labels[i]))
                      for i, l in enumerate(labels)]
    full_labels = list(it.chain(*factor_singles))
    full_labels = list([(x,) for x in full_labels])
    for inter in interactions:
        singles = [factor_singles[i] for i in inter]
        new_terms = list(it.product(*singles))
        full_labels = full_labels + new_terms
    return full_labels, labels, factor_labels

def _generate_cond_refs(labels, comb, cond_labels, ind_sizes, factor_labels):
    labs = np.zeros(len(labels))
    prod_size = np.sum(ind_sizes)
    for i, l in enumerate(labels):
        if i >= prod_size:
            prod = [labs[:prod_size][labels.index((x,))] for x in l]
            labs[i] = np.product(prod)
        else:
            ind = cond_labels.index(l[0][0])
            x = comb[ind]
            if len(l[0]) > 1 and x == factor_labels[ind].index(l[0][1]):
                labs[i] = 1
            elif len(l[0]) > 1:
                labs[i] = 0
            elif x == 1:
                labs[i] = 1
            else:
                labs[i] = -1
    return labs

def condition_mask(data, cond_labels=None, single_conds=(), interactions=(),
                   double_factors=None, full_interactions=False,
                   factor_labels=None):
    """
    Format array data for production of (generalized) linear models.

    Parameters
    ----------
    data : array
        An array of shape (K, N, T, F_1, ..., F_C), where K is the number of 
        trials, N is the number of neurons, and T is the number of timepoints. 
        The remaining dimensions, F_1 ... F_C, are the factors.
    cond_labels : None or list
        Labels for each of the axes past the first three in the data array, 
        will also be used to generate labels for the specified interaction 
        terms; if left blank, will just be populated from the alphabet.
    single_conds : None or list
        List of inds for which to use a single factor with values (1, -1) 
        rather than the default two factor (values 1, 0)
    interactions : list of lists of ints
        Dimensions for which to include interaction terms.
    double_factors : list of boolean
        If square terms should be included, default is true -- typically doesn't
        make sense for single factor conditions.

    Returns 
    -------
    format_data : array
        An array of shape (T, N, K*(F_1 + ... + F_C)) with the data to be fit.
    format_cond : array
        An array of shape (N, K*(F_1 + ... + F_C), F_1*...*F_C) with the 
        conditions to be fit to. 
    """
    if full_interactions:
        interactions = u.generate_all_combinations(len(data.shape) - 3, 2)
    n_trials = data.shape[0]*np.product(data.shape[3:])
    factors = list(data.shape[3:])
    out = _generate_factor_labels(factors, cond_labels, 
                                  interactions,
                                  factor_labels=factor_labels,
                                  double_factors=double_factors)
    labels, cond_labels, factor_labels = out
    n_factors = len(labels)
    format_data = np.zeros((data.shape[2], data.shape[1], n_trials))
    format_cond = np.zeros((data.shape[1], n_trials, n_factors))
    for i in range(data.shape[1]):
        neur = data[:, i]
        ind_sizes = data.shape[3:]
        inds = list(it.product(*[range(x) for x in ind_sizes]))
        for j in range(n_trials):
            comb_ind = int(np.floor(j/data.shape[0]))
            comb = inds[comb_ind]
            trl_ind = j % data.shape[0]
            data_ind = (trl_ind, i, slice(0, data.shape[2])) + comb
            format_data[:, i, j] = data[data_ind]
            cs = _generate_cond_refs(labels, comb, cond_labels, 
                                     ind_sizes, factor_labels)
            format_cond[i, j] = cs
    cm = format_cond[0].sum(axis=0) > 0
    format_cond = format_cond[:, :, cm]
    labels = np.array(labels, dtype=object)[cm]
    return format_data, format_cond, labels

def generate_null_glm_coeffs(data, conds, perms=100, use_stan=False,
                             demean=False, z_score=False, alpha=None):
    if demean:
        coeff_add = 0
    else:
        coeff_add = 1
    null_coeff_pop = np.zeros((perms, data.shape[1], data.shape[0], 
                               conds.shape[2] + coeff_add))
    for i in range(perms):
        shuff_conds = u.resample_on_axis(conds, conds.shape[1], axis=1, 
                                         with_replace=False)
        _, null_coeff_pop[i] = fit_glms(data, shuff_conds, use_stan=use_stan,
                                        demean=demean, z_score=z_score,
                                        alpha=alpha)
    return null_coeff_pop

def fit_glms_with_perm(data, conds, perms=100, use_stan=False, demean=False,
                       z_score=False, alpha=None):
    mp, cp = fit_glms(data, conds, use_stan=use_stan, demean=demean,
                      z_score=z_score, alpha=alpha)
    null_cp = generate_null_glm_coeffs(data, conds, perms, use_stan=use_stan,
                                       demean=demean, z_score=z_score,
                                       alpha=alpha)
    exp_cp = np.expand_dims(cp, axis=0)
    higher = np.sum(exp_cp >= null_cp, axis=0) / perms
    lower = np.sum(exp_cp <= null_cp, axis=0) / perms
    ps = np.min(np.stack((lower, higher), axis=0), axis=0)
    return mp, cp, ps, null_cp

def fit_glms(data, conds, use_stan=False, demean=False, z_score=False,
             alpha=None):
    model_pop = []
    if demean:
        coeff_add = 0
    else:
        coeff_add = 1
    coeffs_pop = np.zeros((data.shape[1], data.shape[0],
                           conds.shape[2] + coeff_add))
    for i, neur in enumerate(conds):
        if use_stan:
            print('neur {} / {}'.format(i + 1, len(conds)))
        ms, cs = generalized_linear_model(data[:, i, :], neur, 
                                          use_stan=use_stan, demean=demean,
                                          z_score=z_score, alpha=alpha)
        model_pop.append(ms)
        coeffs_pop[i] = cs
    return model_pop, coeffs_pop

stan_file_trunk = ('/Users/wjj/Dropbox/research/uc/freedman/analysis/general/'
                   'stan_models/')
stan_file_glm_mean = os.path.join(stan_file_trunk, 'glm_fitting.pkl')
stan_file_glm_nomean = os.path.join(stan_file_trunk, 'glm_fitting_nomean.pkl')

def generalized_linear_model(data, conds, use_stan=False, stan_chains=4, 
                             stan_iters=10000, stan_file=stan_file_glm_mean,
                             demean=False, z_score=False, alpha=None):
    """
    Fit a generalized linear model to data.

    Parameters
    ----------
    data : array
        An array of shape (T, K) where T is timepoints and K is the number of
        samples at each timepoint.
    conds : array
        An array of shape (K, F) where K is the number of samples at each 
        timepoint and F specifies the task conditions for each of those samples.
    use_stan : boolean
        If true, will fit the model using stan; if false, will use linear 
        regression.
    Returns
    -------
    models : list of objects
        The fit model at each timepoint from scikit-learn.
    coeffs : array
        An array of shape (T, F + 1), which are the model coefficients (the 
        labels as well as a mean term) at each timepoint. The coefficients are
        in the order of the labels, with the mean term in the zeroeth position.
    """
    models = []
    if alpha is None:
        alpha = 1
    if demean:
        data = data - np.mean(data, axis=1).reshape((-1, 1))
        fit_inter = False
        coeff_add = 0
    else:
        fit_inter = True
        coeff_add = 1
    if z_score:
        m = np.mean(data, axis=1).reshape((-1, 1))
        dm_data = data - m
        stdvar = np.std(dm_data, axis=1).reshape((-1, 1))
        zs_data = dm_data/stdvar
        data = zs_data + m
        if alpha is None:
            alpha = .1
    coeffs = np.zeros((data.shape[0], conds.shape[1] + coeff_add))    
    for t in range(data.shape[0]):
        if not np.all(np.isnan(data[t, :])):
            if use_stan:
                conds = conds.astype(int)
                if demean:
                    sf = stan_file_glm_nomean
                else:
                    sf = stan_file_glm_mean
                stan_data = {'N':data.shape[1], 'K':conds.shape[1], 'x':conds, 
                             'y':data[t, :]}
                sm = pickle.load(open(sf, 'rb'))
                m = sm.sampling(data=stan_data, iter=stan_iters, 
                                chains=stan_chains)
                coeffs[t] = m.get_posterior_mean()[:conds.shape[1], 0]
                if not u.stan_model_valid(m):
                    m = None
                    coeffs[t] = np.nan
            else:
                m = linear_model.Lasso(fit_intercept=fit_inter, alpha=alpha)
                m.fit(conds, data[t, :])
                if demean:
                    coeffs[t] = m.coef_
                else:
                    coeffs[t, 0] = m.intercept_
                    coeffs[t, 1:] = m.coef_
        else:
            m = None
            coeffs[t, :] = np.nan
        models.append(m)
    return models, coeffs
    
# spiking variability

def _firing_rate_range(x, fr_range):
    out = np.logical_and(len(x) + 1 >= fr_range[0], len(x) + 1 < fr_range[1])
    return out

def fit_distrib_population(ns, fr_ranges, distrib=sts.gengamma, require_isis=50,
                           n_params=4, elim_lessthan=8, elim_greaterthan=1000,
                           boots=10):
    isi_ns = {}
    isi_dist = {}
    for k in ns:
        v = ns[k]
        out = fit_distrib_neur(v, fr_ranges, distrib=distrib, 
                               require_isis=require_isis, n_params=n_params,
                               elim_lessthan=elim_lessthan, 
                               elim_greaterthan=elim_greaterthan,
                               boots=boots)
        isi_ns[k], isi_dist[k] = out
    return isi_ns, isi_dist        

def fit_distrib_neur(neur, fr_ranges, distrib=sts.gengamma, require_isis=50,
                     n_params=4, elim_lessthan=8, elim_greaterthan=1000,
                     boots=10):
    fits = np.zeros((len(fr_ranges), n_params, boots, neur.shape[1]))
    dists = np.zeros((len(fr_ranges), boots, neur.shape[1]), dtype=object)
    for i, fr in enumerate(fr_ranges):
        out = fit_distrib_tc(neur, fr, distrib=distrib, 
                             require_isis=require_isis, 
                             n_params=n_params,
                             elim_lessthan=elim_lessthan, 
                             elim_greaterthan=elim_greaterthan,
                             boots=boots)
        fits[i], dists[i] = out
    return fits, dists

def fit_distrib_tc(isi_tc, fr_range, distrib=sts.gengamma, require_isis=50, 
                   n_params=4, elim_lessthan=8, elim_greaterthan=1000,
                   boots=10):
    params = np.zeros((n_params, boots, isi_tc.shape[1]))
    distribs = np.zeros((boots, isi_tc.shape[1]), dtype=object)
    for i in range(isi_tc.shape[1]):
        frs = list(filter(lambda x: _firing_rate_range(x, fr_range), 
                          isi_tc[:, i]))
        if len(frs) > 0:
            all_frs = np.concatenate(frs)
        else:
            all_frs = np.array([])
        elim_mask = np.logical_and(all_frs >= elim_lessthan, 
                                   all_frs < elim_greaterthan)
        all_frs = all_frs[elim_mask]
        if len(all_frs) > require_isis:
            for j in range(boots):
                samp_frs = u.resample_on_axis(all_frs, len(all_frs))
                params[:, j, i] = distrib.fit(samp_frs)
                distribs[j, i] = samp_frs
        else:
            params[:, :, i] = np.nan
            distribs[:, i] = np.nan
    return params, distribs
            

