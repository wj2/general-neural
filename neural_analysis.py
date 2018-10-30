
import numpy as np
import scipy.stats as sts
import general.utility as u
from sklearn import svm, linear_model
from sklearn import discriminant_analysis as da
from sklearn.decomposition import PCA
from dPCA.dPCA import dPCA
import itertools as it
import string
import os

### ORGANIZE SPIKES ###

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
                          func_out=(bin_spiketimes, None)):
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
    if binstep is not None and binstep < binsize:
        xs = np.arange(pretime + binsize/2., posttime + binsize/2. + binstep,
                       binstep)
    else:
        xs = np.arange(pretime + binsize/2., posttime + binsize/2. + 1, binsize)
    for i, df in enumerate(discrim_funcs):
        mf = marker_funcs[i]
        d_trials = data[df(data)]
        neurs = {}
        for j, run in enumerate(druns):
            all_dictnames = data[data[drunfield] == run][spikefield]
            nr_names = np.concatenate([list(x.keys()) for x in all_dictnames])
            n_names = np.unique(nr_names)
            udt = d_trials[d_trials[drunfield] == run]
            n_kvs = [((run, name), u.nan_array((udt.shape[0], xs.shape[0]),
                                               dtype=out_type)) 
                      for name in n_names]
            neurs.update(n_kvs)
            marks = mf(udt)
            for k, trial in enumerate(udt):
                for l, neurname in enumerate(trial[spikefield].keys()):
                    key = (run, neurname)
                    if not np.isnan(marks[k]):
                        spikes = trial[spikefield][neurname] - marks[k]
                        psth = spk_func(spikes, binsize, 
                                        (pretime, posttime), binstep,
                                        accumulate=cumulative)
                    else:
                        psth = np.ones_like(xs)
                        psth[:] = np.nan
                    neurs[key][k, :] = psth
        all_discs.append(neurs)
    return all_discs, xs
        
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


### SVM ###

def sample_trials_svm(dims, n, with_replace=False):
    trls = np.zeros((len(dims), n, dims[0].shape[1]))
    for i, d in enumerate(dims):
        trl_inds = np.random.choice(d.shape[0], n, replace=with_replace)
        trls[i, :, :] =  d[trl_inds, :]
    return trls

def _fold_model(cat1, cat2, leave_out=1, model=svm.SVC, norm=True, eps=.00001,
                shuff_labels=False, stability=False, params=None, 
                collapse_time=False):
    alltr = np.concatenate((cat1, cat2), axis=1)
    alllabels = np.concatenate((np.zeros(cat1.shape[1]), 
                                np.ones(cat2.shape[1])))
    inds = np.arange(alltr.shape[1])
    np.random.shuffle(inds)
    alltr = alltr[:, inds, :]
    if shuff_labels:
        inds = np.arange(alltr.shape[1])
        np.random.shuffle(inds)
    alllabels = alllabels[inds]
    if norm:
        mu = alltr.mean(1).reshape((alltr.shape[0], 1, alltr.shape[2]))
        sig = alltr.std(1).reshape((alltr.shape[0], 1, alltr.shape[2]))
        sig[sig < eps] = 1.
        alltr = (alltr - mu)/sig
    folds_n = int(np.floor(alltr.shape[1] / leave_out))
    if stability:
        results = np.zeros((folds_n, cat1.shape[2], cat1.shape[2]))
    else:
        results = np.zeros((folds_n, cat1.shape[2]))
    sup_vecs = np.zeros((folds_n, cat1.shape[2], cat1.shape[0]))
    for i in range(folds_n):
        train_tr = np.concatenate((alltr[:, (i+1)*leave_out:], 
                                   alltr[:, :i*leave_out]),
                                  axis=1)
        train_l = np.concatenate((alllabels[(i+1)*leave_out:], 
                                  alllabels[:i*leave_out]))

        test_tr = alltr[:, i*leave_out:(i+1)*leave_out]
        test_l = alllabels[i*leave_out:(i+1)*leave_out]
        results[i], sup_vecs[i] = model_decode_tc(train_tr, train_l, test_tr, 
                                                  test_l, model=model, 
                                                  stability=stability,
                                                  params=params, 
                                                  collapse_time=collapse_time)
    mr = np.mean(results, axis=0)
    return mr, results, alltr, sup_vecs

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
        svs[i] = s.coef_[0]
    return percent_corr, svs

def svm_decoding(cat1, cat2, leave_out=1, require_trials=15, resample=100,
                 with_replace=False, shuff_labels=False, stability=False,
                 kernel='linear', penalty=1, format_=True):
    spec_params = {'C':penalty, 'kernel':kernel}
    model = svm.SVC
    out = decoding(cat1, cat2, leave_out=leave_out, 
                   require_trials=require_trials, resample=resample,
                   with_replace=with_replace, shuff_labels=shuff_labels,
                   stability=stability, params=spec_params, format_=format_)
    return out

def decoding(cat1, cat2, model=svm.SVC, leave_out=1, require_trials=15, 
             resample=100, with_replace=False, shuff_labels=False, 
             stability=False, params=None, collapse_time=False,
             format_=True):
    if format_:
        cat1 = np.array(list(cat1.values()))
        cat2 = np.array(list(cat2.values()))
        bool1 = [x.shape[0] < require_trials for x in cat1]
        bool2 = [x.shape[0] < require_trials for x in cat2]
        combool = np.logical_not(np.logical_or(bool1, bool2))
        cat1_f = cat1[combool]
        cat2_f = cat2[combool]
    else:
        cat1_f = cat1
        cat2_f = cat2
    if stability:
        tcs_shape = (resample, cat1_f[0].shape[1], cat1_f[0].shape[1])
    else:
        tcs_shape = (resample, cat1_f[0].shape[1])
    folds_n = int(np.floor((require_trials*2)/leave_out))
    ms = np.zeros((resample, folds_n, cat1_f[0].shape[1], cat1_f.shape[0]))
    tcs = np.zeros(tcs_shape)
    for i in range(resample):
        cat1_samp = sample_trials_svm(cat1_f, require_trials, with_replace)
        cat2_samp = sample_trials_svm(cat2_f, require_trials, with_replace)
        out = _fold_model(cat1_samp, cat2_samp, leave_out, model=model,
                         shuff_labels=shuff_labels, stability=stability, 
                          params=params, collapse_time=collapse_time)
        tcs[i], _, _, ms[i] = out
    return tcs, cat1_f, cat2_f, ms

def svm_multi_decoding(data, leave_out=1, require_trials=15, resample=50,
                       with_replace=False, shuff_labels=False, stability=False,
                       kernel='linear', penalty=1, collapse_time=False):
    spec_params = {'C':penalty, 'kernel':kernel}
    model = svm.SVC
    out = multi_decoding(data, model=model, leave_out=leave_out, 
                         require_trials=require_trials,
                         resample=resample, with_replace=with_replace, 
                         shuff_labels=shuff_labels, stability=stability,
                         params=spec_params, collapse_time=collapse_time)
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
                   stability=False, params=None, collapse_time=False):
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
                              collapse_time=collapse_time)
            tcs[i, j], _, _, ms[i, j] = out
        used_arrs[i] = arr
    return tcs, ms, used_arrs

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
        neurons with trials > require_trials for each condition and T is the
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
    except:
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
    Bootstrap a function on some colelction of dimensions across some
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
    full_labels = []
    factor_singles = [list(it.product((l,), factor_labels[i]))
                      for i, l in enumerate(labels)]
    for x in factor_singles:
        full_labels = full_labels + x
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
            prod = [labs[:prod_size][labels.index(x)] for x in l]
            labs[i] = np.product(prod)
        else:
            ind = cond_labels.index(l[0])
            x = comb[ind]
            if len(l) > 1 and x == factor_labels[ind].index(l[1]):
                labs[i] = 1
            elif len(l) > 1:
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
        zs_data = dm_data/np.std(dm_data, axis=1).reshape((-1, 1))
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
            

