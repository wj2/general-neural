
import numpy as np
import scipy.stats as sts

import sklearn.model_selection as skms
import sklearn.svm as skc
import sklearn.decomposition as skd
import sklearn.preprocessing as skp
import sklearn.pipeline as skpipe


def sample_trials_pseudo(tf, n_samples=100, test_prop=.1):
    n_samples_test = int(np.ceil(n_samples*test_prop))
    n_conds = len(tf[0])
    train_data = np.zeros((tf.shape[0], n_conds*n_samples,
                           tf[0, 0].shape[1]))
    test_data = np.zeros((tf.shape[0], n_conds*n_samples_test,
                          tf[0, 0].shape[1]))
    for i, neur_conds in enumerate(tf):
        train_trls = []
        test_trls = []
        for j, neur in enumerate(neur_conds):
            n_test = int(np.ceil(neur.shape[0]*test_prop))
            test_trl_inds = np.random.choice(neur.shape[0], n_test,
                                             replace=False)
            test_inds = np.random.choice(test_trl_inds, n_samples_test)
            test_trls.append(neur[test_inds])

            trl_inds = np.arange(neur.shape[0])
            train_ind_mask = np.logical_not(np.isin(trl_inds,
                                                test_inds))
            train_trl_inds = trl_inds[train_ind_mask]
            train_inds = np.random.choice(train_trl_inds, n_samples)
            train_trls.append(neur[train_inds])
        train_data[i] = np.concatenate(train_trls)
        test_data[i] = np.concatenate(test_trls)
    return train_data, test_data

def _neur_trl_splitters(c1, c2):
    n12_data = []
    for i, nc1 in enumerate(c1):
        folds_per_cond = []
        for j, n1 in enumerate(nc1):
            n2 = c2[i][j]
            l_ij = np.concatenate((np.zeros(len(n1)),
                                   np.ones(len(n2))))
            n12 = np.concatenate((n1, n2), axis=0)
            folds_per_cond.append((n12, l_ij))
        n12_data.append(folds_per_cond)
    return n12_data

class BalanceGroupFolder:
    def __init__(self, k_folds, shuffle=True, hard_equal=False, **kwargs):
        self.k_folds = k_folds
        self.shuffle = shuffle
        self.kwargs = kwargs
        self.hard_equal = hard_equal

    def split(self, X, y=None, groups=None):
        if groups is not None:
            groups = y
        group_folders = np.unique(groups)
        folders = []
        for g in group_folders:
            kf_g = skms.KFold(self.k_folds, shuffle=self.shuffle, **self.kwargs)
            g_gen = kf_g.split(X[groups == g])
            folders.append(g_gen)
        for k in range(self.k_folds):
            trains = []
            tests = []
            min_tr_members = np.inf
            min_te_members = np.inf
            for i, gen_i in enumerate(folders):
                g = group_folders[i]
                g_inds = np.where(groups == g)[0]
                tr_inds, te_inds = next(gen_i)
                min_tr_members = np.min((min_tr_members, len(tr_inds)))
                min_te_members = np.min((min_te_members, len(te_inds)))
                trains.append(g_inds[tr_inds])
                tests.append(g_inds[te_inds])
            if self.hard_equal:
                trains = [t[:int(min_tr_members)] for t in trains]
                tests = [t[:int(min_te_members)] for t in tests]
            yield np.concatenate(trains), np.concatenate(tests)

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.k_folds

class PseudoPopulationFolder:

    def __init__(self, k_folds, n_samples=100, hard_equal=False,
                 n_conds=1, shuffle=True, **kwargs):
        test_prop = 1/k_folds
        self.n_samples_test = int(n_conds*np.ceil(n_samples*test_prop/n_conds))
        self.n_samples_train = n_samples - self.n_samples_test
        self.k_folds = k_folds
        self.hard_equal = hard_equal
        self.shuffle = shuffle
        self.kwargs = kwargs

    def split(self, X, reindex=False):
        """ 
        X is N x K array of M_ij x T arrays
        where N is the number of neurons, K is the number of conditions,
        M_ij is the number of trials for neuron i in condition j, and
        T is the number of time points

        Returns
        -------
        X_train 
        X_test
        """
        splitters = []
        for i, neur in enumerate(X):
            cond_splitters = []
            for j, dij in enumerate(neur):
                kf_ij = skms.KFold(self.k_folds, shuffle=self.shuffle,
                                   **self.kwargs)
                split_ij = kf_ij.split(dij)
                cond_splitters.append(split_ij)
            splitters.append(cond_splitters)
        for k in range(self.k_folds):
            trains = []
            tests = []
            for i, s_i in enumerate(splitters):
                trains_i = []
                tests_i = []
                for j, s_ij in enumerate(s_i):
                    train_split, test_split = next(s_ij)
                    if reindex:
                        train_split = X[i][j][train_split]
                        test_split = X[i][j][test_split]
                    trains_i.append(train_split)
                    tests_i.append(test_split)
                trains.append(trains_i)
                tests.append(tests_i)
            yield trains, tests

    def apply_split(self, X, split):
        n_neurs = len(X)
        n_conds = len(X[0])
        n_ts = X[0][0].shape[1]
        split_train, split_test = split
        
        nstrain_cond = int(np.ceil(self.n_samples_train/n_conds))
        nstest_cond = int(np.ceil(self.n_samples_test/n_conds))
        
        ns_train = nstrain_cond*n_conds
        ns_test = nstest_cond*n_conds

        train_data = np.zeros((nstrain_cond, n_conds, n_neurs, n_ts))
        test_data = np.zeros((nstest_cond, n_conds, n_neurs, n_ts))
        for i, s_tr_i in enumerate(split_train):
            if self.hard_equal:
                min_trls_tr = min(len(x) for x in s_tr_i)
                min_trls_te = min(len(x) for x in s_tr_i)
            for j, s_tr_ij in enumerate(s_tr_i):
                s_te_ij = split_test[i][j]
                if self.hard_equal:
                    s_tr_ij = s_tr_ij[:min_trls_tr]
                    s_te_ij = s_te_ij[:min_trls_te]
                neur_ij = X[i][j]
                tr_inds = np.random.choice(s_tr_ij, nstrain_cond)
                train_data[:, j, i] = neur_ij[tr_inds]
                
                te_inds = np.random.choice(s_te_ij, nstest_cond)
                test_data[:, j, i] = neur_ij[te_inds]
                assert len(set(te_inds).intersection(tr_inds)) == 0
        train_data = np.concatenate(list(train_data[:, i]
                                         for i in range(n_conds)),
                                    axis=0)
        test_data = np.concatenate(list(test_data[:, i]
                                        for i in range(n_conds)),
                                   axis=0)
        return train_data, test_data

def _shuffle_trls(trls, labels):
    inds = np.random.choice(len(trls), len(trls), replace=False)
    trls_shuff = trls[inds]
    labels_shuff = labels[inds]
    return trls_shuff, labels_shuff
        
def merge_c1c2(c1_data, c2_data, shuffle_order=True):
    train_c1, test_c1 = c1_data
    train_c2, test_c2 = c2_data
    train_both = np.concatenate((train_c1, train_c2), axis=0)
    train_labels = np.concatenate((np.zeros(train_c1.shape[0]),
                                   np.ones(train_c2.shape[0])))
    test_both = np.concatenate((test_c1, test_c2), axis=0)
    test_labels = np.concatenate((np.zeros(test_c1.shape[0]),
                                  np.ones(test_c2.shape[0])))
    if shuffle_order:
        train_both, train_labels = _shuffle_trls(train_both,
                                                 train_labels)
        test_both, test_labels = _shuffle_trls(test_both,
                                               test_labels)
    return (train_both, train_labels), (test_both, test_labels)

def _inner_cv(c1, c1_spl, c2, c2_spl, k_folds, pipeline, n_samples=100,
              hard_equal=False):
    ppf_c1 = PseudoPopulationFolder(k_folds, n_samples, n_conds=len(c1[0]),
                                    hard_equal=hard_equal)
    ppf_c2 = PseudoPopulationFolder(k_folds, n_samples, n_conds=len(c2[0]),
                                    hard_equal=hard_equal)
    n_total_test = ppf_c1.n_samples_test + ppf_c2.n_samples_test
    e_inner = np.zeros((k_folds, n_total_test,
                        c1[0][0].shape[1]))
    c2_gen = ppf_c2.split(c2_spl, reindex=True)
    for i, spl_c1 in enumerate(ppf_c1.split(c1_spl, reindex=True)):
        spl_c2 = next(c2_gen)
        c1_spl = ppf_c1.apply_split(c1, spl_c1)
        c2_spl = ppf_c2.apply_split(c2, spl_c2)
        train, test = merge_c1c2(c1_spl, c2_spl)
        for j in range(train[0].shape[-1]):
            pipeline.fit(train[0][..., j], train[1])
            pred = pipeline.predict(test[0][..., j])
            e_inner[i, :, j] = pred != test[1]
    return e_inner

def _ncv_format(*cs):
    i_cat = 0
    i_group = 0
    i_pop = 0
    for i, c_i in enumerate(cs):
        for j in range(c_i.shape[1]):
            p_ij = np.stack(c_i[:, j], axis=1)
            g_ij = np.ones(c_i[0, j].shape[0])*i_group
            c_ij = np.ones(c_i[0, j].shape[0])*i_cat
            if i_pop == 0:
                pops = p_ij
                groups = g_ij
                cats = c_ij
            else:
                pops = np.concatenate((pops, p_ij), axis=0)
                groups = np.concatenate((groups, g_ij), axis=0)
                cats = np.concatenate((cats, c_ij), axis=0)
            i_group += 1
            i_pop += 1
        i_cat += 1
    return pops, groups, cats

def nested_cv_shell(c1, c2, **kwargs):
    pops, cats, groups = _ncv_format(c1, c2)
    return nested_cv(pops, cats, groups, **kwargs)

def nested_cv(pops, cats, groups=None, alpha=.05, k_folds=5, n_reps=100,
              pipeline=None, kernel='rbf', pre_pca=None, norm=True,
              class_weight='balanced', hard_equal=True, **kwargs):
    if groups is None:
        groups = cats
    if pipeline is None:
        pipe = []
        if norm:
            pipe.append(skp.StandardScaler())
        if pre_pca is not None:
            pipe.append(skd.PCA(pre_pca))
        pipe.append(skc.SVC(class_weight=class_weight, kernel=kernel))
        pipeline = skpipe.make_pipeline(*pipe)
    n_ts = pops.shape[-1]
    n_conds = int(np.ceil(len(np.unique(groups))/2))
    n_samples = pops.shape[0]
    n_test_outer = int(np.ceil(n_samples/k_folds))
    e_outer = np.zeros((n_reps, k_folds, n_test_outer, n_ts))
    e_inner = np.zeros((n_reps, k_folds, k_folds - 1, n_ts))
    for r in range(n_reps):
        bgf = BalanceGroupFolder(k_folds, hard_equal=hard_equal)
        for i, (spl_tr_i, spl_te_i) in enumerate(bgf.split(pops, cats, groups)):
            for j in range(pops.shape[-1]):
                bgf_inner = BalanceGroupFolder(k_folds - 1,
                                               hard_equal=hard_equal)
                inner_score = 1 - skms.cross_val_score(
                    pipeline, pops[spl_tr_i, ..., j], cats[spl_tr_i],
                    groups=groups[spl_tr_i], cv=bgf_inner)
                e_inner[r, i, :, j] = inner_score
                pipeline.fit(pops[spl_tr_i, ..., j], cats[spl_tr_i])
                pred_outer = pipeline.predict(pops[spl_te_i, ..., j])
                score_outer = pred_outer != cats[spl_te_i]
                e_outer[r, i, :len(score_outer), j] = score_outer
                e_outer[r, i, len(score_outer):, j] = np.nan
    out = _process_ncv(e_inner, e_outer, alpha=alpha)
    return out
        
def nested_cv_pseudo(c1, c2, alpha=.05, k_folds=5, n_reps=100, n_samples=100,
                     pipeline=None, kernel='rbf', pre_pca=None, norm=True,
                     hard_equal=True, class_weight='balanced', **kwargs):
    print(class_weight)
    if pipeline is None:
        pipe = []
        if norm:
            pipe.append(skp.StandardScaler())
        if pre_pca is not None:
            pipe.append(skd.PCA(pre_pca))
        pipe.append(skc.SVC(class_weight=class_weight, kernel=kernel))
        pipeline = skpipe.make_pipeline(*pipe)
    n_ts = c1[0][0].shape[1]
    n_conds = len(c1[0])
    n_samples_inner = int(np.ceil(n_samples*(k_folds - 1)/k_folds))
    n_test_outer = int(np.ceil(n_samples/k_folds)*2)
    n_test_inner = int(n_conds*np.ceil(n_samples_inner/((k_folds - 1)*n_conds))*2)
    e_outer = np.zeros((n_reps, k_folds, n_test_outer, n_ts))
    e_inner = np.zeros((n_reps, k_folds, k_folds - 1, n_test_inner, n_ts))
    for r in range(n_reps):
        ppf_c1 = PseudoPopulationFolder(k_folds, n_samples,
                                        hard_equal=hard_equal,
                                        n_conds=len(c1[0]))
        ppf_c2 = PseudoPopulationFolder(k_folds, n_samples,
                                        hard_equal=hard_equal,
                                        n_conds=len(c2[0]))
        gen_c2 = ppf_c2.split(c2)
        for i, spl_c1 in enumerate(ppf_c1.split(c1)):
            spl_c2 = next(gen_c2)
            e_inner[r, i] = _inner_cv(c1, spl_c1[0], c2, spl_c2[0],
                                      k_folds - 1, pipeline,
                                      n_samples=n_samples_inner,
                                      hard_equal=hard_equal)
            c1_spl = ppf_c1.apply_split(c1, spl_c1)
            c2_spl = ppf_c2.apply_split(c2, spl_c2)
            train, test = merge_c1c2(c1_spl, c2_spl)
            for j in range(train[0].shape[-1]):
                pipeline.fit(train[0][..., j], train[1])
                pred = pipeline.predict(test[0][..., j])
                e_outer[r, i, ..., j] = pred != test[1]
    return _process_ncv(e_inner, e_outer, alpha=alpha)

def _process_ncv(e_inner, e_outer, alpha=.05):
    q = sts.norm(0, 1).ppf(1 - alpha/2)

    k_folds = e_outer.shape[1]
    if len(e_inner.shape) == 5:
        e_inner = np.nanmean(e_inner, axis=3)
    err_outer = np.nanmean(e_outer, axis=2)
    a_list = (np.nanmean(e_inner, axis=2) - err_outer)**2
    entries = np.sum(~np.isnan(e_outer), axis=2)
    b_list = np.nanvar(e_outer, axis=2)/entries 
    # something goes wrong here for real populations... not sure why
    # want to figure it out, but maybe best to just cut losses at this
    # point
    
    mse_hat = np.nanmean(a_list, axis=(0, 1)) - np.nanmean(b_list, axis=(0, 1))
    if np.any(mse_hat < 0):
        print(mse_hat.shape)
        print(mse_hat)
    err_hat = np.nanmean(e_outer, axis=(0, 1, 2))
    bias = (1 + (k_folds - 2)/k_folds)*(err_hat - err_outer)
    conf_lower = err_hat - bias - q*np.sqrt(mse_hat*(k_folds - 1)/k_folds)
    conf_upper = err_hat - bias + q*np.sqrt(mse_hat*(k_folds - 1)/k_folds)
    err_cent = err_hat - bias
    other = (err_hat, mse_hat, bias, (e_outer, e_inner, a_list, b_list))
    err_cent_all = np.nanmean(e_outer, axis=(1, 2))
    return err_cent_all, err_cent, conf_lower, conf_upper, other

        

