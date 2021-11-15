
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import re
from sklearn import svm
import functools as ft

import general.utility as u
import general.neural_analysis as na

class ResultSequence(object):

    def __init__(self, x):
        self.val = list(x)

    def __hash__(self):
        hashable = tuple(tuple(x) for x in self.val)
        return hash(hashable)
        
    def _op(self, x, operator):
        out = ResultSequence(operator(v, x) for v in self.val)
        return out

    def _op_rs(self, x, operator):
        out = ResultSequence(operator(v, x[i])
                             for i, v in enumerate(self.val))
        return out

    def __getitem__(self, key):
        return self.val[key]

    def __len__(self):
        return len(self.val)
    
    def __iter__(self):
        return iter(self.val)

    def __next__(self):
        return next(self.val)
    
    def __repr__(self):
        return str(self.val)
        
    def __lt__(self, x):
        return self._op_dispatcher(x, np.less)

    def __le__(self, x):
        return self._op_dispatcher(x, np.less_equal)

    def __gt__(self, x):
        return self._op_dispatcher(x, np.greater)

    def __ge__(self, x):
        return self._op_dispatcher(x, np.greater_equal)

    def __eq__(self, x):
        return self._op_dispatcher(x, np.equal)

    def __ne__(self, x):
        return self._op_dispatcher(x, np.not_equal)

    def _op_dispatcher(self, x, op):
        try:
            len(x)
            if type(x) == str:
                raise TypeError()
            out = self._op_rs(x, op)
        except TypeError:
            out = self._op(x, op)
        return out

    def one_of(self, x):
        return self._op(x, np.isin)
    
    def rs_and(self, x):
        return self._op_dispatcher(x, np.logical_and)

    def rs_or(self, x):
        return self._op_dispatcher(x, np.logical_or)

    def rs_not(self):
        return ResultSequence(np.logical_not(v) for v in self.val)

    def __sub__(self, x):
        return self._op_dispatcher(x, np.subtract)

    def __add__(self, x):
        return self._op_dispatcher(x, np.add)

def combine_ntrls(*args):
    stacked = np.stack(args, axis=0)
    return np.min(stacked, axis=0)

def _convolve_psth(psth_tm, binsize, xs, causal_xs=False, binwidth=20):
    filt_wid = int(np.round(binsize))
    filt = np.ones(filt_wid)/filt_wid
    n_ts = psth_tm.shape[2]
    out_len = int(n_ts - binsize + 1)
    mult = 1000/binwidth
    smooth_psth = np.zeros(psth_tm.shape[:2] + (out_len,))
    for i, ptm_i in enumerate(psth_tm):
        for j, ptm_ij in enumerate(ptm_i):
            smooth_psth[i, j] = np.convolve(ptm_ij, filt, mode='valid')
    if causal_xs:
        xs_adj = xs[:out_len]
    else:
        step = xs[1] - xs[0]
        cent_offset = step*binsize/2 
        xs_adj = xs + cent_offset
        xs_adj = xs_adj[:out_len]
    return mult*smooth_psth, xs_adj

class Dataset(object):
    
    def __init__(self, dframe, seconds=False):
        self.data = dframe
        try:
            self.session_fields = self.data['data'].iloc[0].columns
        except KeyError as e:
            if len(self.data['data']) == 0:
                raise IOError('no data available')
            else:
                raise e
        self.n_sessions = len(self.data)
        self.data = self.data.sort_values('date', ignore_index=True)
        self.data = self.data.sort_values('animal', ignore_index=True)
        self.seconds = seconds
        self.population_cache = {}
        self.mask_pop_cache = {}

    @classmethod
    def from_dict(cls, seconds=False, **inputs):
        df = pd.DataFrame(data=inputs)
        return cls(df, seconds=seconds)
        
    @classmethod
    def from_readfunc(cls, read_func, *args, seconds=False, **kwargs):
        super_df = read_func(*args, **kwargs)
        return cls.from_dict(seconds=seconds, **super_df)

    def __getitem__(self, key):
        try:
            out = self.data[key]
        except KeyError:
            out = ResultSequence(dd[key] for dd in self.data['data'])
        return out

    def __contains__(self, key):
        top_level = key in self.data.keys()
        session_level = key in self.data['data'].iloc[0].columns
        return top_level or session_level
    
    def session_mask(self, mask):
        return Dataset(self.data[mask], seconds=self.seconds)

    def mask(self, mask):
        df = {}
        for c in self.data.columns:
            df[c] = self.data[c]
        dlist = []
        
        for i, m in enumerate(mask):
            d = self.data['data'][i][m]
            dlist.append(d)
        df['data'] = dlist
        return Dataset.from_dict(**df, seconds=self.seconds)

    def _center_spks(self, spks, tz, tzf):
        if tz is not None:
            spks = spks - tz
        if tzf is not None:
            spks = spks - self[tzf]
        return spks
            
    def get_response_in_window(self, begin, end, time_zero=None,
                               time_zero_field=None):
        spks = self['spikeTimes']
        spks = self._center_spks(spks, time_zero, time_zero_field)
        out = []
        for spk in spks:
            resp_arr = np.array(list(u.get_spks_window(spk, begin, end)))
            out.append(resp_arr)
        return out

    def _get_spikerates(self, spk, binsize, bounds, binstep, accumulate=False,
                        convert_seconds=True, pre_bound=-2, post_bound=2):
        xs = na.compute_xs(binsize, bounds[0], bounds[1], binstep)
        if len(spk.shape) > 2:
            spk = np.squeeze(spk)
        no_spks = np.zeros_like(spk)
        for i, spk_i in enumerate(spk):
            for j, spk_ij in enumerate(spk_i):
                
                spk_sq = np.squeeze(spk_ij)
                if convert_seconds:
                    spk_sq = spk_sq*1000
                no_spks[i, j] = len(spk_sq.shape) == 0 or len(spk_sq) == 0 
                resp_arr = na.bin_spiketimes(spk_sq, binsize, bounds, binstep,
                                             accumulate=accumulate,
                                             spks_per_sec=not self.seconds)
                if i == 0 and j == 0:
                    tlen = len(resp_arr)
                    out_arr = np.zeros(spk.shape + (tlen,))
                out_arr[i, j] = resp_arr
        return out_arr, xs, no_spks
    
    def make_pseudopop(self, outs, n_trls=None, min_trials_pseudo=10,
                       resample_pseudos=10, skl_axs=False, same_n_trls=False):
        if n_trls is None:
            n_trls = list(len(o) for o in outs)
        n_trls = np.array(n_trls)
        n_trls_mask = n_trls >= min_trials_pseudo
        outs_mask = np.array(outs, dtype=object)[n_trls_mask]
        if skl_axs:
            trl_ax = 2
            neur_ax = 0
        else:
            trl_ax = 0
            neur_ax = 1
        if same_n_trls:
            min_trls = np.min(n_trls[n_trls_mask])
        else:
            n_trls_actual = np.array(list(o.shape[trl_ax] for o in outs))
            min_trls = np.min(n_trls_actual[n_trls_mask])
        for i in range(resample_pseudos):
            for j, pop in enumerate(outs_mask):
                trl_inds = np.random.choice(pop.shape[trl_ax], min_trls,
                                            replace=False)
                if skl_axs:
                    ppop_j = pop[:, :, trl_inds]
                else:
                    ppop_j = pop[trl_inds]
                if j == 0:
                    ppop = ppop_j
                else:
                    ppop = np.concatenate((ppop, ppop_j), axis=neur_ax)
            if i == 0:
                out_pseudo = np.zeros((resample_pseudos,) + ppop.shape)
            out_pseudo[i] = ppop
        return out_pseudo

    def get_nneurs(self):
        return self['n_neurs']

    def clear_cache(self):
        self.population_cache = {}

    def mask_population(self, mask, *args, cache=False, **kwargs):
        key = (mask,) + args + tuple(kwargs.values())
        ret = self.mask_pop_cache.get(key)
        if cache and ret is not None:
            data, pop, xs = ret
        else:
            data = self.mask(mask)
            pop, xs = data.get_populations(*args, **kwargs)
        if cache:
            self.mask_pop_cache[key] = (data, pop, xs)
        return data, pop, xs

    def get_psth(self, binsize=None, begin=None, end=None, 
                 skl_axes=False, accumulate=False, time_zero=None,
                 time_zero_field=None, combine_pseudo=False, min_trials_pseudo=10,
                 resample_pseudos=10, repl_nan=False, regions=None,
                 ret_no_spk=False, causal_timing=False, shuffle_trials=False):
        psths = self['psth']
        xs_all = self['psth_timing']
        outs = []
        n_trls = []
        for i, psth_l in enumerate(psths):
            psth = np.stack(psth_l, axis=0)
            xs = xs_all[i]
            if time_zero is not None:
                xs_i = np.repeat(np.expand_dims(xs - time_zero, 0),
                                 len(psth_l), 0)
            elif time_zero_field is not None:
                xs_exp = np.expand_dims(xs, 0)
                tzf = self[time_zero_field][i]
                tzf_exp = np.expand_dims(tzf, 1)
                xs_i = xs_exp - tzf_exp
            else:
                xs_i = np.repeat(np.expand_dims(xs, 0), len(psth_l), 0)
            psth_bs = xs[1] - xs[0]
            time_mask = np.ones_like(xs_i, dtype=bool)
            if begin is not None:
                time_mask = np.logical_and(xs_i >= begin, time_mask)
            if end is not None:
                time_mask = np.logical_and(xs_i < end, time_mask)
            psth_tm = np.zeros(psth.shape[:2]
                               + (np.sum(time_mask, axis=1)[0],))
            for i, trl in enumerate(psth):
                for j, trl_ij in enumerate(trl):
                    psth_tm[i, j] = trl_ij[time_mask[i]]
            xs_reg = xs_i[0, time_mask[0]]
            if binsize is not None:
                assert (binsize % psth_bs) == 0
                psth_tm, xs_reg = _convolve_psth(psth_tm, binsize/psth_bs, xs_reg,
                                                 causal_xs=causal_timing)
            if shuffle_trials:
                rng = np.random.default_rng()
                list(rng.shuffle(psth_tm[:, i]) for i in range(psth_tm.shape[1]))
            if skl_axes:
                psth_tm = np.expand_dims(np.swapaxes(psth_tm, 0, 1), 1)
            outs.append(psth_tm)
            n_trls.append(psth_tm.shape[0])
        if combine_pseudo:
            outs = self.make_pseudopop(outs, n_trls, min_trials_pseudo,
                                       resample_pseudos)
        return outs, xs_reg

    def get_psth_window(self, begin, end, **kwargs):
        psths, xs = self.get_psth(begin=begin, end=end, **kwargs)
        time_filt = np.logical_and(xs >= begin, xs < end)
        psths_collapsed = [] 
        for i, psth in enumerate(psths):
            psth_c = np.sum(psth[..., time_filt], axis=-1)
            psths_collapsed.append(psth_c)
        return psths_collapsed
    
    # @ft.lru_cache(maxsize=10)
    def get_populations(self, *args, cache=False, **kwargs):
        key = args + tuple(kwargs.values())
        if cache and key in self.population_cache.keys():
            out = self.population_cache[key]
        else:
            out = self._get_populations(*args, **kwargs)
        if cache:
            self.population_cache[key] = out
        return out
    
    def _get_populations(self, binsize, begin, end, binstep=None, skl_axes=False,
                         accumulate=False, time_zero=None, time_zero_field=None,
                         combine_pseudo=False, min_trials_pseudo=10,
                         resample_pseudos=10, repl_nan=False, regions=None,
                         ret_no_spk=False, shuffle_trials=False):
        spks = self['spikeTimes']
        spks = self._center_spks(spks, time_zero, time_zero_field)
        if regions is not None:
            regions_all = self['neur_regions']
        outs = []
        n_trls = []
        no_spks = []
        for i, spk in enumerate(spks):
            if len(spk) == 0:
                xs = na.compute_xs(binsize, begin, end, binstep)
                resp_arr = np.zeros((0, self.get_nneurs()[i], len(xs)))
                out = (resp_arr, xs, np.ones_like(resp_arr, dtype=bool))
            else:
                spk_stack = np.stack(spk, axis=0)
                out = self._get_spikerates(spk_stack, binsize, (begin, end),
                                           binstep, accumulate,
                                           convert_seconds=not self.seconds)
            resp_arr, xs, no_spk_mask = out
            if regions is not None and len(resp_arr) > 0:
                regs = regions_all[i].iloc[0]
                mask = np.isin(regs, regions)
                resp_arr = resp_arr[:, mask]
            if repl_nan:
                resp_arr[no_spk_mask] = np.nan
            n_trls.append(resp_arr.shape[0])
            if skl_axes:
                resp_arr = np.expand_dims(np.swapaxes(resp_arr, 0, 1), 1)
            outs.append(resp_arr)
            no_spks.append(no_spk_mask)
        if combine_pseudo:
            outs = self.make_pseudopop(outs, n_trls, min_trials_pseudo,
                                       resample_pseudos)
        if ret_no_spk:
            out = (outs, xs, no_spks)
        else:
            out = (outs, xs)
        return out

    def get_ntrls(self):
        return list(len(o) for o in self['data'])
    
    def decode_masks(self, m1, m2, winsize, begin, end, stepsize, n_folds=20,
                     model=svm.SVC, params=None, pre_pca=None,
                     mean=False, shuffle=False, time_zero_field=None,
                     pseudo=False, min_trials_pseudo=10, resample_pseudo=10,
                     repl_nan=False, impute_missing=False,
                     ret_pops=False, shuffle_trials=False,
                     decode_m1=None, decode_m2=None, decode_tzf=None,
                     **kwargs):
        if params is None:
            # params = {'class_weight':'balanced'}
            params = {}
            params.update(kwargs)            
        cat1 = self.mask(m1)
        cat2 = self.mask(m2)
        if decode_tzf is None:
            decode_tzf = time_zero_field
        if decode_m1 is not None:
            dec_data_c1 = self.mask(decode_m1)
        else:
            dec1 = (None,)*len(cat1)
        if decode_m2 is not None:
            dec_data_c2 = self.mask(decode_m2)
        else:
            dec2 = (None,)*len(cat2)
        if 'spikeTimes' in self:
            out1 = cat1.get_populations(winsize, begin, end, stepsize,
                                        skl_axes=True, repl_nan=repl_nan,
                                        time_zero_field=time_zero_field,
                                        shuffle_trials=shuffle_trials)
            out2 = cat2.get_populations(winsize, begin, end, stepsize,
                                        skl_axes=True, repl_nan=repl_nan,
                                        time_zero_field=time_zero_field,
                                        shuffle_trials=shuffle_trials)
            if decode_m1 is not None:
                dec1 = dec_data_c1.get_populations(
                    winsize, begin, end, stepsize, skl_axes=True,
                    repl_nan=repl_nan, time_zero_field=decode_tzf,
                    shuffle_trials=shuffle_trials)[0]
            if decode_m2 is not None:
                dec2 = dec_data_c2.get_populations(
                    winsize, begin, end, stepsize, skl_axes=True,
                    repl_nan=repl_nan, time_zero_field=decode_tzf,
                    shuffle_trials=shuffle_trials)[0]
        elif 'psth' in self:
            out1 = cat1.get_psth(winsize, begin, end, 
                                 skl_axes=True, repl_nan=repl_nan,
                                 time_zero_field=time_zero_field,
                                 shuffle_trials=shuffle_trials)
            out2 = cat2.get_psth(winsize, begin, end, 
                                 skl_axes=True, repl_nan=repl_nan,
                                 time_zero_field=time_zero_field,
                                 shuffle_trials=shuffle_trials)
            if decode_m1 is not None:
                dec1 = dec_data_c1.get_psth(
                    winsize, begin, end, skl_axes=True,
                    repl_nan=repl_nan, time_zero_field=decode_tzf,
                    shuffle_trials=shuffle_trials)[0]
            if decode_m2 is not None:
                dec2 = dec_data_c2.get_psth(
                    winsize, begin, end, skl_axes=True,
                    repl_nan=repl_nan, time_zero_field=decode_tzf,
                    shuffle_trials=shuffle_trials)[0]
        else:
            raise TypeError('this Dataset has no associated neural data')
        pop1, xs = out1
        pop2, xs = out2
        if pseudo:
            c1_n = cat1.get_ntrls()
            c2_n = cat2.get_ntrls()
            trls_list = (c1_n, c2_n)
            if decode_m1 is not None:
                trls_list = trls_list + (dec_data_c1.get_ntrls(),)
            if decode_m2 is not None:
                trls_list = trls_list + (dec_data_c2.get_ntrls(),)
            comb_n = combine_ntrls(*trls_list)
            pop1 = self.make_pseudopop(pop1, comb_n, min_trials_pseudo,
                                       resample_pseudo, skl_axs=True,
                                       same_n_trls=True)
            pop2 = self.make_pseudopop(pop2, comb_n, min_trials_pseudo,
                                       resample_pseudo, skl_axs=True,
                                       same_n_trls=True)
            if decode_m1 is not None:
                dec1 = self.make_pseudopop(dec1, comb_n, min_trials_pseudo,
                                           resample_pseudo, skl_axs=True)
            else:
                dec1 = (None,)*resample_pseudo
            if decode_m2 is not None:
                dec2 = self.make_pseudopop(dec2, comb_n, min_trials_pseudo,
                                           resample_pseudo, skl_axs=True)
            else:
                dec2 = (None,)*resample_pseudo
        outs = np.zeros((len(pop2), n_folds, len(xs)))
        outs_gen = np.zeros_like(outs)
        for i, p1 in enumerate(pop1):
            out = na.fold_skl(p1, pop2[i], n_folds, model=model, params=params, 
                              mean=mean, pre_pca=pre_pca, shuffle=shuffle,
                              impute_missing=(repl_nan or impute_missing),
                              gen_c1=dec1[i], gen_c2=dec2[i])
            if decode_m1 is not None or decode_m2 is not None:
                outs[i] = out[0]
                outs_gen[i] = out[1]
            else:
                outs[i] = out
        if ret_pops:
            out = (outs, xs, pop1, pop2)
            add = tuple(di for di in (dec1, dec2) if di[0] is not None)
            out = out + add
        else:
            out = (outs, xs)
        if decode_m1 is not None or decode_m2 is not None:
            out = out + (outs_gen,)
        return out

    def estimate_dimensionality(self, mask, winsize, begin, end, stepsize,
                                n_resamples=20):
        data_masked = self.mask(mask)
        pops, xs = data_masked.get_populations(winsize, begin, end, stepsize)
        outs = np.zeros((len(pops), n_resamples, len(xs)))
        for i, p in enumerate(pops):
            outs[i] = na.estimate_participation_ratio(p, n_resamples=n_resamples)
        return outs, xs
