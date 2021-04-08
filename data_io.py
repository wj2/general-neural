
import numpy as np
import scipy.io as sio
import pandas as pd
import os
import re
from sklearn import svm

import general.utility as u
import general.neural_analysis as na

miller_file = '([A-Za-z]+)-([A-Za-z]+)-([0-9]{8})\.mat'
miller_2d = ('spikeTimes',)
miller_af = ('trialInfo',)
miller_af_skips = {miller_af[0]:('Properties',)}
miller_sf = ()

def _add_two_dim_arr_to_dataframe(data, df, double_fields):
    for td in double_fields:
        df[td] = list(spks_trl for spks_trl in data[td])
    return df

def _add_single_to_dataframe(data, df, keys, new_keys=None):
    for i, k in enumerate(keys):
        if new_keys is not None:
            new_k = new_keys[i]
        else:
            new_k = k
        df[new_k] = data[k][:, 0]
    return df

def _add_structured_arr_to_dataframe(data, df, sa_keys, new_keys=None,
                                     skips=None):
    if skips is None:
        skips = {}
    for sak in sa_keys:
        data_names = data[sak].dtype.names
        skips_sak = skips.get(sak, ())
        data_names = filter(lambda x: x not in skips_sak, data_names)
        df = _add_single_to_dataframe(data[sak][0, 0], df, data_names)
    return df

def load_miller_data(folder, template=miller_file,
                     single_fields=miller_sf, arr_fields=miller_af,
                     double_fields=miller_2d, arr_fields_skip=miller_af_skips,
                     max_files=np.inf):
    fls = os.listdir(folder)
    counter = 0
    dates, expers, monkeys, datas = [], [], [], []
    for fl in fls:
        m = re.match(template, fl)
        if m is not None:
            df = pd.DataFrame()
            expers.append(m.group(1))
            monkeys.append(m.group(2))
            date = m.group(3)
            date_f = pd.to_datetime(int(date), format='%m%d%Y')
            dates.append(date_f)
            data = sio.loadmat(os.path.join(folder, fl))
            df = _add_structured_arr_to_dataframe(data, df, arr_fields,
                                                  skips=arr_fields_skip)
            df = _add_two_dim_arr_to_dataframe(data, df, double_fields)
            ain_mask = data['analogChnlInfo']['isAIN'][0,0][:, 0].astype(bool)
            labels = data['analogChnlInfo']['chnlLabel'][0,0][ain_mask]
            for i, l in enumerate(labels):
                label = l[0][0]
                dat_i = data['ain'][:, i].T
                df[label] = list(ain_trl for ain_trl in dat_i)
            datas.append(df)
            counter = counter + 1
            if counter >= max_files:
                break
    super_dict = dict(date=dates, experiment=expers, animal=monkeys,
                      data=datas)
    return super_dict

class ResultSequence(object):

    def __init__(self, x):
        self.val = list(x)

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
        return self._op(x, np.less)

    def __le__(self, x):
        return self._op(x, np.less_equal)

    def __gt__(self, x):
        return self._op(x, np.greater)

    def __ge__(self, x):
        return self._op(x, np.greater_equal)

    def __eq__(self, x):
        return self._op(x, np.equal)

    def __ne__(self, x):
        return self._op(x, np.not_equal)

    def _op_dispatcher(self, x, op):
        try:
            len(x)
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
        spks = self._center_spks(spks, time_zero, time_zero_field)
        out = []
        for spk in spks:
            resp_arr = np.array(list(u.get_spks_window(spk, begin, end)))
            out.append(resp_arr)
        return out

    def _get_spikerates(self, spk, binsize, bounds, binstep, accumulate=False,
                        convert_seconds=True):
        xs = na.compute_xs(binsize, bounds[0], bounds[1], binstep)
        spk = np.squeeze(spk)
        for i, spk_i in enumerate(spk):
            for j, spk_ij in enumerate(spk_i):
                spk_sq = np.squeeze(spk_ij)
                if convert_seconds:
                    spk_sq = spk_sq*1000
                resp_arr = na.bin_spiketimes(spk_sq, binsize, bounds, binstep,
                                             accumulate=accumulate,
                                             spks_per_sec=not self.seconds)
                if i == 0 and j == 0:
                    tlen = len(resp_arr)
                    out_arr = np.zeros(spk.shape + (tlen,))
                out_arr[i, j] = resp_arr
        return out_arr, xs
    
    def combine_ntrls(self, *args):
        stacked = np.stack(args, axis=0)
        return np.min(stacked, axis=0)
    
    def make_pseudopop(self, outs, n_trls=None, min_trials_pseudo=10,
                       resample_pseudos=10, skl_axs=False):
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
    
    def get_populations(self, binsize, begin, end, binstep=None, skl_axes=False,
                        accumulate=False, time_zero=None, time_zero_field=None,
                        combine_pseudo=False, min_trials_pseudo=10,
                        resample_pseudos=10, repl_nan=False):
        spks = self['spikeTimes']
        spks = self._center_spks(spks, time_zero, time_zero_field)
        outs = []
        n_trls = []
        for spk in spks:
            spk_stack = np.stack(spk, axis=0)
            out = self._get_spikerates(spk_stack, binsize, (begin, end),
                                       binstep, accumulate,
                                       convert_seconds=not self.seconds)
            resp_arr, xs = out
            if repl_nan:
                no_spk_mask = np.all(resp_arr == 0, axis=-1)
                resp_arr[no_spk_mask] = np.nan
            n_trls.append(resp_arr.shape[0])
            if skl_axes:
                resp_arr = np.expand_dims(np.swapaxes(resp_arr, 0, 1), 1)
            outs.append(resp_arr)
        if combine_pseudo:
            outs = self.make_pseudopop(outs, n_trls, min_trials_pseudo,
                                       resample_pseudos)
        return outs, xs

    def get_ntrls(self):
        return list(len(o) for o in self['data'])
    
    def decode_masks(self, m1, m2, winsize, begin, end, stepsize, n_folds=20,
                     model=svm.SVC, params=None, pre_pca=None,
                     mean=False, shuffle=False, time_zero_field=None,
                     pseudo=False, min_trials_pseudo=10, resample_pseudo=10,
                     repl_nan=False, impute_missing=False):
        if params is None:
            params = {'class_weight':'balanced'}

        cat1 = self.mask(m1)
        cat2 = self.mask(m2)
        pop1, xs = cat1.get_populations(winsize, begin, end, stepsize,
                                        skl_axes=True, repl_nan=repl_nan,
                                        time_zero_field=time_zero_field)
        pop2, xs = cat2.get_populations(winsize, begin, end, stepsize,
                                        skl_axes=True, repl_nan=repl_nan,
                                        time_zero_field=time_zero_field)
        if pseudo:
            c1_n = cat1.get_ntrls()
            c2_n = cat2.get_ntrls()
            comb_n = self.combine_ntrls(c1_n, c2_n)
            pop1 = self.make_pseudopop(pop1, comb_n, min_trials_pseudo,
                                       resample_pseudo, skl_axs=True)
            pop2 = self.make_pseudopop(pop2, comb_n, min_trials_pseudo,
                                       resample_pseudo, skl_axs=True)
        outs = np.zeros((len(pop2), n_folds, len(xs)))
        for i, p1 in enumerate(pop1):
            out = na.fold_skl(p1, pop2[i], n_folds, model=model, params=params, 
                              mean=mean, pre_pca=pre_pca, shuffle=shuffle,
                              impute_missing=(repl_nan or impute_missing))
            outs[i] = out
        return outs, xs

    def estimate_dimensionality(self, mask, winsize, begin, end, stepsize,
                                n_resamples=20):
        data_masked = self.mask(mask)
        pops, xs = data_masked.get_populations(winsize, begin, end, stepsize)
        outs = np.zeros((len(pops), n_resamples, len(xs)))
        for i, p in enumerate(pops):
            outs[i] = na.estimate_participation_ratio(p, n_resamples=n_resamples)
        return outs, xs
