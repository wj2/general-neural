import numpy as np
import pandas as pd
from sklearn import svm
import sklearn.linear_model as sklm
import sklearn.preprocessing as skp
import quantities as pq
import neo

import general.utility as u
import general.neural_analysis as na


class ResultSequence(object):
    def __init__(self, x):
        self.val = list(x)

    def __hash__(self):
        hashable = tuple(tuple(x) for x in self.val)
        return hash(hashable)

    def _op(self, x, operator):
        out = ResultSequence(pd.Series(operator(v, x)) for v in self.val)
        return out

    def _unary_op(self, operator):
        out = ResultSequence(operator(v) for v in self.val)
        return out

    def _op_rs(self, x, operator):
        out = ResultSequence(operator(v, x[i]) for i, v in enumerate(self.val))
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

    def _op_dispatcher(self, x, op, use_rs=True):
        try:
            assert use_rs
            len(x)
            if isinstance(x, str):
                raise TypeError()
            out = self._op_rs(x, op)
        except (TypeError, AssertionError):
            out = self._op(x, op)
        return out

    def one_of(self, x):
        return self._op_dispatcher(x, np.isin, use_rs=False)

    def rs_isnan(self):
        return self._unary_op(np.isnan)

    def rs_and(self, x):
        return self._op_dispatcher(x, np.logical_and)

    def rs_or(self, x):
        return self._op_dispatcher(x, np.logical_or)

    def rs_xor(self, x):
        return self._op_dispatcher(x, np.logical_xor)

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
    filt = np.ones(filt_wid) / filt_wid
    n_ts = psth_tm.shape[2]
    out_len = int(n_ts - binsize + 1)
    mult = 1000 / binwidth
    smooth_psth = np.zeros(psth_tm.shape[:2] + (out_len,))
    for i, ptm_i in enumerate(psth_tm):
        for j, ptm_ij in enumerate(ptm_i):
            smooth_psth[i, j] = np.convolve(ptm_ij, filt, mode="valid")
    if causal_xs:
        xs_adj = xs[:out_len]
    else:
        step = xs[1] - xs[0]
        cent_offset = step * binsize / 2
        xs_adj = xs + cent_offset
        xs_adj = xs_adj[:out_len]
    return mult * smooth_psth, xs_adj


def _format_for_svm(pops):
    pop_format = []
    neur_mins = []
    for pop in pops:
        pop_format.extend(list(neur[0] for neur in pop))
        neur_mins.extend(np.ones(pop.shape[0]) * pop.shape[2])
    neur_pop = np.expand_dims(np.array(pop_format, dtype=object), 1)
    neur_mins = np.array(neur_mins).astype(int)
    return neur_pop, neur_mins


class Dataset(object):
    def __init__(self, dframe, seconds=False):
        self.data = dframe
        try:
            self.session_fields = self.data["data"].iloc[0].columns
        except KeyError as e:
            if len(self.data["data"]) == 0:
                raise IOError("no data available")
            else:
                raise e
        self.n_sessions = len(self.data)
        self.data = self.data.sort_values("date", ignore_index=True)
        self.data = self.data.sort_values("animal", ignore_index=True)
        self.seconds = seconds
        self.population_cache = {}
        self.mask_pop_cache = {}

    @classmethod
    def from_dict(cls, sort_by=None, seconds=False, **inputs):
        df = pd.DataFrame(data=inputs)
        if sort_by is not None:
            df.sort_values(by=sort_by, inplace=True)
            df.reset_index(drop=True, inplace=True)
        return cls(df, seconds=seconds)

    @classmethod
    def from_readfunc(cls, read_func, *args, seconds=False, sort_by=None, **kwargs):
        super_df = read_func(*args, **kwargs)
        return cls.from_dict(seconds=seconds, sort_by=sort_by, **super_df)

    def __getitem__(self, key):
        try:
            out = self.data[key]
        except KeyError:
            out = ResultSequence(dd[key] for dd in self.data["data"])
        return out

    def __contains__(self, key):
        top_level = key in self.data.keys()
        session_level = key in self.data["data"].iloc[0].columns
        return top_level or session_level

    @property
    def session_keys(self):
        return self.data["data"].iloc[0].columns

    def session_mask(self, mask):
        return Dataset(self.data[mask], seconds=self.seconds)

    def mask(self, mask):
        df = {}
        for c in self.data.columns:
            df[c] = self.data[c]
        dlist = []

        for i, m in enumerate(mask):
            d = self.data["data"][i][m]
            dlist.append(d)
        df["data"] = dlist
        return Dataset.from_dict(**df, seconds=self.seconds)

    def _center_spks(self, spks, tz, tzf):
        if tz is not None:
            spks = spks - tz
        if tzf is not None:
            spks = spks - self[tzf]
        return spks

    def get_response_in_window(self, begin, end, time_zero=None, time_zero_field=None):
        spks = self["spikeTimes"]
        spks = self._center_spks(spks, time_zero, time_zero_field)
        out = []
        for spk in spks:
            resp_arr = np.array(list(u.get_spks_window(spk, begin, end)))
            out.append(resp_arr)
        return out

    def _get_spikerates(
        self,
        spk,
        binsize,
        bounds,
        binstep,
        accumulate=False,
        convert_seconds=True,
        pre_bound=-2,
        post_bound=2,
        use_new=True,
    ):
        xs = na.compute_xs(binsize, bounds[0], bounds[1], binstep)
        if len(spk.shape) > 2:
            spk = np.squeeze(spk)
        no_spks = np.zeros_like(spk, dtype=bool)
        # make 3D histogram by trial, units, time
        if spk.shape[1] == 0:
            out_arr = np.zeros((len(spk), 0, len(xs)))
        if use_new:
            out_arr = na.bin_spiketimes_3d(
                spk,
                binsize,
                bounds,
                binstep,
                accumulate=accumulate,
                spks_per_sec=not self.seconds,
            )
            no_spks = np.sum(out_arr, axis=2) == 0
        else:
            for i, spk_i in enumerate(spk):
                for j, spk_ij in enumerate(spk_i):
                    spk_sq = np.squeeze(spk_ij)
                    # UNCOMMENT IF SOMETHING BREAKS IN BUSCHMAN
                    # if convert_seconds:
                    #     spk_sq = spk_sq * 1000
                    no_spks[i, j] = len(spk_sq.shape) == 0 or len(spk_sq) == 0
                    resp_arr = na.bin_spiketimes(
                        spk_sq,
                        binsize,
                        bounds,
                        binstep,
                        accumulate=accumulate,
                        spks_per_sec=not self.seconds,
                    )
                    if i == 0 and j == 0:
                        tlen = len(resp_arr)
                        out_arr = np.zeros(spk.shape + (tlen,))
                    out_arr[i, j] = resp_arr
        return out_arr, xs, no_spks

    def make_pseudopop(
        self,
        outs,
        n_trls=None,
        min_trials_pseudo=10,
        n_trls_mask=None,
        resample_pseudos=10,
        skl_axs=False,
        same_n_trls=False,
        subsample_neurons=None,
        use_inds=None,
    ):
        rng = np.random.default_rng()
        if n_trls is None:
            n_trls = list(len(o) for o in outs)
        if n_trls_mask is None:
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
            n_trls_actual = np.array(list(o.shape[trl_ax] for o in outs_mask))
            min_trls = np.min(n_trls_actual[n_trls_mask])

        save_inds = []
        fracs = np.zeros((len(outs_mask), 2))
        fracs[:, 0] = min_trls
        for i in range(resample_pseudos):
            for j, pop in enumerate(outs_mask):
                fracs[j, 1] = pop.shape[trl_ax]
                trl_inds = np.random.choice(pop.shape[trl_ax], min_trls, replace=False)
                if skl_axs:
                    ppop_j = pop[:, :, trl_inds]
                else:
                    ppop_j = pop[trl_inds]
                if j == 0:
                    ppop = ppop_j
                else:
                    ppop = np.concatenate((ppop, ppop_j), axis=neur_ax)
            if subsample_neurons is not None:
                if ppop.shape[neur_ax] >= subsample_neurons:
                    if use_inds is None:
                        inds = rng.choice(
                            ppop.shape[neur_ax], size=subsample_neurons, replace=False,
                        )
                    else:
                        inds = use_inds[i]
                    save_inds.append(inds)
                    if neur_ax != 0:
                        raise IOError(
                            "neur_ax cannot be non-zero, it is {}".format(neur_ax)
                        )
                    ppop = ppop[inds]
            if i == 0:
                out_pseudo = np.zeros((resample_pseudos,) + ppop.shape)
            
            out_pseudo[i] = ppop
        out = (out_pseudo, fracs)
        if subsample_neurons is not None:
            out = out + (save_inds,)
        return out

    def get_nneurs(self):
        return self["n_neurs"]

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

    def get_psth(
        self,
        binsize=None,
        begin=None,
        end=None,
        binstep=None,
        skl_axes=False,
        accumulate=False,
        time_zero=None,
        time_zero_field=None,
        combine_pseudo=False,
        min_trials_pseudo=10,
        resample_pseudos=10,
        repl_nan=False,
        regions=None,
        ret_no_spk=False,
        causal_timing=False,
        shuffle_trials=False,
    ):
        psths = self["psth"]
        xs_all = self["psth_timing"]
        if regions is not None:
            regions_all = self["neur_regions"]
        outs = []
        n_trls = []
        for i, psth_l in enumerate(psths):
            if len(psth_l) > 0:
                psth = np.stack(psth_l, axis=0)
            else:
                psth = np.zeros((0, 0, len(xs_all[i])))
            xs = xs_all[i]
            if time_zero is not None:
                xs_i = np.repeat(np.expand_dims(xs - time_zero, 0), len(psth_l), 0)
            elif time_zero_field is not None:
                xs_exp = np.expand_dims(xs, 0)
                tzf = self[time_zero_field][i]
                tzf_exp = np.expand_dims(tzf, 1)
                if len(tzf_exp) == 0:
                    tzf_exp = np.zeros((1, 1))
                xs_i = xs_exp - tzf_exp
            else:
                xs_i = np.repeat(np.expand_dims(xs, 0), len(psth_l), 0)
            psth_bs = xs[1] - xs[0]
            time_mask = np.ones_like(xs_i, dtype=bool)
            if begin is not None:
                time_mask = np.logical_and(xs_i >= begin, time_mask)
            if end is not None:
                time_mask = np.logical_and(xs_i < end, time_mask)
            psth_tm = np.zeros(psth.shape[:2] + (np.sum(time_mask, axis=1)[0],))
            for k, trl in enumerate(psth):
                for j, trl_ij in enumerate(trl):
                    psth_tm[k, j] = trl_ij[time_mask[k]]
            xs_reg = xs_i[0, time_mask[0]]
            if binsize is not None:
                assert (binsize % psth_bs) == 0
                psth_tm, xs_reg = _convolve_psth(
                    psth_tm, binsize / psth_bs, xs_reg, causal_xs=causal_timing
                )
            if binstep is not None:
                xs_0 = xs_reg - xs_reg[0]
                xs_rem = np.mod(xs_0 / binstep, 1)
                step_mask = xs_rem == 0
                psth_tm = psth_tm[..., step_mask]
                xs_reg = xs_reg[step_mask]
            if regions is not None and len(psth_tm) > 0:
                regs = regions_all[i].iloc[0]
                mask = np.isin(regs, regions)
                psth_tm = psth_tm[:, mask]
            if shuffle_trials:
                rng = np.random.default_rng()
                list(rng.shuffle(psth_tm[:, i]) for i in range(psth_tm.shape[1]))
            if skl_axes:
                psth_tm = np.expand_dims(np.swapaxes(psth_tm, 0, 1), 1)
            outs.append(psth_tm)
            n_trls.append(psth_tm.shape[0])
        if combine_pseudo:
            outs, _ = self.make_pseudopop(
                outs, n_trls, min_trials_pseudo, resample_pseudos
            )
        return outs, xs_reg

    def get_psth_window(self, begin, end, **kwargs):
        psths, xs = self.get_psth(begin=begin, end=end, **kwargs)
        time_filt = np.logical_and(xs >= begin, xs < end)
        psths_collapsed = []
        for i, psth in enumerate(psths):
            psth_c = np.sum(psth[..., time_filt], axis=-1)
            psths_collapsed.append(psth_c)
        return psths_collapsed

    def get_spiketrains(
        self,
        begin,
        end,
        time_zero_field=None,
        regions=None,
        time_zero=None,
        repl_nan=False,
        ret_no_spk=False,
    ):
        spks = self["spikeTimes"]
        spks = self._center_spks(spks, time_zero, time_zero_field)
        if regions is not None:
            regions_all = self["neur_regions"]
        outs = []
        n_trls = []
        no_spks = []
        for i, spk in enumerate(spks):
            if len(spk) == 0:
                xs = na.compute_xs(10, begin, end, 10)
                resp_arr = np.zeros((0, self.get_nneurs()[i], len(xs)))
                out = (resp_arr, xs, np.ones_like(resp_arr, dtype=bool))
            else:
                spk_stack = np.stack(spk, axis=0)
                resp_arr = np.zeros(spk_stack.shape, dtype=object)
                no_spk_mask = np.zeros_like(resp_arr, dtype=bool)
                for i, j in u.make_array_ind_iterator(spk_stack.shape):
                    spk_ij = spk_stack[i, j]
                    mask = np.logical_and(spk_ij > begin, spk_ij < end)
                    resp_arr[i, j] = neo.SpikeTrain(
                        spk_ij[mask] * pq.s, end * pq.s, t_start=begin * pq.s
                    )
                    if len(spk_ij[mask]) == 0:
                        no_spk_mask[i, j] = True
            if regions is not None and len(resp_arr) > 0:
                regs = regions_all[i].iloc[0]
                mask = np.isin(regs, regions)
                resp_arr = resp_arr[mask]
            if repl_nan:
                resp_arr[no_spk_mask] = np.nan
            n_trls.append(resp_arr.shape[0])
            outs.append(resp_arr)
            no_spks.append(no_spk_mask)
        if ret_no_spk:
            out = (outs, no_spks)
        else:
            out = outs
        return out

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

    def _get_populations(
        self,
        binsize,
        begin,
        end,
        binstep=None,
        skl_axes=False,
        accumulate=False,
        time_zero=None,
        time_zero_field=None,
        combine_pseudo=False,
        min_trials_pseudo=10,
        resample_pseudos=10,
        repl_nan=False,
        regions=None,
        ret_no_spk=False,
        shuffle_trials=False,
        use_new=False,
    ):
        spks = self["spikeTimes"]
        spks = self._center_spks(spks, time_zero, time_zero_field)
        if regions is not None:
            regions_all = self["neur_regions"]
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
                if regions is not None:
                    regs = regions_all[i].iloc[0]
                    mask = np.isin(regs, regions)
                    spk_stack = spk_stack[:, mask]
                out = self._get_spikerates(
                    spk_stack,
                    binsize,
                    (begin, end),
                    binstep,
                    accumulate,
                    convert_seconds=not self.seconds,
                    use_new=use_new,
                )
            resp_arr, xs, no_spk_mask = out
            # if regions is not None and len(resp_arr) > 0:
            #     regs = regions_all[i].iloc[0]
            #     mask = np.isin(regs, regions)
            #     resp_arr = resp_arr[:, mask]
            if repl_nan:
                resp_arr[no_spk_mask] = np.nan
            n_trls.append(resp_arr.shape[0])
            if skl_axes:
                resp_arr = np.expand_dims(np.swapaxes(resp_arr, 0, 1), 1)
            outs.append(resp_arr)
            no_spks.append(no_spk_mask)
        if combine_pseudo:
            outs, _ = self.make_pseudopop(
                outs, n_trls, min_trials_pseudo, resample_pseudos
            )
        if ret_no_spk:
            out = (outs, xs, no_spks)
        else:
            out = (outs, xs)
        return out

    def get_ntrls(self):
        return list(len(o) for o in self["data"])

    def regress_target_field(
            self,
            target_field,
            *args,
            **kwargs
    ):
        target = self[target_field]
        return self.regress_target(target, *args, **kwargs)

    def regress_target(
            self,
            target,
            winsize,
            begin,
            end,
            stepsize,
            n_folds=20,
            model=sklm.Ridge,
            params=None,
            pre_pca=None,
            mean=False,
            shuffle=False,
            time_zero_field=None,
            gen_tzf=None,
            pseudo=False,
            repl_nan=False,
            impute_missing=False,
            ret_pops=False,
            train_mask=None,
            gen_mask=None,
            regions=None,
            time_accumulate=False,
            min_trials=20,
            ret_pred_targ=False,
            **kwargs,
    ):
        if params is None:
            # params = {'class_weight':'balanced'}
            params = {}
            params.update(kwargs)
        if gen_tzf is None:
            gen_tzf = time_zero_field
        if train_mask is not None:
            data_tr = self.mask(train_mask)
            targs_tr = list(
                target[i][mask_i].to_numpy() for i, mask_i in enumerate(train_mask)
            )
        else:
            data_tr = self
            targs_tr = list(
                t.to_numpy() for t in target
            )
        pops_tr, xs = data_tr.get_neural_activity(
            winsize,
            begin,
            end,
            stepsize,
            skl_axes=True,
            repl_nan=repl_nan,
            time_zero_field=time_zero_field,
            regions=regions,
        )
        if gen_mask is not None:
            data_te = self.mask(gen_mask)
            targs_te = list(target[i][mask_i] for i, mask_i in enumerate(gen_mask))
            pops_te, xs = data_te.get_neural_activity(
                winsize,
                begin,
                end,
                stepsize,
                skl_axes=True,
                repl_nan=repl_nan,
                time_zero_field=time_zero_field,
                regions=regions,
            )
        else:
            pops_te = (None,)*len(pops_tr)
            targs_te = (None,)*len(pops_tr)
        outs = np.zeros((len(pops_tr), n_folds, len(xs)))
        outs_gen = np.zeros_like(outs)
        outs_pred = []
        outs_targ = []
        for i, pop_tr in enumerate(pops_tr):
            labels_tr = targs_tr[i]
            labels_te = targs_te[i]
            if pops_te[i] is not None:
                pop_te_i = np.squeeze(pops_te[i])
            else:
                pop_te_i = pops_te[i]
            if pop_tr.shape[0] == 0 or pop_tr.shape[2] < min_trials:
                out_i = {
                    "score": np.zeros((n_folds, len(xs)))*np.nan,
                    "score_gen": np.zeros((n_folds, len(xs)))*np.nan,
                    "predictions": np.zeros((n_folds, 0, len(xs)))*np.nan,
                    "targets": np.zeros((n_folds, 0, len(xs)))*np.nan,
                }
            else:
                out_i = na.fold_skl_continuous(
                    np.squeeze(pop_tr),
                    labels_tr,
                    folds_n=n_folds,
                    model=model,
                    params=params,
                    mean=mean,
                    pre_pca=pre_pca,
                    shuffle=shuffle,
                    gen_cs=(pop_te_i, labels_te),
                    impute_missing=(repl_nan or impute_missing),
                    time_accumulate=time_accumulate,
                )
            if labels_te is not None:
                outs[i] = out_i["score"]
                outs_gen[i] = out_i["score_gen"]
            else:
                outs[i] = out_i["score"]
            outs_pred.append(out_i["predictions"])
            outs_targ.append(out_i["targets"])
        if ret_pops:
            out = (outs, xs, pops_tr)
            add = tuple(di for di in pops_te if di[0] is not None)
            out = out + (add,)
        else:
            out = (outs, xs)
        if gen_mask is not None:
            out = out + (outs_gen,)
        if ret_pred_targ:
            out = out + (outs_pred, outs_targ)
        return out

    
    def regress_discrete_masks(
        self,
        masks,
        winsize,
        begin,
        end,
        stepsize,
        n_folds=20,
        model=sklm.Ridge,
        params=None,
        pre_pca=None,
        mean=False,
        shuffle=False,
        time_zero_field=None,
        pseudo=False,
        min_trials_pseudo=10,
        resample_pseudo=10,
        repl_nan=False,
        impute_missing=False,
        ret_pops=False,
        shuffle_trials=False,
        decode_masks=None,
        decode_tzf=None,
        regions=None,
        combine=False,
        time_accumulate=False,
        **kwargs
    ):
        if params is None:
            # params = {'class_weight':'balanced'}
            params = {}
            params.update(kwargs)
        cats = list(self.mask(mask_i) for mask_i in masks)
        if decode_tzf is None:
            decode_tzf = time_zero_field
        if decode_masks is not None:
            dec_data = list(self.mask(mask_i) for mask_i in decode_masks)
        else:
            decs = (None,) * len(cats)
        outs = list(
            cat_i.get_neural_activity(
                winsize,
                begin,
                end,
                stepsize,
                skl_axes=True,
                repl_nan=repl_nan,
                time_zero_field=time_zero_field,
                shuffle_trials=shuffle_trials,
                regions=regions,
            )
            for cat_i in cats
        )
        if decode_masks is not None:
            decs = list(
                cat_i.get_neural_activity(
                    winsize,
                    begin,
                    end,
                    stepsize,
                    skl_axes=True,
                    repl_nan=repl_nan,
                    time_zero_field=decode_tzf,
                    shuffle_trials=shuffle_trials,
                    regions=regions,
                )
                for cat_i in dec_data
            )
        xs = outs[0][1]
        if pseudo:
            trls_list = tuple(cat_i.get_ntrls() for cat_i in cats)
            if decode_masks is not None:
                trls_list = trls_list + tuple(cat_i.get_ntrls() for cat_i in dec_data)
            comb_n = combine_ntrls(*trls_list)
            pops = list(
                self.make_pseudopop(
                    pop_i,
                    comb_n,
                    min_trials_pseudo=min_trials_pseudo,
                    resample_pseudos=resample_pseudo,
                    skl_axs=True,
                    same_n_trls=True,
                )[0]
                for pop_i, xs in outs
            )
            if decode_masks is not None:
                decs = list(
                    self.make_pseudopop(
                        dec_i,
                        comb_n,
                        min_trials_pseudo=min_trials_pseudo,
                        resample_pseudos=resample_pseudo,
                        skl_axs=True,
                        same_n_trls=True,
                    )[0]
                    for dec_i, xs in decs
                )
            else:
                decs = (None,) * resample_pseudo
        outs = np.zeros((len(pops[0]), n_folds, len(xs)))
        outs_gen = np.zeros_like(outs)
        for i in range(len(pops[0])):
            if combine:
                ps = list(
                    np.concatenate((pops[j][i], decs[j][i]), axis=2)
                    for j in range(len(pops))
                )
                ds = None
            else:
                ps = list(pop_j[i] for pop_j in pops)
                if decode_masks is not None:
                    ds = list(pop_j[i] for pop_j in decs)
                else:
                    ds = None
            out = na.fold_skl_multi(
                *ps,
                folds_n=n_folds,
                model=model,
                params=params,
                mean=mean,
                pre_pca=pre_pca,
                shuffle=shuffle,
                impute_missing=(repl_nan or impute_missing),
                gen_cs=ds,
                time_accumulate=time_accumulate
            )
            if not combine and (decode_masks is not None):
                outs[i] = out["score"]
                outs_gen[i] = out["score_gen"]
            else:
                outs[i] = out["score"]
        if ret_pops:
            out = (outs, xs, pops)
            add = tuple(di for di in decs if di[0] is not None and not combine)
            out = out + (add,)
        else:
            out = (outs, xs)
        if decode_masks is not None:
            out = out + (outs_gen,)
        return out

    def get_neural_activity(self, winsize, begin, end, stepsize=None, **kwargs):
        if "spikeTimes" in self:
            out = self.get_populations(winsize, begin, end, binstep=stepsize, **kwargs)
        elif "psth" in self:
            out = self.get_psth(winsize, begin, end, binstep=stepsize, **kwargs)
        else:
            raise TypeError("no neural data associated with this dataset")
        return out

    def get_time_features(
            self,
            n_xs,
            skl_axes=False,
            trial_field=None,
            model=skp.SplineTransformer,
            knots_per_trial=.01,
            degree=2,
            use_trs=None
    ):
        if trial_field is None:
            inds = list(row.data.index for _, row in self.data.iterrows())
        else:
            inds = self[trial_field]
        feats = []
        for ind_group in inds:
            ind_group = np.expand_dims(ind_group, 1)
            if use_trs is None:
                n_knots = max(int(np.round(len(ind_group)*knots_per_trial)), 2)
                m = model(n_knots, degree)
                use_trs = m.fit(ind_group)
            if ind_group.shape[0] == 0:
                n_dim = use_trs.transform([[0]]).shape[1]
                feat_ig = np.zeros((0, n_dim))
            else:
                feat_ig = use_trs.transform(ind_group)
            feat_ig = np.expand_dims(feat_ig, 2)
            feat_ig = np.repeat(feat_ig, n_xs, axis=2)
            if skl_axes:
                feat_ig = np.expand_dims(np.swapaxes(feat_ig, 0, 1), 1)
            feats.append(feat_ig)
        return feats, use_trs
    
    def get_dec_pops(
        self,
        winsize,
        begin,
        end,
        stepsize,
        *masks,
        tzfs=None,
        repl_nan=False,
        shuffle_trials=True,
        regions=None,
        use_time=False,
        combined_time=True,
        time_field=None,
    ):
        try:
            assert len(tzfs) == len(masks)
        except AssertionError:
            tzfs = (tzfs,) * len(masks)
        out_pops = []
        if use_time and combined_time:
            _, trs = self.get_time_features(1)
        else:
            trs = None
        for i, m in enumerate(masks):
            if m is not None:
                cat_m = self.mask(m)
                out_m = cat_m.get_neural_activity(
                    winsize,
                    begin,
                    end,
                    stepsize,
                    skl_axes=True,
                    repl_nan=repl_nan,
                    time_zero_field=tzfs[i],
                    shuffle_trials=shuffle_trials,
                    regions=regions,
                )
                pop_m, xs = out_m
                if use_time and not shuffle_trials:
                    t_feat, _ = cat_m.get_time_features(
                        len(xs), skl_axes=True, use_trs=trs,
                    )
                    pops = []
                    for i, pop in enumerate(pop_m):
                        pops.append(np.concatenate((pop, t_feat[i]), axis=0))
                    pop_m = pops
            else:
                pop_m = (None,) * len(self.data)
            out_pops.append(pop_m)
        return xs, out_pops

        # cat1 = self.mask(m1)
        # cat2 = self.mask(m2)
        # if decode_tzf is None:
        #     decode_tzf = time_zero_field
        # if decode_m1 is not None:
        #     dec_data_c1 = self.mask(decode_m1)
        # else:
        #     dec1 = (None,)*len(cat1)
        # if decode_m2 is not None:
        #     dec_data_c2 = self.mask(decode_m2)
        # else:
        #     dec2 = (None,)*len(cat2)
        # out1 = cat1.get_neural_activity(winsize, begin, end, stepsize,
        #                                 skl_axes=True, repl_nan=repl_nan,
        #                                 time_zero_field=time_zero_field,
        #                                 shuffle_trials=shuffle_trials,
        #                                 regions=regions)
        # out2 = cat2.get_neural_activity(winsize, begin, end, stepsize,
        #                                 skl_axes=True, repl_nan=repl_nan,
        #                                 time_zero_field=time_zero_field,
        #                                 shuffle_trials=shuffle_trials,
        #                                 regions=regions)
        # if decode_m1 is not None:
        #     dec1 = dec_data_c1.get_neural_activity(
        #         winsize, begin, end, stepsize, skl_axes=True,
        #         repl_nan=repl_nan, time_zero_field=decode_tzf,
        #         shuffle_trials=shuffle_trials, regions=regions)[0]
        # if decode_m2 is not None:
        #     dec2 = dec_data_c2.get_neural_activity(
        #         winsize, begin, end, stepsize, skl_axes=True,
        #         repl_nan=repl_nan, time_zero_field=decode_tzf,
        #         shuffle_trials=shuffle_trials, regions=regions)[0]
        # pop1, xs = out1
        # pop2, xs = out2

    def decode_masks_upsample(
        self,
        m1,
        m2,
        winsize,
        begin,
        end,
        stepsize,
        model=svm.LinearSVC,
        params=None,
        pre_pca=None,
        mean=False,
        shuffle=False,
        time_zero_field=None,
        pseudo=False,
        min_trials_pseudo=10,
        resample_pseudo=10,
        repl_nan=False,
        impute_missing=False,
        ret_pops=False,
        shuffle_trials=False,
        decode_m1=None,
        decode_m2=None,
        decode_tzf=None,
        regions=None,
        combine=False,
        n_pseudo=500,
        **kwargs
    ):
        out = self.get_dec_pops(
            winsize,
            begin,
            end,
            stepsize,
            m1,
            m2,
            decode_m1,
            decode_m2,
            tzfs=(time_zero_field, time_zero_field, decode_tzf, decode_tzf),
            repl_nan=repl_nan,
            regions=regions,
            shuffle_trials=shuffle_trials,
        )
        xs, pops = out
        pop1, pop2, dec1, dec2 = pops
        if params is None:
            params = {"class_weight": "balanced", "max_iter": 10000}
            params.update(kwargs)

        cat1_f, _ = _format_for_svm(pop1)
        cat2_f, _ = _format_for_svm(pop2)
        if dec1 is not None:
            cat1_gen_f, _ = _format_for_svm(dec1)
        else:
            cat1_gen_f = None
        if dec2 is not None:
            cat2_gen_f, _ = _format_for_svm(dec2)
        else:
            cat2_gen_f = None

        out_dec = na.decoding(
            cat1_f,
            cat2_f,
            format_=False,
            multi_cond=True,
            sample_pseudo=True,
            resample=resample_pseudo,
            pre_pca=pre_pca,
            require_trials=min_trials_pseudo,
            cat1_gen=cat1_gen_f,
            cat2_gen=cat2_gen_f,
            model=model,
            params=params,
        )
        out = (out_dec[0], xs)
        if ret_pops:
            out = out + tuple(pops)
        if dec1 is not None or dec2 is not None:
            out = out + (out_dec[-1],)
        return out

    def make_pseudo_pops(
        self,
        winsize,
        begin,
        end,
        stepsize,
        *masks,
        tzfs=None,
        repl_nan=False,
        shuffle_trials=True,
        regions=None,
        min_trials=10,
        resamples=10,
        skl_axs=True,
        same_n_trls=True
    ):
        xs, pops = self.get_dec_pops(
            winsize,
            begin,
            end,
            stepsize,
            *masks,
            tzfs=tzfs,
            repl_nan=repl_nan,
            shuffle_trials=shuffle_trials,
            regions=regions
        )
        pops_pseudo = self.sample_pseudo_pops(
            *pops,
            min_trials=min_trials,
            resamples=resamples,
            skl_axs=skl_axs,
            same_n_trls=same_n_trls
        )
        return xs, pops_pseudo

    def sample_pseudo_pops(
        self, *pops, min_trials=10, resamples=10, skl_axs=True, same_n_trls=True
    ):
        trls_list = []
        for pop in pops:
            ci_n = list(pop_i.shape[2] for pop_i in pop)
            trls_list.append(ci_n)
        comb_n = combine_ntrls(*trls_list)
        pops_pseudo = []
        for pop in pops:
            pop_i, _ = self.make_pseudopop(
                pop,
                comb_n,
                min_trials,
                resample_pseudos=resamples,
                skl_axs=skl_axs,
                same_n_trls=same_n_trls,
            )
            pops_pseudo.append(pop_i)
        return pops_pseudo

    def decode_masks(
        self,
        m1,
        m2,
        winsize,
        begin,
        end,
        stepsize,
        n_folds=20,
        model=svm.LinearSVC,
        params=None,
        pre_pca=None,
        mean=False,
        shuffle=False,
        time_zero_field=None,
        pseudo=False,
        min_trials_pseudo=10,
        resample_pseudo=10,
        repl_nan=False,
        impute_missing=False,
        ret_pops=False,
        shuffle_trials=False,
        decode_m1=None,
        decode_m2=None,
        decode_tzf=None,
        regions=None,
        combine=False,
        max_iter=10000,
        dec_less=True,
        time_mask=None,
        dec_beg=None,
        dec_end=None,
        collapse_time=False,
        ret_projections=False,
        use_time=False,
        subsample_neurons=None,
        **kwargs
    ):
        out = self.get_dec_pops(
            winsize,
            begin,
            end,
            stepsize,
            m1,
            m2,
            decode_m1,
            decode_m2,
            tzfs=(time_zero_field, time_zero_field, decode_tzf, decode_tzf),
            repl_nan=repl_nan,
            regions=regions,
            shuffle_trials=shuffle_trials,
            use_time=use_time,
        )
        xs, pops = out

        one_of_dec = dec_beg is not None or dec_end is not None
        if collapse_time and time_mask is None and one_of_dec:
            if dec_beg is None:
                dec_beg = xs[0]
            if dec_end is None:
                dec_end = xs[-1]
            beg_mask = dec_beg <= (xs - winsize / 2)
            end_mask = dec_end >= (xs + winsize / 2)
            time_mask = np.logical_and(beg_mask, end_mask)
        pop1, pop2, dec1, dec2 = pops

        # print(pop1)
        if params is None:
            params = {
                "class_weight": "balanced",
                "max_iter": max_iter,
                "dual": "auto",
            }
            # params.update(kwargs)

        if pseudo:
            c1_n = list(pop_i.shape[2] for pop_i in pop1)
            c2_n = list(pop_i.shape[2] for pop_i in pop2)
            trls_list = (c1_n, c2_n)
            trls_list_dec = ()
            if decode_m1 is not None:
                g1_n = list(pop_i.shape[2] for pop_i in dec1)
                if dec_less:
                    trls_list_dec = trls_list_dec + (g1_n,)
                else:
                    trls_list = trls_list + (g1_n,)
            if decode_m2 is not None:
                g2_n = list(pop_i.shape[2] for pop_i in dec2)
                if dec_less:
                    trls_list_dec = trls_list_dec + (g2_n,)
                else:
                    trls_list = trls_list + (g2_n,)

            comb_n = combine_ntrls(*trls_list)
            if dec_less:
                comb_n_dec = combine_ntrls(*trls_list_dec)
            else:
                comb_n_dec = comb_n

            out = self.make_pseudopop(
                pop1,
                comb_n,
                min_trials_pseudo,
                resample_pseudos=resample_pseudo,
                skl_axs=True,
                same_n_trls=True,
                subsample_neurons=subsample_neurons,
            )
            pop1, _ = out[:2]
            if subsample_neurons is not None:
                use_inds = out[-1]
            else:
                use_inds = None
            pop2, _ = self.make_pseudopop(
                pop2,
                comb_n,
                min_trials_pseudo,
                resample_pseudos=resample_pseudo,
                skl_axs=True,
                same_n_trls=True,
                subsample_neurons=subsample_neurons,
                use_inds=use_inds,
            )[:2]
            n_trls_mask = comb_n >= min_trials_pseudo
            if decode_m1 is not None:
                dec1, _ = self.make_pseudopop(
                    dec1,
                    comb_n_dec,
                    n_trls_mask=n_trls_mask,
                    resample_pseudos=resample_pseudo,
                    skl_axs=True,
                    same_n_trls=True,
                    subsample_neurons=subsample_neurons,
                    use_inds=use_inds,
                )[:2]
            else:
                dec1 = (None,) * resample_pseudo
            if decode_m2 is not None:
                dec2, _ = self.make_pseudopop(
                    dec2,
                    comb_n_dec,
                    n_trls_mask=n_trls_mask,
                    resample_pseudos=resample_pseudo,
                    skl_axs=True,
                    same_n_trls=True,
                    subsample_neurons=subsample_neurons,
                    use_inds=use_inds,
                )[:2]
            else:
                dec2 = (None,) * resample_pseudo
        print(pop1[0].shape, pop2[0].shape, dec1[0].shape, dec2[0].shape)
        outs = np.zeros((len(pop2), n_folds, len(xs)))
        outs_gen = np.zeros_like(outs)
        for i, p1 in enumerate(pop1):
            if combine:
                p1 = np.concatenate((p1, dec1[i]), axis=2)
                p2 = np.concatenate((pop2[i], dec2[i]), axis=2)
                d1 = None
                d2 = None
            else:
                p2 = pop2[i]
                d1 = dec1[i]
                d2 = dec2[i]
            cond1 = p1.shape[2] < min_trials_pseudo or p2.shape[2] < min_trials_pseudo
            cond2 = d1.shape[2] == 0 and d2.shape[2] == 0
            if p1.shape[0] == 0 or cond1 or cond2:
                out = {
                    "score": np.zeros((n_folds, len(xs)))*np.nan,
                    "score_gen": np.zeros((n_folds, len(xs)))*np.nan,
                    "predictions": np.zeros((n_folds, 0, len(xs)))*np.nan,
                    "targets": np.zeros((n_folds, 0, len(xs)))*np.nan,
                }
            else:
                if d1.shape[2] == 0:
                    d1 = np.zeros((d2.shape[0],) + d1.shape[1:])
                if d2.shape[2] == 0:
                    d2 = np.zeros((d1.shape[0],) + d2.shape[1:])
                out = na.fold_skl(
                    p1,
                    p2,
                    n_folds,
                    model=model,
                    params=params,
                    mean=mean,
                    pre_pca=pre_pca,
                    shuffle=shuffle,
                    impute_missing=(repl_nan or impute_missing),
                    gen_c1=d1,
                    gen_c2=d2,
                    collapse_time=collapse_time,
                    time_mask=time_mask,
                    ret_projections=ret_projections,
                    **kwargs
                )
            if not combine and (decode_m1 is not None or decode_m2 is not None):
                outs[i] = out["score"]
                outs_gen[i] = out["score_gen"]
            else:
                outs[i] = out["score"]
        if ret_pops:
            out = (outs, xs, pop1, pop2)
            add = tuple(di for di in (dec1, dec2) if di[0] is not None and not combine)
            out = out + add
        else:
            out = (outs, xs)
        if decode_m1 is not None or decode_m2 is not None:
            out = out + (outs_gen,)
        return out

    def estimate_dimensionality(
        self, mask, winsize, begin, end, stepsize, n_resamples=20
    ):
        data_masked = self.mask(mask)
        pops, xs = data_masked.get_populations(winsize, begin, end, stepsize)
        outs = np.zeros((len(pops), n_resamples, len(xs)))
        for i, p in enumerate(pops):
            outs[i] = na.estimate_participation_ratio(p, n_resamples=n_resamples)
        return outs, xs
