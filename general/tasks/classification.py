import numpy as np
import scipy.stats as sts
import itertools as it
import functools as ft
import scipy.special as ss
import sklearn.preprocessing as skp
import sklearn.pipeline as sklpipe
import sklearn.decomposition as skd
import sklearn.metrics.pairwise as skmp

import general.utility as u


def parity(x):
    out = np.mod(np.sum(x, axis=1, keepdims=True), 2, dtype=float)
    return out


class Task:
    """
    A task needs to define a call method that takes in stimuli with shape N x D
    and returns an array with shape N x T
    """

    def __init__(self, t_inds):
        if not u.check_list(t_inds):
            t_inds = np.arange(t_inds)
        self.t_inds = t_inds

    @classmethod
    def make_task_group(cls, n_tasks, *args, **kwargs):
        tasks = []
        for i in range(n_tasks):
            tasks.append(cls(*args, **kwargs))
        return TaskGroup(*tasks)

    def __len__(self):
        return 1


class TaskGroup:
    def __init__(self, *tasks):
        self.tasks = tasks

    def __call__(self, x):
        return np.concatenate(list(t(x) for t in self.tasks), axis=1)

    def __len__(self):
        return len(self.tasks)


class CompositeTask:
    def __init__(self, merger, *tasks):
        self.tasks = tasks
        self.task_group = TaskGroup(*tasks)
        self.merger = merger

    def __call__(self, x):
        return self.merger(self.task_group(x))

    def __len__(self):
        return 1


class ParityTask(Task):
    def __init__(self, t_inds, sign=None):
        super().__init__(t_inds)
        if sign is None:
            sign = np.sign(sts.norm(0, 1).rvs(1))
        self.sign = sign
        
    def __call__(self, x):
        stim = x[:, self.t_inds]
        return self.sign * parity(stim)


class IdentityTask(Task):
    def __call__(self, x):
        stim = x[:, self.t_inds]
        return stim


class ColoringTask(Task):
    def __init__(
        self, t_inds, coloring=None, n_coloring=None, n_values=2, merger=np.sum
    ):
        super().__init__(t_inds)
        if n_coloring is None:
            n_coloring = n_values ** (len(self.t_inds) - 1)
        self.merger = merger
        if coloring is None:
            coloring = generate_many_colorings(n_coloring, len(self.t_inds))
        self.coloring = coloring
        self.c_func = ft.partial(
            ft.partial(
                apply_many_colorings, colorings=self.coloring, merger=self.merger
            )
        )

    def __call__(self, x):
        stim = x[:, self.t_inds]
        return self.c_func(stim)


def make_discrete_order_transform(k, n, order, use_pca=True):
    categories = [np.arange(n, dtype=int)] * k
    if n**k > 10000:
        print("there are {} unique stimuli... this will take awhile".format(n**k))
    binary = list(it.product(range(n), repeat=k))
    ohe = skp.OneHotEncoder(categories=categories, sparse_output=False)
    pf = skp.PolynomialFeatures(
        degree=(order, order), include_bias=False, interaction_only=True
    )
    steps = [ohe, pf]
    if use_pca:
        pca = skd.PCA()
        steps.append(pca)

    pipe = sklpipe.make_pipeline(*steps)
    pipe = pipe.fit(binary)
    mask = np.var(pipe.transform(binary), axis=0) > 1e-10

    def trs(x):
        return pipe.transform(x)[:, mask]

    return trs, np.sum(mask)


class DiscreteOrderTask(Task):
    def __init__(
        self,
        t_inds,
        order,
        n_vals=2,
        task_vec=None,
        offset=None,
        axis_aligned=False,
        offset_var=0,
        scale=2,
    ):
        super().__init__(t_inds)
        self.trs, self.vec_dim = make_discrete_order_transform(
            len(self.t_inds), n_vals, order
        )
        if task_vec is None:
            vec = sts.norm(0, 1).rvs((1, self.vec_dim))
            task_vec = u.make_unit_vector(vec)
        self.task_vec = task_vec
        if offset is None:
            offset = sts.norm(0, np.sqrt(offset_var)).rvs(1, 1)
        self.offset = offset

    def __call__(self, x):
        stim = x[:, self.t_inds]
        rep = self.trs(stim)
        proj = np.sum(self.task_vec * rep, axis=1, keepdims=True)
        return (proj + self.offset) > 0


def make_strict_discrete_order_transform(k, n, order, retries=10, **kwargs):
    for i in range(retries):
        binary, labels = _label_stim(k, n, order, **kwargs)
        if np.all(np.abs(labels) > 0):
            break

    def trs(x):
        ds = skmp.euclidean_distances(x, binary)
        inds = np.argmin(ds, axis=1)
        return labels[inds]

    return trs, None


def _label_stim(k, n, order, balanced=True, exclusion=None):
    if n**k > 10000:
        print("there are {} unique stimuli... this will take awhile".format(n**k))
    if exclusion is None:
        exclusion = ()
    binary = np.array(list(it.product(range(n), repeat=k)))
    labels = np.zeros(len(binary))
    combs = np.array(list(it.combinations(range(k), order)))
    sub_stim = np.array(list(it.product(range(n), repeat=order)))
    sub_stim_all = np.tile(sub_stim, (int(ss.comb(k, order)), 1))
    combs_all = np.repeat(combs, n**order, axis=0)

    exclusion_mask = np.ones(len(combs_all), dtype=bool)
    for fs, vs in exclusion:
        fs_match = np.all(combs_all == np.array(fs)[None], axis=1)
        vs_match = np.all(sub_stim_all == np.array(vs)[None], axis=1)
        exclude = np.logical_and(fs_match, vs_match)
        exclusion_mask = np.logical_xor(exclusion_mask, exclude)

    sub_stim_all = sub_stim_all[exclusion_mask]
    combs_all = combs_all[exclusion_mask]

    rng = np.random.default_rng()
    shuff_inds = rng.permutation(len(combs_all))
    if balanced:
        signs = np.tile((-1, 1), int(np.ceil(len(shuff_inds) / 2)))
    else:
        signs = rng.choice((-1, 1), size=len(shuff_inds))
    j = 0
    for i, si in enumerate(shuff_inds):
        feats_i = np.array(combs_all[si])
        ss_i = np.array(sub_stim_all[si])
        mask = np.all(binary[:, feats_i] == ss_i[None], axis=1)
        if np.all(labels[mask] == 0):
            sign = signs[j]
            j += 1
            labels[mask] = sign
    return binary, labels[:, None]


class DiscreteOrderTaskStrict(Task):
    def __init__(self, t_inds, order, n_vals=2, **kwargs):
        super().__init__(t_inds)
        self.trs, self.vec_dim = make_strict_discrete_order_transform(
            len(self.t_inds),
            n_vals,
            order,
            **kwargs,
        )

    def __call__(self, x):
        stim = x[:, self.t_inds]
        return self.trs(stim) > 0


class LinearTask(Task):
    def __init__(
        self,
        t_inds,
        task_vec=None,
        offset=None,
        axis_aligned=False,
        offset_var=0,
        center=0.5,
        scale=2,
        centered_targets=False,
    ):
        super().__init__(t_inds)
        if task_vec is None:
            vec = sts.norm(0, 1).rvs((1, len(self.t_inds)))
            task_vec = u.make_unit_vector(vec)
        self.task_vec = task_vec
        if offset is None:
            offset = sts.norm(0, np.sqrt(offset_var)).rvs(1, 1)
        self.offset = offset
        self.center = center
        self.scale = scale
        self.centered_targets = centered_targets

    def __call__(self, x):
        stim = self.scale * (x[:, self.t_inds] - self.center)
        proj = np.sum(self.task_vec * stim, axis=1, keepdims=True)
        t = (proj + self.offset) > 0
        if self.centered_targets:
            t = 2 * (t - .5)
        return t
    


class ContextualTask(Task):
    def __init__(self, *tasks, c_inds=None, single_ind=False):
        if single_ind:
            assert len(tasks) == 2
        self.single_ind = single_ind
        if c_inds is None:
            c_inds = np.arange(-len(tasks) + single_ind, 0)
        self.c_inds = c_inds
        self.tasks = tasks

    def __len__(self):
        return len(self.tasks[0])

    def __call__(self, x):
        if self.single_ind:
            task_inds = x[:, self.c_inds]
        else:
            task_inds = np.argmax(x[:, self.c_inds], axis=1)
        task_inds = np.squeeze(task_inds)
        outs_all = np.stack(list(t(x) for t in self.tasks), axis=0)

        targets = outs_all[task_inds, range(len(x))]
        return targets


def make_contextual_task(
    t_inds,
    n_tasks=1,
    n_contexts=2,
    single_ind=False,
    c_inds=None,
    **kwargs,
):
    tasks = []
    for i in range(n_contexts):
        task_i = LinearTask.make_task_group(n_tasks, t_inds, **kwargs)
        tasks.append(task_i)
    contask = ContextualTask(*tasks, single_ind=single_ind)
    return contask


def generate_many_colorings(n_colorings, n_g, n_values=2):
    rng = np.random.default_rng()
    inds = rng.choice(n_values**n_g, n_colorings, replace=False)
    out = np.array(list(it.product(tuple(range(n_values)), repeat=n_g)))[inds]
    return out


def apply_coloring(x, coloring=None):
    return np.all(x == coloring, axis=1)


def apply_many_colorings(x, colorings=None, merger=np.sum):
    out = np.zeros((len(x), len(colorings)))
    for i, coloring in enumerate(colorings):
        out[:, i] = apply_coloring(x, coloring)
    return merger(out, axis=1, keepdims=True) > 0


def group_xor_task(n_g):
    if not u.check_list(n_g):
        n_g = np.arange(n_g)

    def task_func(samps):
        rel_dims = samps[:, n_g]
        targ = parity(rel_dims)
        return targ

    return (task_func,)


def group_coloring_tasks(g1, g2, n_tasks=10, combine=parity, merger=np.sum):
    if not u.check_list(g1):
        g1_use = np.arange(g1)
        g2 = np.arange(g1, g1 + g2)
        g1 = g1_use
    funcs = []
    for i in range(n_tasks):
        c1 = generate_many_colorings(2 ** (len(g1) - 1), len(g1))
        c1_func = ft.partial(apply_many_colorings, colorings=c1, merger=merger)

        c2 = generate_many_colorings(2 ** (len(g2) - 1), len(g2))
        c2_func = ft.partial(apply_many_colorings, colorings=c2, merger=merger)

        def group_func(samps):
            targ = combine(
                np.concatenate(
                    (
                        c1_func(samps[:, g1]),
                        c2_func(samps[:, g2]),
                    ),
                    axis=1,
                )
            )
            return targ

        funcs.append(group_func)
    return funcs
