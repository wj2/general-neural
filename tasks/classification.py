
import numpy as np
import scipy.stats as sts
import itertools as it
import functools as ft

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
    def __call__(self, x):
        stim = x[:, self.t_inds]
        return parity(stim)

class IdentityTask(Task):
    def __call__(self, x):
        stim = x[:, self.t_inds]
        return stim

def generate_many_colorings(n_colorings, n_g):
    rng = np.random.default_rng()
    inds = rng.choice(2**n_g, n_colorings, replace=False)
    out = np.array(list(it.product((0, 1), repeat=n_g)))[inds]
    return out


def apply_coloring(x, coloring=None):
    return np.all(x == coloring, axis=1)


def apply_many_colorings(x, colorings=None, merger=np.sum):
    out = np.zeros((len(x), len(colorings)))
    for i, coloring in enumerate(colorings):
        out[:, i] = apply_coloring(x, coloring)
    return merger(out, axis=1, keepdims=True) > 0


class ColoringTask(Task):
    def __init__(self, t_inds, coloring=None, n_coloring=None, merger=np.sum):
        super().__init__(t_inds)
        if n_coloring is None:
            n_coloring = 2 ** (len(t_inds) - 1)
        self.merger = merger
        if coloring is None:
            coloring = generate_many_colorings(n_coloring, len(self.t_inds))
        self.coloring = coloring
        self.c_func = ft.partial(
            ft.partial(apply_many_colorings, colorings=self.coloring, merger=self.merger)
        )

    def __call__(self, x):
        stim = x[:, self.t_inds]
        return self.c_func(stim)


class LinearTask(Task):
    def __init__(
            self, t_inds, task_vec=None, offset=None, axis_aligned=False, offset_var=0,
    ):
        super().__init__(t_inds)
        if task_vec is None:
            task_vec = u.make_unit_vector(
                sts.norm(0, 1).rvs(1, len(self.t_inds))
            )
        self.task_vec = task_vec
        if offset is None:
            offset = sts.norm(0, np.sqrt(offset_var)).rvs(1, 1)
        self.offset = offset

    def __call__(self, x):
        stim = x[:, self.t_inds]
        proj = np.sum(self.task_vec * stim, axis=1, keepdims=True)
        return (proj + self.offset) > 0





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
                    (c1_func(samps[:, g1]), c2_func(samps[:, g2]),), axis=1,
                )                  
            )
            return targ
        
        funcs.append(
            group_func
        )
    return funcs



