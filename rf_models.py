
import functools as ft
import numpy as np
import scipy.stats as sts
import matplotlib.pyplot as plt
from matplotlib import patches
import itertools as it


def get_population_cents_helper(ind, rf_cents, spaces, rfsizes, offsets, cents):
    for k in range(int(spaces[ind])):
        cents[ind] = rfsizes[ind]*k + offsets[ind]
        if ind < (len(spaces) - 1):
            rf_cents = get_population_cents_helper(ind + 1, rf_cents, spaces, 
                                                   rfsizes, offsets, cents)
        else:
            rf_cents.append(tuple(cents))
    return rf_cents

def get_population_cents_recursive(rfsize, res, spacesize):
    rfsize = np.array(rfsize)
    res = np.array(res)
    spacesize = np.array(spacesize)
    lats = rfsize/res
    lat = np.ceil(np.max(lats))
    orig_spaces = np.ceil(spacesize/rfsize)
    rf_cents = []
    offsets = rfsize/2
    spaces = orig_spaces
    for i in range(int(lat)):
        cents = np.zeros((len(orig_spaces)))
        rf_cents = get_population_cents_helper(0, rf_cents, spaces, rfsize, 
                                               offsets, cents)
        offsets = offsets - res
        spaces = orig_spaces + (offsets + rfsize/2)//rfsize + 1
    return rf_cents, lat

def get_population_cents(rfsize, res, spacesize, lattice='square'):
    rfsize = np.array(rfsize)
    res = np.array(res)
    spacesize = np.array(spacesize)
    lats = rfsize/res
    lat = int(np.ceil(np.max(lats)))    
    grid_points = []
    for i, ss in enumerate(spacesize):
        steps = np.ceil(ss/rfsize[i])*rfsize[i]
        gps = np.arange(-rfsize[i], steps, rfsize[i]) + res[i] + rfsize[i]/2
        grid_points.append(gps)
    n_pts = lat*np.product(np.ceil(spacesize/rfsize) + 1)
    for i in range(lat):
        pts = np.array(list(it.product(*grid_points)))
        offset = i*res
        pts = pts + offset
        if i == 0:
            all_pts = pts
        else:
            all_pts = np.concatenate((all_pts, pts), axis=0)
    assert all_pts.shape[0] == n_pts
    return all_pts, lat
        

def plot_square_rfs(cents, rfsize, spacesize, figsize=(10,10), jitter=.2, 
                    numjit=4, cm='Set3', cs=10, spacecol=(.1, .1, .1),
                    bigwid=2, save=False, savename='space_plot.svg'):
    f = plt.figure(figsize=figsize)
    ax = f.add_subplot(1, 1, 1)
    p_space = patches.Rectangle((0, 0),
                                spacesize[0], spacesize[1], fill=False,
                                linewidth=5, edgecolor=spacecol,
                                label='boundary of visual space')
    ax.add_patch(p_space)
    m = plt.get_cmap(cm)
    m.N = cs
    jitters = np.linspace(0, jitter, numjit)
    for i, c in enumerate(cents):
        jitt = jitters[np.mod(i, numjit)]
        color = np.mod(i, cs)
        c_lowleft = (c[0] - rfsize[0]/2 + jitt, c[1] - rfsize[1]/2 + jitt)
        p = patches.Rectangle(c_lowleft, rfsize[0], rfsize[1], fill=False,
                              edgecolor=m(color), linewidth=bigwid)
        ax.add_patch(p)
    # c = cents[-1]
    # color = np.mod(i+1, cs)
    # c_lowleft = (c[0] - rfsize[0]/2, c[1] - rfsize[1]/2)
    # p = patches.Rectangle(c_lowleft, rfsize[0], rfsize[1], fill=False,
    #                       edgecolor=m(color), linewidth=bigwid)
    # ax.add_patch(p)
    # ax.set_xlim([0, spacesize[0]])
    # ax.set_ylim([0, spacesize[1]])
        
    cs_arr = np.array(cents)
    x_min, x_max = (np.min(cs_arr[:, 0]) - rfsize[0]/2., 
                    np.max(cs_arr[:, 0] + rfsize[0]/2.))
    y_min, y_max = (np.min(cs_arr[:, 1]) - rfsize[1]/2.,
                    np.max(cs_arr[:, 1]) + rfsize[1]/2.)
    ax.set_xlim([x_min, x_max])
    ax.set_ylim([y_min, y_max])
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel('visual space')
    ax.set_ylabel('visual space')
    ax.legend(loc=(0, 1), frameon=False)
    f.tight_layout()
    if save:
        f.savefig(savename, bbox_inches='tight', dpi=150, transparent=True)
    plt.show()

def make_flat_square_rf(cent, sizes, scale, baseline, sub_dim=None):
    def flat_rf(coords):
        if len(coords.shape) == 1:
            coords = np.reshape(coords, (1, -1))
        if sub_dim is not None:
            coords = coords[:, sub_dim]
        ins = np.ones(coords.shape[0])
        for i in range(coords.shape[1]):
            in_dim = np.logical_and(coords[:, i] < cent[i] + sizes[i]/2,
                                    coords[:, i] >= cent[i] - sizes[i]/2)
            ins = ins*in_dim
        out_rf = np.logical_not(ins)
        condlist = [ins, out_rf]
        funclist = [lambda x: scale, lambda x: baseline]
        out = np.piecewise(np.ones(len(coords)), condlist, funclist)
        return out
    return flat_rf

def eval_gaussian_rf(coords, cent, sizes, scale, baseline, sub_dim=None):
    if len(coords.shape) == 1:
        coords = np.reshape(coords, (1, -1))
    if sub_dim is not None:
        coords = coords[:, sub_dim]
    ins = np.ones(coords.shape[0])
    r = np.exp(-np.sum(((coords - cent)**2)/(2*sizes), axis=1))
    r = ((scale - baseline)*r + baseline)
    return r

def eval_vector_rf(coords, rf_func, *rf_params, sub_dim=None):
    coords = np.array(coords)
    if len(coords.shape) == 1:
        coords = np.reshape(coords, (1, -1))
    if sub_dim is not None:
        coords = coords[:, sub_dim]
    coords = np.expand_dims(coords, axis=1)
    r = rf_func(coords, *rf_params)
    return r

def gaussian_func(coords, cents, sizes, scale, baseline):
    r = np.exp(-np.sum(((coords - cents)**2)/(2*sizes), axis=2))
    r = (scale - baseline)*r + baseline
    return r

def gaussian_deriv_func(coords, cents, sizes, scale, baseline):
    inner = -np.sum(((coords - cents)**2)/(2*sizes), axis=2)
    inner = np.expand_dims(inner, axis=2)
    out = -(scale - baseline)*np.exp(inner)*(coords - cents)/sizes
    return out

def ramp_func(coords, extent, scale, baseline):
    slope = (scale - baseline)/extent
    r = slope*coords + baseline
    return r

def ramp_deriv_func(coords, extent, scale, baseline):
    slope = (scale - baseline)/extent
    slope = np.expand_dims(slope, 0)
    return slope

def eval_ramp_vector_rf(coords, extent, scale, baseline, sub_dim=None):
    r = eval_vector_rf(coords, ramp_func, extent, scale, baseline,
                       sub_dim=sub_dim)
    return r
    
def eval_gaussian_vector_rf(coords, cents, sizes, scale, baseline,
                            sub_dim=None):
    r = eval_vector_rf(coords, gaussian_func, cents, sizes, scale, baseline,
                       sub_dim=sub_dim)
    return r

def eval_gaussian_vector_deriv(coords, cents, sizes, scale, baseline,
                               sub_dim=None):
    r = eval_vector_rf(coords, gaussian_deriv_func, cents, sizes, scale, baseline,
                       sub_dim=sub_dim)
    return r

def eval_ramp_vector_deriv(coords, extent, scale, baseline, sub_dim=None):
    r = eval_vector_rf(coords, ramp_deriv_func, extent, scale, baseline,
                       sub_dim=sub_dim)
    return r

def get_output_func_distribution_shapes(n_units_pd, input_distributions):
    means = np.zeros((len(input_distributions), n_units_pd))
    wids = np.zeros_like(means)
    percent_size = 1/n_units_pd
    for j in range(len(input_distributions)):
        for i in range(n_units_pd):
            perc = (i + .5)*percent_size
            m = input_distributions[j].ppf(perc)
            means[j, i] = m
        dm = np.diff(means[j])
        dm_p1 = np.zeros(n_units_pd)
        half = int(np.floor(n_units_pd/2))
        dm_p1[:half] = dm[:half]
        dm_p1[half:] = dm[-half:]
        wids[j] = dm_p1
    means_all = np.array(list(it.product(*means)))
    wids_all = np.array(list(it.product(*wids)))
    return means_all, wids_all

def get_distribution_gaussian_resp_func(n_units_pd, input_distributions, scale=1,
                                        baseline=0):
    ms, ws = get_output_func_distribution_shapes(n_units_pd, input_distributions)
    resp_func, d_resp_func = make_gaussian_vector_rf(ms, ws, scale,
                                                     baseline)
    return resp_func, d_resp_func, ms, ws

def make_gaussian_vector_rf(cents, sizes, scale, baseline, sub_dim=None):
    cents = np.array(cents)
    sizes = np.array(sizes)
    if len(cents.shape) <= 1:
        cents = np.reshape(cents, (-1, 1))
    if len(sizes.shape) <= 1:
        sizes = np.reshape(sizes, (-1, 1))
    cents = np.expand_dims(cents, axis=0)
    sizes = np.expand_dims(sizes, axis=0)
    rfs = ft.partial(eval_gaussian_vector_rf, cents=cents, sizes=sizes,
                     scale=scale, baseline=baseline)
    drfs = ft.partial(eval_gaussian_vector_deriv, cents=cents, sizes=sizes,
                      scale=scale, baseline=baseline)
    return rfs, drfs

def make_ramp_vector_rf(num, extent, scale, baseline, sub_dim=None):
    extents = np.ones((num, 1))*extent
    rfs = ft.partial(eval_ramp_vector_rf, extent=extents, scale=scale,
                     baseline=baseline)
    drfs = ft.partial(eval_ramp_vector_deriv, extent=extents, scale=scale,
                      baseline=baseline)
    return rfs, drfs 

def make_gaussian_rf(cent, sizes, scale, baseline, sub_dim=None):
    sizes = np.reshape(sizes, (1, -1))
    gaussian_rf = ft.partial(eval_gaussian_rf, cent=cent, sizes=sizes,
                             scale=scale, baseline=baseline,
                             sub_dim=sub_dim)
    return gaussian_rf        

def make_2dgaussian_rf(cent, xsize, ysize, scale, baseline):
    def gaussian2d_rf(coords):
        coords = np.array(coords).reshape(-1, 2)
        xs, ys = coords[:, 0], coords[:, 1]
        out = scale*np.exp(-(((xs - cent[0])**2)/(2*xsize**2) 
                             + ((ys - cent[1])**2)/(2*ysize**2))) + baseline
        return out
    return gaussian2d_rf

def fim(pts, deriv_func, noise_var=None, cov=None):
    pts = np.array(pts)
    if len(pts.shape) < 2:
        pts = np.expand_dims(pts, 0)
    d_resps = list(deriv_func(pt) for pt in pts)
    assert not (noise_var is None and cov is None)
    if cov is None and noise_var is not None:
        cov = np.identity(d_resps[0].shape[1])*noise_var
    inv_cov = np.linalg.inv(cov)
    mat_dim = np.product(pts.shape)
    fim_mat = np.zeros((mat_dim, mat_dim))
    mat_combs = it.combinations_with_replacement(range(mat_dim), 2)
    for mc in mat_combs:
        i, j = mc
        val = np.dot(d_resps[i][0, :, 0],
                     np.dot(inv_cov, d_resps[j][0, :, 0].T))
        fim_mat[i, j] = val
        fim_mat[j, i] = val
    return fim_mat
        
class NullNoise(object):
    
    def __init__(self):
        pass

    def pdf(self, exp, val):
        p = exp == val
        return p
        
    def sample(self, val):
        return val

class PoissonNoise(object):
    
    def __init__(self):
        pass

    def sample(self, val):
        s = sts.poisson.rvs(val)
        return s

    def pdf(self, exp, val):
        p = sts.poisson.pmf(val, loc=exp)
        return p

class GaussianNoise(object):
    
    def __init__(self, std):
        self.std = std

    def sample(self, val):
        s_pre = sts.norm.rvs(loc=val, scale=self.std)
        s_stack = np.vstack((s_pre, np.zeros_like(s_pre)))
        s = np.max(s_stack, 0)
        return s

    def pdf(self, exp, val):
        p = np.product(sts.norm.pdf(val, loc=exp, scale=self.std))
        return p

    def pdf_nocorr(self, exp, val):
        return self.pdf(exp, val)

class NDGaussianNoise(object):

    def __init__(self, cov):
        self.cov = cov

    def sample(self, vals):
        s_pre = sts.multivariate_normal.rvs(mean=vals, cov=self.cov)
        s_stack = np.vstack((s_pre, np.zeros_like(s_pre)))
        s = np.max(s_stack, 0)
        return s

    def pdf(self, exp, val):
        p = sts.multivariate_normal.pdf(val, mean=exp, cov=self.cov)
        return p

    def pdf_nocorr(self, exp, val):
        cov_eye = self.cov*np.identity(self.cov.shape[0])
        p = sts.multivariate_normal.pdf(val, mean=exp, cov=cov_eye)
        return p

class RFNeuron(object):
    
    def __init__(self, rf, noise_model=None):
        self.rf = rf
        if noise_model is None:
            noise_model = NullNoise()
        self.noise_model = noise_model

    def sample_stim(self, loc):
        out = self.noise_model.sample(self.rf(loc))
        return out
        
    def pdf(self, resp, loc):
        exp_val = self.rf(loc)
        p = self.noise_model.pdf(exp_val, resp)
        return p

def construct_rf_pop(sig_rfs, sizes, reses=None, resp=1, base=0,
                     sub_dim=None, rf_func=make_flat_square_rf,
                     rf_sizes=None, return_objects=True):
    """
    Construct a population of receptive fields tiling a space of dimensionality
    <= D using a rectangular lattice.

    Parameters
    ----------
    sig_rfs : list, length D
       the size of the receptive fields, changes to this will 
       impact how the space is tiled
    sizes : list, length D
       the size of each dimension to be tiled
    reses : list, length D
       the resolution (fields per unit of size) to tile the space at
    resp : float
       the maximum response of each field, dynamic range is from base to resp
    base : float
       the minimum response of each field, dynamic range is from base to resp
    sub_dim : optional, list, length <= D (default None)
       the subset of the dimensions that the fields respond to
    rf_func : optional, function (default is make_flat_square_rf)
       a receptive field function taking a center point, list of dimension 
       sizes, resp, base, and sub_dim
    rf_sizes : optional
       changes rf size without changing tiling behavior
    return_objects : optional, bool (default True)
       if False, will return a tuple of the rf_func and its argument instead of
       the rf_func objects with frozen parameters
       
    Returns
    -------
    rfs : list of functions
       list of all the created receptive fields
    """
    if reses is None:
        reses = (1,)*len(sizes)
    if rf_sizes is None:
        rf_sizes = sig_rfs
    cents, l = get_population_cents(sig_rfs, reses, sizes)
    cents = np.array(cents)
    rf_sizes = np.array(rf_sizes)
    if return_objects:
        rfs = [rf_func(c, rf_sizes, resp, base, sub_dim=sub_dim)
               for c in cents]
    else:
        rfs = [(rf_func, (c, rf_sizes, resp, base, sub_dim))
               for c in cents]
    return rfs

def get_codewords(stims, rfs):
    words = np.array([rf(stims) for rf in rfs]).T
    return words
