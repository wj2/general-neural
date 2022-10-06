
import functools as ft
import numpy as np
import scipy.stats as sts
import scipy.special as ss
import matplotlib.pyplot as plt
from matplotlib import patches
import itertools as it
import scipy.optimize as sopt
import scipy.integrate as sint
import scipy.signal as sig
import joblib as jl

import general.utility as u

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

def get_random_uniform_fill(n_units, input_distributions, volume_factor=3,
                            wid=None):
    """ not proper for elongated input spaces """
    if wid is None:
        vol = np.product(list(np.diff(distr.interval(1))
                              for distr in input_distributions))
        wid = volume_factor*vol/n_units
    means_all = np.zeros((n_units, len(input_distributions)))
    wids_all = np.ones((n_units, len(input_distributions)))*(wid**2)
    for i, distr in enumerate(input_distributions):
        means_all[:, i] = distr.rvs(n_units)
    return means_all, wids_all

@u.arg_list_decorator
def get_fi_thresh(n_units, pwrs, dims, wid, lam=2, approx_w=False):
    out_fi = np.zeros((len(n_units), len(pwrs), len(dims), len(wid)))
    out_nl = np.zeros_like(out_fi)
    out_w = np.zeros_like(out_fi)
    out_var = np.zeros_like(out_fi)
    for (i, j, k, l) in u.make_array_ind_iterator(out_fi.shape):
        nu_, pwr, dim, w = n_units[i], pwrs[j], dims[k], wid[l]
        if w is None:
            if approx_w:
                w = random_uniform_pwr_trs(nu_, lam, dim)
            else:
                out = min_mse_power(pwr, nu_, dim, max_w=.5, n_ws=200,
                                    lambda_deviation=lam)
                w = out[-2]
        out_w[i, j, k, l] = w 
        out_fi[i, j, k, l] = random_uniform_fi_pwr(nu_, pwr, w, dim)
        out = get_thresh_prob(pwr, nu_, dim, w)
        out_nl[i, j, k, l] = out[0]
        out_var[i, j, k, l] = random_uniform_pwr_var_fix(nu_, pwr, w, dim)
    return out_fi, out_nl, out_w, out_var

@u.arg_list_decorator
def get_pwr_fi_by_param(n_units, wid, dims, scale=1):
    out_pwr = np.zeros((len(n_units), len(wid), len(dims), len(scale)))
    out_fi = np.zeros_like(out_pwr)
    for (i, j, k, l) in u.make_array_ind_iterator(out_pwr.shape):
        nu, w, d, s = n_units[i], wid[j], dims[k], scale[l]
        out_pwr[i, j, k, l] = random_uniform_pwr(nu, w, d, scale=s)
        fim = random_uniform_fi(nu, w, d, scale=s)
        out_fi[i, j, k, l] = fim[0, 0]
    return out_pwr, out_fi

def get_lattice_uniform_pop(total_pwr, n_units, dims, w_use=None,
                            scale_use=None, sigma_n=1, ret_params=False,
                            **kwargs):
    distrs = (sts.uniform(0, 1),)*dims
    stim_distr = u.MultivariateUniform(dims, (0, 1))

    n_units_pd = int(np.round(n_units**(1/dims)))
    ms, ws = get_output_func_distribution_shapes(n_units_pd, distrs,
                                                 wid_scaling=1)

    if w_use is not None:
        ws = np.ones_like(ws)*w_use
    if scale_use is not None:
        scale = scale_use
        titrate_pwr = None
    else:
        scale = total_pwr
        titrate_pwr = stim_distr
    
    rf, drf = make_gaussian_vector_rf(ms, ws**2, scale, 0,
                                      titrate_pwr=titrate_pwr,
                                      **kwargs)
    noise_distr = sts.multivariate_normal(np.zeros(n_units_pd**dims),
                                          sigma_n, allow_singular=True)
    
    out = (stim_distr, rf, drf, noise_distr)
    if ret_params:
        out = out + (ms, ws)
    return out

def get_random_uniform_pop(total_pwr, n_units, dims, w_use=None,
                           scale_use=None, sigma_n=1, ret_params=False,
                           **kwargs):
    dims = int(dims)
    distrs = (sts.uniform(0, 1),)*dims
    stim_distr = u.MultivariateUniform(dims, (0, 1))

    ms, ws = get_random_uniform_fill(n_units, distrs, wid=w_use)
    if scale_use is not None:
        scale = scale_use
        titrate_pwr = None 
    else:
        scale = total_pwr
        titrate_pwr = stim_distr
    rf, drf = make_gaussian_vector_rf(ms, ws, scale, 0,
                                      titrate_pwr=titrate_pwr,
                                      **kwargs)
    noise_distr = sts.multivariate_normal(np.zeros(n_units), sigma_n,
                                          allow_singular=True)
    
    out = (stim_distr, rf, drf, noise_distr)
    if ret_params:
        out = out + (ms, ws)
    return out

def visualize_random_rf_responses(resp, cents, vis_dims=(0, 1), cmap='Blues',
                                  ax=None, normalize_resp=True,
                                  plot_stim=None, ms=5, stim_color='r'):
    cmap = plt.get_cmap(cmap)
    vis_dims = np.array(vis_dims)
    if ax is None:
        f = plt.figure()
        if len(vis_dims) == 3:
            ax = f.add_subplot(1, 1, 1, projection='3d')
        else:
            ax = f.add_subplot(1, 1, 1, aspect='equal')
    if normalize_resp:
        r_ = resp - np.min(resp)
        resp = r_/np.max(r_)
    for i, r in enumerate(resp):
        pt = cents[i, vis_dims]
        if r > 0:
            ax.plot(*pt,
                    'o',
                    color=cmap(r),
                    alpha=r)
    if plot_stim is not None:
        for i, ps in enumerate(plot_stim):
            pt = ps[vis_dims]
            ax.plot(*pt,
                    'o',
                    color=stim_color,
                    markersize=ms)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    if len(vis_dims) == 3:
        ax.set_zlim([0, 1])
    return ax

def ml_decode_rf(reps, func, dim, init_guess=None):
    if init_guess is None:
        init_guess = np.zeros((reps.shape[0], dim))*.5
        
    def f(x):
        x_shaped = np.reshape(x, init_guess.shape)
        err = np.sum((func(x_shaped) - reps)**2)
        return err

    n_vars = np.product(init_guess.shape)
    bounds = ((0, 1),)*n_vars
    # out = sopt.minimize(f, init_guess, bounds=bounds)
    out = sopt.basinhopping(f, init_guess, niter=100,
                            minimizer_kwargs=dict(bounds=bounds))
    x_opt = np.reshape(out.x, init_guess.shape)
    return x_opt

def brute_decode_decouple(reps, func, dim, dim_i=(0,), n_gran=200,
                          init_guess=None, **kwargs):
    guesses = np.ones((n_gran**len(dim_i), dim))*np.nan
    all_guess = np.array(list(it.product(np.linspace(0, 1, n_gran),
                                         repeat=len(dim_i))))
    guesses[:, dim_i] = all_guess
    mg = _min_guess(func, reps, guesses, **kwargs)
    return mg

def _min_guess(func, reps, guesses, add_dc=0, **kwargs):
    g_reps = func(guesses, **kwargs) + add_dc
    g_reps = np.expand_dims(g_reps, 1)
    reps = np.expand_dims(reps, 0)
    g_ind = np.argmin(np.nansum((g_reps - reps)**2, axis=-1), axis=0)
    return guesses[g_ind]


def brute_decode_rf(reps, func, dim, n_gran=200, init_guess=None, **kwargs):
    guesses = np.array(list(it.product(np.linspace(0, 1, n_gran),
                                       repeat=dim)))
    return _min_guess(func, reps, guesses, **kwargs)

def refine_decode_rf(reps, func, dim, n_gran=10, **kwargs):
    guesses = np.array(list(it.product(np.linspace(0, 1, n_gran),
                                       repeat=dim)))
    guesses = _min_guess(func, reps, guesses, **kwargs)

    def _f_surrogate(x):
        x_inp = np.reshape(x, guesses.shape)
        out = func(x_inp)
        return np.sum((reps - out)**2)
    
    bounds = ((0, 1),)*np.product(guesses.shape)
    flat_guesses = np.reshape(guesses, np.product(guesses.shape))
    out = sopt.minimize(_f_surrogate, flat_guesses, bounds=bounds)
    x = np.reshape(out.x, guesses.shape)
    return x

@ft.lru_cache(maxsize=None)
def _thresh_integrate(mu_p, std_p, sigma_n=1):
    def integ_func(d):
        prob = sts.norm(mu_p, std_p).pdf(d)
        cumu = sts.norm(0, 1).cdf(-np.sqrt(d)/(2*sigma_n))
        return prob*cumu

    def integ_func_alt(d):
        prob = sts.norm(mu_p, std_p).pdf(d)
        sd = np.sqrt(d)/(2*sigma_n)
        cumu = .5 + (1/np.sqrt(np.pi))*(-sd + (sd**3)/3)
        return prob*cumu                                        

    return sint.quad(integ_func, 0, np.inf)

def _approx_thresh_integrate(mu_p, std_p, sigma_n=1, bins=500, lam=3,
                             ret_all=False):
    ds = np.linspace(0, mu_p + lam*std_p, bins)

    probs = sts.norm(mu_p, std_p).pdf(ds)
    cumus = sts.norm(0, 1).cdf(-np.sqrt(ds)/(2*sigma_n))
    e1 = probs*cumus*np.diff(ds)[0]
    if not ret_all:
        e1 = np.sum(e1)
    return e1

def rf_volume(wid, dim, wid_factor=2):
    num = ((wid_factor*wid)**dim)*np.pi**(dim/2)
    denom = ss.gamma(dim/2 + 1)
    return num/denom

def get_ws_range(total_pwr, n_units, dim_, n_ws=100, lambda_dev=2):
    ws = np.linspace(.001, .5, n_ws)
    
    fi_mm = np.zeros(n_ws)
    p_v1 = np.zeros_like(fi_mm)
    p_v2 = np.zeros_like(fi_mm)
    p_m2 = np.zeros_like(fi_mm)
    p_thr = np.zeros_like(fi_mm)
    eff_dim = np.zeros_like(fi_mm)
    p_single = np.zeros_like(fi_mm)
    mus_p = np.zeros_like(fi_mm)
    stds_p = np.zeros_like(fi_mm)
    thr_mag = np.zeros_like(fi_mm)

    for i, w in enumerate(ws):
        pwr = random_uniform_pwr(n_units, w, dim_, scale=1)
        rescale = np.sqrt(total_pwr/pwr)
        fi = random_uniform_fi(n_units, w, dim_, scale=rescale)
        out = random_uniform_fi_var(n_units, w, dim_, scale=rescale,
                                        ret_pieces=True)
        fi_v, p_v1[i], p_v2[i], p_m2[i] = out
        fi_mm[i] = fi[0, 0] - lambda_dev*np.sqrt(fi_v[0, 0])
        out = compute_threshold_err_prob(total_pwr, n_units, dim_, w,
                                         resp_scale=rescale,
                                         ret_components=True)
        eff_dim[i], p_single[i], mus_p[i], stds_p[i] = out[2:]
        p_thr[i] = out[0]
        thr_mag = out[1]
    fi_mm[fi_mm < 0] = np.nan
    total_mse = p_thr*thr_mag + (1 - p_thr)/fi_mm
    return ws, total_mse, fi_mm, p_thr, thr_mag, p_single, mus_p, stds_p

def opt_w_approx(pwr, dims, sigma=1, w_factor=2):
    alpha = (1/6)*(sigma/np.sqrt(pwr*np.pi))*np.exp(-pwr/(4*sigma**2))
    beta = dims*ss.gamma(dims/2 + 1)*w_factor**(-dims - 1)/(np.pi**dims/2)
    comb = pwr*alpha*beta/(5*sigma)
    return comb**(1/(dims + 2))

def min_mse_vec(pwr, n_units, dims, wid=None, ret_components=False,
                n_ws=10000, **kwargs):
    if wid is None:
        wid = np.linspace(.001, .5, n_ws)
    out = mse_w_range(pwr, n_units, dims, wid=wid, ret_components=True,
                      **kwargs)
    mse, l_mse, nl_mse, nl_prob, ws = out
    min_ind = np.nanargmin(mse)

    out = mse[min_ind]
    if ret_components:
        out = (out, l_mse[min_ind], nl_mse[min_ind], nl_prob[min_ind],
               wid[min_ind])
    return out

def mse_w_range(pwr, n_units, dims, sigma_n=1, lam=2, w_factor=2, wid=None,
                n_ws=10000, ret_components=False):
    if wid is None:
        wid = np.linspace(.001, .5, n_ws)
    fi = random_uniform_fi_vec(pwr, n_units, wid, dims, sigma_n=sigma_n)
    p, em = compute_threshold_vec(pwr, n_units, dims, wid, sigma_n=sigma_n,
                                  lam=lam)
    out = (1 - p)/fi + p*em
    if ret_components:
        l_mse = 1/fi
        nl_mse = em
        nl_prob = p
        out = (out, l_mse, nl_mse, nl_prob, wid)
    return out

def compute_threshold_vec(pwr, n_units, dim, wid, sigma_n=1, lam=2,
                          stim_scale=1):
    scale = random_uniform_scale_vec(pwr, n_units, wid, dim)

    v_std = np.sqrt(random_uniform_pwr_var(n_units, wid, dim,
                                           scale=scale, vec=True))
    
    v_lam = pwr - (lam/np.sqrt(2))*v_std
    v_lam = np.max([v_lam, np.zeros(v_lam.shape)], axis=0)
    p_pre = (sigma_n/np.sqrt(v_lam*np.pi))*np.exp(-v_lam/(4*sigma_n**2))

    effective_dim = (stim_scale**dim)/rf_volume(wid, dim)

    f = np.min([np.ones(effective_dim.shape)*n_units, effective_dim], axis=0)
    factor = np.max([f, np.zeros(f.shape)], axis=0)
    
    approx_prob = np.min([p_pre*factor, np.ones(factor.shape)], axis=0) 
    err_mag = np.ones(approx_prob.shape)*(stim_scale**2)/6
    return approx_prob, err_mag   
    
def get_thresh_prob(pwr, n_units, dim, w, **kwargs):
    pwr_pre = random_uniform_pwr(n_units, w, dim, scale=1)
    rescale = np.sqrt(pwr/pwr_pre)
    return compute_threshold_err_prob(pwr, n_units, dim, w, resp_scale=rescale,
                                      **kwargs)

def compute_threshold_err_prob(pwr, n_units, dim, w_opt, sigma_n=1, scale=1,
                               lam=2, resp_scale=1, ret_components=False,
                               use_approx=True, print_=False):
    sigma_n = np.sqrt(sigma_n)
    mu_p = np.sqrt(2*pwr)
    std_p = np.sqrt(2*random_uniform_pwr_var(n_units, w_opt, dim,
                                             scale=resp_scale))
    v_std = np.sqrt(random_uniform_pwr_var(n_units, w_opt, dim,
                                           scale=resp_scale))

    if use_approx:
        v_lam = max(mu_p**2 - lam*std_p/np.sqrt(2), 0)
        v_lam = pwr - (lam/np.sqrt(2))*v_std
        v = (sigma_n/np.sqrt(v_lam*np.pi))*np.exp(-v_lam/(4*sigma_n**2))
        # print(pwr, w_opt, mu_p, std_p)
        # print('a', v)
        v_obs = _approx_thresh_integrate(mu_p, std_p, sigma_n=sigma_n)
        # print('b', v)
        err = 0
    else:
        v, err = _thresh_integrate(mu_p, std_p, sigma_n=sigma_n)
    lam = 2
    arg = np.sqrt(mu_p - lam*std_p)/(2*sigma_n)
    effective_dim = (scale**dim)/rf_volume(w_opt, dim)
    
    factor = max(min(n_units, effective_dim) - 1, 0)
    approx_prob = v*factor
    approx_prob = min(approx_prob, 1)
    err_mag = (scale**2)/6
    out = (approx_prob, err_mag)
    if ret_components:
        comp = (effective_dim, v, mu_p, std_p)
        out = out + comp
    return out

@u.arg_list_decorator
def emp_rf_decoding(total_pwr, n_units, dims, sigma_n=1, n_pops=10,
                    n_samps_per_pop=100, n_jobs=-1, give_guess=True,
                    pop_func='random', spk_cost='l2', **kwargs):
    if pop_func == 'random':
        pop_func = get_random_uniform_pop
    elif pop_func == 'lattice':
        pop_func = get_lattice_uniform_pop
    else:
        raise IOError('unrecognized pop_func indicator {}'.format(pop_func))
    if spk_cost == 'l1':
        cost_func = lambda x: np.mean(np.sum(np.abs(x), axis=1))
        titrate_func = lambda x, y: x/y
    elif spk_cost == 'l2':
        cost_func = lambda x: np.mean(np.sum(x**2, axis=1))
        titrate_func = lambda x, y: np.sqrt(x/y)
    else:
        raise IOError('unrecognized spk_cost indicator {}'.format(spk_cost))

    packs = np.zeros((len(total_pwr), len(n_units), len(dims), n_pops),
                     dtype=object)
    configs = {}
    for ind in u.make_array_ind_iterator(packs.shape):
        conf_ind = configs.get(ind[:-1])
        pi, ni, di = ind[:-1]
        if conf_ind is None:
            out = max_fi_power(total_pwr[pi], n_units[ni], int(dims[di]),
                               sigma_n=sigma_n, use_min_func=_min_mse_func)
            fi, _, _, w_opt, rescale_opt = out
            tp, tm = compute_threshold_err_prob(total_pwr[pi], n_units[ni],
                                                dims[di], w_opt,
                                                sigma_n=sigma_n,
                                                resp_scale=rescale_opt)
            conf_ind = fi, w_opt, rescale_opt, tp, tm
            configs[ind[:2]] = conf_ind
        else:
            fi, w_opt, rescale_opt = conf_ind
        stim_d, rf, _, noise = pop_func(total_pwr[pi], n_units[ni],
                                        int(dims[di]),
                                        w_use=w_opt, sigma_n=sigma_n,
                                        scale_use=rescale_opt,
                                        cost_func=cost_func,
                                        titrate_func=titrate_func)
        packs[ind] = (stim_d, rf, noise)
        
    def _emp_rf_pop(stim_i, rf_i, noise_i):
        random_state = np.random.randint(1, 2**32, 1)[0]
        samps_i = stim_i.rvs(n_samps_per_pop,
                             random_state=random_state)
        rep_i = rf_i(samps_i) + noise_i.rvs(n_samps_per_pop,
                                            random_state=random_state)
        if give_guess:
            init = samps_i
        else:
            init = None
        dec_i = brute_decode_rf(rep_i, rf_i, samps_i.shape[1],
                                init_guess=init)
        mse_i = (samps_i - dec_i)**2
        return mse_i
    
    par = jl.Parallel(n_jobs=n_jobs, backend='loky')
    out = par(jl.delayed(_emp_rf_pop)(*packs[ind])
              for ind in u.make_array_ind_iterator(packs.shape))
    out_arr = np.zeros_like(packs)
    for i, ind in enumerate(u.make_array_ind_iterator(out_arr.shape)):
        out_arr[ind] = out[i]
    return out_arr, configs

def random_uniform_pwr_trs_partial(n_units, wid, lam, dim):
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)
    inner_left_num = np.pi - wid*spi*(stwo + 2)/2
    inner_left_denom = spi/stwo - wid/2
    inner_left = inner_left_num/inner_left_denom
    left = 1 + (n_units - 1)*(wid**dim)*inner_left**dim
    inner_right = ((spi - wid)**2)/(spi/stwo - wid/2)
    right = ((1 + lam**2)/(lam**2))*n_units*(wid**dim)*inner_right**dim
    return left, right

def random_uniform_pwr_trs(n_units, lam, dim):
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)
    
    out_denom = stwo*spi*((1/lam**2)*n_units + 1)**(1/dim)
    out = 1/out_denom
    return out

@ft.lru_cache(maxsize=None)
def _integrate_pwr_var(wid):
    int_func = ft.partial(_non_deriv_terms, wid)
    return sint.quad(int_func, 0, 1)

def _taylor_pwr_var(wid):
    # return ((.5*np.sqrt(np.pi)*wid)**2)*(4 - wid*8/np.sqrt(np.pi))
    
    return ((.5*np.sqrt(np.pi)*wid)**2)*(4 - wid*(4 + np.pi)/np.sqrt(np.pi)
                                         + (np.pi/8)*wid**2)

def _approx_pwr_var(wid, do_integral=False, eps=1e-3):
    # return ((.5*np.sqrt(np.pi)*wid)**2)*(4 - wid*8/np.sqrt(np.pi))
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)
    a = (2/spi)*wid*np.exp(-1/wid**2)*ss.erf(1/wid)
    b = ss.erf(1/wid)**2
    c = -(stwo/spi)*wid*ss.erf(stwo/wid)
    if do_integral:
        _f = lambda x: ss.erf((1 - x)/wid)*ss.erf(x/wid)
        d, err = sint.quad(_f, 0, 1)
        assert err < eps
    else:
        d = ss.erf(1/(2*wid)) - (2/spi)*wid*(1 - np.exp(-1/(4*wid**2)))
    out = 2*(a + b + c + d)*(.5*spi*wid)**2
    
    return out

def random_uniform_pwr_var_approx(n_units, pwr, wid, dims):
    prefix = (pwr**2)/n_units
    inner = 1/((wid*np.sqrt(2*np.pi))**dims) - 1
    return prefix*inner

def random_uniform_pwr_var_fix(n_units, pwr, wid, dims):
    pwr_pre = random_uniform_pwr(n_units, wid, dims, scale=1)
    rescale = np.sqrt(pwr/pwr_pre)
    out = random_uniform_pwr_var(n_units, wid, dims, scale=rescale)
    return out

def random_uniform_pwr_var(n_units, wid, dims, scale=1, use_taylor=False,
                           use_approx=True, vec=False):
    pwr = random_uniform_pwr(n_units, wid, dims, scale=scale, vec=vec)
    a = scale**4
    b = (np.sqrt(np.pi/2)*wid*ss.erf(np.sqrt(2)/wid)
         - .5*(wid**2)*(1 - np.exp(-2/(wid**2))))
    if use_taylor:
        off_term = _taylor_pwr_var(wid)
        err = 0
    elif use_approx:
        off_term = _approx_pwr_var(wid)
        err = 0 
    else:
        off_term, err = _integrate_pwr_var(wid)
    if (not vec) and u.check_list(wid):
        c = np.product(b)
        off_term_prod = np.product(off_term)
    else:
        c = b**dims
        off_term_prod = off_term**dims
    pwr2 = a*(n_units*c + n_units*(n_units - 1)*off_term_prod)
    r_var = pwr2 - pwr**2
    return r_var

def _min_mse_func_nostd(w, n_units=None, dims=None, total_pwr=None,
                        lambda_deviation=None):
    w = w[0]
    pwr_pre = random_uniform_pwr(n_units, w, dims, scale=1)
    rescale = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w, dims, scale=rescale)
    fi_var = random_uniform_fi_var(n_units, w, dims, scale=rescale)
    fi_corr = fi[0, 0]

    pwr_end = random_uniform_pwr(n_units, w, dims, scale=rescale)
    prob, em = compute_threshold_err_prob(total_pwr, n_units, dims, w,
                                          resp_scale=rescale)
    threshold_mse = prob*em
    fi_mse = 1/fi_corr
    if fi_mse < 0:
        fi_mse = np.inf
    m_prob = min(prob, 1)
    loss = (1 - m_prob)*fi_mse + threshold_mse
    return loss    

def _min_mse_func(w, n_units=None, dims=None, total_pwr=None,
                  lambda_deviation=None, ret_pieces=False,
                  sigma_n=1):
    w = w[0]
    pwr_pre = random_uniform_pwr(n_units, w, dims, scale=1)
    rescale = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w, dims, scale=rescale,
                           sigma_n=sigma_n)
    fi_var = random_uniform_fi_var(n_units, w, dims, scale=rescale,
                                   sigma_n=sigma_n)
    fi_corr = fi[0, 0] - lambda_deviation*np.sqrt(fi_var[0, 0])
    
    out = compute_threshold_err_prob(total_pwr, n_units, dims, w,
                                     resp_scale=rescale,
                                     sigma_n=sigma_n,
                                     ret_components=True)
    prob, em, ed, v, mu_p, std_p = out
    fi_mse = 1/fi_corr
    if fi_mse < 0:
        fi_mse = np.inf
    loss = (1 - prob)*fi_mse + prob*em
    if ret_pieces:
        out = (loss, fi_mse, fi, fi_var, prob, em, ed, v, mu_p, std_p)
    else:
        out = loss
    return out 

def _min_mse_func_inverted(w, n_units=None, dims=None, total_pwr=None,
                           lambda_deviation=None):
    w = w[0]
    pwr_pre = random_uniform_pwr(n_units, w, dims, scale=1)
    rescale = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w, dims, scale=rescale)
    fi_var = random_uniform_fi_var(n_units, w, dims, scale=rescale)
    fi_corr = fi[0, 0] - lambda_deviation*np.sqrt(fi_var[0, 0])
    
    prob, em = compute_threshold_err_prob(total_pwr, n_units, dims, w,
                                          resp_scale=rescale)
    
    m_prob = min(prob, 1)
    loss = -(1 - m_prob)*fi_corr - m_prob*(1/em)
    return loss    

def _min_func(w, n_units=None, dims=None, total_pwr=None,
              lambda_deviation=None):
    w = w[0]
    pwr_pre = random_uniform_pwr(n_units, w, dims, scale=1)
    rescale = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w, dims, scale=rescale)
    fi_var = random_uniform_fi_var(n_units, w, dims, scale=rescale)
    pwr_end = random_uniform_pwr(n_units, w, dims, scale=rescale)
    loss = -fi[0, 0] + lambda_deviation*np.sqrt(fi_var[0, 0])
    return loss 

def min_mse_power(total_pwr, n_units, dims, sigma_n=1, eps=1e-4,
                  lambda_deviation=2, local_min_max=False, n_ws=1000,
                  max_w=1):
    min_func = ft.partial(_min_mse_func, n_units=n_units, dims=dims,
                          total_pwr=total_pwr, sigma_n=sigma_n,
                          lambda_deviation=lambda_deviation)

    ws = np.linspace(eps, max_w, n_ws)
    mses = list(min_func((w_i,)) for w_i in ws)
    if not np.all(np.isnan(mses)):
        w_opt = ws[np.nanargmin(mses)]
    else:
        w_opt = np.nan

    out = _min_mse_func((w_opt,), n_units, dims, total_pwr,
                        lambda_deviation*local_min_max,
                        ret_pieces=True)
    local_mse = out[1]
    nonlocal_mse = out[5]
    nonlocal_prob = out[4]
    return local_mse, nonlocal_mse, nonlocal_prob, w_opt, out

def max_fi_power(total_pwr, n_units, dims, sigma_n=1, max_snr=2, eps=1e-4,
                 volume_mult=2, lambda_deviation=2, ret_min_max=False,
                 n_ws=200, n_iters=10, T=.35, opt_kind='brute',
                 use_min_func=_min_mse_func, use_w=None, max_w=.5):
    max_pwr = max_snr*sigma_n
    min_func = ft.partial(use_min_func, n_units=n_units, dims=dims,
                          total_pwr=total_pwr, lambda_deviation=lambda_deviation)

    if use_w is not None:
        w_opt = use_w
    elif opt_kind == 'basinhop':
        minimizer_kwargs = {'bounds':((eps, max_w),)}
        res = sopt.basinhopping(min_func, (.5,), niter=n_iters,
                                minimizer_kwargs=minimizer_kwargs,
                                T=T)
        w_opt = res.x[0]
    elif opt_kind == 'brute':
        # pre_ws = np.linspace(eps, 1, n_ws)
        pre_ws = np.linspace(eps, max_w, n_ws)
        fis = list(min_func((w_i,)) for w_i in pre_ws)
        w_opt = pre_ws[np.nanargmin(fis)]
    elif opt_kind == 'peak_finding':
        pre_ws = np.linspace(eps, max_w, n_ws)
        fis = list(min_func((w_i,)) for w_i in pre_ws)
        peaks, _ = sig.find_peaks(fis)
        if len(peaks) == 0:
            starting_ws = (.5,)
        else:
            starting_ws = pre_ws[peaks]
        candidate_ws = np.zeros(len(starting_ws))
        candidate_loss = np.zeros_like(candidate_ws)
        for i, w_i in enumerate(starting_ws):
            res = sopt.minimize(min_func, (w_i,), bounds=((eps, 1),))
            candidate_ws[i] = res.x[0]
            candidate_loss[i] = res.fun
        w_opt = candidate_ws[np.argmin(candidate_loss)]
    else:
        raise TypeError('opt_kind must be one of "basinhop", "brute", or '
                        '"peak_finding"')
    pwr_pre = random_uniform_pwr(n_units, w_opt, dims, scale=1)
    rescale_opt = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w_opt, dims, scale=rescale_opt,
                           print_=False)
    fi_var = random_uniform_fi_var(n_units, w_opt, dims, scale=rescale_opt)
    pwr = random_uniform_pwr(n_units, w_opt, dims, scale=rescale_opt)
    if ret_min_max:
        fi = min_func([w_opt])
    return fi, fi_var, pwr, w_opt, rescale_opt

def random_uniform_unit_mean(wid, dims, scale=1):
    b = (np.sqrt(2*np.pi)*wid*ss.erf(1/(np.sqrt(2)*wid))
         - 2*(wid**2)*(1 - np.exp(-1/(2*wid**2))))
    if u.check_list(wid):
        c = np.product(b)
    else:
        c = b**dims
    return scale*c

def random_uniform_unit_var(pwr, n_units, wid, dims, modules=1):
    scale = random_uniform_scale_vec(pwr, n_units, wid, dims, modules=modules)
    v2 = random_uniform_pwr(1, wid, dims, scale=scale)
    v = random_uniform_unit_mean(wid, dims, scale=scale)
    return v2 - v**2

def random_uniform_pwr(n_units, wid, dims, scale=1, vec=True):
    a = scale**2
    b = np.sqrt(np.pi)*wid*ss.erf(1/wid) - (wid**2)*(1 - np.exp(-1/(wid**2)))
    if (not vec) and u.check_list(wid):
        c = np.product(b)
    else:
        c = b**dims
    return n_units*a*c

def random_uniform_fi_ms(pwr, wid, sigma=1):
    a = pwr*.5
    b = sigma*(wid**2)
    return a/b    

def random_uniform_fi_simplified(pwr, wid, sigma=1):
    a = pwr*(.5*np.sqrt(np.pi) - wid)
    b = sigma*(wid**2)*(np.sqrt(np.pi) - wid)
    return a/b

def random_uniform_fi_pwr(n_units, pwr, wid, dims, sigma_n=1, ret_num=True):
    p = random_uniform_pwr(n_units, wid, dims, scale=1)
    rescale = np.sqrt(pwr/p)
    out = random_uniform_fi(n_units, wid, dims, scale=rescale, sigma_n=sigma_n)
    if ret_num:
        out = out[0, 0]
    return out

def random_uniform_scale_vec(pwr, n_units, wid, dims, modules=1):
    denom1 = np.sqrt(np.pi)*wid*ss.erf(1/wid) - (wid**2)*(1 - np.exp(-1/(wid**2)))
    denom2 = (modules - 1)*(np.sqrt(np.pi*2)*wid*ss.erf(1/(np.sqrt(2)*wid))
                            - 2*(wid**2)*(1 - np.exp(-1/(2*(wid**2)))))
    denom = (denom1**dims + denom2**(2*dims))*n_units
    scale = np.sqrt(pwr/denom)
    return scale

def random_uniform_fi_vec(pwr, n_units, wid, dims, sigma_n=1):
    scale = random_uniform_scale_vec(pwr, n_units, wid, dims)
    wid_i2 = wid**2
    wid_i = wid
    b_pre = np.sqrt(np.pi)*wid*ss.erf(1/wid) - (wid**2)*(1 - np.exp(-1/(wid**2)))
    b = b_pre**(dims - 1)
    
    a = n_units*(scale**2)/((sigma_n**2)*wid_i2)**2
    c = .5*wid_i2*(np.sqrt(np.pi)*wid_i*ss.erf(1/wid_i)
                   - 2*np.exp(-1/wid_i2))
    d = wid_i2*(wid_i2 - (wid_i2 + 1)*np.exp(-1/wid_i2))
    fi = a*b*(c - d)
    return fi

def random_uniform_fi(n_units, wid, dims, scale=1, sigma_n=1, print_=False):
    fi = np.zeros((dims, dims))
    b_pre = np.sqrt(np.pi)*wid*ss.erf(1/wid) - (wid**2)*(1 - np.exp(-1/(wid**2)))
    for i in range(dims):
        if u.check_list(wid):
            mask = np.arange(dims) != i
            b = np.product(b_pre[mask])
            wid_i = wid[i]
        else:
            b = b_pre**(dims - 1)
            wid_i = wid
        wid_i2 = wid_i**2
        a = n_units*(scale**2)/(sigma_n*wid_i2)**2
        c = .5*wid_i2*(np.sqrt(np.pi)*wid_i*ss.erf(1/wid_i)
                       - 2*np.exp(-1/wid_i2))
        d = wid_i2*(wid_i2 - (wid_i2 + 1)*np.exp(-1/wid_i2))
        fi[i, i] = a*b*(c - d)
    if print_:
        print('b: {}   c: {}   d: {}'.format(b, c, d))
    return fi

def _non_deriv_terms(wid, m):
    return (.5*np.sqrt(np.pi)*wid*(ss.erf((1 - m)/wid) + ss.erf(m/wid)))**2

def _deriv_term(wid, m):
    pref = .25/(wid**2)
    ft = -2*m*np.exp(-(m**2)/(wid**2))
    st = 2*(m - 1)*np.exp(-((m - 1)**2)/(wid**2))
    tt = np.sqrt(np.pi)*wid*(ss.erf((1 - m)/wid) + ss.erf(m/wid))
    return (pref*(ft + st + tt))**2

def _cov_terms_func(wid, i, dims, m):
    ndts = _non_deriv_terms(wid, m)
    if u.check_list(wid):
        mask = np.arange(dims) != i
        ndt = np.product(ndts[mask])
        wid_i = wid[i]
    else:
        ndt = ndts**(dims - 1)
        wid_i = wid
    dt = _deriv_term(wid_i, m)
    return (ndt*dt)**2

def ni_non_deriv_term(w, approx=False):
    spi = np.sqrt(np.pi)
    a = (2*w/spi)*np.exp(-1/(w**2))*ss.erf(1/w)
    b = ss.erf(1/w)**2
    c = -(np.sqrt(2)/spi)*w*ss.erf(np.sqrt(2)/w)
    d = ss.erf(1/(2*w))
    e = 2*(w/spi)*(np.exp(-1/(4*w**2)) - 1)

    if approx:
        de = d + e
    else:
        _f = lambda x: ss.erf((1 - x)/w)*ss.erf(x/w)
        de = sint.quad(_f, 0, 1)[0]
    
    return np.pi*(w**2)*(a + b + c + de)/2

def ni_deriv_term(w, approx=False):
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)
    pref = 1/(16*w**4)

    a = .25*(w**2)*(stwo*spi*w*ss.erf(stwo/w) - 4*np.exp(-2/(w**2)))
    b = .5*w*np.exp(-1/(w**2))*(2*w - stwo*spi*np.exp(1/(2*w**2))*(w**2 - 1)
                              *ss.erf(1/(stwo*w)))
    c = -spi*(w**3)*(ss.erf(1/w) - stwo*np.exp(-1/(2*w**2))
                     *ss.erf(1/(stwo*w)))
    d = -.5*spi*(w**3)*(stwo*ss.erf(stwo/w) - 2*np.exp(-1/(w**2))*ss.erf(1/w))

    e = b
    f = a
    g = d
    h = c

    i = c
    j = g
    k = np.pi*(w**2)*((2*w/spi)*np.exp(-1/(w**2))*ss.erf(1/w) + ss.erf(1/w)**2
                      - (stwo/spi)*w*ss.erf(stwo/w))
    if approx:
        l = np.pi*(w**2)*(ss.erf(1/(2*w)) + (2*w/spi)*(np.exp(-1/(4*(w**2))) - 1))
    else:
        _f = lambda x: ss.erf((1 - x)/w)*ss.erf(x/w)
        l = np.pi*(w**2)*sint.quad(_f, 0, 1)[0]

    m = d
    n = h
    o = l
    p = k
    
    big_sum = (a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p)
    return pref*np.sum(big_sum)

def non_term_m(wid, i, dims):
    out = []
    if u.check_list(wid):
        for j in range(dims):
            w = wid[j]
            if i == j:
                b = (4*np.sqrt(np.pi) - np.sqrt(np.pi*2)*15/2
                     - 4*np.sqrt(np.pi)*np.exp(-1/(4*w**2)))
                v = (1/(16*w**2))*(4*np.pi + b*w)
            else:
                a = (np.sqrt(2) - 2 + 2*np.exp(-1/(4*w**2)))
                v = ((w**2)*np.pi/2)*(2 - a*w/np.sqrt(np.pi))
            out.append(v)
        var = np.product(out)
    else:
        w = wid
        b = (4*np.sqrt(np.pi) - np.sqrt(np.pi*2)*15/2
             - 4*np.sqrt(np.pi)*np.exp(-1/(4*w**2)))
        dt = (1/(16*w**2))*(4*np.pi + b*w)
        a = (-np.sqrt(2) - 2 + 2*np.exp(-1/(4*w**2)))
        ndt = ((w**2)*np.pi/2)*(2 - a*w/np.sqrt(np.pi))
        var = dt*(ndt**(dims - 1))
    return var    

def non_integrate_m(wid, i, dims):
    out = []
    if u.check_list(wid):
        for j in range(dims):
            if i == j:
                v = ni_deriv_term(wid[j])
            else:
                v = ni_non_deriv_term(wid[j])
            out.append(v)
        var = np.product(out)
    else:
        dt = ni_deriv_term(wid)
        ndt = ni_non_deriv_term(wid)
        var = dt*(ndt**(dims - 1))
    # print('dt ', dt)
    # print('ndt', ndt)
    return var

@ft.lru_cache(maxsize=None)
def integrate_m(wid, i, dims):
    out = []
    if u.check_list(wid):
        for j in range(dims):
            if i == j:
                f = ft.partial(_deriv_term, wid[j])
            else:
                f = ft.partial(_non_deriv_terms, wid[j])
            v, err = sint.quad(f, 0, 1)
            out.append(v)
        var = np.product(out)
    else:
        f = ft.partial(_deriv_term, wid)
        dt, err = sint.quad(f, 0, 1)
        f = ft.partial(_non_deriv_terms, wid)
        ndt, err = sint.quad(f, 0, 1)
        var = dt*(ndt**(dims - 1))        
    # print('dt ', dt)
    # print('ndt', ndt)    
    return var, 0

def _full_cov_terms(wid, i, dims, *zs):
    zs = np.array(zs)
    zs_m = zs[:dims]
    zs_n = zs[dims:2*dims]
    mu = zs[2*dims:]
    non_deriv = np.exp(-((zs_m - mu)**2 + (zs_n - mu)**2)/(wid**2))
    t1 = np.product(non_deriv)
    if u.check_list(wid):
        wid_i = wid[i]
    else:
        wid_i = wid
    t2 = ((zs_m[i] - mu[i])**2)*((zs_n[i] - mu[i])**2)/(wid_i**8)
    return t1*t2

@ft.lru_cache(maxsize=None)
def full_integrate_m(wid, i, dims):
    f = ft.partial(_full_cov_terms, wid, i, dims)
    ranges = ((0, 1),)*(3*dims)
    val, err = sint.nquad(f, ranges)
    return val, err

# def _wid_irrel_integ(wid, n_samps=1000):
#     rng = np.random.default_rng()
#     diff_samps = np.diff(rng.uniform(0, 1, size=(n_samps, 2)), axis=1)**2
#     return np.mean(np.exp(-2*diff_samps/wid**2), axis=0)

@ft.lru_cache(maxsize=None)
def _wid_irrel_integ(wid, n_samps=1000):
    func = lambda diff: (2 - 2*diff)*np.exp(-2*diff**2/wid**2)
    out, err = sint.quad(func, 0, 1)
    return out

@ft.lru_cache(maxsize=None)
def _wid_rel_integ(wid, n_samps=1000):
    func = lambda diff: (2 - 2*diff)*(diff**4)*np.exp(-2*diff**2/wid**2)/(wid**4)
    out, err = sint.quad(func, 0, 1)
    return out

@ft.lru_cache(maxsize=None)
def _wid_full_integ(wid):
    wid_i = wid[0]
    wid_j1 = wid[1]
    wid_j2 = wid[2]
    def func(z_i, z_j1, z_j2):
        out_i = (2 - 2*z_i)*(z_i**4)*np.exp(-2*z_i**2/wid_i**2)/(wid_i**4)
        out_j1 = (2 - 2*z_j1)*np.exp(-2*z_j1**2/wid_j1**2)
        out_j2 = (2 - 2*z_j2)*np.exp(-2*z_j2**2/wid_j2**2)
        return out_i*out_j1*out_j2
    out, err = sint.tplquad(func, 0, 1, 0, 1, 0, 1)
    
    return out

@ft.lru_cache(maxsize=None)
def _wid_full_integ2(wid):
    wid_i = wid[0]
    wid_j1 = wid[1]
    wid_j2 = wid[2]
    
    def func(z_i, z_j1, z_j2):
        out_i = (2 - 2*z_i)*((z_i**2)*np.exp(-z_i**2/wid_i**2)/(wid_i**2))**2
        out_j1 = (2 - 2*z_j1)*np.exp(-z_j1**2/wid_j1**2)**2 
        out_j2 = (2 - 2*z_j2)*np.exp(-z_j2**2/wid_j2**2)**2
        return (out_i*out_j1*out_j2)
    out, err = sint.tplquad(func, 0, 1, 0, 1, 0, 1)    
    return out

def random_uniform_fi_var_simp(n_units, w, dims, total_pwr, sigma_n=1):
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)

    pref1 = (total_pwr**2)/(4*(n_units**2)*(sigma_n**2)*(w**3))
    pref2 = ((spi - w)*w)**(-2*dims)
    pref = pref1*pref2

    a = w*(n_units**2)*(spi - 2*w)**2
    b = ((spi - w)*w)**(2*(dims - 1))
    ab = a*b

    pref_cde = 2**(-2 - dims)*n_units
    c = (3*stwo*spi - 8*w)*((stwo*spi - w)*w)**(dims - 1)
    d = ((n_units - 1)*np.pi**(dims - .5)*w
         *(w**2*(1 + (-2 - stwo + 2*np.exp(-1/(4*w**2)))*w/spi
                 + ss.erf(1/(2*w))))**(dims - 1))
    e = (4*spi - 16*w - 7*stwo*w + 8*np.exp(-1/(4*w**2))*w +
         4*spi*ss.erf(1/(2*w)) + 8*stwo*np.exp(-1/(2*w**2))*w*ss.erf(1/(stwo*w)))
        
    return -pref*(ab - pref_cde*(c + d*e))

def random_uniform_fi_var_simp2(n_units, w, dims, total_pwr, sigma_n=1):
    spi = np.sqrt(np.pi)
    stwo = np.sqrt(2)

    pref1 = (total_pwr**2)/(4*(n_units**2)*(sigma_n**2)*(w**3))
    pref2 = ((spi - w)*w)**(-2*dims)
    pref = pref1*pref2

    a = w*(n_units**2)*(spi - 2*w)**2
    b = ((spi - w)*w)**(2*(dims - 1))
    ab = a*b

    pref_cde = 2**(-2 - dims)*n_units
    c = (3*stwo*spi - 8*w)*((stwo*spi - w)*w)**(dims - 1)
    d = ((n_units - 1)*np.pi**(dims - .5)*w
         *(w**2*(2 + (-2 - stwo)*w/spi))**(dims - 1))
    e = (8*spi - 16*w - 7*stwo*w)
        
    return -pref*(ab - pref_cde*(c + d*e))

def random_uniform_fi_var_pwr(n_units, pwr, wid, dims, **kwargs):
    pre_pwr = random_uniform_pwr(n_units, wid, dims, scale=1)
    rescale = np.sqrt(pwr/pre_pwr)
    return random_uniform_fi_var(n_units, wid, dims, scale=rescale,
                                 **kwargs)[0, 0]

def random_uniform_fi_var(n_units, wid, dims, scale=1, sigma_n=1, err_thr=1e-3,
                          ret_pieces=False, non_integrate=False):
    fi_mean = random_uniform_fi(n_units, wid, dims, scale=scale,
                                sigma_n=sigma_n)
    fi_v2 = np.zeros_like(fi_mean)
    b_pre = (np.sqrt(np.pi/2)*wid*ss.erf(np.sqrt(2)/wid)
             - .5*(wid**2)*(1 - np.exp(-2/(wid**2))))
    if ret_pieces:
        piece_fiv = np.zeros(dims)
        piece_off = np.zeros(dims)
        piece_fim = np.zeros(dims)
    for i in range(dims):
        if u.check_list(wid):
            mask = np.arange(dims) != i
            jt = np.product(b_pre[mask])
            wid_i = wid[i]
        else:
            jt = b_pre**(dims - 1)
            wid_i = wid
        wid_i2 = wid_i**2
        wid_i4 = wid_i**4
        a = 3*np.sqrt(2*np.pi)*(wid_i**3)*ss.erf(np.sqrt(2)/wid_i)
        b = 4*(3*wid_i2 + 4)*np.exp(-2/wid_i2)
        c = 32*wid_i2

        d = wid_i4 - (wid_i4 + 2*wid_i2 + 2)*np.exp(-2/wid_i2)
        e = 4*wid_i2
        fiv = (((a - b)/c) - (d/e))*jt
        if non_integrate:
            off_diag = non_integrate_m(wid, i, dims)
            err = 0
        else:
            off_diag, err = integrate_m(wid, i, dims)
        # print('od', off_diag)
        assert err < err_thr
        pref = (scale**4)/((sigma_n**4))
        # look at off_diag and fiv for a large RF vs small RF
        fi_v2[i, i] = pref*((n_units)*fiv/wid_i4
                            + (n_units - 1)*n_units*off_diag)
        if ret_pieces:
            piece_fiv[i] = pref*n_units*fiv/wid_i4
            piece_off[i] = pref*off_diag*n_units*(n_units - 1)
            piece_fim[i] = fi_mean[i, i]**2
    fi_var = fi_v2 - fi_mean**2
    out = fi_var
    if ret_pieces:
        out = (fi_var, np.mean(piece_fiv), np.mean(piece_off),
               np.mean(piece_fim))
    else:
        out = fi_var
    return out

def get_output_func_distribution_shapes(n_units_pd, input_distributions,
                                        wid_scaling=1):
    means = np.zeros((len(input_distributions), n_units_pd))
    wids = np.zeros_like(means)
    percent_size = 1/n_units_pd
    for j in range(len(input_distributions)):
        for i in range(n_units_pd):
            perc = (i + .5)*percent_size
            m = input_distributions[j].ppf(perc)
            means[j, i] = m
        dm = np.diff(means[j])/wid_scaling
        dm_p1 = np.zeros(n_units_pd)
        half = int(np.floor(n_units_pd/2))
        dm_p1[:half] = dm[:half]
        dm_p1[-half:] = dm[-half:]
        if n_units_pd % 2 == 1:
            dm_p1[half] = dm[half]
        wids[j] = dm_p1**2
    means_all = np.array(list(it.product(*means)))
    wids_all = np.array(list(it.product(*wids)))
    return means_all, wids_all

def get_distribution_gaussian_resp_func(n_units_pd, input_distributions, scale=1,
                                        baseline=0, wid_scaling=1,
                                        random_widths=False, rand_frac=.5):
    ms, ws = get_output_func_distribution_shapes(n_units_pd, input_distributions,
                                                 wid_scaling=wid_scaling)
    if random_widths:
        rng = np.random.default_rng()
        min_w = np.min(ws)
        for ind in u.make_array_ind_iterator(ws.shape):
            w_now = ws[ind]
            w_new =  rng.normal(w_now, w_now*rand_frac, size=1)
            ws[ind] = max(w_new, min_w)
        
    resp_func, d_resp_func = make_gaussian_vector_rf(ms, ws, scale,
                                                     baseline)
    return resp_func, d_resp_func, ms, ws

def make_gaussian_vector_rf(cents, sizes, scale, baseline, sub_dim=None,
                            titrate_pwr=None, n_samps=10000, cost_func=None,
                            titrate_func=None):
    cents = np.array(cents)
    sizes = np.array(sizes)
    if len(cents.shape) <= 1:
        cents = np.reshape(cents, (-1, 1))
    if len(sizes.shape) <= 1:
        sizes = np.reshape(sizes, (-1, 1))
    cents = np.expand_dims(cents, axis=0)
    sizes = np.expand_dims(sizes, axis=0)
    if titrate_pwr is not None:
        rfs = ft.partial(eval_gaussian_vector_rf, cents=cents, sizes=sizes,
                         scale=1, baseline=baseline)
        if cost_func is None:
            pwr = np.mean(np.sum(rfs(titrate_pwr.rvs(n_samps))**2, axis=1))
        else:
            pwr = cost_func(rfs(titrate_pwr.rvs(n_samps)))
        if titrate_func is None:
            new_scale = np.sqrt(scale/pwr)
        else:
            new_scale = titrate_func(scale, pwr)
    else:
        new_scale = scale
    rfs = ft.partial(eval_gaussian_vector_rf, cents=cents, sizes=sizes,
                     scale=new_scale, baseline=baseline)
    drfs = ft.partial(eval_gaussian_vector_deriv, cents=cents, sizes=sizes,
                      scale=new_scale, baseline=baseline)
    if titrate_pwr is not None:
        pwr = np.mean(np.sum(rfs(titrate_pwr.rvs(n_samps))**2, axis=1))
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
