
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
                            scale_use=None, sigma_n=1, **kwargs):
    distrs = (sts.uniform(0, 1),)*dims
    stim_distr = u.MultivariateUniform(dims, (0, 1))

    n_units_pd = int(np.round(n_units**(1/dims)))
    ms, ws = get_output_func_distribution_shapes(n_units_pd, distrs,
                                                 wid_scaling=1)
    scale = total_pwr
    titrate_pwr = stim_distr
    
    rf, drf = make_gaussian_vector_rf(ms, ws**2, scale, 0,
                                      titrate_pwr=titrate_pwr,
                                      **kwargs)
    noise_distr = sts.multivariate_normal(np.zeros(n_units_pd**dims),
                                          sigma_n)
    
    return stim_distr, rf, drf, noise_distr

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
    noise_distr = sts.multivariate_normal(np.zeros(n_units), sigma_n)
    
    out = (stim_distr, rf, drf, noise_distr)
    if ret_params:
        out = out + (ms, ws)
    return out

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

def brute_decode_rf(reps, func, dim, n_gran=200, init_guess=None):
    guesses = np.array(list(it.product(np.linspace(0, 1, n_gran),
                                       repeat=dim)))
    g_reps = func(guesses)
    g_reps = np.expand_dims(g_reps, 1)
    reps = np.expand_dims(reps, 0)
    g_ind = np.argmin(np.sum((g_reps - reps)**2, axis=-1), axis=0)
    return guesses[g_ind]

@ft.lru_cache(maxsize=None)
def compute_threshold_err_prob(pwr, n_units, dim, w_opt, sigma_n=1, scale=1,
                               resp_scale=1, ret_components=False):
    mu_p = 2*pwr
    std_p = np.sqrt(2*random_uniform_pwr_var(n_units, w_opt, dim,
                                             scale=resp_scale))
    def integ_func(d):
        prob = sts.norm(mu_p, std_p).pdf(d)
        cumu = sts.norm(0, 1).cdf(-np.sqrt(d)/(2*sigma_n))
        return prob*cumu

    v, err = sint.quad(integ_func, 0, np.inf)
    effective_dim = (scale/(2*w_opt))**dim
    factor = max(min(n_units, effective_dim) - 1, 0)
    approx_prob = v*factor # min(v*factor, 1)
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

def random_uniform_pwr_var(n_units, wid, dims, scale=1):
    pwr = random_uniform_pwr(n_units, wid, dims, scale=scale)
    a = scale**4
    b = (np.sqrt(np.pi/2)*wid*ss.erf(np.sqrt(2)/wid)
         - .5*(wid**2)*(1 - np.exp(-2/(wid**2))))
    f = ft.partial(_non_deriv_terms, wid)
    off_term, err = sint.quad(f, 0, 1)
    if u.check_list(wid):
        c = np.product(b)
        off_term_prod = np.product(off_term)
    else:
        c = b**dims
        off_term_prod = off_term**dims
    pwr2 = a*(n_units*c + n_units*(n_units - 1)*off_term_prod) - pwr**2
    # r_var = (pwr2 - (pwr/n_units)**2
    r_var = pwr2
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
                        lambda_deviation=None):
    w = w[0]
    pwr_pre = random_uniform_pwr(n_units, w, dims, scale=1)
    rescale = np.sqrt(total_pwr/pwr_pre)
    fi = random_uniform_fi(n_units, w, dims, scale=rescale)
    fi_var = random_uniform_fi_var(n_units, w, dims, scale=rescale)
    fi_corr = fi[0, 0] - lambda_deviation*np.sqrt(fi_var[0, 0])
    
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

def max_fi_power(total_pwr, n_units, dims, sigma_n=1, max_snr=2, eps=1e-4,
                 volume_mult=2, lambda_deviation=2, ret_min_max=False,
                 n_ws=5000, n_iters=10, T=.35, opt_kind='brute',
                 use_min_func=_min_func, use_w=None):
    max_pwr = max_snr*sigma_n
    min_func = ft.partial(use_min_func, n_units=n_units, dims=dims,
                          total_pwr=total_pwr, lambda_deviation=lambda_deviation)

    if use_w is not None:
        w_opt = use_w
    elif opt_kind == 'basinhop':
        minimizer_kwargs = {'bounds':((eps, 1),)}
        res = sopt.basinhopping(min_func, (.5,), niter=n_iters,
                                minimizer_kwargs=minimizer_kwargs,
                                T=T)
        w_opt = res.x[0]
    elif opt_kind == 'brute':
        pre_ws = np.linspace(eps, 1, n_ws)
        fis = list(min_func((w_i,)) for w_i in pre_ws)
        w_opt = pre_ws[np.nanargmin(fis)]
    elif opt_kind == 'peak_finding':
        pre_ws = np.linspace(eps, 1, n_ws)
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

def random_uniform_pwr(n_units, wid, dims, scale=1):
    a = scale**2
    b = np.sqrt(np.pi)*wid*ss.erf(1/wid) - (wid**2)*(1 - np.exp(-1/(wid**2)))
    if u.check_list(wid):
        c = np.product(b)
    else:
        c = b**dims
    return n_units*a*c

## try to gain intuition for falling precision with increasing dimensionality
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
        a = n_units*(scale**2)/(sigma_n*wid_i2**2)
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


def random_uniform_fi_var(n_units, wid, dims, scale=1, sigma_n=1, err_thr=1e-3,
                          ret_pieces=False):
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
        off_diag, err = integrate_m(wid, i, dims)
        assert err < err_thr
        pref = (scale**4)/((sigma_n**2))
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
        wids[j] = dm_p1
    means_all = np.array(list(it.product(*means)))
    wids_all = np.array(list(it.product(*wids)))
    return means_all, wids_all

def get_distribution_gaussian_resp_func(n_units_pd, input_distributions, scale=1,
                                        baseline=0, wid_scaling=1):
    ms, ws = get_output_func_distribution_shapes(n_units_pd, input_distributions,
                                                 wid_scaling=wid_scaling)
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
