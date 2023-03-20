
import numpy as np
import scipy.stats as sts
import sklearn.decomposition as skd
import itertools as it
import matplotlib as mpl
import matplotlib.pyplot as plt
import general.utility as u
import general.neural_analysis as na
from matplotlib.collections import LineCollection
import matplotlib.colors as mplcolor
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import matplotlib.patches as patches
import matplotlib.tri as tri
import scipy.stats as sts
import colorsys
import ternary

def set_ax_color(ax, color, sides=('bottom', 'top', 'left', 'right')):
    for s in sides:
        ax.spines[s].set_color(color)

def line_speed_func(*args):
    diffs = np.diff(np.stack(args, axis=1), axis=0)
    speeds = np.sqrt(np.sum(diffs**2, axis=1))
    speeds = np.concatenate((speeds[:1], speeds), axis=0)
    return speeds

def simplefy(p, how='heatmap', ax=None, cmap=None, border='k', color=None,
             lw=2, **kwargs):
    """
    Plot a heatmap or contour of 3-simplex. Indices of rows of p are:
        
             3
            /\
           /  \ 
         1 --- 2 
           
    p: (N, 3) probabilities
    how: 'heatmap' (default) or 'contour'
    cmap: colormap for heatmap
    border: color of triangle border (string)

    "code from the collection MATTEO by matteo"
    """
    p = p.T
    if ax is None:
        ax = plt.gca()
    if cmap is None and color is not None:
        cmap = make_linear_cmap(color)
    elif cmap is None:
        cmap = 'Greys'

    
    simplx_basis = np.array([[-1,1, 0],[-0.5,-0.5, 1]])
    simplx_basis /= np.linalg.norm(simplx_basis,axis=1,keepdims=True)
    
    simp = simplx_basis@p
    
    kd_pdf = sts.gaussian_kde(simp) # estimate density
    
    ## plotting
    xmin = -0.5*np.sqrt(2)
    xmax = 0.5*np.sqrt(2)
    ymin = np.sqrt(6)/3 - np.sqrt(1.5)
    ymax = np.sqrt(6)/3
    
    if how=='contours':
        xx, yy = np.meshgrid(np.linspace(xmin,xmax,100),np.linspace(ymin,ymax,100))
        foo = (np.stack([xx.flatten(),yy.flatten()]).T@simplx_basis) + [1/3,1/3,1/3]
        if color is None:
            color = 'k'
        
        support = np.linalg.norm(foo,1, axis=-1)<1.001
        
        zz = np.where(support, kd_pdf(np.stack([xx.flatten(),yy.flatten()])), np.nan)

        ax.contour(xx,yy,zz.reshape(100,100,order='A'), [.95],
                   colors=(color,),
                   linestyles=['solid','dotted'],
                   linewidths=(lw,))
        
    elif how == 'heatmap':
        
        grid = tri.Triangulation(simplx_basis[0,:], simplx_basis[1,:])
        grid = tri.UniformTriRefiner(grid).refine_triangulation(subdiv=6)
        foo = np.stack([grid.x,grid.y])[:,grid.triangles]
        foo = foo.mean(-1).T@simplx_basis + [1/3,1/3,1/3]
        msk = np.linalg.norm(foo,1, axis=-1)>=1.001
        grid.set_mask(msk)
        
        zz = kd_pdf(np.stack([grid.x,grid.y]))
        
        ax.tripcolor(grid, zz, rasterized=True, cmap=cmap, **kwargs)

    ax.plot([xmin,xmax,0,xmin], [ymin, ymin, ymax, ymin], color=border)        
    ax.set_ylim([ymin*1.1,ymax*1.1])
    ax.set_xlim([xmin*1.1,xmax*1.1])
    ax.set_aspect('equal')
    ax.set_axis_off()

def plot_colored_pts(xs, ys, colors=None, ax=None, line_col=(.9, .9, .9),
                     linewidth=1,
                     **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if colors is None:
        colors = (None,)*len(xs)
    ax.plot(xs, np.mean(ys, axis=0), color=line_col, lw=linewidth)
    for i, (x, y) in enumerate(zip(xs, ys.T)):
        y = np.expand_dims(y, 1)
        plot_trace_werr([x], y, points=True, fill=False,
                        color=colors[i], ax=ax, **kwargs)
    return ax        

def plot_colored_line(xs, ys, zs=None, col_inds=None, cmap='Blues',
                      norm=plt.Normalize(0., 1.), func=None, ax=None,
                      **kwargs):
    if u.check_list(cmap):
        cmap = plt.get_cmap(cmap)
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if func is not None:
        col_inds = func(xs, ys)
    elif col_inds is None:
        col_inds = np.linspace(0, 1, len(xs))
    if norm is not None:
        norm.autoscale(col_inds)
    else:
        norm = plt.Normalize(0, 1)
    if zs is None:
        points = np.array([xs, ys]).T.reshape(-1, 1, 2)
        lc_func = LineCollection
    else:
        points = np.array([xs, ys, zs]).T.reshape(-1, 1, 3)
        lc_func = Line3DCollection
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = lc_func(segments, array=col_inds, cmap=cmap, norm=norm,
                 **kwargs)
    ax.add_collection(lc)
    if zs is not None:
        ax.plot(xs, ys, zs, alpha=0)
    else:
        ax.plot(xs, ys, alpha=0)
    return lc
    

def plot_single_units(xs, sus, labels, colors=None, style=(), show=False,
                      errorbar=True, alpha=.5, trial_color=(.8, .8, .8),
                      trial_style='dashed', trial_struct=None, ax_spec=None,
                      fs=None, min_trials=None):
    """
    construct a plot according to a particular style

    xs - array<float> N, providing the label for the x-axis
    sus - list<dict<tuple, array<float> KxN>> C, a list of dictionaries
      in which the C dictionaries are C different conditions; in each 
      dictionary, the key identifies neurons (and should be common across
      the C dicts) while the array is K trials at N time points
    labels - list<string> C, labels to use for each condition being compared
    colors - list<color> C, list of items that can be interpreted as colors
      by matplotlib, defining the color to use for each condition
    style - matplotlib style to use for the plots
    """
    with plt.style.context(style):
        if colors is None:
            colors = (None,)*len(sus)
        if ax_spec is None:
            ax_spec = (1,1,1)
        if fs is None:
            fs = list([plt.figure() for k in sus[0].keys()])
        axs = list([f.add_subplot(*ax_spec) for f in fs])
        for j, k in enumerate(sus[0].keys()):
            for i, c in enumerate(sus):
                ax = plot_trace_werr(xs, sus[i][k], colors[i], label=labels[i],
                                     show=False, errorbar=True, alpha=alpha,
                                     error_func=sem, title=k, ax=axs[j])
            if trial_struct is not None:
                plot_trial_structure(trial_struct['transitions'],
                                     trial_struct['labels'],
                                     color=trial_color,
                                     linestyle=trial_style,
                                     ax=axs[j])
    if show:
        plt.show(block=False)
    return fs

def axs_adder(func, rows, cols, **ax_kwargs):
    if rows == 1 and cols == 1:
        axs_wrapper = ax_adder(func)
    else:        
        def axs_wrapper(*args, axs=None, fwid=3, **kwargs):
            if axs is None:
                f, axs = plt.subplots(rows, cols, figsize=(fwid*cols, fwid*rows),
                                      **ax_kwargs)
            out = func(*args, axs=axs, **kwargs)
            if out is None:
                new_out = axs
            else:
                new_out = (out, axs)
            return new_out
    return axs_wrapper

def ax_3d_adder(func):
    def ax_wrapper(*args, ax=None, fwid=3, **kwargs):
        if ax is None:
            f = plt.figure(figsize=(fwid, fwid))
            ax = f.add_subplots(1, 1, 1, projection='3d')
        out = func(*args, ax=ax, **kwargs)
        if out is None:
            new_out = ax
        else:
            new_out = (out, ax)
        return new_out
    return ax_wrapper

def ax_adder(func):
    def ax_wrapper(*args, ax=None, fwid=3, **kwargs):
        if ax is None:
            f, ax = plt.subplots(1, 1, figsize=(fwid, fwid))
        out = func(*args, ax=ax, **kwargs)
        if out is None:
            new_out = ax
        else:
            new_out = (out, ax)
        return new_out
    return ax_wrapper

def plot_population_average(xs, sus, labels, colors=None, style=(),
                            errorbar=True, alpha=.5, trial_color=(.8, .8, 8),
                            trial_style='dashed', trial_struct=None,
                            ax_spec=None, fs=None, boots=1000,
                            subsample_min=None, include_min=None):
    with plt.style.context(style):
        if colors is None:
            colors = (None,)*len(sus)
        if ax_spec is None:
            ax_spec = (1,1,1)
        if fs is None:
            fs = plt.figure()
        ax = fs.add_subplot(*ax_spec)
        for i, c in enumerate(sus):
            su_avgs = np.zeros((boots, len(xs)))
            for y in range(boots):
                su_ms = np.zeros((len(c.keys()), len(xs)))
                for j, k in enumerate(c.keys()):
                    if subsample_min is not None:
                        n_samps = subsample_min
                    elif include_min is not None:
                        n_samps = max(c[k].shape[0], include_min)
                    else:
                        n_samps = c[k].shape[0]
                    spks = u.resample_on_axis(c[k], n_samps, axis=0,
                                              with_replace=True)
                    su_ms[j] = np.nanmean(spks, axis=0)
                su_avgs[y] = np.nanmean(su_ms, axis=0)
                kb = ~np.isnan(su_ms[:, 0])
            if i == 0:
                mkb = kb
            else:
                mkb = mkb*kb
            print(np.sum(mkb))
            _ = plot_trace_werr(xs, su_avgs, colors[i], label=labels[i],
                                show=False, errorbar=True, alpha=alpha,
                                error_func=conf95_interval,
                                ax=ax)
    return fs        

def plot_collection_views(xs_l, sus_l, labels, colors=None, style=(),
                          errorbar=True, alpha=.5,
                          trial_color=(.8, .8, .8), trial_style='dashed',
                          trial_struct=None, ax_gs=None,
                          add_expansion=True, plotter_func=None):
    fig_list = []
    n_views = len(xs_l)
    if ax_gs is None:
        ax_gs = list((n_views, 1, i + 1) for i in range(n_views))
    elif add_expansion:
        ax_gs = list((x,) for x in ax_gs)
    if colors is None:
        colors = (None,)*n_views
    if trial_struct is None:
        trial_struct = (None,)*n_views
    if plotter_func is None:
        plotter_func = plot_single_units
    for i, sus in enumerate(sus_l):
        xs = xs_l[i]
        labs = labels[i]
        cols = colors[i]
        ts = trial_struct[i]
        if i == 0:
            fs = None
        fs = plotter_func(xs, sus, labs, cols, style, errorbar=errorbar,
                          alpha=alpha, trial_color=trial_color,
                          trial_style=trial_style,
                          trial_struct=trial_struct[i], ax_spec=ax_gs[i],
                          fs=fs)
    return fs    
                
def sem(dat, axis=0, sub=1):
    err_1d = np.nanstd(dat, axis=axis)/np.sqrt(dat.shape[axis] - sub)
    err = np.vstack((err_1d, -err_1d))
    return err

def std(dat, axis=0):
    err_1d = np.nanstd(dat, axis=axis)
    err = np.vstack((err_1d, -err_1d))
    return err

def biased_sem(dat, axis=0):
    err = sem(dat, axis=axis, sub=0)
    return err
    
def conf95_interval(dat, axis=0):
    return u.conf_interval(dat, axis=0, perc=95)

def plot_trial_structure(transition_times=(), labels=(), transition_dict=None,
                         ax=None, style=(), linestyle='dashed', 
                         color=(.1, .1, .1), fontsize=12, alpha=.8):
    if transition_dict is not None:
        transition_times = transition_dict['transitions']
        labels = transition_dict['labels']
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        xmin, xmax = ax.get_xlim()
        ymin, ymax = ax.get_ylim()
        lns = ax.vlines(transition_times, ymin, ymax, linestyle=linestyle,
                        color=color, alpha=alpha)
        
        for i, lab in enumerate(labels):
            ax.text(transition_times[i], ymax, lab, color=color, alpha=alpha,
                    fontsize=fontsize)
    return ax

def add_vlines(pos, ax, color=(.8, .8, .8), alpha=1, **kwargs):
    yl = ax.get_ylim()
    ax.vlines(pos, yl[0], yl[1], color=color, alpha=alpha, **kwargs)
    ax.set_ylim(yl)

def add_hlines(pos, ax, color=(.8, .8, .8), alpha=1, **kwargs):
    xl = ax.get_xlim()
    ax.hlines(pos, xl[0], xl[1], color=color, alpha=alpha, **kwargs)
    ax.set_xlim(xl)

def gen_circle_pts(n, r=1):
    angs = np.linspace(0, 2*np.pi, n)
    pts = np.stack((np.cos(angs), np.sin(angs)), axis=1)
    return r*pts

def pcolormesh_axes(axvals, val_len, diff_ind=0, append=True):
    axvals = np.array(axvals)
    if len(axvals) == val_len:
        if len(axvals) < 2:
            diff = 0
        else:
            diff = np.diff(axvals)[diff_ind]
        axvals_shift = axvals - diff/2
        if append:
            axvals = np.append(axvals_shift, (axvals_shift[-1] + diff))
    return axvals

def plot_highdim_trace(*args, plot_points=False, plot_line=True,
                       ax=None, ms=2, p=None, central_tendence=np.mean,
                       dim_red_mean=True, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1, 1, 1, projection='3d')
    if len(args[0].shape) == 2:
        args = list(np.expand_dims(arg, 0) for arg in args)
    if p is None:
        all_samps = np.concatenate(args, axis=1)
        if dim_red_mean:
            all_samps = np.mean(all_samps, axis=0)
        else:
            all_samps = np.concatenate(all_samps, axis=0)
        p = skd.PCA(3)
        p.fit(all_samps)
    for i, y in enumerate(args):
        colors = kwargs.get('color')
        if colors is None:
            color = None
        else:
            color = colors[i]
        y_mu = p.transform(np.mean(y, axis=0))
        y_pts = p.transform(np.concatenate(y, axis=0))
        if plot_line:
            l = ax.plot(*y_mu.T, color=color, **kwargs)
            color = l[0].get_color()
        if plot_points:
            ax.plot(*y_pts.T, 'o', ms=ms, color=color,
                    **kwargs)
    return ax, p
    

def pcolormesh(xs, ys, data, ax, diff_ind=0, append=True, equal_bins=False,
               **kwargs):
    if equal_bins:
        xs_bins = np.arange(len(xs))
        ys_bins = np.arange(len(ys))
    else:
        xs_bins = xs
        ys_bins = ys
    if len(data.shape) == 1:
        lens = [len(xs), len(ys)]
        dim = np.argmin(lens)
        data = np.repeat(np.expand_dims(data, dim), lens[dim], axis=dim).T
    xs_ax = pcolormesh_axes(xs_bins, len(xs_bins), diff_ind=diff_ind,
                            append=append)
    ys_ax = pcolormesh_axes(ys_bins, len(ys_bins), diff_ind=diff_ind,
                            append=append)
    img = ax.pcolormesh(xs_ax, ys_ax, data, **kwargs)

    if equal_bins:
        ax.set_xticks(xs_bins)
        ax.set_xticklabels(xs)
        
        ax.set_yticks(ys_bins)
        ax.set_yticklabels(ys)
    else:
        ax.set_xticks(xs)
        ax.set_yticks(ys)
    return img

def plot_decoding_heatmap(xs, decmat, colormap=None, show=False, title='',
                          ax=None, style=(), colorbar=True, cb_wid=.05,
                          cutoff=.5, vmax=1.):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1, aspect='equal')
        if len(xs) == decmat.shape[1]:
            diff = np.diff(xs)[0]
            xs = xs - diff/2
            xs = np.append(xs, (xs[-1] + diff))
        trmap = np.nanmean(decmat, axis=0)
        trmap[trmap < cutoff] = cutoff
        img = ax.pcolormesh(xs, xs, trmap, cmap=colormap, 
                            vmin=cutoff, vmax=vmax, shading='flat')
        ax.set_title(title)
        if colorbar:
            f = ax.get_figure()
            pos = ax.get_position()
            x1, y0, y1 = pos.x1, pos.y0, pos.y1
            rect = (x1, y0, cb_wid, y1 - y0)
            cb_ax = f.add_axes(rect, label='cbarax')
            cb = f.colorbar(img, cax=cb_ax, orientation='vertical')
            cb.set_ticks([.5, 1.])
            cb.set_ticklabels(['<.5', '1'])
    if show:
        plt.show(block=False)
    return ax

def plot_trajectories(mean_traj, indiv_trajs, color=None, label='', show=False,
                      title='', alpha=.1, ax=None, style=(), marker=None):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        l = ax.plot(mean_traj[0, :], mean_traj[1, :], color=color, label=label,
                    marker=marker)
        if color is None:
            color = l[0].get_color()
        for i, t in enumerate(indiv_trajs):
            _ = ax.plot(t[0, :], t[1, :], color=color, alpha=alpha,
                        marker=marker)
    if show:
        plt.show(block=False)
    return ax

def plot_smooth_cumu(dat, bins='auto', color=None, label='', title='', 
                     ax=None, legend=True, linestyle=None, normed=True,
                     start=None, log_x=False, log_y=False):
    cts, base = np.histogram(dat, bins)
    cum_cts = np.cumsum(cts)
    cum_cts = cum_cts/np.max(cum_cts)
    base = base[:-1] + np.diff(base)[0]/2
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    if start is not None:
        base = np.concatenate(([start[0]], base))
        cum_cts = np.concatenate(([start[1]], cum_cts))
    ax.plot(base, cum_cts, color=color, label=label, linestyle=linestyle)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    if log_x:
        ax.set_xscale('log')
    if log_y:
        ax.set_yscale('log')
    if len(label) > 0 and legend:
        ax.legend(frameon=False)
    if len(title) > 0:
        ax.set_title(title)
    return ax    

def add_color(color, add_amt):
    new_col = color + add_amt
    nc = np.min(np.stack((new_col, np.ones_like(new_col)), axis=0),
                axis=0)
    return nc

def plot_pt_werr(x, data, central_tendency=np.nanmean, ax=None, **kwargs):
    if ax is None:
        f = plt.figure()
        ax = f.add_subplot(1,1,1)
    xs = np.expand_dims(x, 0)
    dats = np.expand_dims(data, 1)
    l = plot_trace_werr(xs, dats, ax=ax, fill=False,
                        **kwargs)
    ax.plot(xs, central_tendency(dats, axis=0), 'o', color=l[0].get_color())
    return ax

def plot_trace_wpts(x, data, ax=None, color=None, **kwargs):
    l = plot_trace_werr(x, data, ax=ax, **kwargs, fill=False,
                        errorbar=False, color=color)
    for i, data_i in enumerate(data):
        ax.plot(x, data_i, 'o', color=l[0].get_color(), **kwargs)

def make_linear_cmap(b1, b2=None, name=''):
    if b2 is None:
        b_start = (1,)*len(b1)
        b_end = b1
    else:
        b_start = b1
        b_end = b2
    cmap = mpl.colors.LinearSegmentedColormap.from_list(
        name,
        [b_start, b_end]
    )
    return cmap

def set_ylim(ax, low=None, up=None):
    return _set_lim(ax.get_ylim, ax.set_ylim, low=low, up=up)

def set_xlim(ax, low=None, up=None):
    return _set_lim(ax.get_xlim, ax.set_xlim, low=low, up=up)

def _set_lim(getter, setter, low=None, up=None):
    o_low, o_up = getter()
    if low is None:
        low = o_low
    if up is None:
        up = o_up
    return setter([low, up])

def _cs_convert(color, conv_func):
    if len(color.shape) == 1:
        color = np.expand_dims(color, 0)
    color = color
    new_color = np.zeros_like(color)
    for i, col in enumerate(color):
        new_color[i] = conv_func(*col)
    return np.squeeze(new_color)

def rgb_to_hls(color):
    return _cs_convert(color, colorsys.rgb_to_hls)

def hls_to_rgb(color):
    return _cs_convert(color, colorsys.hls_to_rgb)

def get_next_n_colors(n):
    cycler = plt.rcParams['axes.prop_cycle'].by_key()['color']
    return cycler[:n]

def add_color_value(color, amt):
    hsv = rgb_to_hls(color)
    hsv[..., -2] = hsv[..., -2] + amt
    hsv[..., -2] = np.min(np.stack((np.ones_like(hsv[..., -2]),
                                    hsv[..., -2]), axis=0),
                          axis=0)
    hsv[..., -2] = np.max(np.stack((np.zeros_like(hsv[..., -2]),
                                    hsv[..., -2]), axis=0),
                          axis=0)
    return hls_to_rgb(hsv)

def plot_trace_werr(xs_orig, dat, color=None, label='', show=False, title='', 
                    errorbar=True, alpha=.5, ax=None, error_func=sem,
                    style=(), central_tendency=np.nanmean, legend=True,
                    fill=True, log_x=False, log_y=False, line_alpha=1,
                    jagged=False, points=False, elinewidth=1, conf95=False,
                    err=None, err_x=None, polar=False, **kwargs):
    if conf95:
        error_func = conf95_interval
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
        xs_orig = np.array(xs_orig)
        if jagged:
            dat = np.array(dat, dtype=object)
        else:
            dat = np.array(dat)
        if jagged:
            tr = np.array(list(central_tendency(d) for d in dat))
            er = np.array(list(error_func(d)[:, 0] for d in dat)).T
        elif len(dat.shape) > 1:
            tr = central_tendency(dat, axis=0)
            er = error_func(dat, axis=0)
        elif err is not None:
            tr = dat
            er = err
        else: 
            tr = dat
        if len(xs_orig.shape) > 1:
            xs_er = error_func(xs_orig, axis=0)
            xs = central_tendency(xs_orig, axis=0)
        else:
            xs = xs_orig
        if err_x is not None:
            xs_er = err_x
        if polar:
            xs = np.concatenate((xs, xs[..., 0:1]), axis=-1)
            tr = np.concatenate((tr, tr[..., 0:1]), axis=-1)
            er = np.concatenate((er, er[..., 0:1]), axis=-1)
            
        trl = ax.plot(xs, tr, label=label, color=color, alpha=line_alpha,
                      **kwargs)
        if color is None:
            color = trl[0].get_color()
        if points:
            ax.plot(xs, tr, 'o', color=color, alpha=line_alpha, **kwargs)
        alpha = min(line_alpha, alpha)
        if len(dat.shape) > 1 or jagged or err is not None:
            if fill:
                ax.fill_between(xs, tr+er[1, :], tr+er[0, :], color=color, 
                                alpha=alpha)
            else:
                ax.errorbar(xs, tr, (-er[1, :], er[0, :]), color=color,
                            elinewidth=elinewidth, alpha=alpha, **kwargs)
        if len(xs_orig.shape) > 1 or err_x is not None:
            if fill:
                ax.fill_betweenx(tr, xs+xs_er[1, :], xs+xs_er[0, :], 
                                 color=color, alpha=alpha)
            else:
                ax.errorbar(xs, tr, yerr=None, xerr=(-xs_er[1, :], xs_er[0, :]),
                            color=color, elinewidth=elinewidth, alpha=alpha,
                            **kwargs)
        ax.set_title(title)
        if not polar:
            ax.spines['right'].set_visible(False)
            ax.spines['top'].set_visible(False)
        if len(label) > 0 and legend:
            ax.legend(frameon=False)
    if show:
        plt.show(block=False)
    return trl

def plot_parameter_sweep_results(res_dicts, names, errorbar=True, 
                                 error_func=conf95_interval, style=(), 
                                 central_tendency=np.nanmean, f=None,
                                 y_axis_label='', figsize=(12, 4),
                                 sharey=False, colors=None, linestyles=None):
    entries = len(res_dicts[0])
    if f is None:
        f = plt.figure(figsize=figsize)
    for i, k in enumerate(res_dicts[0].keys()):
        if i > 0 and sharey:
            ax_i = f.add_subplot(1, entries, i+1, sharey=ax_i)
        else:
            ax_i = f.add_subplot(1, entries, i+1)
        ax_i.set_xlabel(k)
        if sharey and i > 0:
            plt.setp(ax_i.get_yticklabels(), visible=False)
        else:
            ax_i.set_ylabel(y_axis_label)
        if i > 0:
            legend = False
        else: 
            legend = True
        for j, res_dict in enumerate(res_dicts):
            x_ax, y_ax = res_dict[k]
            if colors is not None:
                col = colors[j]
            else: 
                col = None
            if linestyles is not None:
                linest = linestyles[j]
            else:
                linest = None
            plot_trace_werr(np.array(x_ax)[0], np.array(y_ax).T, 
                            label=names[j], error_func=error_func, 
                            central_tendency=central_tendency, errorbar=errorbar,
                            ax=ax_i, legend=legend, color=col, linestyle=linest)
    return f

def plot_distrib(dat, color=None, bins=None, label='', show=False, title='', 
                 errorbar=True, alpha=1, ax=None, style=(), histtype='step',
                 normed=True, **kwargs):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
        hist = ax.hist(dat, bins=bins, color=color, alpha=alpha, 
                       histtype=histtype, label=label, density=normed,
                       **kwargs)
        if len(label) > 0:
            ax.legend()
    if show:
        plt.show(block=False)
    return ax
    
def plot_dpca_results(xs, res, dims=(1,), color_dict=None, title_dict=None,
                      style=(), axes=None, show=False):
    xs_use = np.arange(len(xs))
    with plt.style.context(style):
        keys = list(res.keys())
        n_plots = len(keys)
        if axes is None:
            f = plt.figure(figsize=(10, 15))
            axes = [f.add_subplot(n_plots, 1, i+1) for i in range(n_plots)]
        if color_dict is None:
            color_dict = dict((k, [None]*res[k].shape[1]) for k in keys)
        if title_dict is None:
            title_dict = dict((k, k) for k in keys)
        for i, k in enumerate(keys):
            ax = axes[i]
            tc = res[k]
            for j in range(tc.shape[1]):
                k_tc = tc[dims, j]
                ax.plot(xs_use, k_tc.T, color=color_dict[k][j])
            ax.set_xlabel('time (ms)')
            xs_labels = xs[ax.get_xticks().astype(int)[:-1]]
            ax.set_xticklabels(xs_labels)
            ax.set_title(title_dict[k])
    if show:
        plt.show(block=False)
    return ax

def label_plot_line(xs, ys, text, ax, mid_samp=None, **kwargs):
    if mid_samp is not None:
        ms0 = mid_samp[0]
        ms1 = mid_samp[1]
    else:
        ms0 = 0
        ms1 = -1
    start = np.array([xs[ms0], ys[ms0]])
    end = np.array([xs[ms1], ys[ms1]])
    return label_line(start, end, text, ax, **kwargs)

def label_line(start, end, text, ax, buff=0, lat_offset=0, **kwargs):
    t_start = ax.transData.transform_point(start)
    t_end = ax.transData.transform_point(end)
    t_lv = t_end - t_start
    t_lv = t_lv/np.sqrt(np.sum(t_lv**2))
    t_center_pt = np.mean(np.array((t_start, t_end)),
                          axis=0)
    ov = np.array((1, -t_lv[0]/t_lv[1]))
    ov = ov/np.sqrt(np.sum(ov**2))
    ang = -u.vector_angle(t_lv, (1,0), degrees=True)
    if ang < -90:
        ang = -(180 - ang)
        ov = -ov
    t_txt_pt = t_center_pt + ov*buff
    txt_pt = ax.transData.inverted().transform_point(t_txt_pt)
    ax.text(txt_pt[0], txt_pt[1], text, 
            rotation=ang, verticalalignment='bottom',
            horizontalalignment='center', rotation_mode='anchor', **kwargs)
    return ax

def plot_glm_coeffs(coeffs, ps, pcoeffs=(1, 2), p_thr=.05, style=(), ax=None, 
                    show=False, xlabel='', ylabel='', ns_alpha=.2, 
                    legend=True, legend_labels=('ns', 'coeff_1', 'coeff_2', 
                                                'both'),
                    colors=None):
    p_thr = p_thr/2
    coeffs = coeffs[:, pcoeffs]
    ps = ps[:, pcoeffs]
    f_sig_o = ps[:, 0] < p_thr
    s_sig_o = ps[:, 1] < p_thr
    f_sig = np.logical_and(f_sig_o, np.logical_not(s_sig_o))
    s_sig = np.logical_and(s_sig_o, np.logical_not(f_sig_o))
    b_sig = np.logical_and(f_sig_o, s_sig_o)
    n_sig = np.logical_not(b_sig)
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        if colors is None:
            colors = (None,)*4
        ax.plot(coeffs[n_sig, 0], coeffs[n_sig, 1], 'o', alpha=ns_alpha, 
                label=legend_labels[0], color=colors[0])
        ax.plot(coeffs[f_sig, 0], coeffs[f_sig, 1], 'o', label=legend_labels[1],
                color=colors[1])
        ax.plot(coeffs[s_sig, 0], coeffs[s_sig, 1], 'o', label=legend_labels[2],
                color=colors[2])
        ax.plot(coeffs[b_sig, 0], coeffs[b_sig, 1], 'o', label=legend_labels[3],
                color=colors[3])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend(frameon=False)
    if show:
        plt.show(block=False)
    return ax

def _cent_selectivity(inds, ps, mags, p_thr, eps, central_func=np.nanmedian):
    ps = ps[inds]
    mags = mags[inds]
    mask = np.logical_and(np.logical_or(mags > eps, mags < -eps),
                          ps < p_thr)
    use_d = mags[mask]
    return central_func(use_d)
                            
def set_violin_color(vp, color):
    for b in vp['bodies']:
        b.set_facecolor(color)
        b.set_edgecolor(color)
    cm = vp.get('cmedians')
    if cm is not None:
        cm.set_color(color)

def visualize_simplex_2d(pts, ax=None, ax_labels=None, thr=1,
                         pt_col=(.7, .7, .7),
                         line_grey_col=(.6, .6, .6),
                         colors=None, bottom_x=.8,
                         bottom_y=-1.1, top_x=.35, top_y=1,
                         legend=False, plot_mean=False,
                         plot_bounds=True,
                         plot_outline=False,
                         ms=2,
                         outline_factor=1.5,
                         **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
        # ax.set_aspect('equal')
    if colors is None:
        colors = (None,)*pts.shape[1]
    if ax_labels is None:
        ax_labels = ('',)*pts.shape[1]
    if plot_bounds:
        ax.plot([-1, 0], [-1, 1], color=line_grey_col)
        ax.plot([0, 1], [1, -1], color=line_grey_col)
        ax.plot([-1, 1], [-1, -1], color=line_grey_col)
    
    pts_x = pts[:, 1] - pts[:, 0]
    pts_y = pts[:, 2] - (pts[:, 0] + pts[:, 1])
    if plot_outline:
        ax.plot(pts_x, pts_y, 'o', color='k', ms=outline_factor*ms,
                **kwargs)
    ax.plot(pts_x, pts_y, 'o', color=pt_col, ms=ms, **kwargs)
    if plot_mean:
        pt_mu = np.mean(pts, axis=0, keepdims=True)
        pt_mu_x = pt_mu[:, 1] - pt_mu[:, 0]
        pt_mu_y = pt_mu[:, 2] - (pt_mu[:, 0] + pt_mu[:, 1])
        ax.plot(pt_mu_x, pt_mu_y, 'o', color=pt_col, ms=2*ms, **kwargs)
        
    
    for i in range(pts.shape[1]):
        mask = pts[:, i] > thr
        ax.plot(pts_x[mask], pts_y[mask], 'o', color=colors[i],
                label=ax_labels[i], **kwargs)
    if legend:
        ax.legend(frameon=False)
    clean_plot(ax, 1)
    clean_plot_bottom(ax, 0)
    ax.text(bottom_x, bottom_y, ax_labels[1], verticalalignment='top',
            horizontalalignment='center')
    ax.text(-bottom_x, bottom_y, ax_labels[0], verticalalignment='top',
            horizontalalignment='center')
    ax.text(top_x, top_y, ax_labels[2], verticalalignment='top',
            horizontalalignment='center', rotation=-60)
    ax.set_aspect('equal')
    return ax
        
def violinplot(vp_seq, positions, ax=None, color=None, label=[''],
               markerstyles=None, markersize=5, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    if color is None:
        color = (None,)*len(vp_seq)
    for i, vps in enumerate(vp_seq):
        p = ax.violinplot(vps, positions=[positions[i]],
                          **kwargs)
        if markerstyles is not None:
            ax.plot([positions[i]], [np.median(vps)],
                    marker=markerstyles[i],
                    markersize=markersize,
                    color=color[i])
        set_violin_color(p, color[i])
        
def clean_plot_bottom(ax, keeplabels=False):
    ax.spines['bottom'].set_visible(False)
    ax.xaxis.set_tick_params(size=0)
    if not keeplabels:
        plt.setp(ax.get_xticklabels(), visible=False)

def remove_ticks_3d(ax):
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])
    return ax

def add_x_background(beg, end, ax, color=(.1, .1, .1), alpha=.1, zorder=-1,
                     **kwargs):
    y_low, y_high = ax.get_ylim()
    rect = patches.Rectangle((beg, y_low), end - beg, (y_high - y_low),
                             facecolor=color, alpha=alpha,
                             zorder=zorder, **kwargs)
    ax.add_patch(rect)

def set_3d_background(ax, background_color=(1, 1, 1, 1), line_color=None):
    ax.w_xaxis.set_pane_color(background_color)
    ax.w_yaxis.set_pane_color(background_color)
    ax.w_zaxis.set_pane_color(background_color)
    if line_color is not None:
        ax.xaxis.pane.set_edgecolor(line_color)
        ax.yaxis.pane.set_edgecolor(line_color)
        ax.zaxis.pane.set_edgecolor(line_color)
    return ax
        
def make_3d_bars(ax, center=(0, 0, 0), bar_len=.1, bar_wid=.7):
    ax.plot([center[0], center[0] + bar_len],
            [center[1], center[1]],
            [center[2], center[2]], color='k', linewidth=bar_wid)
    ax.plot([center[0], center[0]],
            [center[1], center[1] + bar_len],
            [center[2], center[2]], color='k', linewidth=bar_wid)
    ax.plot([center[0], center[0]],
            [center[1], center[1]],
            [center[2], center[2] + bar_len], color='k', linewidth=bar_wid)
    ax.set_axis_off()
        
def clean_3d_plot(ax):
    pc = (1., 1., 1., 0.)
    ax.xaxis.set_pane_color(pc)
    ax.yaxis.set_pane_color(pc)
    ax.zaxis.set_pane_color(pc)
    ax.xaxis._axinfo['grid']['color'] = pc
    ax.yaxis._axinfo['grid']['color'] = pc
    ax.zaxis._axinfo['grid']['color'] = pc
        
def clean_plot(ax, i, max_i=None, ticks=True, spines=True, horiz=True):
    if spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if horiz:
        if i > 0:
            if spines:
                ax.spines['left'].set_visible(False)
            if ticks:
                plt.setp(ax.get_yticklabels(), visible=False)
                ax.yaxis.set_tick_params(size=0)
    else:
        if max_i is not None and i < max_i:
            if ticks:
                plt.setp(ax.get_xticklabels(), visible=False)
                ax.xaxis.set_tick_params(size=0)

def digitize_vars(xs, ys, n_bins=10, cent_func=np.mean, eps=1e-10,
                  use_max=None, use_min=None, ret_all_y=True):
    if use_max is None:
        use_max = np.nanmax(xs)
    if use_min is None:
        use_min = np.nanmin(xs)
    bins = np.linspace(use_min, use_max + eps, n_bins + 1)
    bs = np.digitize(xs, bins)
    u_bs = np.unique(bs)
    u_bs = u_bs[:n_bins]
    u_bs = u_bs[u_bs > 0]
    if ret_all_y:
        y_cents = np.zeros(len(u_bs), dtype=object)
    else:
        y_cents = np.zeros(len(u_bs))
    x_cents = np.zeros(len(u_bs))
    for i, b in enumerate(u_bs):
        mask = bs == b
        x_cents[i] = cent_func(xs[mask])
        if ret_all_y:
            y_cents[i] = ys[mask]
        else:
            y_cents[i] = cent_func(ys[mask])
    return x_cents, y_cents
                
def plot_scatter_average(xs, ys, *args, ax=None, n_bins=10, cent_func=np.mean,
                         eps=1e-10, use_max=None, use_min=None, **kwargs):
    if ax is None:
        f, ax = plt.subplots(1, 1)
    x_cs, y_cs = digitize_vars(xs, ys, n_bins=n_bins, cent_func=cent_func,
                               eps=eps, use_max=use_max, use_min=use_min,
                               ret_all_y=True)
    out = plot_trace_werr(x_cs, y_cs, *args, jagged=True, ax=ax, **kwargs)
    return out    
                
def make_xaxis_scale_bar(ax, magnitude=None, double=True, anchor=0, bottom=True,
                         true_scale=False, label='', text_buff=.22):
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    if magnitude is None:
        ext = np.abs(xl[1] - xl[0])
        if double:
            magnitude = ext/4
        else:
            magnitude = ext/2
    if double:
        new_ticks = [-magnitude + anchor, anchor, anchor + magnitude]
    else:
        new_ticks = [anchor, anchor + magnitude]
    ax.set_xticks(new_ticks)
    if bottom:
        ax.hlines(yl[0], new_ticks[0], new_ticks[-1], color='k')
        ax.spines['bottom'].set_visible(False)
        y_pt = yl[0]
    else:
        ax.hlines(yl[1], new_ticks[0], new_ticks[-1], color='k')
        ax.spine['top'].set_visible(False)
        y_pt = yl[1]
    if len(label) > 0:
        x_pt = new_ticks[0] + (new_ticks[-1] - new_ticks[0])/2
        disp_pt = ax.transData.transform((x_pt, y_pt))
        txt_pt = ax.transAxes.inverted().transform(disp_pt)
        ax.text(txt_pt[0], txt_pt[1] - text_buff, label, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='top')
    ax.set_ylim(yl)
    
def get_corr_conf95(as_list, bs_list, n_boots=1000, func=np.corrcoef,
                    rm_nan=False, confounders=None):
    if confounders is None:
        use_confounders = False
        confounders = np.zeros((len(as_list), 1), dtype=bool)
    else:
        use_confounders = True
    if rm_nan:
        mask = np.logical_or(np.isnan(as_list),
                             np.isnan(bs_list))
        mask = np.logical_or(mask, np.any(np.isnan(confounders), axis=1))
        mask = np.logical_not(mask)
        as_list = as_list[mask]
        bs_list = bs_list[mask]
        confounders = confounders[mask]
    if use_confounders:
        f = lambda x: na.partial_correlation(x[:, 0], x[:, 1], x[:, 2:])
        inp = np.stack((as_list, bs_list), axis=1)
        if len(confounders.shape) == 1:
            confounders = np.expand_dims(confounders, 1)
        inp = np.concatenate((inp, confounders), axis=1)
    else:
        f = lambda x: func(x[:, 0], x[:, 1])[1,0]
        inp = np.stack((as_list, bs_list), axis=1)
    cc = u.bootstrap_list(inp, f, n=n_boots)
    cent = f(inp)
    interv = conf95_interval(cc)
    upper = cent + interv[0, 0]
    lower = cent + interv[1, 0]
    return cent, lower, upper, cc

def _get_pval(bs, comp_pt, comparator=np.greater):
    p = 1 - np.sum(comparator(bs, comp_pt))/len(bs)
    if p == 0:
        pt = 'p < {}'.format(1/len(bs))
    else:
        pt = 'p = {}'.format(p)
    return pt

def _greater_than_zero(x):
    return np.greater(x, 0)

def print_sig_fraction(boot_arr, subj, text, func=_greater_than_zero, p_thr=.05,
                       two_sided=True):
    n_samps = boot_arr.shape[1]
    n_feats = boot_arr.shape[0]
    if two_sided:
        factor = 2
    else:
        factor = 1
    p_comp = 1 - p_thr/factor
    prop = np.sum(func(boot_arr), axis=1)/n_samps
    n_sig = np.sum(prop > p_comp)
    s = '{} {}: {:0.2f}% ({}/{})'.format(subj, text, 100*n_sig/n_feats, n_sig,
                                         n_feats)
    print(s)
    return s   

def print_corr_conf95(as_list, bs_list, subj, text, n_boots=1000, func=np.corrcoef,
                      round_result=2, confounders=None, comp_pt=0,
                      comparator=np.greater):
    cent, lower, upper, cc = get_corr_conf95(as_list, bs_list, n_boots=n_boots,
                                             func=func, confounders=confounders)
    pt = _get_pval(cc, comp_pt, comparator)
    s = '{} {}: {:0.2f} [{:0.2f}, {:0.2f}], {}'.format(subj, text, cent,
                                                       lower, upper, pt)
    print(s)
    return s

def _get_sem_from_bound(err, use_lower=True, p_thr=.05):
    factor = sts.norm(0, 1).ppf(1 - p_thr/2)
    ind = 1 if use_lower else 0
    sem = err[ind]/factor
    return sem    

def print_proportion(num, denom, subj, text, *args):
    perc = 100*num/denom
    rep_str = '{:0.2f}% ({}/{})'.format(perc, num, denom)
    s = text.format(subj, rep_str, *args)
    print(s)
    return s

def print_mean_conf95(bs_list, subj, text, n_boots=1000, func=np.nanmean,
                      preboot=False, round_result=2, comp_pt=0,
                      comparator=np.greater, supply_err=False,
                      diff=False):
    if supply_err:
        factor = sts.norm(0, 1).ppf(1 - .025)
        if diff:
            (cent1, err1), (cent2, err2) = bs_list
            sem1 = _get_sem_from_bound(err1)
            sem2 = _get_sem_from_bound(err2)
            cent = cent1 - cent2
            sem = np.sqrt(sem1**2 + sem2**2)
            upper = cent + factor*sem
            lower = cent - factor*sem
        else:
            cent, err = bs_list
            upper = cent + err[0]
            lower = cent + err[1]
            sem = err[1]/factor
        p = sts.norm(cent, np.sqrt(sem**2)).cdf(comp_pt)
        pt = 'p = {}'.format(p)
    else:
        bs_list = np.array(bs_list)
        if  preboot:
            cent = func(bs_list)
            bs = bs_list
        else:
            cent = func(bs_list)
            bs = u.bootstrap_list(bs_list, func, n_boots)
        interv = conf95_interval(bs)
        upper = cent + interv[0, 0]
        lower = cent + interv[1, 0]
        pt = _get_pval(bs, comp_pt, comparator)
    s = '{} {}: {:0.2f} [{:0.2f}, {:0.2f}], {}'.format(subj, text, cent,
                                                       lower, upper, pt)
    print(s)
    return s

def print_diff_conf95(b_list, a_list, subj, text, n_boots=1000,
                      func=np.nanmean, preboot=False, round_result=2,
                      comp_pt=0, comparator=np.greater):
    if preboot:
        b = b_list
        a = a_list
        diff = b - a
    else:
        diff = u.bootstrap_diff(b_list, a_list, func, n_boots)
    interv = conf95_interval(diff)
    cent = func(diff)
    upper = cent + interv[0, 0]
    lower = cent + interv[1, 0]        
    pt = _get_pval(diff, comp_pt, comparator)
    s = '{} {}: {:0.2f} [{:0.2f}, {:0.2f}], {}'.format(subj, text, cent, lower,
                                                       upper, pt)
    print(s)
    return s

def make_yaxis_scale_bar(ax, magnitude=None, double=True, anchor=0, left=True,
                         label='', text_buff=.15):
    xl = ax.get_xlim()
    yl = ax.get_ylim()
    if magnitude is None:
        ext = np.abs(yl[1] - yl[0])
        if double:
            magnitude = ext/4
        else:
            magnitude = ext/2
    if double:
        new_ticks = [-magnitude + anchor, anchor, anchor + magnitude]
    else:
        new_ticks = [anchor, anchor + magnitude]
    ax.set_yticks(new_ticks)
    if left:
        ax.vlines(xl[0], new_ticks[0], new_ticks[-1], color='k')
        ax.spines['left'].set_visible(False)
        x_pt = xl[0]
    else:
        ax.hlines(xl[1], new_ticks[0], new_ticks[-1], color='k')
        ax.spine['right'].set_visible(False)
        x_pt = xl[1]
    if len(label) > 0:
        y_pt = new_ticks[0] + (new_ticks[-1] - new_ticks[0])/2
        disp_pt = ax.transData.transform((x_pt, y_pt))
        txt_pt = ax.transAxes.inverted().transform(disp_pt)
        ax.text(txt_pt[0] - text_buff, txt_pt[1], label, transform=ax.transAxes,
                horizontalalignment='center', verticalalignment='center',
                rotation=90)
    ax.set_xlim(xl)

def plot_horiz_conf_interval(y, x_distr, ax, color=None,
                             error_func=conf95_interval,
                             central_tend=np.nanmedian, **kwargs):
    if len(x_distr.shape) < 2:
        x_distr = x_distr.reshape((-1, 1))
    l = plot_trace_werr(x_distr, np.array([y]), central_tendency=central_tend,
                        error_func=error_func, ax=ax, color=color, fill=False,
                        **kwargs)
    col = l[0].get_color()
    l = ax.plot(central_tend(x_distr), [y], '|', color=col)
    return l

def plot_conf_interval(x, y_distr, ax, color=None, error_func=conf95_interval,
                       central_tend=np.nanmedian, **kwargs):
    if len(y_distr.shape) < 2:
        y_distr = y_distr.reshape((-1, 1))
    l = plot_trace_werr(np.array([x]), y_distr, central_tendency=central_tend,
                        error_func=error_func, ax=ax, color=color, fill=False,
                        **kwargs)
    col = l[0].get_color()
    l = ax.plot([x], central_tend(y_distr), '_', color=col)
    return l

def _preprocess_glm(coeffs, ps, subgroups=None, p_thr=.05, eps=None,
                    repl_with=0):
    if subgroups is not None:
        all_use = np.concatenate(subgroups)
    else:
        all_use = np.arange(coeffs.shape[1])
        subgroups = (all_use,)
    ps = ps[:, all_use]
    coeffs = coeffs[:, all_use]
    sig_filt = np.any(ps < p_thr, axis=1)
    coeffs[ps > p_thr] = repl_with
    use_pop = coeffs[sig_filt]
    use_ps = ps[sig_filt]
    if eps is not None:
        mask = np.any(use_pop > eps, axis=1)
        use_pop = use_pop[mask]
    return use_pop, subgroups, all_use, use_ps

def format_glm_labels(labels, label_dict, subgroups=None, separator='-'):
    new_labels = []
    for l in labels:
        nl = separator.join(('{}',)*len(l))
        nl = nl.format(*list(label_dict[l_i] for l_i in l))
        new_labels.append(nl)
    if subgroups is not None:
        grouped_labels = []
        for sg in subgroups:
            grouped_labels.append(tuple(new_labels[sg_i] for sg_i in sg))
        new_labels = grouped_labels
    return new_labels

@ax_adder
def scatter_error(x_data, y_data, x_sem, y_sem, ax=None, elinewidth=1, **kwargs):
    ax.plot(x_data, y_data, 'o', **kwargs)
    ax.errorbar(x_data, y_data, fmt='none', xerr=x_sem, yerr=y_sem,
                elinewidth=elinewidth, **kwargs)

def plot_stanglm_selectivity_scatter(ms, params, labels, ax=None, figsize=None,
                                     time_ind=None, param_funcs=None,
                                     check_valid=False, format_labels=True,
                                     link_string=None, conn=True,
                                     conn_alpha=.3):
    if ax is None:
        f = plt.figure(figsize=figsize)
        ax = f.add_subplot(1,1,1)
    if param_funcs is None:
        param_funcs = (lambda x, y: x[y],)*len(params)
        params = tuple((p,) for p in params)
    all_pairs = np.zeros((len(ms), len(ms[0]), len(params)))
    for i, m in enumerate(ms):
        if time_ind is not None:
            m = (m[time_ind],)
        n_pairs = np.zeros((len(m), len(params)))
        for j, t in enumerate(m):
            if (t is not None and ((not check_valid)
                or (check_valid and u.stan_model_valid(t)))):
                pm = np.mean(t.get_posterior_mean(), axis=1)
                for k, p in enumerate(params):
                    n_pairs[j, k] = param_funcs[k](pm, *p)
            else:
                n_pairs[j] = np.nan
        all_pairs[i] = n_pairs
        l = ax.plot(n_pairs[:, 0], n_pairs[:, 1], 'o')
        if conn:
            col = l[0].get_color()
            ax.plot(n_pairs[:, 0], n_pairs[:, 1], alpha=conn_alpha, color=col,
                    linewidth=3)
    if format_labels:
        ax.set_xlabel(format_lm_label(params[0], labels,
                                      link_string=link_string))
        ax.set_ylabel(format_lm_label(params[1], labels,
                                      link_string=link_string))
    else:
        ax.set_xlabel(list(labels[p] for p in params[0]))
        ax.set_ylabel(list(labels[p] for p in params[1]))
    return ax, all_pairs

def format_lm_label(params, labels, second_only=True, link_string=None,
                    cat_l_joiner='-', interaction_joiner=' x '):
    if link_string is None:
        link_string = ' '.join(('{}',)*len(params))
    ls = []
    for i, p in enumerate(params):
        lg = []
        for l_groups in labels[p]:
            if second_only:
                l = l_groups[1]
            else:
                l = cat_l_joiner.join(l_groups)
            lg.append(l)
        lg_l = interaction_joiner.join(lg)
        ls.append(lg_l)
    full_l = link_string.format(*ls)
    return full_l            

def plot_glm_pop_selectivity_prop(coeffs, ps, subgroups=None, p_thr=.05,
                                  boots=10000, figsize=None, colors=None,
                                  eps=.001, group_xlabels=None, ylabel=None,
                                  group_term_labels=None, fig=None,
                                  label_rotation='horizontal'):
    use_pop, subgroups, all_use, ps = _preprocess_glm(coeffs, ps, subgroups,
                                                      p_thr)
    use_pop = np.abs(use_pop)
    if fig is None:
        f = plt.figure(figsize=figsize)
        rows = 1
        i_buff = 1
        share_x = (None,)*len(subgroups)
    else:
        f = fig
        rows = 2
        share_x = f.get_axes()
        i_buff = len(share_x) + 1
    if colors is None:
        colors = (None,)*len(subgroups)
    share_ax = None
    for i, sg in enumerate(subgroups):
        ax = f.add_subplot(rows, len(subgroups), i + i_buff, sharey=share_ax,
                           sharex=share_x[i])
        share_ax = ax
        sg_means = np.zeros(len(sg))
        for ind, j in enumerate(sg):
            sgm = use_pop[:, j]
            psm = ps[:, j]
            distr_func = lambda x: np.mean(np.logical_and(sgm[x] > eps,
                                                          psm[x] < p_thr))
            inds = np.arange(len(sgm))
            prop_distr = u.bootstrap_list(inds, distr_func, n=boots)
            plot_conf_interval(j, prop_distr, ax, color=colors[i])
            sg_means[ind] = np.mean(prop_distr)
        if group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        clean_plot(ax, i, ticks=True, spines=True)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
    return f

def plot_glm_coeff_tc(xs, coeffs, ps, subgroups=None, p_thr=.05, boots=1000,
                      figsize=None, colors=None, eps=.001, group_xlabels=None,
                      ylabel=None, group_term_labels=None, combined_fig=False,
                      axs=None, f=None, use_abs=True, diff=False):
    coeff_arr = np.zeros((boots, len(xs), coeffs.shape[-1]))
    nm_ax = lambda x: np.nanmean(x, axis=0)
    for i, t in enumerate(xs):
        out = _preprocess_glm(coeffs[:, i], ps[:, i], subgroups=None,
                              p_thr=p_thr, eps=eps, repl_with=np.nan)
        up, _, all_use, use_ps = out
        coeff_arr[:, i] = u.bootstrap_list(up, nm_ax, n=boots,
                                           out_shape=up.shape[1:])
        
    if axs is None:
        f, axs = plt.subplots(1, len(subgroups), figsize=figsize)
        
    for i, sg in enumerate(subgroups):
        ax = axs[i]
        for j, sg_j in enumerate(sg):
            if diff:
                sg_arr = coeff_arr[..., sg_j[0]] - coeff_arr[..., sg_j[1]]
            else:
                sg_arr = coeff_arr[:, :, sg_j]
            if group_term_labels is not None:
                label = group_term_labels[i][j]
            else:
                label = ''
            if colors is not None:
                color = colors[i][j]
            else:
                color = None
            plot_trace_werr(xs, sg_arr, ax=ax, conf95=True, label=label,
                            color=color)
    return axs         

def plot_glm_pop_selectivity_mag(coeffs, ps, subgroups=None, p_thr=.05,
                                 boots=1000, figsize=None, colors=None,
                                 eps=.001, group_xlabels=None, ylabel=None,
                                 group_term_labels=None, combined_fig=False,
                                 label_rotation='horizontal', group_test=False,
                                 group_test_n=5000, axs=None, f=None,
                                 use_abs=True):
    out = _preprocess_glm(coeffs, ps, subgroups, p_thr)
    use_pop, subgroups, all_use, ps = out
    if use_abs:
        use_pop = np.abs(use_pop)
    if combined_fig:
        rows = 2
    else:
        rows = 1
    if axs is None:
        f = plt.figure(figsize=figsize)
    if colors is None:
        colors = (None,)*len(subgroups)
    share_ax = None
    coeff_groups = {}
    for i, sg in enumerate(subgroups):
        if axs is None:
            ax = f.add_subplot(rows, len(subgroups), i + 1, sharey=share_ax)
        else:
            ax = axs[i]
        ax.set_yticks([0, .5, 1])
        share_ax = ax
        sg_mags = ()
        sg_violin = ()
        for j in sg:
            sgm = use_pop[:, j]
            psm = ps[:, j]
            mask = np.logical_and(np.logical_or(sgm > eps, sgm < -eps),
                                  psm < p_thr)
            pop_mags = tuple(sgm[mask])
            if len(pop_mags) > 0:
                sg_mags = sg_mags + (pop_mags,)
                sg_violin = sg_violin + (j,)
            distr_func = lambda x: _cent_selectivity(x, psm, sgm, p_thr,
                                                     eps, np.nanmedian)
            inds = np.arange(len(sgm))
            sg_cent_dist = u.bootstrap_list(inds, distr_func, n=boots)
            plot_conf_interval(j, sg_cent_dist, ax, color=colors[i])
        
        p = ax.violinplot(sg_mags, positions=sg_violin, showmedians=False,
                          showextrema=False)
        set_violin_color(p, colors[i])
        coeff_groups[(i, sg)] = np.concatenate(sg_mags)
        if (not combined_fig) and group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        clean_plot(ax, i, ticks=True, spines=True)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if (not combined_fig) and group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
        if combined_fig:
            plt.setp(ax.get_xticklabels(), visible=False)
    if group_test:
        test_ps = {}
        for a, b in it.combinations(coeff_groups.keys(), 2):
            p = u.bootstrap_test(coeff_groups[a], coeff_groups[b], np.nanmean,
                                 n=group_test_n)
            test_ps[(a,b)] = p
        out = (f, test_ps)
    else:
        out = f
    return out

def plot_glm_indiv_selectivity(coeffs, ps, subgroups=None, p_thr=.05,
                               sort=True, figsize=None, cmap='RdBu',
                               group_xlabels=None, ylabel=None,
                               group_term_labels=None, sep_cb=False,
                               cb_size=(1.2, .8), label_rotation='horizontal',
                               cb_label='', remove_nans=True, cb_wid=8,
                               axs=None, f=None):
    if remove_nans:
        neur_mask = np.logical_not(np.sum(np.isnan(coeffs), axis=1) > 0)
        coeffs = coeffs[neur_mask]
        ps = ps[neur_mask]
    use_pop, subgroups, all_use, ps = _preprocess_glm(coeffs, ps, subgroups,
                                                      p_thr)
    abs_coeffs = np.abs(use_pop)
    maxterm_order = np.argsort(np.argmax(abs_coeffs, axis=1))
    sorted_pop = use_pop[maxterm_order]
    vmax = np.max(abs_coeffs)
    vmin = -vmax
    if axs is None:
        f = plt.figure(figsize=figsize)
    yvals = pcolormesh_axes(np.arange(use_pop.shape[0]) + 1, use_pop.shape[0])
    ax_list = ()
    for i, sg in enumerate(subgroups):
        xvals = pcolormesh_axes(sg, len(sg))
        submap = sorted_pop[:, sg]
        if axs is None:
            ax = f.add_subplot(1, len(subgroups), i + 1)
        else:
            ax = axs[i]
        ax.invert_yaxis()
        p = ax.pcolormesh(xvals, yvals, submap, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_list = ax_list + (ax,)
        if group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        clean_plot(ax, i, ticks=True, spines=False)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
    if sep_cb or axs is not None:
        f2 = plt.figure(figsize=cb_size)
        ax_list = None
        ret = (f, f2)
    else:
        f2 = f
        ret = (f,)
    colbar = f2.colorbar(p, ax=ax_list, aspect=cb_wid)
    colbar.set_ticks([round(vmin, 1) + .1, 0, round(vmax, 1) - .1])
    colbar.set_label(cb_label)
    return ret
