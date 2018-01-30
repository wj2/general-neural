
import numpy as np
import matplotlib.pyplot as plt

def plot_single_units(xs, sus, labels, colors, style=(), show=False,
                      errorbar=True, alpha=.5, trial_color=(.8, .8, .8),
                      trial_style='dashed', trial_struct=None):
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
        for k in sus[0].keys():
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
            for i, c in enumerate(sus):
                ax = plot_trace_werr(xs, sus[i][k], colors[i], label=labels[i],
                                     show=False, errorbar=True, alpha=alpha,
                                     error_func=sem, title=k, ax=ax)
            if trial_struct is not None:
                plot_trial_structure(trial_struct['transitions'],
                                     trial_struct['labels'],
                                     color=trial_color,
                                     linestyle=trial_style,
                                     ax=ax)
    if show:
        plt.show(block=False)
    return f
                
def sem(dat, axis=0):
    err_1d = np.nanstd(dat, axis=axis)/np.sqrt(dat.shape[axis] - 1)
    err = np.vstack((err_1d, -err_1d))
    return err

def conf_interval(dat, axis=0, perc=95):
    lower = (100 - perc) / 2.
    upper = lower + perc
    lower_err = np.nanpercentile(dat, lower, axis=axis)
    upper_err = np.nanpercentile(dat, upper, axis=axis)
    err = np.vstack((upper_err - np.nanmean(dat, axis), 
                     lower_err - np.nanmean(dat, axis)))
    return err

def conf95_interval(dat, axis=0):
    return conf_interval(dat, axis=0, perc=95)

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

def pcolormesh_axes(axvals, val_len):
    if len(axvals) == val_len:
        diff = np.diff(axvals)[0]
        axvals_shift = axvals - diff/2
        axvals = np.append(axvals_shift, (axvals_shift[-1] + diff))
    return axvals

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
        
def plot_trace_werr(xs_orig, dat, color=None, label='', show=False, title='', 
                    errorbar=True, alpha=.5, ax=None, error_func=sem,
                    style=(), central_tendency=np.nanmean, legend=True,
                    linestyle=None, fill=True):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        if len(dat.shape) > 1:
            tr = central_tendency(dat, axis=0)
            er = error_func(dat, axis=0)
        else: 
            tr = dat
        if len(xs_orig.shape) > 1:
            xs_er = error_func(xs_orig, axis=0)
            xs = central_tendency(xs_orig, axis=0)
        else:
            xs = xs_orig
        trl = ax.plot(xs, tr, label=label, color=color, linestyle=linestyle)
        if len(dat.shape) > 1:
            if color is None:
                color = trl[0].get_color()
            if fill:
                ax.fill_between(xs, tr+er[1, :], tr+er[0, :], color=color, 
                                alpha=alpha)
            else:
                ax.errorbar(xs, tr, (-er[1, :], er[0, :]), color=color,
                            linestyle=linestyle)
        if len(xs_orig.shape) > 1:
            if fill:
                ax.fill_betweenx(tr, xs+xs_er[1, :], xs+xs_er[0, :], 
                                 color=color, alpha=alpha)
            else:
                ax.errorbar(xs, tr, yerr=None, xerr=(-er[1, :], er[0, :]),
                            color=color, linestyle=linestyle)
        ax.set_title(title)
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
                 normed=True):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1, 1, 1)
        hist = ax.hist(dat, bins=bins, color=color, alpha=alpha, 
                       histtype=histtype, label=label, normed=normed)
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

def plot_glm_coeffs(coeffs, ps, pcoeffs=(1, 2), p_thr=.05, style=(), ax=None, 
                    show=False, xlabel='', ylabel='', ns_alpha=.2, 
                    legend=True, legend_labels=('ns', 'coeff_1', 'coeff_2', 
                                                'both')):
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
        ax.plot(coeffs[n_sig, 0], coeffs[n_sig, 1], 'o', alpha=ns_alpha, 
                label=legend_labels[0])
        ax.plot(coeffs[f_sig, 0], coeffs[f_sig, 1], 'o', label=legend_labels[1])
        ax.plot(coeffs[s_sig, 0], coeffs[s_sig, 1], 'o', label=legend_labels[2])
        ax.plot(coeffs[b_sig, 0], coeffs[b_sig, 1], 'o', label=legend_labels[3])
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if legend:
            ax.legend()
    if show:
        plt.show(block=False)
    return ax
