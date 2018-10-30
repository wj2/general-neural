
import numpy as np
import matplotlib.pyplot as plt
import general.utility as u

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
        
def plot_trace_werr(xs_orig, dat, color=None, label='', show=False, title='', 
                    errorbar=True, alpha=.5, ax=None, error_func=sem,
                    style=(), central_tendency=np.nanmean, legend=True,
                    fill=True, log_x=False, log_y=False, **kwargs):
    with plt.style.context(style):
        if ax is None:
            f = plt.figure()
            ax = f.add_subplot(1,1,1)
        if log_x:
            ax.set_xscale('log')
        if log_y:
            ax.set_yscale('log')
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
        trl = ax.plot(xs, tr, label=label, color=color, **kwargs)
        if len(dat.shape) > 1:
            if color is None:
                color = trl[0].get_color()
            if fill:
                ax.fill_between(xs, tr+er[1, :], tr+er[0, :], color=color, 
                                alpha=alpha)
            else:
                ax.errorbar(xs, tr, (-er[1, :], er[0, :]), color=color,
                            **kwargs)
        if len(xs_orig.shape) > 1:
            if fill:
                ax.fill_betweenx(tr, xs+xs_er[1, :], xs+xs_er[0, :], 
                                 color=color, alpha=alpha)
            else:
                ax.errorbar(xs, tr, yerr=None, xerr=(-er[1, :], er[0, :]),
                            color=color, **kwargs)
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

def label_line(start, end, text, ax, buff=.1, lat_offset=0, **kwargs):
    lv = end - start
    lv = lv/np.sqrt(np.sum(lv**2))
    center_pt = np.mean(np.array((start, end)), axis=0)
    orth_ang = ax.transData.transform_angles(np.array((90,)),
                                             center_pt.reshape((1,2)))[0]
    
    ov = np.array((1, -lv[0]/lv[1]))
    ov = ov/np.sqrt(np.sum(ov**2))
    center_pt = center_pt + lat_offset*lv
    ang = -u.vector_angle(lv, (1,0), degrees=True)
    if ang < -90:
        ang = -(180 - ang)
        ov = -ov
    txt_pt = center_pt + ov*buff
    trans_ang = ax.transData.transform_angles(np.array((ang,)),
                                              txt_pt.reshape((1,2)))[0]
    ax.text(txt_pt[0], txt_pt[1], text, 
            rotation=trans_ang, verticalalignment='center',
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
    use_d = mags[np.logical_and(ps < p_thr, mags > eps)]
    return central_func(use_d)
                            
def _set_violin_color(vp, color):
    for b in vp['bodies']:
        b.set_facecolor(color)
        b.set_edgecolor(color)

def _clean_plot(ax, i, ticks=True, spines=True):
    if spines:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    if i > 0:
        if spines:
            ax.spines['left'].set_visible(False)
        if ticks:
            plt.setp(ax.get_yticklabels(), visible=False)
            ax.yaxis.set_tick_params(size=0)
        
def plot_conf_interval(x, y_distr, ax, color=None, error_func=conf95_interval,
                       central_tend=np.nanmedian):
    if len(y_distr.shape) < 2:
        y_distr = y_distr.reshape((-1, 1))
    plot_trace_werr(np.array([x]), y_distr, central_tendency=central_tend,
                    error_func=error_func, ax=ax, color=color, fill=False)
    ax.plot([x], central_tend(y_distr), '_', color=color)
    return ax

def _preprocess_glm(coeffs, ps, subgroups=None, p_thr=.05, eps=None):
    if subgroups is not None:
        all_use = np.concatenate(subgroups)
    else:
        all_use = np.arange(coeffs.shape[1])
        subgroups = (all_use,)
    ps = ps[:, all_use]
    coeffs = coeffs[:, all_use]
    sig_filt = np.any(ps < p_thr, axis=1)
    use_pop = coeffs[sig_filt]
    use_ps = ps[sig_filt]
    if eps is not None:
        mask = np.any(use_pop > eps, axis=1)
        use_pop = use_pop[mask]
    return use_pop, subgroups, all_use, use_ps

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
                n_pairs[j] = np.nan # do I want to set all entries to nan? 
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
    return ax

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
                                  boots=1000, figsize=None, colors=None,
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
        for j in sg:
            sgm = use_pop[:, j]
            psm = ps[:, j]
            distr_func = lambda x: np.mean(np.logical_and(sgm[x] > eps,
                                                          psm[x] < p_thr))
            inds = np.arange(len(sgm))
            prop_distr = u.bootstrap_list(inds, distr_func, n=boots)
            plot_conf_interval(j, prop_distr, ax, color=colors[i])
        if group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        _clean_plot(ax, i, ticks=True, spines=True)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
    return f

def plot_glm_pop_selectivity_mag(coeffs, ps, subgroups=None, p_thr=.05,
                                 boots=1000, figsize=None, colors=None,
                                 eps=.001, group_xlabels=None, ylabel=None,
                                 group_term_labels=None, combined_fig=False,
                                 label_rotation='horizontal'):
    out = _preprocess_glm(coeffs, ps, subgroups, p_thr)
    use_pop, subgroups, all_use, ps = out
    use_pop = np.abs(use_pop)
    if combined_fig:
        rows = 2
    else:
        rows = 1
    f = plt.figure(figsize=figsize)
    if colors is None:
        colors = (None,)*len(subgroups)
    share_ax = None
    for i, sg in enumerate(subgroups):
        ax = f.add_subplot(rows, len(subgroups), i + 1, sharey=share_ax)
        share_ax = ax
        sg_mags = ()
        for j in sg:
            sgm = use_pop[:, j]
            psm = ps[:, j]
            pop_mags = tuple(sgm[np.logical_and(sgm > eps, psm < p_thr)])
            sg_mags = sg_mags + (pop_mags,)
            distr_func = lambda x: _cent_selectivity(x, psm, sgm, p_thr,
                                                     eps, np.nanmedian)
            inds = np.arange(len(sgm))
            sg_cent_dist = u.bootstrap_list(inds, distr_func, n=boots)
            plot_conf_interval(j, sg_cent_dist, ax, color=colors[i])
        p = ax.violinplot(sg_mags, positions=sg, showmedians=False,
                          showextrema=False)
        _set_violin_color(p, colors[i])
        if (not combined_fig) and group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        _clean_plot(ax, i, ticks=True, spines=True)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if (not combined_fig) and group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
        if combined_fig:
            plt.setp(ax.get_xticklabels(), visible=False)
    return f

def plot_glm_indiv_selectivity(coeffs, ps, subgroups=None, p_thr=.05,
                               sort=True, figsize=None, cmap='RdBu',
                               group_xlabels=None, ylabel=None,
                               group_term_labels=None, sep_cb=False,
                               cb_size=(.75, 1), label_rotation='horizontal',
                               cb_label=''):
    use_pop, subgroups, all_use, ps = _preprocess_glm(coeffs, ps, subgroups,
                                                      p_thr)
    abs_coeffs = np.abs(use_pop)
    maxterm_order = np.argsort(np.argmax(abs_coeffs, axis=1))
    sorted_pop = use_pop[maxterm_order]

    vmax = np.max(abs_coeffs)
    vmin = -vmax
    f = plt.figure(figsize=figsize)
    yvals = pcolormesh_axes(np.arange(use_pop.shape[0]) + 1, use_pop.shape[0])
    ax_list = ()
    for i, sg in enumerate(subgroups):
        xvals = pcolormesh_axes(sg, len(sg))
        submap = sorted_pop[:, sg]

        ax = f.add_subplot(1, len(subgroups), i + 1)
        ax.invert_yaxis()
        p = ax.pcolormesh(xvals, yvals, submap, cmap=cmap, vmin=vmin, vmax=vmax)
        ax_list = ax_list + (ax,)
        if group_term_labels is not None:
            ax.set_xticks(sg)
            ax.set_xticklabels(group_term_labels[i], rotation=label_rotation)
        _clean_plot(ax, i, ticks=True, spines=False)
        if i == 0 and ylabel is not None:
            ax.set_ylabel(ylabel)
        if group_xlabels is not None:
            ax.set_xlabel(group_xlabels[i])
    if sep_cb:
        f2 = plt.figure(figsize=cb_size)
        ax_list = None
        ret = (f, f2)
    else:
        f2 = f
        ret = (f,)
    colbar = f2.colorbar(p, ax=ax_list)
    colbar.set_ticks([round(vmin, 1), 0, round(vmax, 1)])
    colbar.set_label(cb_label)
    return ret
