
import numpy as np
import matplotlib.pyplot as plt

import general.plotting_styles as gps
import general.utility as u
import os
import re

def split_gridspec(n_rows, start, end, spacing):
    free_space = (end - start) - (n_rows - 1)*spacing
    hei = np.floor(free_space/n_rows)
    out = np.zeros((n_rows, 2))
    for i in range(n_rows):
        top = start + i*hei + i*spacing
        bottom = top + hei
        out[i] = (top, bottom)
    return out.astype(int)

def make_mxn_gridspec(gs, n_rows, n_cols, top, bottom, left, right,
                      v_space, h_space):
    v_bounds = split_gridspec(n_rows, top, bottom, v_space)
    h_bounds = split_gridspec(n_cols, left, right, h_space)
    gss = np.zeros((v_bounds.shape[0], h_bounds.shape[0]), dtype=object)
    for ind in u.make_array_ind_iterator(gss.shape):
        t, b = v_bounds[ind[0]]
        l, r = h_bounds[ind[1]]
        gss[ind] = gs[t:b, l:r]
    return gss

class Figure:

    def __init__(self, fsize, params, data=None, bf=None,
                 colors=None, style_func=gps.set_paper_style,
                 find_panel_keys=True):
        if style_func is not None:
            style_func(colors)
        self.f = plt.figure(figsize=fsize)
        self.gs = self.f.add_gridspec(100, 100)
        self.params = params
        if bf is None:
            self.bf = params.get('basefolder')
        else:
            self.bf = bf
        if data is None:
            self.data = {}
        else:
            self.data = data
        self.make_gss()
        if find_panel_keys:
            self.panel_keys = tuple(m for m in dir(self)
                                    if re.match('panel_.*', m) is not None)

    def make_panels(self, panel_keys=None):
        if panel_keys is None:
            panel_keys = self.panel_keys
        list(getattr(self, pk)() for pk in panel_keys)
        
    def make_gss(self):
        pass

    def get_axs(self, grids, sharex=None, sharey=None, plot_3ds=None,
                all_3d=False, squeeze=False, exp_dim=0, share_ax_x=None,
                share_ax_y=None, **all_ax_kwargs):
        grid_arr = np.array(grids)
        if len(grid_arr.shape) == 1:
            grid_arr = np.expand_dims(grid_arr, exp_dim)
        ax_arr = np.zeros_like(grid_arr, dtype=object)
        share_ax = None
        for i, ind in enumerate(u.make_array_ind_iterator(grid_arr.shape)):
            ax_kwarg = dict()
            ax_kwarg.update(all_ax_kwargs)
            if plot_3ds is not None and plot_3ds[ind] or all_3d:
                ax_kwarg['projection'] = '3d'
            if share_ax_x is not None:
                ax_kwarg['sharex'] = share_ax_x
            if share_ax_y is not None:
                ax_kwarg['sharey'] = share_ax_y
            if i > 0 and sharex is not None:
                ax_kwarg['sharex'] = share_ax
            if i > 0 and sharey is not None:
                ax_kwarg['sharey'] = share_ax
            ax_arr[ind] = self.f.add_subplot(grid_arr[ind], **ax_kwarg)
            if i == 0 and sharex == 'all':
                share_ax_x = ax_arr[ind]
            if i == 0 and sharey == 'all':
                share_ax_y = ax_arr[ind]
            if ind[0] == 0 and sharex == 'horizontal':
                share_ax_x = ax_arr[ind]
            if ind[0] == 0 and sharey == 'horizontal':
                share_ax_y = ax_arr[ind]
            if ind[1] == 0 and sharex == 'vertical':
                share_ax_x = ax_arr[ind]
            if ind[1] == 0 and sharey == 'vertical':
                share_ax_y = ax_arr[ind]
                
        if squeeze:
            ax_arr = np.squeeze(ax_arr)
        return ax_arr
    
    def generate(self, panels=None):
        pass

    def save(self, file_=None, bbox_inches='tight', transparent=True):
        if file_ is None:
            file_ = self.fig_key
        fname = os.path.join(self.bf, file_)
        self.f.savefig(fname, bbox_inches=bbox_inches, transparent=transparent)
        
    def get_data(self):
        return self.data
    
