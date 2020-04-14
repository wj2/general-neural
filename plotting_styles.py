
import numpy as np
from cycler import cycler
import matplotlib.pyplot as plt

nms_colors = np.array([(127,205,187),
                       (65,182,196),
                       (29,145,192),
                       (34,94,168),
                       (37,52,148),
                       (8,29,88)])/256

assignment_num_colors = np.array([(127,205,187),
                                  (125, 152, 112),
                                  (101, 101, 101),
                                  (114, 87, 82),
                                  (212, 223, 199),
                                  (135, 142, 136)])/256


def turn_off_max_figure():
    plt.rcParams.update({'figure.max_open_warning': 0})

def set_poster_style(colors):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('axes', prop_cycle=cycler('color', colors))
    plt.rcParams.update({'font.size': 16})
    plt.rcParams.update({'axes.labelsize':22})
    plt.rcParams.update({'xtick.labelsize':15})
    plt.rcParams.update({'ytick.labelsize':15})
    plt.rcParams.update({'lines.linewidth':6})

def set_paper_style(colors):
    plt.rcParams.update(plt.rcParamsDefault)
    plt.rc('axes', prop_cycle=cycler('color', colors))
    plt.rcParams.update({'font.size': 8})
    plt.rcParams.update({'axes.labelsize':8})
    plt.rcParams.update({'xtick.labelsize':6})
    plt.rcParams.update({'ytick.labelsize':6})
    plt.rcParams.update({'lines.linewidth':2.5})
    plt.rcParams.update({'lines.markersize':5})
    
