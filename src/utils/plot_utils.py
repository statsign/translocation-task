import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from cycler import cycler
import numpy as np

LOG_FLOOR = 1e-30 # minimum value on pdf graph

def init_plot_style():

    plt.style.use('ggplot')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    linestyles = ['-', '--', '-.', ':'] 

    style_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    plt.rc('axes', prop_cycle=style_cycler)


def plot_pdf(ax, t, p, log_scale=False, floor=LOG_FLOOR, **kwargs):
    """
    Settings for pdf plots (optimization results)
    """
    t = np.asarray(t).ravel()
    p = np.asarray(p, dtype=float).ravel()
    if log_scale:
        p = np.where(p > floor, p, np.nan) 
        ax.set_yscale('log')
    return ax.plot(t, p, **kwargs)
