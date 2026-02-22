
import matplotlib.pyplot as plt
from cycler import cycler

def init_plot_style():

    plt.style.use('ggplot')

    colors = [
        '#1f77b4', '#ff7f0e', '#2ca02c',
        '#d62728', '#9467bd', '#8c564b',
        '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
    ]
    linestyles = ['-', '--', '-.', ':'] * 3  

    style_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    plt.rc('axes', prop_cycle=style_cycler)
