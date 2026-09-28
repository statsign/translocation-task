import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
from cycler import cycler

def init_plot_style():

    plt.style.use('ggplot')

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] 
    linestyles = ['-', '--', '-.', ':'] 

    style_cycler = cycler(color=colors) + cycler(linestyle=linestyles)
    plt.rc('axes', prop_cycle=style_cycler)
