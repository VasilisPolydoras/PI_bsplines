# plot_style.py
import matplotlib.pyplot as plt
import matplotlib as mpl

def set_plot_style():
    mpl.rcParams.update({
        "text.usetex": False,  # Set to True if you have LaTeX installed
        "font.family": "serif",
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "legend.fontsize": 13,
        "xtick.labelsize": 13,
        "ytick.labelsize": 13,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.alpha": 0.3,
        "lines.linewidth": 3,
        "lines.markersize": 6,
        "figure.figsize": (6, 4),  # Default size (in inches)
        "figure.dpi": 100,
        "savefig.dpi": 300,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "legend.fancybox": True,
        "legend.edgecolor": 'gray',
    })
