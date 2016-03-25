"""
All functions related to plotting
"""

import numpy as np
import networkx as nx

import matplotlib as mpl
import matplotlib.pylab as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plot_matrix(mat, ax):
    """ Plot system evolution
    """
    cmap = plt.cm.coolwarm
    cmap.set_under('white')

    im = ax.imshow(
        mat, vmin=0,
        interpolation='nearest', cmap=cmap)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.2)
    plt.colorbar(im, cax=cax)

    ax.set_xlabel(r'$i$')
    ax.set_ylabel(r'$j$')

    ax.set_title('Sign-switch of dynamic connectivity matrix')

def plot_graph(graph, ax):
    """ Plot graph
    """
    # generate some node properties
    labels = {}
    for n in graph.nodes():
        labels[n] = n

    # compute layout
    pos = nx.nx_pydot.graphviz_layout(graph, prog='neato')

    # draw graph
    nx.draw(
        graph, pos,
        node_color='lightskyblue', node_size=800,
        font_size=20,
        ax=ax)
    nx.draw_networkx_labels(graph, pos, labels)

def plot_evolutions(sols, ts, ax):
    """ Plot system evolution
    """
    ax.imshow(
        sols, aspect='auto',
        cmap=plt.cm.gray, interpolation='nearest')

    # let x-labels correspond to actual time steps
    ts = np.append(ts, ts[-1]+(ts[-1]-ts[-2]))
    formatter = plt.FuncFormatter(lambda x, pos: int(ts[x]))
    ax.xaxis.set_major_formatter(formatter)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\Theta_i$')

def plot_correlation_matrix(cmat, ts, ax):
    """ Plot individual correlation matrix
    """
    for i, row in enumerate(cmat):
        for j, sol in enumerate(row):
            ax.plot(ts, sol, label='{},{}'.format(i, j))

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle \cos \left(\Theta_i(t) - \Theta_j(t)\right)\rangle$')
    ax.set_ylim((-1, 1.1))

def plot_series(series, ts, ax):
    """ Plot single time series
    """
    ax.plot(ts, series)

    ax.set_xlabel(r'$t$')
    ax.set_ylabel(r'$\langle\frac{N^2}{N^2 - 2 \sum_{ij} \sin\left( \theta_i - \theta_j \right)^2}\rangle$')

def plot_result(data):
    """ Plot final result
    """
    # main overview
    fig = plt.figure(figsize=(30, 10))
    gs = mpl.gridspec.GridSpec(2, 3, width_ratios=[1, 1, 2])

    plot_graph(data.graph, plt.subplot(gs[:, 0]))
    plot_matrix(data.syncs, plt.subplot(gs[:, 1]))
    plot_evolutions(data.sol, data.ts, plt.subplot(gs[0, 2]))
    plot_correlation_matrix(data.cmats, data.ts, plt.subplot(gs[1, 2]))

    plt.tight_layout()
    fig.savefig('result.pdf')
    fig.savefig('foo.png')

    # variance investigation
    fig = plt.figure(figsize=(20, 10))
    gs = mpl.gridspec.GridSpec(2, 1)

    plot_evolutions(data.sol, data.ts, plt.subplot(gs[0]))
    plot_series(data.vser, data.ts, plt.subplot(gs[1]))

    plt.tight_layout()
    fig.savefig('cluster_num.pdf')
    fig.savefig('bar.png')
