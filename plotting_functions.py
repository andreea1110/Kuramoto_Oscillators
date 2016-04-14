import numpy as np
import matplotlib.pylab as plt
import networkx as nx
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import make_axes_locatable

def plot_graph(graph, ax):
    """ Plot the network used for the current simulation
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
        node_color='lightskyblue', node_size=400,
        font_size=20, 
        ax=ax)
    nx.draw_networkx_labels(graph, pos, labels)

def plot_corr_vals(corr_vals, t):
	plt.figure()
	for i in range(np.size(corr_vals, 0)):
		plt.plot(t, corr_vals[i, :])
	plt.xlabel('t')
	plt.ylabel('correlation values')
	plt.savefig('images/corrValAvg.png')
	plt.savefig('images/corrValAvg.pdf')

def plot_time_evolution(sol, t):
	""" Plot the time evolution of the system
	"""
	plt.figure()
	nf = np.size(sol, 0)
	for i in range(nf):
		plt.plot(t, sol[i, :], label = "f %d" % (i))
	plt.legend(loc = 'upper right', prop = {'size':7})
	plt.xlabel("t")
	plt.ylabel(r"$\theta$")
	#plt.xlim(0, 10)
	plt.savefig('images/timeSol.png')
	plt.savefig('images/timeSol.pdf')

def plot_corr_mat_dcm(G, sum_rho_avg, D_average, no_runs):
	""" Plot the sum of the correlation matrices, 
		avergaed over a number of runs
	"""
	plt.figure()
	gs = mpl.gridspec.GridSpec(1, 3)

	# plot the network graphs
	plot_graph(G, plt.subplot(gs[:, 0]))

	# plot the averaged sum of the correlation matrices
	ax = plt.subplot(gs[:, 1])
	im = ax.imshow(sum_rho_avg, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.2)
	plt.colorbar(im, cax=cax)
	ax.set_title('Correlation matrix')

	# plot the time to correlation matrix
	ax = plt.subplot(gs[:, 2])
	plt.imshow(D_average, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	divider = make_axes_locatable(ax)
	cax = divider.append_axes('right', size='5%', pad=0.2)
	plt.colorbar(im, cax=cax)
	ax.set_title('Time to synchronization in DCM')

	plt.suptitle('Average over %s runs' % (no_runs))
	plt.savefig('images/corrMat_dcm.png')
	plt.savefig('images/corrMat_dcm.pdf')

