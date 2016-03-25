import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from cairocffi import *
from scipy.integrate import odeint
import networkx as nx
import time
from numpy import linalg as la
from plotter import *

def systemGenerator(w, A, nf):
	""" Generates the system of ODEs for the 
	Kuramoto's model 
	"""
	def f(theta, t = 0):
		""" dtheta_i/dt = d(theta_i) 
		"""
		state = np.zeros(nf)
		for i in range(nf):
			for j in range(nf):
				state[i] += A[i, j]*np.sin(theta[j] - theta[i])
			state[i] += w[i]
		return state
	return f

def correlationMatrix(theta_t):
	n = np.size(theta_t)
	rho = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			rho[i][j] = np.cos(theta_t[i] - theta_t[j])
	return rho


def plotSol(sol, t, nf):
	plt.figure(1)
	for i in range(nf):
		plt.plot(t, sol[i, :], label = "f %d" % (i))
	#plt.legend(loc = "upper right")
	plt.xlabel("t")
	plt.ylabel(r"$\theta$")
	#plt.xlim(0, 10)
	plt.savefig('images/timeSol_ring.png')
	plt.savefig('images/timeSol_ring.pdf')

def plot_phase_shifts(sol, t, nf):
	plt.figure()
	for i in range(nf - 1):
		plt.plot(t, np.abs(sol[i, :] - sol[i + 1, :]))
	plt.plot(t, np.abs(sol[nf - 1, :] - sol[0, :]))
	plt.xlabel("t")
	plt.ylabel(r"$\theta_i - \theta_{i + 1}$")
	#plt.xlim(0, 10)
	plt.savefig('images/phase_shifts.png')
	plt.savefig('images/phase_shifts.pdf')		

def plot_graph(graph):
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
        font_size=20)
    nx.draw_networkx_labels(graph, pos, labels)

def plot_evolutions(sols, ts):
    """ Plot system evolution
    """
    plt.imshow(
        sols, aspect='auto',
        cmap=plt.cm.gray, interpolation='nearest', 
        extent = (0, ts[-1], 0, np.size(sols, 0)))

    # let x-labels correspond to actual time steps
    ts = np.append(ts, ts[-1]+(ts[-1]-ts[-2]))
    #formatter = plt.FuncFormatter(lambda x, pos: int(ts[x]))
    #plt.xaxis.set_major_formatter(formatter)

    plt.xlabel(r'$t$')
    plt.ylabel(r'$\Theta_i$')

def load_network():
	""" Creating a graph representing a network of oscillators
	"""
	G = nx.Graph()
	for i in range(1, 13):
		G.add_node(i)
	edges_list = [(1, 2), (1, 4), (2, 4), (2, 5), (2, 3),
	(3, 5), (3, 4), (5, 6), (6, 9), (4, 11), (7, 8),
	(7, 9), (8, 10), (8, 9), (7, 10), (9, 10), (10, 11), 
	(11, 12), (10, 12), (6, 2), (6, 3), (6, 1), (9, 11), (9, 12), (4, 6), (11, 7)]
	G.add_edges_from(edges_list)
	return G

def load_ring_network(size):
	""" Creating a ring-structured network
	"""
	G = nx.Graph()
	for i in range(size):
		G.add_node(i)

	edges_list = [];
	for i in range(size - 1):
		edges_list.append((i, i + 1));

	edges_list.append((size - 1, 0));
	G.add_edges_from(edges_list)
	return G


def main():
	start_time = time.time()

	#nf = 9
	#w = 16*np.ones(nf)
	#w = 0.3*np.ones(nf)
	# w1 = 16*np.ones(nf//2)
	# w2 = 8*np.ones(nf - nf//2)
	# w = np.concatenate([w1, w2])

	# A = np.array([
	# 	[0, 1, 1, 1, 1, 1, 1, 1, 1],
	# 	[1, 0, 0, 0, 0, 1, 1, 0, 0], 
	# 	[1, 0, 0, 0, 0, 0, 0, 1, 1], 
	# 	[1, 0, 0, 0, 1, 0, 0, 0, 0], 
	# 	[1, 0, 0 ,1, 0, 0, 0, 0, 0], 
	# 	[1, 1, 0, 0, 0, 0, 1, 0, 0], 
	# 	[1, 1, 0, 0, 0, 1, 0, 0, 0], 
	# 	[1, 0, 1, 0, 0, 0, 0, 0, 1], 
	# 	[1, 0, 1, 0, 0, 0, 0, 1, 0]
	# 	])
	#A = np.ones((nf, nf))
	#A = np.eye(nf, nf)
	#A = np.random.randint(2, size = (nf, nf))

	G = load_ring_network(20)
	A = nx.adjacency_matrix(G).todense()

	#G = nx.from_numpy_matrix(A)

	nf = np.size(A, 0)
	w = 0.3*np.ones(nf)

	# printing the network
	plt.figure(2)
	plot_graph(G)
	plt.savefig('images/network_ring.png')
	plt.savefig('images/network_ring.pdf')

	# solving ODEs
	tmin = 0
	tmax = 25
	dt = 0.01
	t = np.arange(tmin, tmax, dt)
	f = systemGenerator(w, A, nf)
	init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle
	sol = odeint(f, init, t).T
	sol = sol % (2*np.pi)


	# computing correlation matrix & the dynamic connectivity matrix
	rho_sum = np.zeros((nf, nf))
	corr_vals = np.zeros((nf*nf, np.size(sol, 1)))
	D = np.zeros((nf, nf))
	T = 0.9 # threshold 
	for i in range(np.size(sol, 1)):
		rho_t = correlationMatrix(sol[:, i])
		corr_vals[:, i] = rho_t.flatten()
		rho_sum += rho_t
		idx = np.argwhere(rho_t > T)
		for j in idx:
			if D[tuple(j)] == 0:
	 			D[tuple(j)] = t[i]

	# plotting the values in the correlation matrix, for each point in time
	plt.figure(3)
	for i in range(nf*nf):
		plt.plot(corr_vals[i, :])
	plt.savefig('images/corrVal_ring.png')
	plt.savefig('images/corrVal_ring.pdf')

	# plotting the sum of the correlation matrices
	plt.figure(4)
	plt.imshow(rho_sum, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	plt.colorbar()
	plt.title('Correlation matrix')
	plt.savefig('images/corrMat3_ring.png')
	plt.savefig('images/corrMat3_ring.pdf')


	# plotting the dynamic connectivity matrix
	plt.figure(5)
	plt.imshow(D, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	plt.colorbar()
	plt.title('Time to synchronization in dcm')
	plt.savefig('images/dynCon.png')
	plt.savefig('images/dynCon.pdf')

	L = nx.laplacian_matrix(G).todense()
	eigVals, eigVec = la.eig(L)
	eigVals.sort()

	pairs = []
	for i, eig in enumerate(eigVals):
		if abs(eig) < 1e-5: continue
		inv_eig = 1 / eig
		pairs.append((inv_eig, i))
    	

	plt.figure(6)
	plt.scatter(*zip(*pairs))
	plt.xlabel(r'$\frac{1}{\lambda_i}$')
	plt.ylabel(r'rank index')
	plt.savefig('images/lapEigevalEpectrum.pdf')
	
	plotSol(sol, t, nf)

	plt.figure(20)
	plot_evolutions(sol, t)
	plt.savefig('images/heat_map_evolutions.png')

	plot_phase_shifts(sol, t, nf)

	plt.show()
	print("--- Execution time: %s seconds ---" % (time.time() - start_time))




if __name__ == "__main__":
	main()