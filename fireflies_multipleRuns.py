import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from cairocffi import *
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import trange

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

def load_network():
	G = nx.Graph()
	for i in range(1, 13):
		G.add_node(i)
	edges_list = [(1, 2), (1, 4), (2, 4), (2, 5), (2, 3),
	(3, 5), (3, 4), (5, 6), (6, 9), (4, 11), (7, 8),
	(7, 9), (8, 10), (8, 9), (7, 10), (9, 10), (10, 11), 
	(11, 12), (10, 12), (6, 2), (6, 3), (6, 1), (9, 11), (9, 12), (4, 6), (11, 7)]
	G.add_edges_from(edges_list)
	return G

def load_network2():
	G = nx.Graph()
	for i in range(1, 13):
		G.add_node(i)
	edges_list = [(1, 2), (1, 3), (1, 4), (2, 3), (3, 4), (2, 4), (2, 7), 
	(5, 7), (5, 6), (5, 8), (6, 7), (6, 8), (7, 8), (7, 9), (9, 10), 
	(9, 11), (9, 12), (10, 11), (10, 12), (11, 12)]
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
        node_color='lightskyblue', node_size=400,
        font_size=20)
    nx.draw_networkx_labels(graph, pos, labels)

def plotSol(sol, t, nf):
	plt.figure(1)
	for i in range(nf):
		plt.plot(t, sol[i, :], label = "f %d" % (i))
	plt.legend(loc = "upper right")
	plt.xlabel("t")
	plt.ylabel(r"$\theta$")
	#plt.xlim(0, 10)
	plt.savefig('images/timeSol.png')
	plt.savefig('images/timeSol.pdf')
	plt.show()	


def main():
	start_time = time.time()


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

	
	#G = nx.from_numpy_matrix(A)
	# plt.figure(4)
	# nx.draw_networkx(G)
	# plt.savefig('network.png')

	tmin = 0
	tmax = 8
	dt = 0.01
	t = np.arange(tmin, tmax, dt)
	no_runs = 10

	G = load_ring_network(80)
	A = nx.adjacency_matrix(G).todense()

	nf = np.size(A, 0)
	w = 0.3*np.ones(nf)

	f = systemGenerator(w, A, nf)
	rho_average = np.zeros((nf, nf))
	D_average = np.zeros((nf, nf))
	corr_valsAvg = np.zeros((nf*nf, len(t)))

	for idx in trange(no_runs):

		init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle

		sol = odeint(f, init, t).T
		sol = sol % (2*np.pi)

		rho_sum = np.zeros((nf, nf))
		corr_vals = np.zeros((nf*nf, np.size(sol, 1)))
		D = np.zeros((nf, nf))
		T = 0.95 # threshold 
		# computing correlation matrix & the dynamic connectivity matrix
		for i in range(np.size(sol, 1)):
			rho_t = correlationMatrix(sol[:, i])
			corr_vals[:, i] = rho_t.flatten()
			rho_sum += rho_t
			idx = np.argwhere(rho_t > T)
			for j in idx:
	 			if D[tuple(j)] == 0:
	 				D[tuple(j)] = t[i]

		rho_average += rho_sum
		D_average += D

	rho_average = rho_average/no_runs
	D_average = D_average/no_runs

	plt.figure(3)
	for i in range(nf*nf):
		plt.plot(corr_vals[i, :])
	plt.xlabel('t')
	plt.ylabel('correlation values')
	plt.savefig('images/corrValAvg.png')
	plt.savefig('images/corrValAvg.pdf')

	plt.figure(2)
	plt.subplot(121)
	plot_graph(G)
	plt.subplot(122)
	plt.imshow(rho_average, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	plt.colorbar()
	plt.suptitle('Correlation matrix (average over %s runs)' % (no_runs))
	plt.savefig('images/corrMatAvg.png')
	plt.savefig('images/corrMatAvg.pdf')

	# plotting the average dynamic connectivity matrix
	plt.figure(5)
	plt.subplot(121)
	plot_graph(G)
	plt.subplot(122)
	plt.imshow(D_average, 
		cmap = plt.cm.coolwarm, 
		interpolation='nearest')
	plt.colorbar()
	plt.suptitle('Time to sync in dcm (average over %s runs)' % (no_runs))
	plt.savefig('images/dynConAvg_withNet.png')
	plt.savefig('images/dynConAvg_withNet.pdf')

	print("--- Execution time: %s seconds ---" % (time.time() - start_time))
	plt.show()

	#plotSol(sol, t, nf)




if __name__ == "__main__":
	main()