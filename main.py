import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from cairocffi import *
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import trange

from plotting_functions import*
from network_generator import*

def system_generator(w, A, B, Phi, nf):
	""" Generate the system of ODEs for the 
		Kuramoto's model of coupled oscillators 
	"""
	def f(theta, t = 0):
		""" dtheta_i/dt = d(theta_i) 
		"""
		state = np.zeros(nf)
		for i in range(nf):
			for j in range(nf):
				state[i] += A[i, j]*np.sin(theta[j] - theta[i])
			state[i] += w[i] + B[i] * np.sin(Phi(t) - theta[i])
		return state
	return f

def correlation_matrix(theta_t):
	""" Compute the corelation matrix at one point in time
	"""
	n = np.size(theta_t)
	rho = np.zeros((n, n))
	for i in range(n):
		for j in range(n):
			rho[i][j] = np.cos(theta_t[i] - theta_t[j])
	return rho

def sum_correlation_matrices(no_runs, t, w, A, B, Phi):
	""" Compute the sum of the correlation matrices, 
		averaged over the number of runs
	"""
	nf = np.size(A, 0)
	f = system_generator(w, A, B, Phi, nf)
	rho_average = np.zeros((nf, nf))
	for idx in trange(no_runs):
		#initial values of the phases
		init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle
		sol = (odeint(f, init, t).T) % (2*np.pi) #solve the system of ODEs

		rho_sum = np.zeros((nf, nf)) # initialize the matrix which will hold the sum of the connectivity matrices
		D = np.zeros((nf, nf))
		T = 0.95 # synchronization threshold 
		for i in range(np.size(sol, 1)):
			rho_sum += correlation_matrix(sol[:, i])

		rho_average += rho_sum
	rho_average = rho_average/no_runs
	return rho_average

def time_to_sync_matrix(no_runs, t, w, A, B, Phi):
	""" Compute the time to synchronization of each two nodes in the network
		(time to value-switch from 0 to some nonzero number in the dynamic 
		connectivity matrix DCM)
	"""
	nf = np.size(A, 0)
	f = system_generator(w, A, B, Phi, nf)
	D_average = np.zeros((nf, nf)) # initialize the DCM
	for idx in trange(no_runs):
		#initial values of the phases
		init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle
		sol = (odeint(f, init, t).T) % (2*np.pi) #solve the system of ODEs
		D = np.zeros((nf, nf)) # actual tyme-to-sync matrix
		T = 0.95 # synchronization threshold 
		for i in range(np.size(sol, 1)): # iterating through the time series
			rho_t = correlation_matrix(sol[:, i]) # the correlation matrix at time t
			idx = np.argwhere(rho_t > T)
			for j in idx:
	 			if D[tuple(j)] == 0:
	 				D[tuple(j)] = t[i]
		D_average += D
	D_average = D_average/no_runs
	return D

def solve_system(t, w, A, B, Phi, correlations = False):
	""" Solves the system of ODEs describing the Kuramoto oscillators, 
		to determine the phases of the oscillators.
	"""
	nf = np.size(A, 0)
	f = system_generator(w, A, B, Phi, nf)
	init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle
	sol = (odeint(f, init, t).T) % (2*np.pi) #solve the system of ODEs

	if correlations == False:
		return sol
	else:
		corr_vals = np.zeros((nf*nf, np.size(sol, 1)))
		for i in range(np.size(sol, 1)):
			rho_t = correlation_matrix(sol[:, i]) # the correlation matrix at time t
			corr_vals[:, i] = rho_t.flatten()
		return sol, corr_vals
	

def main():
	start_time = time.time()

	# set simulation time
	tmin = 0
	tmax = 40
	dt = 0.01 # time step
	t = np.arange(tmin, tmax, dt) # the simulation time

	no_runs = 5 # number of times the simulation is run (used where applicable)

	# load network of oscillators
	G = load_cycle_graph(10) 
	A = nx.adjacency_matrix(G).todense()
	#plot_graph(G)

	# the parameters of the external driver
	Omega = 3
	Phi = lambda t: Omega*t
	B = [1.00465019, 1.4795114, 3.92223887, 3.99427941, 4.8777141, 2.59484127, 3.6070913, 4.51140533, 3.38070504, 3.65192458]

	nf = np.size(A, 0) # number of fireflies
	w = 0.4611143833383792*np.ones(nf) # the natural frequencies of the oscillators
	
	

	# solve the system and plot the time evolution 
	sol, corr_vals = solve_system(t, w, A, B, Phi, True)
	plot_time_evolution(sol, t)

	# plot the correlation values for the system
	plot_corr_vals(corr_vals, t)

	# compute and plot the sum of the correlation matrices, 
	# averaged over no_runs runs
	sum_rho_avg = sum_correlation_matrices(no_runs, t, w, A, B, Phi)
	

	D_average = time_to_sync_matrix(no_runs, t, w, A, B, Phi)
	plot_corr_mat_dcm(G, sum_rho_avg, D_average, no_runs)

	plt.show()


	# f = system_generator(w, A, nf) # generate the system of ODEs for the given parameters
	# rho_average = np.zeros((nf, nf))
	# D_average = np.zeros((nf, nf))

	# for idx in trange(no_runs):

	# 	init = np.random.uniform(0, 2*np.pi, nf) % (2*np.pi) # restricting the angles to the unit circle

	# 	sol = odeint(f, init, t).T
	# 	sol = sol % (2*np.pi)

	# 	rho_sum = np.zeros((nf, nf))
	# 	corr_vals = np.zeros((nf*nf, np.size(sol, 1)))
	# 	D = np.zeros((nf, nf))
	# 	T = 0.95 # synchronization threshold 
	# 	# computing correlation matrix & the dynamic connectivity matrix
	# 	for i in range(np.size(sol, 1)):
	# 		rho_t = correlationMatrix(sol[:, i])
	# 		corr_vals[:, i] = rho_t.flatten()
	# 		rho_sum += rho_t
	# 		idx = np.argwhere(rho_t > T)
	# 		for j in idx:
	#  			if D[tuple(j)] == 0:
	#  				D[tuple(j)] = t[i]

	# 	rho_average += rho_sum
	# 	D_average += D

	# rho_average = rho_average/no_runs
	# D_average = D_average/no_runs

	# plt.figure(3)
	# for i in range(nf*nf):
	# 	plt.plot(corr_vals[i, :])
	# plt.xlabel('t')
	# plt.ylabel('correlation values')
	# plt.savefig('images/corrValAvg.png')
	# plt.savefig('images/corrValAvg.pdf')

	# # plot the sum of the correlation matrix
	# plt.figure(2)
	# plt.subplot(121)
	# plot_graph(G)
	# plt.subplot(122)
	# plt.imshow(rho_average, 
	# 	cmap = plt.cm.coolwarm, 
	# 	interpolation='nearest')
	# plt.colorbar()
	# plt.suptitle('Correlation matrix (average over %s runs)' % (no_runs))
	# plt.savefig('images/corrMatAvg.png')
	# plt.savefig('images/corrMatAvg.pdf')

	# # plotting the average dynamic connectivity matrix
	# plt.figure(5)
	# plt.subplot(121)
	# plot_graph(G)
	# plt.subplot(122)
	# plt.imshow(D_average, 
	# 	cmap = plt.cm.coolwarm, 
	# 	interpolation='nearest')
	# plt.colorbar()
	# plt.suptitle('Time to sync in dcm (average over %s runs)' % (no_runs))
	# plt.savefig('images/dcaAvg.png')
	# plt.savefig('images/dcaAvg.pdf')

	# print("--- Execution time: %s seconds ---" % (time.time() - start_time))
	

	



if __name__ == "__main__":
	main()