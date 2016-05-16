import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import trange


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