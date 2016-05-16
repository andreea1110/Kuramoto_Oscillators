import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import trange
from plotting_functions import*
from network_generator import*
from reconstruction import*



def main():
	""" Main program.
	"""
	start_time = time.time()

	# set simulation time
	tmin = 0
	tmax = 30
	dt = 0.01 # time step
	t = np.arange(tmin, tmax, dt) # the simulation time

	no_runs = 1 # number of times the simulation is run (used where applicable)

	# load network of oscillators
	G = load_simple_network()

	#G = nx.gnp_random_graph(5, 0.6)
	#A = nx.adjacency_matrix(G).todense()
	A = nx.to_numpy_matrix(G)
	nz = np.nonzero(A)
	A[nz] = np.random.uniform(0.5, 3, size=len(nz[0]))
	

	# the parameters of the system of oscillators
	nf = np.size(A, 0) # number of fireflies
	#w = 0.4611143833383792*np.ones(nf) # the natural frequencies of the oscillators
	w = 3*np.ones(nf)

	# the parameters of the external driver
	Omega = 2.9
	Phi = lambda t: Omega*t
	#B = np.array([2, 10, 5, 8])
	#B = np.linspace(10, 20, nf)
	B = np.random.uniform(10, 20, size=nf)
	#B = np.random.uniform(0, 5, size=nf)
	#B = np.array([7, 0])
	#B = [1.00465019, 1.4795114, 3.92223887, 3.99427941, 4.8777141, 2.59484127, 3.6070913, 4.51140533, 3.38070504, 3.65192458]
	
	
	# solve the system and plot the time evolution and the corresponding correlation values
	sol, corr_vals = solve_system(t, w, A, B, Phi, True)
	plot_time_corr(G, sol, corr_vals, Omega, t)

	phi = compute_phase_diffs(sol, t, Omega)

	#Omega_vec = np.array([0.3, 0.5, 1.3, 1.7, 2.0])
	#Omega_vec = np.linspace(0.2, 0.8, 20)
	Omega_vec = [2.9,3.05,3.1,3.2]
	Arec, Brec = reconstruct_coeffs_mask(t, Omega_vec, w, A, B)
	compute_reconstruction_error(A, B, Arec, Brec)

	plt.figure()
	gs = mpl.gridspec.GridSpec(1, 1)
	plot_time_evolution(sol, t, Omega, plt.subplot(gs[0, 0]))
	plt.savefig('images/phases.png')

	# compute and plot the sum of the correlation matrices and 
	# the time to syncronization in the DCM matrix, 
	# averaged over no_runs runs
	#sum_rho_avg = sum_correlation_matrices(no_runs, t, w, A, B, Phi)
	#D_average = time_to_sync_matrix(no_runs, t, w, A, B, Phi)
	#plot_corr_mat_dcm(G, sum_rho_avg, D_average, no_runs)

	print("--- Execution time: %s seconds ---" % (time.time() - start_time))
	plt.show()

if __name__ == "__main__":
	main()