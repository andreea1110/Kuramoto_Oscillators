import numpy as np
import matplotlib.pylab as plt
import matplotlib as mpl
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import trange

from plotting_functions import*
from network_generator import*
from sys_dynamics_functions import*


def main():
	start_time = time.time()

	# set simulation time
	tmin = 0
	tmax = 20
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

	# the parameters of the system of oscillators
	nf = np.size(A, 0) # number of fireflies
	w = 0.4611143833383792*np.ones(nf) # the natural frequencies of the oscillators
	
	
	# solve the system and plot the time evolution and the corresponding correlation values
	sol, corr_vals = solve_system(t, w, A, B, Phi, True)
	plot_time_corr(G, sol, corr_vals, t)

	# compute and plot the sum of the correlation matrices and 
	# the time to syncronization in the DCM matrix, 
	# averaged over no_runs runs
	sum_rho_avg = sum_correlation_matrices(no_runs, t, w, A, B, Phi)
	D_average = time_to_sync_matrix(no_runs, t, w, A, B, Phi)
	plot_corr_mat_dcm(G, sum_rho_avg, D_average, no_runs)

	print("--- Execution time: %s seconds ---" % (time.time() - start_time))
	plt.show()


if __name__ == "__main__":
	main()