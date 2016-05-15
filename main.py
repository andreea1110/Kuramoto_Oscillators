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


def compute_phase_diffs(sol, t, Omega):
	""" Compute the phase differences  of the oscillators
		to the driver, assuming that the oscillators have locked
		in frequency to the driver.
		Arguments:
			sol = solution matrix, with the rows representing oscillators and the angles at each tiem point in columns
			t = time vector
			Omega = natural frequency of the driver
	"""
	Phi = lambda t: (Omega*t) 
	dt = t[-1] - t[-2]
	nf = np.size(sol, 0)
	phi = np.zeros(nf) # initializing vector of phase differences

	for i in range(nf):
		phi[i] = (Phi(t[-1]) % (2*np.pi)) - sol[i, -1]
		if phi[i] < 0:
			phi[i] += 2*np.pi

	print("phi = ", phi)
	return phi

def reconstruct_coefs(t, Omega_vec, omega, A, B):
	"""
	"""
	nf = np.size(A,0)
	print("Omega_vec = ", Omega_vec)
	print("np.repeat(Omega_vec, nf) = ", np.repeat(Omega_vec, nf))
	b = np.repeat(Omega_vec, nf) - omega[1]

	M = np.zeros((len(Omega_vec)*nf, nf**2))
	for idx, Omega in enumerate(Omega_vec):
		Phi = lambda t: Omega*t
		sol = solve_system(t, omega, A, B, Phi)
		phi = compute_phase_diffs(sol, t, Omega)
		# for i in range(nf):
		# 	row = np.array([])
		# 	for j in range(nf):
		# 		if i == j:
		# 			continue
		# 		row = np.append(row, np.sin(phi[i] - phi[j]))
		# 	row = np.append(row, np.sin(phi[i]))
		# 	row = np.append(row, np.zeros(nf**2 - nf))
		# 	A[i, :] = row
		row0 = np.array([np.sin(phi[0] - phi[1]), np.sin(phi[0]), 0, 0])
		row1 = np.array([0, 0, np.sin(phi[1] - phi[0]), np.sin(phi[1])])
		M[idx*2, :] = row0
		M[idx*2 + 1, :] = row1

	print("M = \n", M)
	print("b = \n", b)
	x = np.linalg.lstsq(M, b)[0]
	print(x)

	Arec = np.array([[0, x[0]], [x[2],0]])
	Brec = np.array([x[1], x[3]])

	print("original A = \n", A)
	print("reconstructed A = \n", Arec)
	print("original B = \n", B)
	print("reconstructed B = \n", Brec)




def main():
	start_time = time.time()

	# set simulation time
	tmin = 0
	tmax = 50
	dt = 0.01 # time step
	t = np.arange(tmin, tmax, dt) # the simulation time

	no_runs = 1 # number of times the simulation is run (used where applicable)

	# load network of oscillators
	G = load_simple_network()
	A = nx.adjacency_matrix(G).todense()
	

	# the parameters of the system of oscillators
	nf = np.size(A, 0) # number of fireflies
	#w = 0.4611143833383792*np.ones(nf) # the natural frequencies of the oscillators
	w = 1*np.ones(nf)

	# the parameters of the external driver
	Omega = 0.3
	Phi = lambda t: Omega*t
	B = np.array([7, 0])
	#B = [1.00465019, 1.4795114, 3.92223887, 3.99427941, 4.8777141, 2.59484127, 3.6070913, 4.51140533, 3.38070504, 3.65192458]
	
	
	# solve the system and plot the time evolution and the corresponding correlation values
	sol, corr_vals = solve_system(t, w, A, B, Phi, True)
	#plot_time_corr(G, sol, corr_vals, Omega, t)
	print(sol.shape)
	print(t[-1])
	print("theta_0 = ", sol[0, t[-1]/dt])
	print("theta_1 = ", sol[1, t[-1]/dt])
	print("Phi = ",Phi(t[-1]) % (2*np.pi))

	phi = compute_phase_diffs(sol, t, Omega)
	print(phi)

	Omega_vec = np.array([0.3, 0.5, 1.3, 1.7, 2.0])
	#Omega_vec = np.arange(0.1, 2.0, 0.1)
	print(Omega_vec.shape)
	reconstruct_coefs(t, Omega_vec, w, A, B)

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