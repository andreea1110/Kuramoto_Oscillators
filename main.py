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
	return phi

def reconstruct_coefs(t, Omega_vec, omega, A, B):
	""" Recontruct the coupling coefficients strored in matrices A and B, 
		using the phase differences between the oscillators and the driver
		obtained for each Omega in Omega_vec.
	"""
	nf = np.size(A,0)
	b = np.repeat(Omega_vec, nf) - omega[1]

	print("Solving a system of {} equations and {} unknowns.".format(len(Omega_vec)*nf, nf**2))

	M = np.zeros((len(Omega_vec)*nf, nf**2))
	for idx, Omega in enumerate(Omega_vec):
		Phi = lambda t: Omega*t
		sol = solve_system(t, omega, A, B, Phi)
		phi = compute_phase_diffs(sol, t, Omega)
		for i in range(nf):
			row = np.array([])
			for j in range(nf):
				if i == j:
					continue
				row = np.append(row, np.sin(phi[i] - phi[j]))

			row = np.append(row, np.sin(phi[i]))
			M[idx*nf + i, i*nf : i*nf + nf] = row

	evals = np.linalg.eigvals(np.array(M).T.dot(M))
	cn = abs(max(evals) / min(evals))
	print('Condition number:', cn, np.log10(cn))
	
	x = np.linalg.lstsq(M, b)[0]

	x = np.reshape(x, (nf, nf))

	Brec = x[:, -1]

	Ahelp = x[:, 0:nf-1]
	Arec = np.zeros((nf, nf))
	for i in range(np.size(Ahelp, 0)):
		Arec[i, :] = np.insert(Ahelp[i, :], i, 0)

	print("original A = \n", A)
	print("reconstructed A = \n", Arec)
	print("original B = \n", B)
	print("reconstructed B = \n", Brec)

	return (Arec, Brec)




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
	B = np.array([0.1, 10, 0.1, 0.1])
	#B = np.random.uniform(0, 5, size=nf)
	#B = np.array([7, 0])
	#B = [1.00465019, 1.4795114, 3.92223887, 3.99427941, 4.8777141, 2.59484127, 3.6070913, 4.51140533, 3.38070504, 3.65192458]
	
	
	# solve the system and plot the time evolution and the corresponding correlation values
	sol, corr_vals = solve_system(t, w, A, B, Phi, True)
	plot_time_corr(G, sol, corr_vals, Omega, t)

	phi = compute_phase_diffs(sol, t, Omega)

	#Omega_vec = np.array([0.3, 0.5, 1.3, 1.7, 2.0])
	Omega_vec = np.linspace(0.3, 2.0, 10)
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