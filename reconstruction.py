import numpy as np
from scipy.integrate import odeint
import networkx as nx
import time
from tqdm import*
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

def reconstruct_coeffs(t, Omega_vec, omega, A, B):
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

def reconstruct_coeffs_mask(t, Omega_vec, omega, A, B):
	""" Recontruct the coupling coefficients strored in matrices A and B, 
		using the phase differences between the oscillators and the driver
		obtained for each Omega in Omega_vec.
	"""
	nf = np.size(A,0)
	b = np.repeat(Omega_vec, nf*len(B)) - omega[1]

	print("Solving a system of {} equations and {} unknowns.".format(len(Omega_vec)*nf*len(B), nf**2))

	M = np.zeros((len(Omega_vec)*nf*len(B), nf**2))
	idxM = 0
	for Omega in tqdm(Omega_vec):	
		Phi = lambda t: Omega*t
		for idx2, Bval in enumerate(B):
			Bmask = np.zeros((nf))
			Bmask[idx2] = Bval
			sol = solve_system(t, omega, A, Bmask, Phi)
			phi = compute_phase_diffs(sol, t, Omega)
			for i in range(nf):
				row = np.array([])
				for j in range(nf):
					if i == j:
						continue
					row = np.append(row, np.sin(phi[i] - phi[j]))
				if i == idx2:	
					row = np.append(row, np.sin(phi[i]))
				else:
					row = np.append(row, 0)
				M[idxM, i*nf : i*nf + nf] = row
				idxM += 1
				

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

def compute_reconstruction_error(A, B, Arec, Brec):
    """
    Compute relative error of reconstruction.
    Arguments:
    	A = original A
    	B = original B
    	Arec = reconstructed A
    	Brec = reconstructed B
    """
    rel_err_A = abs(A - Arec) / (1 + abs(A))
    rel_err_B = abs(B - Brec) / (1 + abs(B))

    print("Mean reconstruction error A: ", np.mean(rel_err_A.flatten()))
    print("Mean reconstruction error B: ", np.mean(rel_err_B))

    return (rel_err_A, rel_err_B)