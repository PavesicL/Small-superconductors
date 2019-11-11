"""
Written for solving a discrete superconducting hamiltonian
	H = sum_j eps_j c_j^dag c_j  -  g sum_ij c_i,up^dag c_i,down^dag c_j,down c_j,up.
Diagonalises it in the Nambu basis and solves the equation for Delta iteratively. The result is the superconduting gap Delta and the eigenenergies of the system. 
"""

import numpy as np 
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from numpy.linalg import eigh, eig
############################################################
#Parameters

#N = 	#number of levels
delta = 0.2	#initial guess for the gap
maxIteration = 100 	#maximum number of iterations

############################################################
#Functions

def eps(i, N):
	"""Dispersion."""
	i=i+1	#THIS IS SO THE DISPERSION IS THE SAME AS IN mathematica
	return -1 + (2/(N+1) * i)

def block(i, delta, N):
	"""A block of the Hamiltonian."""
	return np.array([	[eps(i, N), delta],
						[delta, -eps(i, N)]])

def Hamiltonian(delta, N):
	"""Constructs the Hamiltonian in matrix form."""

	return block_diag(*[block(i, delta, N) for i in range(N)])

def newDelta(delta, N):
	"""One iteration of the computation of delta. 
	Diagonalises the Hamiltonian for a given delta, and computes a new one. Returns the eigenenergies also."""

	u,v = np.zeros((N,N)), np.zeros((N,N))

	energies, vectors = eigh(Hamiltonian(delta, N))
	vectors = np.transpose(vectors)

	for i in range(N):
		u[i] = vectors[i, ::2]
		v[i] = vectors[i, 1::2]	

	delta=0
	for i in range(N):
		for j in range(N):
			delta +=  u[i,j]*v[i,j]
	
	g = 2*0.5/N

	return -g* delta, energies

############################################################

def optimizeDelta(N, initDelta, precision=1e-6, maxIteration=100, verbosity=False):
	"""The function diagonalises the hamiltonian for N particles and calculates the band gap (delta) iteratively, using the argument initDelta as the initial guess. 
	If verbosity==True, prints the values between iterations. Returns the last delta and eigenenergies.
	precision: iteration stops when delta changes by less than precision from one iteration to the next
	maxIteration: max number of iterations calculated."""

	oldDelta,iteration=1,0
	delta=initDelta
	while np.abs(oldDelta/delta -1) > precision and iteration < maxIteration:

		oldDelta = delta	
		delta, energies = newDelta(delta, N)

		if verbosity:
			print("Delta: " +str (delta))
		
		iteration+=1

		if iteration==maxIteration:
			print("maxIteration limit hit, result is not necessarily converged.")

	if verbosity:		
		print("Iterations:"+str(iteration))

	return delta, energies  


def deltaofN(start, stop, step, initDelta, precision=1e-6, maxIteration=100):
	"""Returns a list of calculated band gaps for N in the interval [start, stop) with given step."""

	deltaList=[]
	for N in range(start, stop, step):
		deltaList.append(optimizeDelta(N, initDelta, precision=precision, maxIteration=maxIteration)[0])	

	return deltaList	

############################################################
print_delta = 1
plot_DOS = 0
plot_deltaOfN = 0
############################################################
N = 5


if print_delta:
	delta, energies = optimizeDelta(N, 0.1)

	print("delta: " + str(delta))
	print("energies:")
	print(energies)

if plot_DOS:
	delta, energies = optimizeDelta(N, 0.1)

	plt.hist(energies, bins=40, range=(-1, 1))
	plt.show()

if plot_deltaOfN:
	start, stop, step = 1, 100, 1

	B = deltaofN(start, stop, step, 0.1)

	plt.grid()

	plt.scatter([i for i in range(start, stop, step)], B)
	plt.xlabel(r"$N$")
	plt.ylabel(r"$\Delta$")

	#plt.yscale("log")
	#plt.xscale("log")


	plt.show()









