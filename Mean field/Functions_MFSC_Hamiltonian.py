"""
These are functions used to diagonalise the hamiltonian
	H = sum_j eps_j c_j^dag c_j  -  g sum_j c_j,up^dag c_j,down^dag c_j,down c_j,up
in the nambu basis, and optimizing the value of delta through iterative solving.
"""

import numpy as np 
import matplotlib.pyplot as plt

from scipy.linalg import block_diag
from numpy.linalg import eigh, eig
############################################################
#Parameters

delta = 0.2	#initial guess for the gap

############################################################
#Functions

def eps(i, N):
	"""Dispersion."""
	i=i+1	#THIS IS SO THE DISPERSION IS THE SAME AS IN mathematica
	return -1 + (2/(N+1) * i)

def block(i, delta, mu, N):
	"""A block of the Hamiltonian."""
	return np.array([	[eps(i, N)-mu, delta],
						[delta, -eps(i, N)+mu]])

def Hamiltonian(delta, mu, N):
	"""Constructs the Hamiltonian in matrix form."""

	return block_diag(*[block(i, delta, mu, N) for i in range(N)])


def newDelta(delta, mu, N):
	"""One iteration of the computation of delta. 
	Diagonalises the Hamiltonian for a given delta, and computes a new one. Returns the eigenenergies also."""

	u,v = np.zeros((N,N)), np.zeros((N,N))

	energies, vectors = eigh(Hamiltonian(delta, mu, N))
	vectors = np.transpose(vectors)

	for i in range(N):
		u[i] = vectors[i, ::2]
		v[i] = vectors[i, 1::2]	

	delta=0
	for i in range(N):
		for j in range(N):
			delta +=  u[i,j]*v[i,j]
	
	g = 2*0.5/N

	return -g* delta, energies, u, v

def optimizeDelta(N, initDelta, mu=0, precision=1e-6, maxIteration=100, verbosity=False):
	"""The function diagonalises the hamiltonian for N particles and calculates the band gap (delta) iteratively, using the argument initDelta as the initial guess. 
	If verbosity==True, prints the values between iterations. Returns the last delta and eigenenergies.
	
	N - number of energy levels
	initDelta - initial guess for delta
	precision - iteration stops when delta changes by less than precision from one iteration to the next
	maxIteration - max number of iterations performed."""

	oldDelta, iteration = initDelta+1, 0
	delta = initDelta
	while np.abs(delta/oldDelta -1) > precision and iteration < maxIteration:

		oldDelta = delta	
		delta, energies, u, v = newDelta(delta, mu, N)

		if verbosity:
			print("\nIteration: "+str(iteration))
			print("Delta: "+str(delta))
		
		iteration+=1

		if iteration==maxIteration:
			print("WARNING: maxIteration limit hit, result is not necessarily converged.")

	if verbosity:		
		print("Iterations:"+str(iteration))

	return delta, energies, u, v


def deltaofN(start, stop, step, initDelta, precision=1e-6, maxIteration=100):
	"""Returns a list of calculated band gaps for N in the interval [start, stop) with given step."""

	deltaList=[]
	for N in range(start, stop, step):
		deltaList.append(optimizeDelta(N, initDelta, precision=precision, maxIteration=maxIteration)[0])	

	return deltaList

