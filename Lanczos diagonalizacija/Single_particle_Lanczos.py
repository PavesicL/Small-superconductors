"""
Attempt to reproduce results from: A. Mastellone, G. Falce, R. Fazio, Small Superconducting Grain in the Canonical Ensemble, Phys. Rev. Lett., 1998.
Using Lanczos diagonalisation of the superconducting hamiltonian. 

IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which have a specified amount (n) of particles. This is done by the makeBase() function.  
A state is represented by a vector of probability amplitudes, accompanied by basisList, which holds information of what basis state each index in the state vector represents. 
Finding the inverse information, what is the index of a given basis state, is not simple. Currently this is done by np.searchsorted(). Using a dictionary slows the process by
quite a lot (approximately factor of 10).
The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). 

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a state i, s is given as the 2i+s'th bit of the basis state, written in binary, where (spin) s=0 for UP and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s).

DIAGONALISATION
Diagonalisation of the Hamiltonian is implemented in two ways.
The exactDiag() function brute-forces all the matrix elements and diagonalises them using a numpy function. This works, bot not for cases much above N=n=5.
The alternative is the Lanczos algorithm. To perform Lanczos diagonalisation, one only needs to know how a linear operator acts on a state. This is implemented
using the numpy class LinearOperator. It allows exactly this - to define a linear operator and diagonalise it, using scipy.sparse.linalg.eigsh. 
The second option is much faster. 

NUMBA
As numba dislikes dictionaries, this is a reworked verison of the same program. The previous version used a dictionary (basisDict), as a way to map between basis states and 
indeces of the state vector. This works faster using numba. Numba also dislikes the numpy function for the binomial symbol (np.comb()), which is why the basis creation does
not work with jit.
"""

################################################################################
print("The function LanczosDiag() is actually not called anywhere. Every calculation has the same three lines copy pasted.") 
print("This script is kind of useless (== works much slower than the one with pairs with no advantages) anyway.")
print()
################################################################################

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh, eigs

from numba import jit
# UTILITY ######################################################################

def lengthOfBasisFunc(Nlevels, nParticles):
	"""
	Returns the number of basis states for a given number of levels and a given number of particles. 
	"""
	a = comb(2*Nlevels, nParticles, exact=True)
	return a

def defineBase(Nlevels, nParticles, lengthOfBasis):
	"""
	From all basis states for a system with N levels,  returns a list of basis states with nParticles number of particles.
	This defines a subspace of the Hilbert space with n-filling.
	"""

	resList = np.zeros(lengthOfBasis, dtype=int)

	#build a list of basis states
	count=0
	for i in range(2**(2*Nlevels)):
		if countSetBits(i) == nParticles:
			resList[count] = i
			count+=1


	return resList

def makeBase(Nlevels, nParticles):
	"""	
	Calls the two previous functions, and creates a basis - returns its length and a list of states.
	"""	
	lengthOfBasis = lengthOfBasisFunc(Nlevels, nParticles)
	basisList = defineBase(Nlevels, nParticles, lengthOfBasis)

	return lengthOfBasis, basisList

# BIT-WISE #####################################################################

@jit
def flipBit(n, offset):
	"""Flips the bit at position offset in the integer n."""
	mask = 1 << offset
	return(n ^ mask)

@jit
def countSetBits(n): 
	"""Counts the number of bits that are set to 1 in a given integer."""
	count = 0
	while (n): 
		count += n & 1
		n >>= 1
	return count 

# OPERATORS ####################################################################

@jit
def crcranan(i, j, m, N):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a basis vector m, given as an integer.
	"""

	offi = 2*(N-i)-1	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-j)-1

	if offi>2*N or offj>2*N:
		print("WARNING: the offset is too large!")

	#at each step, the if statement gets TRUE if the operation is valid (does not destroy the state)
	m1 = flipBit(m, offj-0)
	if m>m1:
		m2 = flipBit(m1, offj-1)
		if m1>m2:
			m3 = flipBit(m2, offi-1)
			if m2<m3:
				m4 = flipBit(m3, offi)
				if m3<m4:
					return m4

	return 0  

@jit
def crcrananOnState(i, j, state, N, basisList, lengthOfBasis):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a state.
	"""

	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):

		coef = state[k]
		if coef!=0:

			m = crcranan(i, j, basisList[k], N)

			if m!=0:
				"""
				THIS IS ONE OF THE BOTTLENECKS - given a resulting state (m), find which index in basisList it corresponds to. 
				The solution with the dictionary (of type { state : index_in_basisList }) turns out to be slow. So is the usage 
				of the numpy function np.where(), which finds all occurences of a given value in a list. The current solution is 
				using searchsorted, a function which returns (rouglhy) the first position of the given value, but needs the list 
				to be sorted. basisList is sorted by construction, so this works. 
				"""
				res = np.searchsorted(basisList, m)

				new_state[res] += coef
				
	return new_state
	
@jit					
def countingOp(i, s, m, N):
	"""
	Calculates the application of the counting operator to a basis state m. Returns 0 or 1, according to the occupation of the energy level.
	"""
	m1 = flipBit(m, 2*(N-i)-1-s)
	if m1 < m:
		return 1
	else:
		return 0

@jit
def CountingOpOnState(i, s, state, N, basisList, lengthOfBasis):
	"""	
	Calculates the application of the counting operator on a given state vector state.
	"""
	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += countingOp(i, s, basisList[k], N)*coef
	
	return new_state		

# HAMILTONIAN ##################################################################

#@jit
def HonState(d, alpha, state, N, D, basisList, lengthOfBasis):
	"""
	Calculates the action of the Hamiltonian to a given state.
	INPUT:
	d, alpha - physical constants (float)
	state - the state vector (vector)
	N - number of levels (int). There is 2*N available single-particle states (2 for spin)
	basisList - a list of all basis states (list)
	basisDict - a dictionary of positions of basis states in basisList (dictionary)
	lengthOfBasis - the length of the state vector (int)
	"""

	kinetic, interaction = 0, 0

	for i in range(N):
		kinetic += eps(i, d, D) * (CountingOpOnState(i, 0, state, N, basisList, lengthOfBasis) + CountingOpOnState(i, 1, state, N, basisList, lengthOfBasis)) 
		
		for j in range(N):
			interaction += crcrananOnState(i, j, state, N, basisList, lengthOfBasis)

	return kinetic - d*alpha*interaction

class HLinOP(LinearOperator):
	"""
	This is a class, built-in in scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	"""	
	def __init__(self, d, alpha, N, D, basisList, lengthOfBasis, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)
		self.d = d
		self.alpha = alpha
		self.N = N
		self.D = D
		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis

	def _matvec(self, state):
		return HonState(self.d, self.alpha, state, self.N, self.D, self.basisList, self.lengthOfBasis)

# DIAGONALISATION ##############################################################

def exactDiag(d, alpha, N, D, basisList, lengthOfBasis):
	"""Diagonalises the hamiltonian using exact diagonalisation."""

	matrika1=[]
	for i in range(lengthOfBasis):
		
		stanje1 = np.zeros(lengthOfBasis)
		stanje1[i] = 1

		vrstica = []
		for j in range(lengthOfBasis):
			stanje2 = np.zeros(lengthOfBasis)
			stanje2[j] = 1

			vrstica.append(np.dot(stanje2, HonState(d, alpha, stanje1, N, D, basisList, lengthOfBasis)))
			

		matrika1.append(vrstica)	

	val1, vec1 = np.linalg.eigh(matrika1)
 
	return val1, vec1

# DATA ANALYSIS ################################################################

def findGap(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first spectroscopic gap, bigger than PRECISION.
	"""

	for i in range(1, len(values)):
		difference = values[i] - values[i-1]
		if difference > precision:
			return difference

# PHYSICS ######################################################################

@jit
def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return -D + d*i + d/2

################################################################################

def LanczosDiag(D, N, n, d, alpha):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	"""

	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, N, D, basisList, lengthOfBasis) 
	values = eigsh(LinOp, k=20, which="SA", return_eigenvectors=False)[::-1]

	return values


def LanczosDiagStates(D, N, n, d, alpha):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	"""

	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, N, D, basisList, lengthOfBasis) 
	val, vec = eigsh(LinOp, k=20, which="SA")

	return val, vec

# CALCULATION ##################################################################

time_test=0
GS_comparison_plot=0
parity_gap_plot=0
spectroscopic_gap_plot=0

################################################################################

if 0:
	D, N = 1, 4
	n = 5
	d = 2*D/N
	alpha = 1


	val = LanczosDiag(D, N, n, d, alpha)

	print(val)

if 1:
	D = 1
	N = 8
	n = N
	d = 2*D/N
	alpha = 1

	lengthOfBasis, basisList = makeBase(N, n)

	val, vectors = LanczosDiagStates(D, N, n, d, alpha)
	#vec = vec.transpose()

	print(val)
	for i in range(len(val)):

		vec = vectors[:, i]

		degenerate=0
		if i<len(val)-1 and i>0:
			if np.abs(val[i]-val[i+1])<1e-5 or np.abs(val[i]-val[i-1])<1e-5:
				degenerate=1
		
		a = np.multiply(val[i],vec)
		Hs = HonState(d, alpha, vec, N, D, basisList, lengthOfBasis)

		print(norm(Hs - a), degenerate)





if time_test:

	import time
	
	print("STARTED")

	N = 9
	n = N 
	
	lengthOfBasis, basisList = makeBase(N, n)

	d, alpha = 1, 1

	timetable=[]
	for i in range(5):
		print(i)
		start = time.time()

		LinOp = HLinOP(d, alpha, N, basisList, lengthOfBasis) 
		values = eigsh(LinOp, k=4, which="SA", return_eigenvectors=False)[::-1]

		end = time.time()

		timetable.append(end-start)

	print(timetable)	
	print(np.average(timetable))
	print(min(timetable))

if GS_comparison_plot:

	N = 11
	n = N


	lengthOfBasis, basisList = makeBase(N, n)
	d, alpha = 1, 1

	print("PARAMETERS:")
	print(N, n, lengthOfBasis)
	
	
	print("Lanczos...")
	LinOp = HLinOP(d, alpha, N, basisList, lengthOfBasis) 
	values = eigsh(LinOp, k=3, which="SA", return_eigenvectors=False)[::-1]
	
	print(values)
	#values, vectors = eigsh(LinOp, k=20, which="SA")
	v0 = vectors[:, 0]
	v1 = vectors[:, 1]
	v2 = vectors[:, 2]


	print("EIGENVALUES:")
	print(values)
	print()
	print("EIGENVECTORS")
	print(v0)
	print(v1)
	print(v2)
		


	sez, sez1, sez2 = [], [], []
	for i in range(len(v0)):
		if np.abs(v0[i])>1e-2:
			sez.append(i)
		if np.abs(v1[i])>1e-2:
			sez1.append(i)
		if np.abs(v2[i])>1e-2:
			sez2.append(i)

	print()
	print("MOST PROMINENT EIGENSTATES:")
	print(sez)	
	print(sez1)	
	print(sez2)
	print([bin(basisList[i]) for i in sez])
	print([bin(basisList[i]) for i in sez1])
	print([bin(basisList[i]) for i in sez2])

	plt.semilogy(np.abs(v0), label="0")
	plt.semilogy(np.abs(v1), label="1")
	plt.semilogy(np.abs(v2), label="2")

	plt.legend()

	plt.grid()
	plt.show()

if parity_gap_plot:
	print("NE DELA")
	NN = 5
	n = NN

	DeltaP=[]

	d = 1
	x=np.logspace(-0.5, 1, 10)
	alphalist = [1/np.arcsinh(0.25*NN*d/Delta) for Delta in x]
	dDelta=[]
	for alpha in alphalist:
		
		#PARAMETERS	
		
		omegaD = 0.5*NN*d	
		Delta = omegaD/(2*np.sinh(1/alpha))	
		dDelta.append(d/Delta)
		
		print(d/Delta)

		#CALCULATION OF THE PARITY GAP
		GSE=[]
		for N in [NN-1, NN, NN+1]:
			print(N)
			lengthOfBasis, basisList = makeBase(N, n)
			
			LinOp = HLinOP(d, alpha, N, basisList, lengthOfBasis) 
			values = eigsh(LinOp, k=3, which="SA", return_eigenvectors=False)[::-1]
			GSE.append(values[0]/(0.5*N*d))	
		DeltaP.append((GSE[1] - 0.5*(GSE[0] + GSE[2]))/Delta)
		print(GSE)
		print(Delta)
		print((GSE[1] - 0.5*(GSE[0] + GSE[2]))/Delta)
		print()

	plt.scatter(dDelta, DeltaP)	
	plt.grid()
	plt.show()

if spectroscopic_gap_plot:
		
	for N in [8, 9]:
		print(N)
		n = N
				
		lengthOfBasis, basisList = makeBase(N, n)

		d = 1

		x=np.logspace(-1.3, 0.5, 20)
		alphalist = [1/np.arcsinh(0.25*N*d/Delta) for Delta in x]
		dDelta, gapList = [], []
		for alpha in alphalist:

			omegaD = 0.5*N*d	
			Delta = omegaD/(2*np.sinh(1/alpha))	
			dDelta.append(d/Delta)

			print(d/Delta)


			LinOp = HLinOP(d, alpha, N, basisList, lengthOfBasis) 
			values = eigsh(LinOp, k=4, which="SA", return_eigenvectors=False)[::-1]

			#print(values)

			gap = findGap(values, precision=1e-10)
			#print(gap)
			gapList.append(gap/d)

		print()

		plt.scatter(dDelta, gapList, label=N)	

	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$E_G/d$")	

	plt.hlines(xmin= 0, xmax=20, y=1, linestyle="dashed")

	plt.ylim(0, 4)
	plt.xlim(0,20)

	plt.legend()	
	plt.grid()

	plt.tight_layout()
	plt.show()
	print("DONE")


		











