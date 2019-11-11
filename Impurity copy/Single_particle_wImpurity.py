"""
IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |S_z,imp, n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which have a specified amount (n) of particles. This is done by the makeBase() function.  
A state is represented by a vector of probability amplitudes, accompanied by basisList, which hold information of what basis state each index in the state vector represents. 
The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). The spin operators are also represented as fermionic
operators (which works for S=1/2 at least), using S+ and S-.

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a state i, s is given as the 2i+s'th bit of the basis state, written in binary, where (spin) s=0 for UP and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s). The impurity spin state is at offset 2N.

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

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

#from Functions_fermionic_multiplication import *

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh, eigs

from numba import jit
# UTILITY ######################################################################

def lengthOfBasisFunc(Nlevels, nParticles):
	"""
	Returns the number of basis states for a given number of levels and a given number of particles. The number is equal to 
	the number of possible configurations where we select nParticles on Nlevels, times 2 for the two states of the impurity.
	"""
	a = 2 * comb(2*Nlevels, nParticles, exact=True)
	return a

def defineBase(Nlevels, nParticles, lengthOfBasis):
	"""
	From all basis states for a system with N levels, returns a list of basis states with nParticles number of particles.
	This defines a subspace of the Hilbert space with n-filling. The basis has all basis states as the system without the 
	impurity (when the impurity is in state 0), plus all the basis states with an additional 1 attached in front of the number
	(when the impurity is in state 1). The second part of the basis is obtained by adding 2^2N to all numbers in the original basis.
	"""
	resList = np.zeros(lengthOfBasis//2, dtype=int) 	#resList is doubled later in this function

	#build a list of basis states
	count=0
	for i in range(2**(2*Nlevels)):
		if countSetBits(i) == nParticles:
			resList[count] = i
			count+=1

	addImp = 2**(2*Nlevels)		
	resList = np.concatenate((resList, [i + addImp for i in resList]))		

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

# ELECTRON OPERATORS ###########################################################

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

# IMPURITY OPERATORS ###########################################################

@jit
def impSz(m):
	"""
	Given a basis state, returns the spin of the impurity. 0=UP, 1=DOWN.
	"""
	m1 = flipBit(m, 2*N)
	if m1>m:
		return 0
	else:
		return 1

@jit
def spinSpsm(i, j, m, N):
	"""
	Calculates the result of action of the operator S+ s-_ij to a basis state m.
	"""
	m1 = flipBit(m, 2*N)	#check if the impurity spin is down
	if m1<m:
		if countingOp(i, 1, m1, N) == 0:
			if countingOp(j, 0, m1, N) == 1:
				m1 = flipBit(m1, 2*(N-i)-1-1)
				m1 = flipBit(m1, 2*(N-j)-1)

				return m1				
	return 0			

@jit
def spinSmsp(i, j, m, N):
	"""
	Calculates the result of action of the operator S- s+_ij to a basis state m.
	"""
	m1 = flipBit(m, 2*N)	#check if the impurity spin is up
	if m1>m:
		if countingOp(i, 0, m1, N) == 0:
			if countingOp(j, 1, m1, N) == 1:
				m1 = flipBit(m1, 2*(N-i)-1)
				m1 = flipBit(m1, 2*(N-j)-1-1)
				
				return m1				
	return 0	

@jit
def spinSzsz(i, j, m, N):
	"""
	Calculates the application of the term Sz ( c_iUP^dag c_jUP - c_iDOWN^dag c_JDOWN ) to a basis state m.
	"""
	#term with spin UP
	if countingOp(i, 0, m, N) == 0 and countingOp(j, 0, m, N) == 1:
		m1 = flipBit(m, 2*(N-i)-1)
		m1 = flipBit(m1, 2*(N-j)-1)		

	else:
		m1 = 0

	#term with spin DOWN	
	if countingOp(i, 1, m, N) == 0 and countingOp(j, 1, m, N) == 1: 
		m2 = flipBit(m, 2*(N-i)-1-1)
		m2 = flipBit(m2, 2*(N-j)-1-1)	

	else:
		m2 = 0

	return m1, m2	

@jit
def spinInteractionOnState(i, j, state, N, basisList, lengthOfBasis):
	"""
	Calculates the result of the spin interacting term on sites i, j on a state and impState.
	"""
	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
				
			#S+ s-		
			m1 = spinSpsm(i, j, basisList[k], N)
			if m1!=0:
				#Check crcrananOnState() for comments
				new_state[np.searchsorted(basisList, m1)] += 0.5 * coef

			#S- s+	
			m2 = spinSmsp(i, j, basisList[k], N)	
			if m2!=0:
				#Check crcrananOnState() for comments
				new_state[np.searchsorted(basisList, m2)] += 0.5 * coef
			
			#Sz sz
			impSCoef = -2*impSz(basisList[k]) + 1	#gives 1 for Sz=0 (UP) and -1 for Sz=1 (DOWN)
			m3, m4 = spinSzsz(i, j, basisList[k], N)
			if m3 != 0:
				new_state[np.searchsorted(basisList, m3)] += 0.5 * impSCoef * coef
				
			if m4 != 0:	
				new_state[np.searchsorted(basisList, m4)] -= 0.5 * impSCoef * coef
				
	return new_state		

# HAMILTONIAN ##################################################################

#@jit
def HonState(d, alpha, J, state, N, basisList, lengthOfBasis):
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

	kinetic, interaction, impurity = 0, 0, 0

	for i in range(N):
		niUP = CountingOpOnState(i, 0, state, N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, N, basisList, lengthOfBasis)

		#kinetic term
		kinetic += eps(i, d, N) * (niUP + niDOWN)
		
		for j in range(N):
			interaction += crcrananOnState(i, j, state, N, basisList, lengthOfBasis)

			impurity += spinInteractionOnState(i, j, state, N, basisList, lengthOfBasis)

	return kinetic - d*alpha*interaction - J*impurity

class HLinOP(LinearOperator):
	"""
	This is a class, built-in in scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	"""	
	def __init__(self, d, alpha, J, N, basisList, lengthOfBasis, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)
		self.d = d
		self.alpha = alpha
		self.J = J
		self.N = N
		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis

	def _matvec(self, state):
		return HonState(self.d, self.alpha, self.J, state, self.N, self.basisList, self.lengthOfBasis)

# DIAGONALISATION ##############################################################

def exactDiag(d, alpha, J, N, basisList, lengthOfBasis):
	"""Diagonalises the hamiltonian using exact diagonalisation."""

	matrika1=[]
	for i in range(lengthOfBasis):
		
		stanje1 = np.zeros(lengthOfBasis)
		stanje1[i] = 1

		vrstica = []
		for j in range(lengthOfBasis):
			stanje2 = np.zeros(lengthOfBasis)
			stanje2[j] = 1

			vrstica.append(np.dot(stanje2, HonState(d, alpha, J, stanje1, N, basisList, lengthOfBasis)))
			

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

def findFirstExcited(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first excited state (state with smallest energy,
	different than ground state). The resolution is set by precision.
	"""
	for i in range(1, len(values)):
		if abs(values[i] - values[0]) > precision:
			return values[i]

	print("NOTHING FOUND; PROBLEM IN SPECTRUM?")

def findSecondExcited(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first excited state (state with smallest energy,
	different than ground state). The resolution is set by precision.
	"""
	E1 = findFirstExcited(values, precision)

	for i in range(1, len(values)):
		if values[i] - E1 > precision:
			return values[i]

	print("NOTHING FOUND; PROBLEM IN SPECTRUM?")

# PHYSICS ######################################################################

@jit
def eps(i, d, N):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return d*(i - ((N-1)/2))

################################################################################

def LanczosDiag(N, n, d, alpha, J, NofValues=4):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	"""

	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
	values = eigsh(LinOp, k=NofValues, which="SA", return_eigenvectors=False)[::-1]

	return values

def LanczosDiag_states(N, n, d, alpha, J, NofValues=4):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	"""

	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
	values, vectors = eigsh(LinOp, k=NofValues, which="SA")

	return values, vectors, basisList

# CALCULATION ##################################################################
states_print = 0
spectrum_J_plot = 0
spectrum_alpha_plot = 1
GSE1E2_plot = 0
################################################################################

if states_print:

	N=7
	n=N
	d, alpha = 1, 0.0000001
	J = 1.5
	
	omegaD = 0.5*N*d	
	Delta = omegaD/(2*np.sinh(1/alpha))	

	print(r"d/\Delta", d/Delta)

	val, vec, basisList = LanczosDiag_states(N, n, d, alpha, J, NofValues=10)

	print(val)

	print()
	for i in range(len(val)):
		v = vec[:, i]
		print("STATE", i)
		for k in range(len(v)):
			if abs(v[k])>0.09:
				print(k, v[k], bin(basisList[k]))
		print()

if spectrum_J_plot:

	NofStates = 20

	d = 1
	Jmin, Jmax = 0, 5

	for N in [7]:
		for alpha in [1, 3]:
			print(N, alpha)

			omegaD = 0.5*N*d	
			Delta = omegaD/(2*np.sinh(1/alpha))	
			StateLists = [[] for i in range(NofStates)]
			JList = np.linspace(Jmin, Jmax, 20)
			for J in JList:
				print(J)

				values = LanczosDiag(N, N, d, alpha, J, NofStates)

				for i in range(NofStates):
					
					StateLists[i].append((values[i]-values[0])/d)
						

			for i in range(NofStates):
				lw = 5-3*i/NofStates
				if i%2:
					ls = "--"
				else:
					ls = "-."	
				plt.plot(JList, StateLists[i], linestyle=ls)

			plt.legend()

			plt.xlabel("J/d")
			plt.ylabel(r"$(E-E_{GS})/d$")
			plt.title(r"$N={0}, \alpha={1}, d/\Delta={2}$".format(N, alpha, d/Delta))

			plt.grid()
			plt.tight_layout()
			
			name = "Spectrum_J_alpha{0}_N{1}.pdf".format(alpha, N)
			plt.savefig("Slike/"+name)
			plt.close()
			#plt.show()

if spectrum_alpha_plot:

	NofStates = 20

	d = 1
	for N in [7]:
		GSList, E1List, E2List = [], [], []
		# = 0.2
		for J in [0, 0.5, 1, 1.5, 2, 2.5]:
			print(N, J)
			print()
			x=np.logspace(-1.3, 0.5, 10)
			alphalist = [1/np.arcsinh(0.25*N*d/Delta) for Delta in x]
			dDelta = []
			StateLists = [[] for i in range(NofStates)]
			for alpha in alphalist:
				
				omegaD = 0.5*N*d	
				Delta = omegaD/(2*np.sinh(1/alpha))	
				dDelta.append(d/Delta)
				#print(d/Delta)

				
				values = LanczosDiag(N, N, d, alpha, J, NofStates)

				
				for i in range(NofStates):


					StateLists[i].append((values[i]-values[0])/d)
						

			for i in range(NofStates):
				if i%3==0:
					ls = "--"
				elif i%3==1:
					ls = "-."	
				else:
					ls = "-"
				plt.plot(dDelta, StateLists[i], linestyle=ls)


			plt.legend()

			plt.xlabel(r"$d/\Delta$")
			plt.ylabel(r"$(E-E_{GS})/d$")
			plt.title(r"$N={0}, J={1}d$".format(N, J))

			plt.grid()
			plt.tight_layout()
			name = "Spectrum_alpha_J{0}_N{1}.pdf".format(J, N)
			plt.savefig("Slike/"+name)
			plt.close()
			#plt.show()

if GSE1E2_plot:

	N=6
	d, alpha = 1, 1
	GSList, E1List, E2List = [], [], []
	Jmin, Jmax = 0, 1


	omegaD = 0.5*N*d	
	Delta = omegaD/(2*np.sinh(1/alpha))	

	print(d/Delta)

	JList = np.linspace(Jmin, Jmax, 10)
	for J in JList:
		
		values = LanczosDiag(N, N, d, alpha, J)

		GS = values[0]
		E1 = findFirstExcited(values, precision=1e-10)
		E2 = findSecondExcited(values, precision=1e-10)

		print(GS)
		print(E1)
		print()

		GSList.append(GS)
		E1List.append(E1)
		E2List.append(E2)

	plt.plot(JList, GSList, label="GS")
	plt.plot(JList, E1List, label="E1")
	plt.plot(JList, E2List, label="E2")


	plt.legend()

	plt.xlabel("J")
	plt.title("N="+str(N))

	plt.grid()
	plt.show()









