"""
Finds the spectrum and the eigenstates of the hamiltonian of a small superconductor, coupled to an impurity. In this specific implementation, one has control over the number of
particles in the system and the total spin of the system. 

IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |n_impUP, n_impDOWN, n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which has a specified amount of particles with spin up (nUP) and spin down (nDOWN). 
This is done by the makeBase() function. A state is represented by a vector of probability amplitudes, accompanied by basisList, which hold information of what basis state 
each index in the state vector represents. The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). The spin 
operators are also represented as fermionic operators (which works for S=1/2 at least), using S+ and S-. Either we can extract results for a set combination of nUP and nDOWN,
or take into account all possible pairs (nUP, nDOWN), to obtain the entire spectrum.

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a single particle state (i, s) is given as the 2i+s'th bit of the basis state, written in binary (and counted from the left), where (spin) s=0 for UP 
and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s) = 2(N-i)-1-s. The impurity spin state is at offset 2N and 2N+1.

DIAGONALISATION
Diagonalisation is implemented using the Lanczos algorithm. The linear operator H is implemented using the numpy class LinearOperator. It allows us to define a linear operator
and diagonalise it, using scipy.sparse.linalg.eigsh. 
The complete diagonalisation, in the functions LanczosDiag_states() and LanczosDiag_states() (if one wants to obtain the eigenvectors too) is done subspace by subspace. The
smallest subspace of the Hamiltonian is one with defined number of particles (n) and total spin z of the system. Alternatively, one defines n and the number of particles with
spin UP/DOWN in the system. Here, we have to count the impurity as a particle with spin in either direction, as H consists of terms like S+ s-ij, which conserve only the spin
of the entire system. This is why we have nwimpUP and nwimpDOWN (number of particles with given spin, including the impurity) and it holds that 
	nwimpUP + nwimpDOWN = n + 1,
	1/2 (nwimpUP + nwimpDOWN) = Sz.
All quantities in these two equations are constant in a given subspace. 
It is possible to only obtain the spectrum of the system in the specified state (n, nwimpUP, nwimpDOWN), using LanczosDiag_nUPnDOWN or LanczosDiagStates_nUPnDOWN.

NUMBA
The jit operator works nicely with everything.

SPECTRAL TRANSITION


"""

################################################################################

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh

from numba import jit

# UTILITY ######################################################################

def printV(vector, basisList, prec=0.1):
	"""
	Prints the most prominent elements of a vector and their amplitudes.
	"""
	for i in range(len(vector)):
		if abs(vector[i])>prec:
			print(basisList[i], bin(basisList[i]), vector[i])

def checkParams(N, n, nwimpUP, nwimpDOWN):
	"""
	Checks if the parameter values make sense.
	"""
	allOK = 1

	if n>2*N:
		print("WARNING: {0} particles is too much for {2} levels!".format(n, N))
		allOK=0

	if nwimpUP + nwimpDOWN != n+1:
		print("WARNING: some mismatch in the numbers of particles!")
		allOK=0

	if allOK:
		print("Param check OK.")
					
# BASIS ########################################################################

@jit
def makeBase(N, nwimpUP, nwimpDOWN):
	"""
	Creates a basis
	"""
	resList = []

	for m in range(2**((2*N)+2)):
		if impurityBits(m, N)==(0, 1) or impurityBits(m, N)==(1, 0):	#only one spin on the impurity		 
			if countSetBits(m) == nwimpUP + nwimpDOWN:					#correct number of spins
				if spinUpBits(m, N, allBits=True) == nwimpUP and spinDownBits(m, N, allBits=True) == nwimpDOWN:
					resList.append(m)

	lengthOfBasis = len(resList)
	resList = np.array(resList)

	return lengthOfBasis, resList

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

@jit
def bit(m, off):
	"""
	Returns the value of a bit at offset off in integer m.
	"""

	if m & (1 << off):
		return 1
	return 0

@jit
def impurityBits(m, N):
	"""
	Returns the values of the impurity bits.
	"""

	return bit(m, 2*N+1), bit(m, 2*N)

@jit
def spinUpBits(m, N, allBits=False):
	"""
	Counts the number of spin down electrons in the state. If allBits, the impurity level is also counted.
	"""

	count=0
	
	if allBits:
		for i in range(1, (2*N)+2, 2):
			if bit(m, i)==1:
				count+=1
	else:
		for i in range(1, 2*N, 2):
			if bit(m, i)==1:
				count+=1

	return count		

@jit
def spinDownBits(m, N, allBits=False):
	"""
	Counts the number of spin down electrons in the state. If allBits, the impurity level is also counted.
	"""

	count=0

	if allBits:
		for i in range(0, (2*N)+2, 2):
			if bit(m, i)==1:
				count+=1	

	else:
		for i in range(0, 2*N, 2):
			if bit(m, i)==1:
				count+=1

	return count	

@jit
def clearBitsAfter(m, off, length):
	"""Clears all bits of a number m with length length with offset smaller OR EQUAL off. Used to determine the fermionic +/- prefactor."""
	clearNUm = 0
	for i in range(off+1, length):
		clearNUm += 2**(i)

	return m & clearNUm

@jit
def prefactor_offset(m, off, N):
	"""
	Calculates the fermionic prefactor for a fermionic operator, acting on site given with the offset off. Sets all succeeding bits to zero and count the rest. 
	"""

	#turns off the impurity bit, as it does not contribute
	turnOffImp = (2**(2*N))-1	#this is the number 100000..., where 1 is at the position of the impurity.
	m = m & turnOffImp
	#set bits to zero
	m = clearBitsAfter(m, off, 2*N)

	#count the remaining 1s
	count = countSetBits(m)
	
	return (-1)**count

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
	return bit(m, 2*(N-i)-1-s)

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
def impSz(m, N):
	"""
	Given a basis state, returns the spin Sz of the impurity. 0=UP, 1=DOWN.
	"""
	return bit(m, 2*N)	#this bit specifies if the impurity has spin down. If it is 1, Sz=DOWN, otherwise, Sz=UP

@jit
def spinSpsm(i, j, m, N):
	"""
	Calculates the result of action of the operator S+ s-_ij to a basis state m.
	"""

	m1 = flipBit(m, 2*N)	
	m1 = flipBit(m1, (2*N)+1)	#flip the impurity spin if it is down
	if m1>m:
		prefactor = 1
		m2 = flipBit(m1, 2*(N-j)-1)	#c_jUP operator
		if m2<m1:
			prefactor *= prefactor_offset(m1, 2*(N-j)-1, N)
			m3 = flipBit(m2, 2*(N-i)-1-1)	#c_iDOWN^dag
			if m3>m2:
				prefactor *= prefactor_offset(m2, 2*(N-i)-1-1, N)

				return prefactor, m3		
	
	return 0, 0			

@jit
def spinSmsp(i, j, m, N):
	"""
	Calculates the result of action of the operator S- s+_ij to a basis state m.
	"""
	m1 = flipBit(m, 2*N)	
	m1 = flipBit(m1, (2*N)+1)	#flip the impurity spin if it is up
	if m1<m:
		prefactor = 1
		m2 = flipBit(m1, 2*(N-j)-1-1)	#C_jDOWN operator
		if m2<m1:
			prefactor *= prefactor_offset(m1, 2*(N-j)-1-1, N)
			m3 = flipBit(m2, 2*(N-i)-1)	#c_iUP operator
			if m3>m2:
				prefactor *= prefactor_offset(m2, 2*(N-i)-1, N)

				return prefactor, m3				
	
	return 0, 0

@jit
def SzszUp(i, j, m, N):
	"""
	Calculates the result of action of the operator Sz c_iUP c_jUP to a basis state m.
	"""
	
	#term with spin DOWN	
	prefactor = 1
	m11 = flipBit(m, 2*(N-j)-1)
	if m11 < m:
		m12 = flipBit(m11, 2*(N-i)-1)
		if m12 > m11:
			prefactor *= prefactor_offset(m, 2*(N-j)-1, N)
			prefactor *= prefactor_offset(m11, 2*(N-i)-1, N)
			
			return prefactor, m12
	return 0, 0
			
@jit
def SzszDown(i, j, m, N):
	"""
	Calculates the result of action of the operator Sz c_iDOWN c_jDOWN to a basis state m.
	"""

	#term with spin DOWN
	prefactor=1	
	m21 = flipBit(m, 2*(N-j)-1-1)
	if m21 < m:
		m22 = flipBit(m21, 2*(N-i)-1-1)
		if m22 > m21:
			prefactor *= prefactor_offset(m, 2*(N-j)-1-1, N)
			prefactor *= prefactor_offset(m21, 2*(N-i)-1-1, N)
			
			return prefactor, m22
	return 0, 0
	
@jit
def spinInteractionOnState(i, j, state, N, basisList, lengthOfBasis):
	"""
	Calculates the result of the spin interacting term on sites i, j on a vector state.
	"""
	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			
			#S+ s-		
			prefactor1, m1 = spinSpsm(i, j, basisList[k], N)
			if m1!=0:
				#Check crcrananOnState() for comments
				new_state[np.searchsorted(basisList, m1)] += 0.5 * coef * prefactor1 
				
				if basisList[np.searchsorted(basisList, m1)] != m1:
					print("NOT IN basisList!")

			#S- s+	
			prefactor2, m2 = spinSmsp(i, j, basisList[k], N)	
			if m2!=0:
				#Check crcrananOnState() for comments
				new_state[np.searchsorted(basisList, m2)] += 0.5 * coef * prefactor2
				
				if basisList[np.searchsorted(basisList, m2)] != m2:
					print("NOT IN basisList!")			

			#Sz sz			
			impSCoef = -2*impSz(basisList[k], N) + 1	#gives 1 for Sz=0 (UP) and -1 for Sz=1 (DOWN)
			prefactor3, m3 = SzszUp(i, j, basisList[k], N)
			if m3 != 0:
				new_state[np.searchsorted(basisList, m3)] += 0.5 * impSCoef * 0.5 * coef * prefactor3  
				
				if basisList[np.searchsorted(basisList, m3)] != m3:
					print("NOT IN basisList!")
			
			prefactor4, m4 = SzszDown(i, j, basisList[k], N)
			if m4 != 0:	
				new_state[np.searchsorted(basisList, m4)] += -0.5 * impSCoef * 0.5 * coef * prefactor4 
				
				if basisList[np.searchsorted(basisList, m4)] != m4:
					print("NOT IN basisList!")

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
	OUTPUT:
	the resulting vector, equal to H|state> (np.array)
	"""

	kinetic, interaction, impurity = np.zeros(lengthOfBasis), np.zeros(lengthOfBasis), np.zeros(lengthOfBasis)

	for i in range(N):
		niUP = CountingOpOnState(i, 0, state, N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, N, basisList, lengthOfBasis)

		#kinetic term
		kinetic += eps(i, d, N) * (niUP + niDOWN)
		
		for j in range(N):
			if d*alpha!=0:
				interaction += crcrananOnState(i, j, state, N, basisList, lengthOfBasis)

			if J!=0:	
				impurity += spinInteractionOnState(i, j, state, N, basisList, lengthOfBasis)


	return kinetic - d*alpha*interaction + (J/N)*impurity

class HLinOP(LinearOperator):
	"""
	This is a class, built-in to scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	The operator can be diagonalised using the function scipy.sparse.linalg.eigsh().
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

# PHYSICS ######################################################################

@jit
def eps(i, d, N):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return d*(i - ((N-1)/2))

# DIAGONALISATION ##############################################################

def LanczosDiag_nUPnDOWN(N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	N - number of levels
	n - number of particles in the system (EXCLUDING THE IMPURITY)
	nwimpUP, nwimpDOWN - number of particles with spin UP/DOWN, INCLUDING THE IMPURITY SPIN!
	d, alpha, J - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(N, nwimpUP, nwimpDOWN)

	if verbosity:
		checkParams(N, n, nwimpUP, nwimpDOWN)

	if lengthOfBasis==1:
		Hs = HonState(d, alpha, J, np.array([1]), N, basisList, lengthOfBasis)
		values = np.dot(basisList[0], Hs)

	else:	
		LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
		values = eigsh(LinOp, k=min(lengthOfBasis-1, NofValues), which="SA", return_eigenvectors=False)[::-1]

	return values

def LanczosDiagStates_nUPnDOWN(N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	N - number of levels
	n - number of particles in the system (EXCLUDING THE IMPURITY)
	nwimpUP, nwimpDOWN - number of particles with spin UP/DOWN, INCLUDING THE IMPURITY SPIN!
	d, alpha, J - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(N, nwimpUP, nwimpDOWN)
	
	if verbosity:
		checkParams(N, n, nwimpUP, nwimpDOWN)

	if lengthOfBasis==1:
		Hs = HonState(d, alpha, J, np.array([1]), N, basisList, lengthOfBasis)
		values = np.dot(basisList[0], Hs)
		vectors = [np.array([1])]

	else:	
		LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
		values, vectors = eigsh(LinOp, k=min(lengthOfBasis-1, NofValues), which="SA")

	return values, vectors, basisList

def LanczosDiag(N, n, d, alpha, J, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	Computed as a combination of eigenenergies from all smallest subspace with set number of particles (n) and 
	total system spin (Sz = 1/2 (nwimpUP - nwimpDOWN)).
	"""

	val=[]

	for nwimpUP in range(max(0, n-N), min(N+1, n+1) + 1):	#+1 in range to take into account also the last case	

		nwimpDOWN = n - nwimpUP + 1

		if verbosity:
			print(nwimpUP, nwimpDOWN)

		val.extend(LanczosDiag_nUPnDOWN(N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=NofValues, verbosity=verbosity))

	return np.sort(val)	

def LanczosDiag_states(N, n, d, alpha, J, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	"""

	values, vectors, basisLists = [], [], []

	for nwimpUP in range(max(0, n-N), min(N+1, n+1) + 1):	#+1 in range to take into account also the last case	

		nwimpDOWN = n - nwimpUP + 1

		if verbosity:
			print(nwimpUP, nwimpDOWN)

		val, vec, basisList = LanczosDiagStates_nUPnDOWN(N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=verbosity)

		values.extend(val)
		vectors.extend(np.transpose(vec))
		basisLists.extend([basisList for i in range(len(val))])


	sortedIndices = np.argsort(values)	
	values, vectors, basisLists = np.asarray(values), np.asarray(vectors), np.asarray(basisLists)


	SortedValues = np.take_along_axis(values, sortedIndices, axis=0)
	SortedVectors = np.take_along_axis(vectors, sortedIndices, axis=0)
	SortedBasisLists = np.take_along_axis(basisLists, sortedIndices, axis=0)


	return SortedValues, SortedVectors, SortedBasisLists

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
	From a given set of eigenvalues, finds the first excited state (state with smallest energy
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

def transitionE(N, d, alpha, J, initn, finaln):
	"""
	Calculates the difference between ground state energies of a system with initn and finaln particles. If normalized=True, 
	it returns the energy, normalized with the value without the impurity. 
	"""
	
	initValue = LanczosDiag(N, initn, d, alpha, J, NofValues=2)[0]#/initn
	finalValue = LanczosDiag(N, finaln, d, alpha, J, NofValues=2)[0]#/finaln

	if initn>finaln:	#n->n-1 process
		return initValue - finalValue
	elif initn<finaln:	#n->n+1 process
		return finalValue - initValue
			
# CALCULATION ##################################################################
states_print = 0
spectrum_J_plot = 0
spectrum_alpha_plot = 0
spectral_transition_ofJ_plot = 0
spectral_transition_of_alpha_plot = 0
################################################################################

if states_print:

	N=4
	n=N
	d, alpha = 1, 0
	J = N * 0.1
	NNN = 20

	val, vec, bas = LanczosDiag_states(N, n, d, alpha, J, NofValues=4)
	
	#val, vec = exactDiag(N, n, d, alpha, J)

	print(val)
	print()
	
	for i in range(len(val)):
		print("STATE {0}".format(i))
		printV(vec[i], bas[i], prec=0.1)
		print()

if spectrum_J_plot:
	save=0
	
	NofStates = 30

	d = 1
	Jmin, Jmax = 0, 5

	for N in [6, 7]:
		n=N
		for dDelta in [0.1]:
			omegaD = 0.5*N*d	
			alpha = 1/(np.arcsinh(dDelta*0.5*omegaD/d))
			Delta = omegaD/(2*np.sinh(1/alpha))	
			Delta2 = d/dDelta

			print()
			print(N, dDelta)


			StateLists = [[] for i in range(NofStates)]
			JList = np.linspace(Jmin, Jmax, 10)
			for J in JList:
			#for J in [1.7777777777777777]:
				print(J)

				values = LanczosDiag(N, n, d, alpha, J, NofValues=NofStates, verbosity=False)

				#print(values)

				for i in range(NofStates):
					
					StateLists[i].append((values[i]-values[0])/d)
						

			for i in range(NofStates):
				if i%2:
					ls = "--"
				else:
					ls = "-."
				plt.plot(JList, StateLists[i], linestyle=ls)

			plt.legend()

			plt.xlabel("J/d")
			plt.ylabel(r"$(E-E_{GS})/d$")
			plt.title(r"$N={0}, d/\Delta={1}$".format(N, d/Delta))

			plt.grid()
			plt.tight_layout()
			
			if save:
				name = "Spectrum_J_dDelta{0}_N{1}.pdf".format(dDelta, N)
				plt.savefig("Slike/"+name)
				plt.close()
			else:	
				plt.show()
				#plt.close()

if spectrum_alpha_plot:
	save=0

	NofStates = 20

	d = 1
	for N in [6, 7]:
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

				

				values = LanczosDiag(N, n, d, alpha, J, NofValues=NofStates, verbosity=False)

				
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
			
			if save:
				plt.savefig("Slike/"+name)
				plt.close()
			else:
				plt.show()

if spectral_transition_ofJ_plot:
	
	print("PREMISLI TO")


if spectral_transition_of_alpha_plot:

	print("PREMISLI TO")

