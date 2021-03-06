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

NOTE (POSSIBLE CONFUSION!)
In the program, the impurity spin is represented like a fermion (two states with spin UP and DOWN), where the occupation number of the level is always exactly 1. This is NOT
counted as a particle or a level in the system (N and n do not incorporate the impurity), while for the purposes of total spin in the system, one has to count the impurity spin.

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

def setAlpha(N, d, dDelta):
	"""
	Returns alpha for a given dDelta. 
	"""
	omegaD = 0.5*N*d	
	Delta = d/dDelta
	return 1/(np.arcsinh(omegaD/(Delta)))

# BASIS ########################################################################

@jit
def makeBase(N, nwimpUP, nwimpDOWN):
	"""
	Creates a basis of a system with N levels, nwimpUP fermions with spin UP and nwimpDOWN fermions with spin DOWN, including the impurity. 
	The impurity level is restricted to exactly one fermion, and is not included in the N levels fo the system. 
	The resulting basis defines the smallest ls subset of the Hamiltonian, where the number of particles and the total spin z of the system 
	are good quantum numbers.
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
def countSetBits(m): 
	"""Counts the number of bits that are set to 1 in a given integer."""
	count = 0
	while (m): 
		count += m & 1
		m >>= 1
	return count 

@jit
def countBitsFromOffset(m, off, N):
	count=0
	for i in range(off+1, 2*N):
		count+=bit(m, i)
	
	return (-1)**count	

@jit
def bit(m, off):
	"""
	Returns the value of a bit at offset off in integer m.
	"""

	if m & (1 << off):
		return 1
	else:
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
	Counts the number of spin up electrons in the state. If allBits, the impurity level is also counted.
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
	count=0
	for i in range(off+1, 2*N):	#count bits from offset to (not including) the impurity
		count+=bit(m, i)
	
	"""
	#THIS WORKS MUCH SLOWER BUT IS CLEARER TO UNDERSTAND
	#turns off the impurity bit, as it does not contribute
	turnOffImp = (2**(2*N))-1	#this is the number 100000..., where 1 is at the position of the impurity.
	m = m & turnOffImp
	#set bits to zero
	
	#m = clearBitsAfter(m, off, 2*N)

	#count the 1s of the cleared bit
	count = countSetBits(clearBitsAfter(m, off, 2*N))
	"""

	return -(2 * (count%2)-1)

# ELECTRON OPERATORS ###########################################################

@jit
def crcranan(i, j, m, N):
	"""
	Calculates the action of c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP on a basis vector m, given as an integer.
	"""

	offi = 2*(N-i)-1	#offset of the bits representing the i-th and j-th energy levels, not accounting for spin
	offj = 2*(N-j)-1

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
			m3 = flipBit(m2, 2*(N-i)-1-1)	#c_iDOWN^dag
			if m3>m2:
				prefactor *= prefactor_offset(m1, 2*(N-j)-1, N)
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
			m3 = flipBit(m2, 2*(N-i)-1)	#c_iUP operator
			if m3>m2:
				prefactor *= prefactor_offset(m1, 2*(N-j)-1-1, N)
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

			#S- s+	
			prefactor2, m2 = spinSmsp(i, j, basisList[k], N)	
			if m2!=0:
				#Check crcrananOnState() for comments
				new_state[np.searchsorted(basisList, m2)] += 0.5 * coef * prefactor2		

			#Sz sz			
			impSCoef = -2*impSz(basisList[k], N) + 1	#gives 1 for Sz=0 (UP) and -1 for Sz=1 (DOWN)
			prefactor3, m3 = SzszUp(i, j, basisList[k], N)
			if m3 != 0:
				new_state[np.searchsorted(basisList, m3)] += 0.5 * impSCoef * 0.5 * coef * prefactor3  

			
			prefactor4, m4 = SzszDown(i, j, basisList[k], N)
			if m4 != 0:	
				new_state[np.searchsorted(basisList, m4)] += -0.5 * impSCoef * 0.5 * coef * prefactor4 


	return new_state		

# HAMILTONIAN ##################################################################

#@jit
#@profile	
def HonState(d, alpha, J, state, N, D, basisList, lengthOfBasis):
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
		kinetic += eps(i, d, D) * (niUP + niDOWN)
		
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
	def __init__(self, d, alpha, J, N, D, basisList, lengthOfBasis, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)
		self.d = d
		self.alpha = alpha
		self.J = J
		self.N = N
		self.D = D
		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis

	def _matvec(self, state):
		return HonState(self.d, self.alpha, self.J, state, self.N, self.D, self.basisList, self.lengthOfBasis)

# PHYSICS ######################################################################

@jit
def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	#return d*(i - ((N-1)/2))
	return -D + d*i + d/2

# DIAGONALISATION ##############################################################

def LanczosDiag_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False):
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
		LinOp = HLinOP(d, alpha, J, N, D, basisList, lengthOfBasis) 
		values = eigsh(LinOp, k=min(lengthOfBasis-1, NofValues), which="SA", return_eigenvectors=False)[::-1]

	return values

def LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False):
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
		Hs = HonState(d, alpha, J, np.array([1]), N, D, basisList, lengthOfBasis)
		values = np.dot(basisList[0], Hs)
		vectors = [np.array([1])]

	else:	
		LinOp = HLinOP(d, alpha, J, N, D, basisList, lengthOfBasis) 
		values, vectors = eigsh(LinOp, k=min(lengthOfBasis-1, NofValues), which="SA")

	return values, vectors, basisList

def LanczosDiag(D, N, n, d, alpha, J, NofValues=4, verbosity=False):
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

		val.extend(LanczosDiag_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=NofValues, verbosity=verbosity))

	return np.sort(val)	

def LanczosDiag_states(D, N, n, d, alpha, J, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	Eigenvectors are already transposed, so SortedVectors[i] corresponds to the eigenstate with energy SortedValues[i], with the
	basis given with SortedBasisLists[i].
	"""

	values, vectors, basisLists = [], [], []

	for nwimpUP in range(max(0, n-N), min(N+1, n+1) + 1):	#+1 in range to take into account also the last case	

		nwimpDOWN = n - nwimpUP + 1

		if verbosity:
			print(nwimpUP, nwimpDOWN)

		val, vec, basisList = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=verbosity)

		values.extend(val)
		vectors.extend(np.transpose(vec))
		basisLists.extend([basisList for i in range(len(val))])


	sortedIndices = np.argsort(values)	
	values, vectors, basisLists = np.asarray(values), np.asarray(vectors), np.asarray(basisLists)


	SortedValues = np.take_along_axis(values, sortedIndices, axis=0)
	SortedVectors = np.take_along_axis(vectors, sortedIndices, axis=0)
	SortedBasisLists = np.take_along_axis(basisLists, sortedIndices, axis=0)


	return SortedValues, SortedVectors, SortedBasisLists

# TRANSITION ENERGY ############################################################

def addElectron(D, N, n, d, alpha, J, add=1):
	"""
	Calculates the energy of transition (= difference of ground state energies) between ground states of 
	the subspace with n and n+add particles.
	"""	

	#FIND THE GS FOR THE INITIAL STATE
	#initVal = LanczosDiag(N, n, d, alpha, J, NofValues=4)[0]	#this is the ground state of the initial system	
	initVal, b1, c1 = LanczosDiag_states(D, N, n, d, alpha, J, NofValues=4)
	initVal = initVal[0]

	#ADD ELECTRONS
	n = n + add

	#FIND THE GS FOR THE FINAL STATE
	#finalVal = LanczosDiag(N, n, d, alpha, J, NofValues=4)[0]	#this is the ground state of the final system	
	finalVal, b2, c2 = LanczosDiag_states(D, N, n, d, alpha, J, NofValues=4)
	finalVal = finalVal[0]

	return finalVal - initVal

def addElectronDefindedFinalSpin(D, N, n, spin, d, alpha, J, add=1):
	"""
	Calculates the energy of transition (= difference of ground state energies) between ground states of 
	the subspace (n, any Sz) and the subspace (n=n+1, Sz=Sz'+1); where Sz' is spin of the computed GS 
	with n particles. 
	If add==-1, this calculates the energy of the process of taking away an electron. Actually, the 
	function might work with add as any whole number. The process is then adding add electrons, where
	add<0 corresponds to taking away the specified number of electrons (not tested).
	"""

	print("GS JE LAHKO DEGENERIRANO, PREVERI KAJ TAKRAT!")

	#FIND THE GS FOR THE INITIAL STATE
	a, b, c = LanczosDiag_states(D, N, n, d, alpha, J, NofValues=4, verbosity=False)
	initVal, initVec, initBasisList = a[0], b[0], c[0]		#this is the ground state of the initial system

	print("INIT")
	print(initVal)
	printV(initVec, initBasisList, prec=0.1)
	print()

	#FIND QUANTUM NUMBERS OF THE GS
	n, nwimpUP, nwimpDOWN = findSector(initBasisList, N)

	if spin == "up":

		nwimpUP = nwimpUP + 1*add
		n = n + 1*add
		#FIND THE STATE AFTER TRANSITION
		a, b, c = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False)
		finalVal, finalVec, finalBasisList = a[0], b[:, 0], c

	elif spin == "down":

		nwimpDOWN = nwimpDOWN + 1*add
		n = n + 1*add
		#FIND THE STATE AFTER TRANSITION
		a, b, c = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False)
		finalVal, finalVec, finalBasisList = a[0], b[:, 0], c

	print("FINAL")
	print(finalVal)
	printV(finalVec, finalBasisList, prec=0.1)
	print()
	print("FINAL 2")
	print(a[1])
	printV(finalVec, finalBasisList, prec=0.1)
	print()

	return finalVal - initVal, finalVal, initVal	

def addElectronDefinedInitialFinalSpin(D, N, n, nwimpUP, nwimpDOWN, spin, d, alpha, J, add=1):
	"""
	Calculates the energy of transition (= difference of ground state energies) between ground states of 
	the subspace (n, Sz) and the subspace (n=n+1, Sz=Sz+1).
	If add==-1, this calculates the energy of the process of taking away an electron. Actually, the 
	function might work with add as any whole number. The process is then adding add electrons, where
	add<0 corresponds to taking away the specified number of electrons (not tested).
	"""

	print("GS JE LAHKO DEGENERIRANO, PREVERI KAJ TAKRAT!")

	#FIND THE GS FOR THE INITIAL STATE
	a, b, c = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False)
	initVal, initVec, initBasisList = a[0], b[0], c 			#this is the ground state of the initial system


	if spin == "up":

		nwimpUP = nwimpUP + 1*add
		n = n + 1*add
		#FIND THE STATE AFTER TRANSITION
		a, b, c = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False)
		finalVal, finalVec, finalBasisList = a[0], b[:, 0], c

	elif spin == "down":

		nwimpDOWN = nwimpDOWN + 1*add
		n = n + 1*add
		#FIND THE STATE AFTER TRANSITION
		a, b, c = LanczosDiagStates_nUPnDOWN(D, N, n, nwimpUP, nwimpDOWN, d, alpha, J, NofValues=4, verbosity=False)
		finalVal, finalVec, finalBasisList = a[0], b[:, 0], c	

	print("FINAL:")
	print(finalVal)
	print(printV(finalVec, finalBasisList, prec=0.1))
	print()


	return finalVal - initVal, finalVal, initVal

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

def findSector(basisList, N):
	"""
	For a state vector, written in the basis basisList, find which subspace of the hamiltonian it is from.
	Returns the number of particles n and the total spin of the system Sz.
	The assumption is that all states in the basis are from the same subspace.
	"""

	m = basisList[0]

	n = countSetBits(m) - 1	#impurity is not counted as a particle

	nwimpUP, nwimpDOWN = 0, 0
	for off in range((2*N)+2):
		if off%2==0 and bit(m, off):		#even offset are spin down states 
			nwimpDOWN += 1

		elif off%2!=0 and bit(m, off):	#odd offset correspond to spin up states
			nwimpUP += 1

	return n, nwimpUP, nwimpDOWN		

# CALCULATION ##################################################################
states_print = 1
spectrum_J_plot = 0
spectrum_alpha_plot = 0
spectral_transition_ofJ_plot = 0
spectral_transition_of_alpha_plot = 0
parity_gap_odd_plot = 0
parity_gap_even_plot = 0
################################################################################

if states_print:

	D=1
	N=4
	n=3
	d = 2*D/N
	alpha = 1
	J = 0

	val, vec, bas = LanczosDiag_states(D, N, n, d, alpha, J, NofValues=np.infty)
	
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

	for N in [6]:
		n=N
		for dDelta in [10]:
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
	save=0

	print("N, dDelta")
	for N in [5]:
		print()
		
		n = N
		d = 1 
		
		for dDelta in [0.3]:	
			alpha = setAlpha(N, d, dDelta)
			alpha = 0
			print(N, dDelta)

			Jmin, Jmax = 0*d, 5*d

			Elist1, Elist2 = [], []
			average = []
			Jlist = np.linspace(Jmin, Jmax, 10)
			for J in Jlist:

				print(J)

				a = addElectron(N, n, d, alpha, J, add=1)
				b = addElectron(N, n, d, alpha, J, add=-1)		
				c = addElectron(N, n, d, alpha, 0, add=1)
				e = addElectron(N, n, d, alpha, 0, add=-1)

				if N%2==0:
					parityGap = 0.5*c + 0.5*e 
				else:
					parityGap = -1*(0.5*c + 0.5*e) 



				Elist1.append(a/parityGap)	
				Elist2.append(b/parityGap)	
				average.append(((0.5*a) + (0.5*b))/parityGap)

			Jlist = Jlist/d

			plt.plot(Jlist, Elist1, label=r"$n \rightarrow n+1$")
			plt.scatter(Jlist, Elist1)
			plt.plot(Jlist, Elist2, label=r"$n \rightarrow n-1$")
			plt.scatter(Jlist, Elist2)
			plt.plot(Jlist, average, linestyle="dashed")


			plt.title(r"$N={0}$, $\alpha={1}$, $d/\Delta={2}$".format(N, alpha, dDelta))
			plt.xlabel(r"$J/d$")
			plt.ylabel(r"$\Delta E/\Delta_P$")

			plt.legend()

			plt.grid()
			plt.tight_layout()
			
			plt.ylim(-1.3, 1.3)

			if save:
				name = "TransitionE_N{0}_dDelta{1}.pdf".format(N, dDelta)
				plt.savefig("Slike/"+name)
				plt.close()	
				print("SAVED")
			else:
				plt.show()

if spectral_transition_of_alpha_plot:

	N = 6
	n = N

	d, J = 1, 0
	dDeltamin, dDeltamax = 0.1, 10

	Elist1, Elist2 = [], []

	dDeltaList = np.linspace(dDeltamin, dDeltamax, 10)
	for dDelta in dDeltaList:

		Delta = d/dDelta
		omegaD = 0.5*N*d	
	
		alpha = 1/(np.arcsinh(omegaD/(2*Delta)))

		print(dDelta, Delta)

		a = addElectron(N, n, d, alpha, J, add=1)
		b = addElectron(N, n, d, alpha, J, add=-1)

		print(a, b)
		print()

		Elist1.append(a/Delta)	
		Elist2.append(b/Delta)	
	

	plt.plot(dDeltaList, Elist1, label=r"$n \rightarrow n+1$")
	plt.scatter(dDeltaList, Elist1)
	plt.plot(dDeltaList, Elist2, label=r"$n \rightarrow n-1$")
	plt.scatter(dDeltaList, Elist2)

	plt.legend()

	plt.title(r"$N={0}$".format(N))
	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$\Delta E$")

	plt.grid()
	plt.tight_layout()
	plt.show()

if parity_gap_odd_plot:
	saved=0

	N = 5
	n = N

	d, J = 1, 0
	dDeltamin, dDeltamax = 0.1, 5

	parityGapList = []

	dDeltaList = np.linspace(dDeltamin, dDeltamax, 5)
	for dDelta in dDeltaList:

		Delta = d/dDelta
		omegaD = 0.5*N*d	
	
		alpha = 1/(np.arcsinh(omegaD/(Delta)))

		print(dDelta, Delta, alpha)

		a = addElectron(N, n, d, alpha, J, add=1)
		b = addElectron(N, n, d, alpha, J, add=-1)

		parityGap = -1*(0.5*a + 0.5*b) 
		parityGapList.append(parityGap/Delta)	
	
	

	plt.plot(dDeltaList, parityGapList)
	plt.scatter(dDeltaList, parityGapList)
		

	plt.legend()

	plt.title(r"$N={0}$".format(N))
	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$\Delta_P/\Delta$")

	plt.grid()
	plt.tight_layout()

	if saved:
		name = "ParityGap_odd.pdf"
		plt.savefig("Slike/"+name)
		plt.close()	
		print("SAVED")
	else:
		plt.show()

if parity_gap_even_plot:
	save=0

	N = 6
	for N in [8]:
		n = N
		print()
		print(N)
		d, J = 1, 0
		dDeltamin, dDeltamax = 0.3, 8

		parityGapList = []

		dDeltaList = np.linspace(dDeltamin, dDeltamax, 10)
		for dDelta in dDeltaList:

			Delta = d/dDelta
			omegaD = 0.5*N*d	
		
			alpha = 1/(np.arcsinh(omegaD/(Delta)))

			print(dDelta, Delta, alpha)

			a = addElectron(N, n, d, alpha, J, add=1)
			b = addElectron(N, n, d, alpha, J, add=-1)

			parityGap = 0.5*a + 0.5*b 
			parityGapList.append(parityGap/Delta)	
				

		#plt.plot(dDeltaList, parityGapList)
		plt.scatter(dDeltaList, parityGapList, label=r"$N={0}$".format(N))
		

	plt.legend()

	#plt.title(r"$N={0}$".format(N))
	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$\Delta_P/\Delta$")

	plt.grid()
	plt.tight_layout()

	plt.xlim(0, 8)
	plt.ylim(0, 6)

	if save:
		name = "ParityGap_even.pdf"
		plt.savefig("Slike/"+name)
		plt.close()	
		print("SAVED")
	else:
		plt.show()



