"""
Finds the spectrum and the eigenstates of the hamiltonian of a small superconductor, coupled to an impurity. In this specific implementation, one has control over the number of
particles in the system and the total spin of the system. The implemented impurity coupling is SIAM.

IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |n_impUP, n_impDOWN, n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which has a specified amount of particles with spin up (nUP) and spin down (nDOWN). 
This is done by the makeBase() function. A state is represented by a vector of probability amplitudes, accompanied by basisList, which hold information of what basis state 
each index in the state vector represents. The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). 

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a single particle state (i, s) is given as the 2i+s'th bit of the basis state, written in binary (and counted from the left), where (spin) s=0 for UP 
and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s) = 2(N-i)-1-s. The impurity spin state is at offset 2N and 2N+1.

DIAGONALISATION
Diagonalisation is implemented using the Lanczos algorithm. The linear operator H is implemented using the numpy class LinearOperator. It allows us to define a linear operator
and diagonalise it, using scipy.sparse.linalg.eigsh. 
The complete diagonalisation, in the functions LanczosDiag_states() and LanczosDiag_states() (if one wants to obtain the eigenvectors too) is done subspace by subspace. The
smallest subspace of the Hamiltonian is one with defined number of particles (n) and total spin z of the system. Alternatively, one defines n and the number of particles with
spin UP/DOWN in the system. Here, we have to count the particles at the impurity levels too.
We have:
	n = nUP + nDOWN,
	1/2 (nUP + nDOWN) = Sz.
All quantities in these two equations are constant in a given subspace. 
It is possible to only obtain the spectrum of the system in the specified state (n, nUP, nDOWN), using LanczosDiag_nUPnDOWN or LanczosDiagStates_nUPnDOWN.

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
def makeBase(M, nUP, nDOWN):
	"""
	Creates a basis of a system with N levels, nwimpUP fermions with spin UP and nwimpDOWN fermions with spin DOWN, including the impurity. 
	The impurity level is restricted to exactly one fermion, and is not included in the N levels fo the system. 
	The resulting basis defines the smallest ls subset of the Hamiltonian, where the number of particles and the total spin z of the system 
	are good quantum numbers.
	"""
	resList = []

	for m in range(2**(2*M)):
		if countSetBits(m) == nUP + nDOWN:					#correct number of spins UP and DOWN
			if spinUpBits(m, N, allBits=True) == nUP and spinDownBits(m, N, allBits=True) == nDOWN:
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

@jit
def prefactor_offset_imp(m, s, N):
	"""
	Calculates the fermionic prefactor for a fermionic operator, acting on the impurity site. 
	"""

	if s==1 and bit(m, 2*N+1):
		return -1
	return 1

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


# ELECTRON-IMPURITY OPERATORS ##################################################

@jit
def cranimp(i, s, m, N):
	"""
	Calculates the result of c_i,s^dag a_s acting on an integer m. Returns the new basis state and the fermionic prefactor.
	Spin: UP - s=0, DOWN - s=1.
	"""

	offi = 2*(N-i)-1-s
	offimp = 2*(N+1)-1-s

	m1 = flipBit(m, offimp)
	if m1<m:
		m2=flipBit(m1, offi)
		if m2>m1:
			prefactor = prefactor_offset(m1, offi, N)
			prefactor *= prefactor_offset_imp(m, s, N)

			return prefactor, m2
	return 0, 0

@jit
def crimpan(i, s, m, N):	
	"""
	Calculates the result of a_s^dag c_i,s acting on an integer m. Returns the new basis state and the fermionic prefactor.
	Spin: UP - s=0, DOWN - s=1.
	"""

	offi = 2*(N-i)-1-s
	offimp = 2*(N+1)-1-s	

	m1 = flipBit(m, offi)
	if m1<m:
		m2=flipBit(m1, offimp)
		if m2>m1:
			prefactor = prefactor_offset(m, offi, N)
			prefactor *= prefactor_offset_imp(m1, s, N)

			return prefactor, m2
	return 0, 0		

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

# SIAM OPERATORS ###############################################################

@jit
def impurityEnergyOnState(state, Eimp, U, N, basisList, lengthOfBasis):
	"""
	Calculates the contribution of the impurity to energy (kinetic and potential energy).
	"""

	#impurity is at position i=-1
	nimpUP = CountingOpOnState(-1, 0, state, N, basisList, lengthOfBasis)
	nimpDOWN = CountingOpOnState(-1, 1, state, N, basisList, lengthOfBasis)
	nimpUPnimpDOWN = CountingOpOnState(-1, 0, CountingOpOnState(-1, 1, state, N, basisList, lengthOfBasis), N, basisList, lengthOfBasis)
	
	new_state = Eimp*(nimpUP + nimpDOWN) + U*nimpUPnimpDOWN

	return new_state

@jit
def impurityInteractionOnState(i, state, N, basisList, lengthOfBasis):
	"""
	Calculates the contribution of the interaction term between the impurity and the system.
	"""
	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	

			for s in [0, 1]:
				
				prefactor1, m1 = cranimp(i, s, basisList[k], N)
				prefactor2, m2 = crimpan(i, s, basisList[k], N)

				if m1!=0:
					new_state[np.searchsorted(basisList, m1)] += coef * prefactor1 

				if m2!=0:
					new_state[np.searchsorted(basisList, m2)] += coef * prefactor2


	return new_state

# HAMILTONIAN ##################################################################

#@jit
#@profile	
def HonState(state, M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis):
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


	if M == N+1:
		#impurity energy
		impurity += impurityEnergyOnState(state, Eimp, U, N, basisList, lengthOfBasis)

	for i in range(N):
		niUP = CountingOpOnState(i, 0, state, N, basisList, lengthOfBasis)
		niDOWN = CountingOpOnState(i, 1, state, N, basisList, lengthOfBasis)

		#kinetic term
		kinetic += eps(i, d, D) * (niUP + niDOWN)
		
		if M == N+1:
			#impurity interaction
			impurity += (V/np.sqrt(N))*impurityInteractionOnState(i, state, N, basisList, lengthOfBasis)
			
		for j in range(N):
			if d*alpha!=0:
				interaction += crcrananOnState(i, j, state, N, basisList, lengthOfBasis)

	return kinetic - d*alpha*interaction + impurity

class HLinOP(LinearOperator):
	"""
	This is a class, built-in to scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	The operator can be diagonalised using the function scipy.sparse.linalg.eigsh().
	"""	
	def __init__(self, M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)
		self.M = M
		self.N = N
		self.D = D
		self.d = d
		self.alpha = alpha
		self.Eimp = Eimp
		self.U = U
		self.V = V
		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis

	def _matvec(self, state):
		return HonState(state, self.M, self.N, self.D, self.d, self.alpha, self.Eimp, self.U, self.V, self.basisList, self.lengthOfBasis)

# PHYSICS ######################################################################

@jit
def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	#return d*(i - ((N-1)/2))
	return -D + d*i + d/2

# DIAGONALISATION ##############################################################

def LanczosDiag_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=4, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	D - half-width of the banc
	N - number of levels in the SC
	M - number of levels including the impurity
	n - number of particles in the system
	nwimpUP, nwimpDOWN - number of particles with spin UP/DOWN, INCLUDING ON THE IMPURITY
	d, alpha, Eimp, U, V - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(M, nUP, nDOWN)

	if verbosity:
		checkParams(N, n, nUP, nDOWN)

	if lengthOfBasis==1:
		Hs = HonState(np.array([1]), M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis)
		values = [np.dot(np.array([1]), Hs)]

	else:		
		LinOp = HLinOP(M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis) 
		values = eigsh(LinOp, k=max(1, min(lengthOfBasis-1, NofValues)), which="SA", return_eigenvectors=False)[::-1]

	return values

def LanczosDiagStates_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=4, verbosity=False):
	"""
	Returns the spectrum of the Hamiltonian in the subspace, defined by
	D - half-width of the banc
	N - number of levels in the SC
	M - number of levels including the impurity
	n - number of particles in the system
	nUP, nDOWN - number of particles with spin UP/DOWN, INCLUDING ON THE IMPURITY
	d, alpha, Eimp, U, V - physical constants.
	"""
	
	lengthOfBasis, basisList = makeBase(M, nUP, nDOWN)
	
	if verbosity:
		checkParams(N, n, nUP, nDOWN)

	if lengthOfBasis==1:
		Hs = HonState(np.array([1]), M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis)
		values = [np.dot(np.array([1]), Hs)]
		vectors = [np.array([1])]

	else:	
		LinOp = HLinOP(M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis) 
		values, vectors = eigsh(LinOp, k=max(1, min(lengthOfBasis-1, NofValues)), which="SA")

	return values, vectors, basisList

def LanczosDiag(M, N, D, n, d, alpha, Eimp, U, V, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	Computed as a combination of eigenenergies from all smallest subspace with set number of particles (n) and 
	total system spin (Sz = 1/2 (nUP - nDOWN)).
	"""

	val=[]

	for nUP in range(max(0, n-min(M, n)), min(M, n) + 1):	#+1 in range to take into account also the last case	

		nDOWN = n - nUP

		if verbosity:
			print(nUP, nDOWN)

		val.extend(LanczosDiag_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=NofValues, verbosity=verbosity))

	return np.sort(val)	

def LanczosDiag_states(M, N, D, n, d, alpha, Eimp, U, V, NofValues=4, verbosity=False):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	Eigenvectors are already transposed, so SortedVectors[i] corresponds to the eigenstate with energy SortedValues[i], with the
	basis given with SortedBasisLists[i].
	"""

	values, vectors, basisLists = [], [], []

	for nUP in range(max(0, n-min(M, n)), min(M, n) + 1):	#+1 in range to take into account also the last case	

		nDOWN = n - nUP

		if verbosity:
			print(nUP, nDOWN)

		val, vec, basisList = LanczosDiagStates_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=4, verbosity=verbosity)

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

#print("FIX THIS!")	

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
energies_print = 0
states_transition_print = 0
states_print = 0
################################################################################

if 1:
	#Setup to define a state and calculate the action of H on it

	M=10
	N=M-1

	D=1
	d = 2*D/N
	rho=1/(2*D)

	alpha = 0.1
	U = 10
	Eimp = -U/2 
	Gamma = 1
	V = np.sqrt(Gamma/(np.pi*rho))

	n = M

	import time

	for alpha in [0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:

		start = time.time()
		a = LanczosDiag(M, N, D, n, d, alpha, Eimp, U, V, NofValues=4, verbosity=False)
		end = time.time()

		print(alpha, end-start)
	
	#print(a)
	


if 0:
	#Setup to define a state and calculate the action of H on it

	M=8
	N=M-1

	D=1
	d = 2*D/N
	rho=1/(2*D)

	alpha = 1.0
	U = 10
	Eimp = -U/2 
	Gamma = 1
	V = np.sqrt(Gamma/(np.pi*rho))

	n = 6
	nUP, nDOWN = 3, 3

	lengthOfBasis, basisList = makeBase(M, nUP, nDOWN)
	
	state = np.random.random(lengthOfBasis)
	state = state/norm(state)


	a = HonState(state, M, N, D, d, alpha, Eimp, U, V, basisList, lengthOfBasis)
	
	print(a)
	print(np.dot(state, a))

	"""
	for i in range(1):
		printV(vec[i], bas[i], prec=0.1)
		print()
	"""

if energies_print:

	M=9
	N=M-1

	D=1

	print("M = {0}".format(M))
	for n in [M, M+1, M-1]:

		d = 2*D/N
		rho=1/(2*D)

		alpha = 0.1
		U = 10
		Eimp = -U/2 
		Gamma = 0.1

		V = np.sqrt(Gamma/(np.pi*rho))

		val = LanczosDiag(M, N, D, n, d, alpha, Eimp, U, V, NofValues=1, verbosity=False)
		
		#print(val)
		print("n = {0}  E= {1}".format(n, (val+5)[0]))
	

if states_transition_print:

	M=5
	N=M-1

	D=1
	
	for n in [M, M+1, M-1]:

		d = 2*D/N
		rho=1/(2*D)

		alpha = 0
		U = 10
		Eimp = -U/2 
		Gamma = 0.01
		V = np.sqrt(Gamma/(np.pi*rho))


		#val, vec, bas = LanczosDiagStates_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=4, verbosity=False)
		#vec = np.transpose(vec)
		
		val, vec, bas = LanczosDiag_states(M, N, D, n, d, alpha, Eimp, U, V, NofValues=4, verbosity=False)
		
		#val, vec = exactDiag(N, n, d, alpha, J)
		
		print(n, val[0]+5)
		#print(vec)
		
		for i in range(1):
			printV(vec[i], bas[i], prec=0.1)
			print()



if states_print:

	M=5
	N=M-1

	D=1
	
	n=M

	nUP=n/2
	nDOWN=n-nUP

	d = 2*D/N
	rho=1/(2*D)

	alpha = 0
	U = 10
	Eimp = -U/2 
	Gamma = 0
	V = np.sqrt(Gamma/(np.pi*rho))


	#val, vec, bas = LanczosDiagStates_nUPnDOWN(M, N, D, n, nUP, nDOWN, d, alpha, Eimp, U, V, NofValues=4, verbosity=False)
	#vec = np.transpose(vec)
	
	val, vec, bas = LanczosDiag_states(M, N, D, n, d, alpha, Eimp, U, V, NofValues=4, verbosity=False)
	
	#val, vec = exactDiag(N, n, d, alpha, J)
	
	print(val)
	#print(vec)
	print()
	
	for i in range(len(val)):
		print("STATE {0}".format(i))
		printV(vec[i], bas[i], prec=0.1)
		print()




