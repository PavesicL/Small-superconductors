"""
IMPLEMENTATION
The calculation is performed in the basis of occupation numbers: |S_z,imp, n_0UP, n_0DOWN, n_1UP, n_1DOWN, ... >.
First we find all basis vectors in the Hilbert space of the system with N levels, which have a specified amount (n) of particles. This is done by the makeBase() function.  
A state is represented by a vector of probability amplitudes, accompanied by basisList, which hold information of what basis state each index in the state vector represents. 
The fermionic operators are implemented using bit-wise operations, specifically functions flipBit() and countSetBits(). The spin operators are also represented as fermionic
operators (which works for S=1/2 at least), using S+ and S-.

INDEXING OF STATES IN THE STATE VECTOR: 
The occupancy number of a state i, s is given as the 2i+s'th bit of the basis state, written in binary, where (spin) s=0 for UP and s=1 for down. 
The offset of the given bit (counted from right to left) in the binary representation is (2N-1) - (2i+s) = 2(N-i)-1-s. The impurity spin state is at offset 2N.

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

# BASIS ########################################################################

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
	Calls the two previous functions and creates a basis - returns its length and a list of states.
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
	turnOffImp = (2**(2*N))-1	#this is the number 100000..., where 1 is at the position of the impurity
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
def impSz(m, N):
	"""
	Given a basis state, returns the spin Sz of the impurity. 0=UP, 1=DOWN.
	"""
	if m>=2**(2*N):
		return 1
	else:
		return 0

@jit
def spinSpsm(i, j, m, N):
	"""
	Calculates the result of action of the operator S+ s-_ij to a basis state m.
	"""

	m1 = flipBit(m, 2*N)	#flip the impurity spin if it is down
	if m1<m:
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
	m1 = flipBit(m, 2*N)	#flip the impurity spin if it is up
	if m1>m:
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
			
			return m12, prefactor
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
			
			return m22, prefactor
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
			m3, prefactor3 = SzszUp(i, j, basisList[k], N)
			if m3 != 0:
				new_state[np.searchsorted(basisList, m3)] += 0.5 * impSCoef * 0.5 * coef * prefactor3  
			
			m4, prefactor4 = SzszDown(i, j, basisList[k], N)
			if m4 != 0:	
				new_state[np.searchsorted(basisList, m4)] += -0.5 * impSCoef * 0.5 * coef * prefactor4 

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
			interaction += crcrananOnState(i, j, state, N, basisList, lengthOfBasis)

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

def exactDiag(N, n, d, alpha, J):
	"""
	Diagonalises the hamiltonian using exact diagonalisation.
	"""

	lengthOfBasis, basisList = makeBase(N, n)

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

def LanczosDiag(N, n, d, alpha, J, NofValues=4):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies of the Hamiltonian.
	"""

	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
	values = eigsh(LinOp, k=NofValues, which="SA", ncv = 10*NofValues, return_eigenvectors=False)[::-1]

	return values

def LanczosDiag_states(N, n, d, alpha, J, NofValues=4):
	"""
	For a given number of levels N, and given number of particles n, return the eigenenergies and eigenstates of the Hamiltonian.
	"""
	lengthOfBasis, basisList = makeBase(N, n)
	LinOp = HLinOP(d, alpha, J, N, basisList, lengthOfBasis) 
	values, vectors = eigsh(LinOp, k=NofValues, which="SA")

	return values, vectors, basisList

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
spectral_transition_of_alpha_plot = 1
################################################################################

if 0:
	NN = 5
	nn = NN
	dd, aalpha, JJ = 1, 0, 0
	NofValues = 6

	print("ENERGIJE")
	print([eps(i, dd, NN) for i in range(NN)])
	print()

	for k in range(1):
		lengthOfBasis, basisList = makeBase(NN, nn)

		a1, b1, c1 = LanczosDiag_states(NN, nn, dd, aalpha, JJ, NofValues=NofValues)

		a2, b2, c2 = LanczosDiag_states(NN, nn+1, dd, aalpha, JJ, NofValues=NofValues)

		#print(basisList)
		#for i in basisList:
		#	print(bin(i))
		val, vec = exactDiag(NN, nn, dd, aalpha, JJ)
		
		print(a1[0], a2[0])
		print(a1-a2)
		print(a1 - val[:NofValues])
		print(a2 - val[:NofValues])
		print(val[:NofValues])
		"""
		#print(a1-a2)
		print("ENERGIJA")
		print(val[:5])

		print("VEKTOR GS")
		printV(vec[:, 0], basisList, prec=0.1)
		
		print("VEKTOR ES 1")
		printV(vec[:, 1], basisList, prec=0.1)
		
		print("VEKTOR ES 2")
		printV(vec[:, 2], basisList, prec=0.1)
		"""

if states_print:

	N=4
	n=N
	d, alpha = 1, 0
	J = 0
	NNN = 20

	val, vec, basisList = LanczosDiag_states(N, n, d, alpha, J, NofValues=NNN)
	print(val)
	print()
	
	for i in range(len(val)):
		v = vec[:, i]
		print("STATE", i, val[i])
		for k in range(len(v)):
			if abs(v[k])>0.1:
				print(k, v[k], bin(basisList[k]))
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

				if N<5:
					lengthOfBasis, basisList = makeBase(N, n)
					values, vec = exactDiag(d, alpha, J, N, basisList, lengthOfBasis)
				else:
					values = LanczosDiag(N, n, d, alpha, J, NofValues=NofStates)
				#values, vec, basisList = LanczosDiag_states(N, n, d, alpha, J, NofValues=NofStates)

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
			
			if save:
				plt.savefig("Slike/"+name)
				plt.close()
			else:
				plt.show()

if spectral_transition_ofJ_plot:
	
	N=7
	n=9
	d, alpha = 1, 1

	initn1, finaln1 = n, n+1
	initn2, finaln2 = n, n-1

	DeltaJ1 = transitionE(N, d, alpha, 0, initn1, finaln1)
	DeltaJ2 = transitionE(N, d, alpha, 0, initn2, finaln2)


	EpList, EmList = [], []	

	Jmin, Jmax = 0, 3
	Jlist = np.linspace(Jmin, Jmax, 10)
	for J in Jlist:
		print(N, J)
		
		EpList.append(transitionE(N, d, alpha, J, initn1, finaln1))
		EmList.append(transitionE(N, d, alpha, J, initn2, finaln2))

	print(EpList)
	print(EmList)

	plt.plot(Jlist, EpList/DeltaJ1, label=r"$n={0}\rightarrow {1}$".format(initn1, finaln1))

	plt.plot(Jlist, EmList/DeltaJ2, label=r"$n={0}\rightarrow {1}$".format(initn2, finaln2))	
		

	plt.xlabel(r"$J$")
	plt.ylabel(r"$\Delta E / \Delta E(J=0)$")

	plt.title(r"$N={0}$".format(N))

	plt.legend()
	plt.grid()
	plt.tight_layout()

	plt.show()

if spectral_transition_of_alpha_plot:

	N=7
	n=7
	d, J = 1, 0.6

	initn1, finaln1 = n, n+1
	initn2, finaln2 = n, n-1

	EpList, EmList = [], []	

	dDeltaMin, dDeltaMax = 0.1, 10
	dDeltaList = np.linspace(dDeltaMin, dDeltaMax, 10)
	for dDelta in dDeltaList:

		omegaD = 0.5*N*d	
		alpha = 1/(np.arcsinh(dDelta*0.5*omegaD/d))
		Delta = omegaD/(2*np.sinh(1/alpha))	
		
		print(N, dDelta)
		
		Ep = transitionE(N, d, alpha, J, initn1, finaln1)
		Em = transitionE(N, d, alpha, J, initn2, finaln2)

		DeltaJ1 = transitionE(N, d, alpha, 0, initn1, finaln1)
		DeltaJ2 = transitionE(N, d, alpha, 0, initn2, finaln2)
	
		EpList.append(Ep/DeltaJ1)
		EmList.append(Em/DeltaJ2)
	

	print(EpList)
	print(EmList)

	plt.plot(dDeltaList, EpList, label=r"$n={0}\rightarrow {1}$".format(initn1, finaln1))

	plt.plot(dDeltaList, EmList, label=r"$n={0}\rightarrow {1}$".format(initn2, finaln2))	
	
	#plt.xscale("log")
	#plt.yscale("log")	

	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$\Delta E / \Delta E(J=0)$")

	plt.title(r"$N={0}$".format(N))

	plt.legend()
	plt.grid()
	plt.tight_layout()

	plt.show()


