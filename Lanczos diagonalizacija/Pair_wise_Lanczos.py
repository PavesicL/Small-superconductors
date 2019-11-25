"""
Attempt to reproduce results from: A. Mastellone, G. Falce, R. Fazio, Small Superconducting Grain in the Canonical Ensemble, Phys. Rev. Lett., 1998.
Using Lanczos diagonalisation of the superconducting hamiltonian. 
Instead of using the simple (naive) approach of taking into account all of the single particle states, here, we use the fact that the hamiltonian is decoupled into non-interacting
singly occupied levels, and interacting doubly occupied levels. 
Then, for some configuration of singly occupied levels, one only diagonalises the problem of interacting pairs on the remaining levels. These have a different set of energies 
(for N=4, if the second level is singly occupied, the remaining levels, available for pairs have energies E0, E1, E3). This is a problem with a (exponentially) smaller basis.
The procedure is repeated for all possible numbers of pairs, and at each number of pairs for all possible configurations of singly occupied levels.

IMPLEMENTATION
Numbers of levels are denoted by N (all), Ns (singly occupied) and Np (doubly occupied, available for pairs);
numbers of particles are denoted by n, ns, npair (same).
For given N and n, we do the following: 
	for each possible npair, which is from 0 to n//2:
		-find all possible configuration of positions of singly occupied levels, 
		for each configuration:
			-define the dispersion of the Np levels, available for pairs (take out the singly occupied levels)
			-diagonalise the system and obtain the eigenenergies
			-add the kinetic contribution of the non paired electrons (singly occupied particles) to all eigenenergies; this is independent of the pair interaction and only depends
			 on which levels are singly occupied

	-combine all of the eigenenergies into a sorted list, these form the spectrum of the entire hamiltonian

NUMBA
Mostly works fine, except in the basis definition part. I think it dislikes the numpy function for the binomial symbol (np.comb()). Solving this might provide a SLIGHT speed-up. The basis
is not created many times, the main bottleneck is the application of the hamiltonian to the state, which works fine with jit().

POSSIBLE PROBLEM
The getEigenstates() function, which is used to calculate the eigenstates of the system skips over the case of npair=0. This might cause problems if such state (with each electron in a
seperate level) would be one of the low laying eigenstates. It is probably not, except (probably) in the limit of d->0 and alpha*d >> D, which is not physical anyway.
"""
	
################################################################################

import numpy as np 
import scipy
import matplotlib.pyplot as plt

from scipy.special import comb
from numpy.linalg import norm

from scipy.sparse.linalg import LinearOperator
from scipy.sparse.linalg import eigsh, eigs

from numba import jit

# PHYSICS ######################################################################

@jit
def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return -D + d*i + d/2

# UTILITY ######################################################################

@jit
def countSetBits(n): 
	"""
	Counts the number of bits that are set to 1 in a given integer.
	"""
	count = 0
	while (n): 
		count += n & 1
		n >>= 1
	return count 

@jit
def flipBit(n, offset):
	"""
	Flips the bit at position offset in the integer n.
	"""
	mask = 1 << offset
	return(n ^ mask)

@jit
def testBit(n, offset):
	"""
	DOES NOT RETURN ONLY 1 OR 0! Returns 0 when the bit at offset is 0, but 2^offset when the bit is 1. (If the bit at offset is 1, it returns the number
	corresponding to the binary string 0...010...0, where 1 is at position=offset.)
	"""
	mask = 1 << offset
	return(n & mask)

def findGap(values, precision=1e-16):
	"""
	From a given set of eigenvalues, finds the first spectroscopic gap, bigger than PRECISION.
	"""

	for i in range(1, len(values)):
		difference = values[i] - values[i-1]
		if difference > precision:
			return difference

# BASIS DEFINITION #############################################################

def allSinglyOccupiedPossibilities(N, n, npair):
	"""
	Returns a list of lists of all permutations of singly occupied levels. There should be comb(n, (n-2np)) of them.
	"""

	ns = n - 2*npair

	resList = np.zeros(comb(N, ns, exact=True), dtype=int)

	count=0
	for i in range(2**N):
		if countSetBits(i) == ns:
			resList[count] = i
			count+=1		

	return resList		

def pairLevelsDispersion(d, N, D, singlyOccupiedList):
	"""
	Given the list of lists of the singly occupied levels, returns a list of lists of energies for levels available for pair interaction, and a list of lists of energies of singly occupied levels.
	"""

	pairEnergies, singleEnergies = [[] for i in singlyOccupiedList], [[] for i in singlyOccupiedList]

	for j in range(len(singlyOccupiedList)):
		for i in range(N):
			if testBit(singlyOccupiedList[j], N-i-1)==0:
				pairEnergies[j].append(eps(i, d, D))	
			else:
				singleEnergies[j].append(eps(i, d, D))

	pairEnergies = np.array([np.array(i) for i in pairEnergies])
	singleEnergies = np.array([np.array(i) for i in singleEnergies])


	return pairEnergies, singleEnergies

def defineBase(N, n, npair):
	"""
	Finds the subspace of the Hilbert space with npair pairs on Np available levels, where N = Np + Ns and ns = Ns. (N - number of levels, Ns/p - number of singly/pair occupied levels)
	"""
	Np = N - (n - 2*npair)
	lengthOfBasis = comb(Np, npair, exact=True)


	resList = np.zeros(lengthOfBasis, dtype=int)


	count=0
	for i in range(2**Np):
		if countSetBits(i) == npair:
			#res.append(i)
			resList[count] = i
			count+=1

	return resList, lengthOfBasis

# OPERATORS ####################################################################

@jit
def cran(i, j, m, Npair):
	"""
	Calculates the application of the operator c_i^dag c_j to a basis state m, where c (c^dag) are PAIR anihilation (creation) operators (destroy/create a pair).
	INPUT: 
	i, j - positions of operators, integer
	m - a basis state, integer (its binary form is a representation of the vector in the |...n_i...> basis)
	N - number of energy levels, available for pair hopping, integer
	OUTPUT
	res - a resulting state, integer
	0 - if the calculation gives 0 - this should be safe, as the case of 0 pairs (where 0 is a relevant state) is trivial and computed seperately. The state 0 should
		not appear in any other subspace.
	"""

	m1 = flipBit(m, Npair-j-1)    #flips j-th bit in state m
	if m1 < m:
		m2 = flipBit(m1, Npair-i-1)
		if m2 > m1:
			return m2
	
	return 0

@jit
def cranOnState(i, j, state, Npair, basisList, lengthOfBasis):
	"""
	Applies the operator c_i^dag c_j to a basis state m, where c (c^dag) are PAIR anihilation (creation) operators.
	INPUT: 
	i, j - positions of operators, integer
	state - a state vector, list (np array)
	OUTPUT
	new_state - resulting state vector, list (np array)
	"""

	new_state = np.zeros(lengthOfBasis)

	for k in range(lengthOfBasis):

		coef = state[k]
		if coef!=0:

			m = cran(i, j, basisList[k], Npair)

			if m!=0:
				res = np.searchsorted(basisList, m)

				new_state[res] += coef
				
	return new_state

@jit
def countingOp(i, m, Npair):
	"""
	Calculates the application of the counting operator to a basis state m. Returns 0 or 1, according to the occupation of the energy level.
	"""

	m1 = flipBit(m, Npair-i-1)
	if m1 < m:
		return 1
	else:
		return 0

@jit
def countingOpOnState(i, state, Npair, basisList, lengthOfBasis):
	"""	
	Calculates the application of the counting operator on a given state vector state.
	"""
	new_state = np.zeros(lengthOfBasis)
	
	for k in range(lengthOfBasis):
		coef = state[k]

		if coef!=0:	
			new_state[k] += countingOp(i, basisList[k], Npair)*coef
	
	return new_state	

# HAMILTONIAN ##################################################################

def HonState(state, d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies):
	"""
	Calculates the action of the Hamiltonian to a given state of pairs. Calculates only the pair hamiltonian, the sum of singly occupied energies should be added later!
	INPUT:
	d, alpha - physical constants (float)
	state - the state vector (list)
	N - number of levels (int)
	n - number of particles in the system (int)
	npair - number of pairs in the system (int)
	basisList - a list of all basis states (list)
	lengthOfBasis - the length of the state vector (int)
	pairEnergies - a set of energies of pair levels (list)
	"""

	Npair = N - (n - 2*npair)	#number of levels, available for pairs

	kineticPair, interactionPair = 0, 0

	for i in range(Npair):

		kineticPair += 2 * pairEnergies[i] * countingOpOnState(i, state, Npair, basisList, lengthOfBasis)

		for j in range(Npair):
			interactionPair += cranOnState(i, j, state, Npair, basisList, lengthOfBasis)

	return kineticPair - d*alpha*interactionPair

class HLinOP(LinearOperator):
	"""
	This is a class, built-in in scipy, which allows for a representation of a given function as a linear operator. The method _matvec() defines how it acts on a vector.
	"""
	def __init__(self, d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies, dtype='float64'):
		self.shape = (lengthOfBasis, lengthOfBasis)
		self.dtype = np.dtype(dtype)
		self.d = d
		self.alpha = alpha
		self.N = N
		self.n = n
		self.npair = npair
		self.basisList = basisList
		self.lengthOfBasis = lengthOfBasis
		self.pairEnergies = pairEnergies

	def _matvec(self, state):
		return HonState(state, self.d, self.alpha, self.N, self.n, self.npair, self.basisList, self.lengthOfBasis, self.pairEnergies)

# VISUALISATION ################################################################

def occupationByEigenstates(N, n, d, alpha, NofEgienstates=2):
	"""
	Calcualtes the expected occupation number for each energy level, for first NofEigenstates eigenstates.
	"""

	CompleteListOfStatesOrdered, spectrumOrdered  = getEigenstatesOLD(N, n, d, alpha, NofEgienstates)
																#This is a list, where each element is a list of expected occupation numbers by energy level of an eigenstate. 
	occupationOfLevelsByStates = np.zeros((NofEgienstates, N))	#Elements are ordered by the energy of the state; occupationOfLevelsByStates[0] for GS etc.	  
	for i in range(NofEgienstates):
		state = CompleteListOfStatesOrdered[i][0]
		basisList = CompleteListOfStatesOrdered[i][1]
		singlyOcc = CompleteListOfStatesOrdered[i][2]

		for j in range(len(basisList)):	#for each basis vector
			count=0
			for k in range(N):	#each energy level
				
				if testBit(singlyOcc, N-k-1)!=0:	#level is singly occupied
					occupationOfLevelsByStates[i, k] += (state[j])**2

				else:	#level is not singly occupied	
					if testBit(basisList[j], count)==0:
						count+=1
					else:
						occupationOfLevelsByStates[i, k] += 2*(state[j])**2
						count+=1

	return occupationOfLevelsByStates				

def occupationNumberVector(D, N, n, d, alpha, NofEgienstates=2):
	"""
	Transforms the eigenvectors into a "human readable" form of the occupation number representation.
	"""

	CompleteListOfStatesOrdered, spectrumOrdered  = getEigenstatesOLD(D, N, n, d, alpha, NofEgienstates=NofEgienstates)
																#This is a list, where each element is a list of expected occupation numbers by energy level of an eigenstate. 
	occupationOfLevelsByStates = np.zeros((NofEgienstates, N))	#Elements are ordered by the energy of the state; occupationOfLevelsByStates[0] for GS etc.	  
	for i in range(NofEgienstates):
		state = CompleteListOfStatesOrdered[i][0]
		basisList = CompleteListOfStatesOrdered[i][1]
		singlyOcc = CompleteListOfStatesOrdered[i][2]

		for j in range(len(basisList)):	#for each basis vector
			count=0
			for k in range(N):	#each energy level
				
				if testBit(singlyOcc, N-k-1)!=0:	#level is singly occupied
					occupationOfLevelsByStates[i, k] += 0

				else:	#level is not singly occupied	
					if testBit(basisList[j], count)==0:	#level does not have a pair
						count+=1
					else:								#level has a pair
						occupationOfLevelsByStates[i, k] += state[j]
						count+=1

	return spectrumOrdered, occupationOfLevelsByStates, CompleteListOfStatesOrdered	

################################################################################

def getSpectrum(D, N, n, d, alpha, verbosity=False):
	"""
	For a given number of levels N and number of particles n, returns the spectrum of eigenenergies.
	INPUT:
	N - number of energy levels, with dispersion, defined in eps() (int)
	n - number of particles (int)
	d, alpha - physical constants (float)
	verbostity - if True, print the parameters before the calculation
	OUTPUT:
	spectrum - a list of eigenenergies of the Hamiltonian, sorted from smallest to biggest
	"""
	if verbosity:
		print("D, N, n, d, alpha")
		print(D, N, n, d, alpha)
		print()

	spectrum=[]

	for npair in range(max(n-N, 0), min(n//2, N)+1):
		#The case with zero pairs:

		if npair==0:
			spectrum.append(sum([eps(i, d, D) for i in range(N)]))
			continue

		#Cases with more than zero pairs	
		singlyOccupiedList = allSinglyOccupiedPossibilities(N, n, npair)
		pairEnergies, singleEnergies = pairLevelsDispersion(d, N, D, singlyOccupiedList)
		basisList, lengthOfBasis = defineBase(N, n, npair)


		for i in range(len(pairEnergies)):

			if lengthOfBasis == 1:
				
				values = np.array([pairEnergies[i, kk] for kk in range(len(pairEnergies[i]))])
				values += sum(singleEnergies[i])
				spectrum.extend(values)
			
			else:	
				LinOp = HLinOP(d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies[i]) 
				values = eigsh(LinOp, k=min(lengthOfBasis-1, 5), which="SA", return_eigenvectors=False)[::-1]

				values += sum(singleEnergies[i])

				spectrum.extend(values)

	return sorted(spectrum)

def getEigenstatesOLD(D, N, n, d, alpha, NofEgienstates=2):
	"""
	Calculates the spectrum and its eigenstates. Returns orderes eigenstates, accompanied by a list of integers, which represent singly occupied levels of each eigenstate. 
	Deprecated but needed for the occupation number pictures/functions.
	"""
	
	spectrum=[]
	CompleteListOfStates=[]
	#FOR ALL POSSIBLE NUMBERS OF PAIRS
	for npair in range(max(n-N, 0), min(n//2, N)+1):
		print()
		print(npair)

		#The case with zero pairs:
		if npair==0:
			#spectrum.append(sum([eps(i, d, N) for i in range(N)]))
			#this step is skipped because this case has an empty basisList, so a base vector could not be written. 
			continue

		#FIND ALL CONFIGURATIONS - POSITIONS OF DOUBLY AND SINGLY OCCUPIED LEVELS

		#Cases with more than zero pairs	
		singlyOccupiedList = allSinglyOccupiedPossibilities(N, n, npair)

		pairEnergies, singleEnergies = pairLevelsDispersion(d, N, D, singlyOccupiedList)
		basisList, lengthOfBasis = defineBase(N, n, npair)
	
		#ITERATE OVER ALL CONFIGURATIONS
		for i in range(len(pairEnergies)):

			#IF THE CONFIGURATION HAS ONLY ONE POSSIBLE POSITION FOR PAIRS
			if lengthOfBasis == 1:

				values = np.array([pairEnergies[i, kk] for kk in range(len(pairEnergies[i]))])
				values += sum(singleEnergies[i])
				spectrum.extend(values)
				
				CompleteListOfStates.append([np.array([1]), basisList, singlyOccupiedList[i]])

				for kkk in range(len(values)):
					print(values[-kkk], CompleteListOfStates[-kkk])

			else:
				print("AAA", i)	
				LinOp = HLinOP(d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies[i]) 
				values, vectors = eigsh(LinOp, k=min(lengthOfBasis-1, 5), which="SA")

				values += sum(singleEnergies[i])

				spectrum.extend(values)
				#eigenvectors.extend(vectors)

				for j in range(len(values)):
					CompleteListOfStates.append([vectors[:, j], basisList, singlyOccupiedList[i]])

				for kkk in range(len(values)):
					print(values[-kkk], CompleteListOfStates[-kkk])	

					
	#find the indexes which sort the array:
	sortedIndeces = np.argsort(spectrum)
	#create a list of eigenvectors and a list of singly occupied states in the sorted order 
	CompleteListOfStatesOrdered=[]
	spectrumOrdered=[]
	for i in range(len(spectrum)):
		CompleteListOfStatesOrdered.append(CompleteListOfStates[sortedIndeces[i]])

		spectrumOrdered.append(spectrum[sortedIndeces[i]])
	

	print()
	print()	

	for i in range(len(spectrum)):	
		print(spectrumOrdered[i], CompleteListOfStatesOrdered[i])

	print()
	print()	

	return CompleteListOfStatesOrdered, spectrumOrdered 

def getEigenstates(D, N, n, d, alpha, NofEgienstates=2):
	"""
	Calculates the spectrum and its eigenstates. Returns orderes eigenstates, accompanied by a list of integers, which represent singly occupied levels of each eigenstate. 
	This function returns 
	"""
	
	spectrum=[]
	pairSpectrum=[]
	CompleteListOfStates=[]
	#FOR ALL POSSIBLE NUMBERS OF PAIRS
	for npair in range(max(n-N, 0), min(n//2, N)+1):

		#The case with zero pairs:
		if npair==0:
			#spectrum.append(sum([eps(i, d, N) for i in range(N)]))
			#this step is skipped because this case has an empty basisList, so a base vector could not be written. 
			continue

		#FIND ALL CONFIGURATIONS - POSITIONS OF DOUBLY AND SINGLY OCCUPIED LEVELS

		#Cases with more than zero pairs	
		singlyOccupiedList = allSinglyOccupiedPossibilities(N, n, npair)

		pairEnergies, singleEnergies = pairLevelsDispersion(d, N, D, singlyOccupiedList)
		basisList, lengthOfBasis = defineBase(N, n, npair)
	
		#ITERATE OVER ALL CONFIGURATIONS
		for i in range(len(pairEnergies)):

			#IF THE CONFIGURATION HAS ONLY ONE POSSIBLE POSITION FOR PAIRS
			if lengthOfBasis == 1:

				values = np.array([pairEnergies[i, kk] for kk in range(len(pairEnergies[i]))])

				pairSpectrum.extend(values)


				values += sum(singleEnergies[i])
				
				spectrum.extend(values)
				
				CompleteListOfStates.append([np.array([1]), basisList, pairEnergies[i]])

			else:
				LinOp = HLinOP(d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies[i]) 
				values, vectors = eigsh(LinOp, k=min(lengthOfBasis-1, 5), which="SA")

				pairSpectrum.extend(values)

				values += sum(singleEnergies[i])

				spectrum.extend(values)
				#eigenvectors.extend(vectors)

				for j in range(len(values)):
					CompleteListOfStates.append([vectors[:, j], basisList, pairEnergies[i]])

					
	#find the indexes which sort the array:
	sortedIndeces = np.argsort(spectrum)
	#create a list of eigenvectors and a list of singly occupied states in the sorted order 
	CompleteListOfStatesOrdered=[]
	spectrumOrdered=[]
	pairSpectrumOrdered=[]
	for i in range(len(spectrum)):
		CompleteListOfStatesOrdered.append(CompleteListOfStates[sortedIndeces[i]])

		spectrumOrdered.append(spectrum[sortedIndeces[i]])
		pairSpectrumOrdered.append(pairSpectrum[sortedIndeces[i]])

	return CompleteListOfStatesOrdered, spectrumOrdered, pairSpectrumOrdered 

################################################################################
spectroscopic_gap_plot = 0
eigenstates_plot = 0
eigenstates_of_alpha_plot = 0
################################################################################

if 0:
	print("START")
	N = 15
	n = N
	D = 1
	d = 2*D/N
	alpha = 1
	
	a = getSpectrum(D, N, n, d, alpha)

	print(a[:10])

	print("DONE")

if 1:
	print("START")
	N = 15
	n = N
	D = 1
	d = 2*D/N
	
	#alpha = 0.1
	
	for alpha in [0, 0.1, 1]:
		print()
		print("alpha: ", alpha)

		vec, val, pairE = getEigenstates(D, N, n, d, alpha, NofEgienstates=2)

		print(val[:20])
		#print(pairE)
		#print(vec)
		print()

		for i in range(20):

			npair = countSetBits(vec[i][1][0])
			basisList = vec[i][1]
			lengthOfBasis = len(basisList)
			pairEnergies = vec[i][2]
			state = np.array(vec[i][0])

			Hs = HonState(state, d, alpha, N, n, npair, basisList, lengthOfBasis, pairEnergies)

			print(val[i], norm(pairE[i]*state - Hs))
		print()	

	print("DONE")	

if spectroscopic_gap_plot:

	for N in [9, 10]:
		print(N)

		n = N
		d = 1

		x=np.logspace(-1.3, 0.5, 20)
		alphalist = [1/np.arcsinh(0.25*N*d/Delta) for Delta in x]
		dDelta, gapList = [], []
		for alpha in alphalist:
			gap = findGap(getSpectrum(N, n, d, alpha))


			omegaD = 0.5*N*d	
			Delta = omegaD/(2*np.sinh(1/alpha))	
			dDelta.append(d/Delta)
			gapList.append(gap)


		plt.scatter(dDelta, gapList, label="N="+str(N))	

	plt.xlabel(r"$d/\Delta$")
	plt.ylabel(r"$E_G/d$")	

	plt.hlines(xmin= 0, xmax=20, y=1, linestyle="dashed")
	
	plt.ylim(0, 4)
	plt.xlim(0,20)
			
	plt.legend()
	plt.grid()
	plt.show()	

if eigenstates_plot:

	N = 4
	n = N
	d, alpha = 1, 0.00001
	NofEgienstates = 2

	occupationLists = occupationByEigenstates(N, n, d, alpha, NofEgienstates=NofEgienstates)

	fig = plt.figure()    
	ax = fig.add_subplot(1,1,1)
	
	for i in range(NofEgienstates-1, -1, -1):
		plt.plot([N-i-1 for i in range(N)], occupationLists[i], label=r"$E_{0}$".format(i))
		plt.fill_between([N-i-1 for i in range(N)], occupationLists[i], alpha=0.5)

	plt.vlines(ymin=0, ymax=2, x=(N-1)/2, linestyle="dashed", label=r"$E_F$")

	plt.xlabel("i")
	plt.ylabel(r"$\langle n_i \rangle$")	

	ax.set_axisbelow(True)

	a=0.05
	plt.ylim(0-a, 2+a)

	plt.legend()
	ax.set_xticks([N-i-1 for i in range(N)], minor=False)
	plt.grid()


	plt.tight_layout()
	plt.show()	

if eigenstates_of_alpha_plot:
	
	#PARAMS
	N = 10
	n = N
	d = 1
	NofEgienstates = 2

	#FIGURE SETUP
	fig = plt.figure()    
	ax = fig.add_subplot(1,1,1)
	
	#FOR LOOP
	x=np.logspace(-0.6, 0.8, 5)
	alphalist = [1/np.arcsinh(0.25*N*d/Delta) for Delta in x]

	dDelta = []
	for alpha in alphalist:

		omegaD = 0.5*N*d	
		Delta = omegaD/(2*np.sinh(1/alpha))	
		dDelta.append(d/Delta)

		print(d/Delta)

		occupationLists = occupationByEigenstates(N, n, d, alpha, NofEgienstates=2)

		plt.plot([N-i-1 for i in range(N)], occupationLists[1], label=r"$d/\Delta={0:.2f}$".format(d/Delta))
		plt.fill_between([N-i-1 for i in range(N)], occupationLists[1], alpha=0.5)
	
	plt.vlines(ymin=-1, ymax=3, x=(N-1)/2, linestyle="dashed", label=r"$E_F$", alpha=0.5)


	plt.xlabel("i")
	plt.ylabel(r"$\langle n_i \rangle$")	

	ax.set_axisbelow(True)

	a=0.05
	plt.ylim(0-a, 2+a)
	b=0.05
	plt.xlim(0-b, N-1+b)

	plt.legend()

	ax.set_xticks([N-i-1 for i in range(N)], minor=False)
	plt.grid()

	plt.title(r"$N={0}$".format(N))



	plt.tight_layout()
	plt.show()	



