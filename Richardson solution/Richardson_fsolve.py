"""
Solves the Richardson's equations using scipy's fsolve function.
"""

import numpy as np
from scipy.optimize import fsolve
from scipy.special import comb

################################################################################

def countSetBits(n): 
	"""
	Counts the number of bits that are set to 1 in a given integer.
	"""
	count = 0
	while (n): 
		count += n & 1
		n >>= 1
	return count 

def flipBit(n, offset):
	"""
	Flips the bit at position offset in the integer n.
	"""
	mask = 1 << offset
	return(n ^ mask)

def testBit(n, offset):
	"""
	DOES NOT RETURN ONLY 1 OR 0! Returns 0 when the bit at offset is 0, but 2^offset when the bit is 1. (If the bit at offset is 1, it returns the number
	corresponding to the binary string 0...010...0, where 1 is at position=offset.)
	"""
	mask = 1 << offset
	return(n & mask)

################################################################################

def pairLevelsDispersion(d, N, D, singlyOccupiedList):
	"""
	Given the list of the singly occupied levels, returns a list of lists of energies for levels available for pair interaction, and a list of lists of energies of singly occupied levels.
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

################################################################################

def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return -D + d*i + d/2

################################################################################

def equations(E, *data):
	"""
	A system of equations, describing the energies of npair pairs, where dispersionList is a list of available energies for the pairs.
	*data is a tuple of arguments, * unpacks the tuple into seperate args.
	E is a np.array of npair energies.
	"""
	g, n = data

	result = np.zeros(n)
	for nu in range(n):

		eq = 1

		for j in range(n):
			eq += -g / (2*eps(j, d, D) - E[nu])

			if j!=nu:
				eq += 2*g / (E[j] - E[nu])
				

		result[nu] = eq	

	return result

print("NE DELA")

################################################################################
D=1
N, n = 4, 4
alpha, d = 1, 2*D/N
g = alpha*d

npair = 2

singlyOccupiedList = allSinglyOccupiedPossibilities(N, n, npair)
pairEnergies, singleEnergies = pairLevelsDispersion(d, N, D, singlyOccupiedList)
initGuess = pairEnergies[0] 	#at least for small alpha, the result should be close to the non interacting energies 

print(initGuess)


res = sorted(fsolve(equations, initGuess, args=(g, n)))
print(res)

print(res[0]+res[1])


def energies(D, N, n, alpha, d):

	for npair in range(max(n-N, 0), min(n//2, N)+1):

		if npair==0:
			spectrum.append(sum([eps(i, d, D) for i in range(N)]))
			continue

		#Cases with more than zero pairs	
		singlyOccupiedList = allSinglyOccupiedPossibilities(N, n, npair)
		pairEnergies, singleEnergies = pairLevelsDispersion(d, N, D, singlyOccupiedList)

		initGuess = pairEnergies[:npair] + 0.1	#at least for small alpha, the result should be close to the non interacting energies 


		return None




