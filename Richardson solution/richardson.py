import scipy
import numpy as np
from scipy.optimize import fsolve, root


################################################################################

def eps(i, d, D):
	"""
	Dispersion; the spacing between levels is d. This is used to compute the energy for the singly occupied levels.
	"""

	return -D + d*i + d/2

################################################################################
#N - number of levels
#np - number of pairs

def eq(nu, E, g, N, npair, d, D, available_levels):

	sum1, sum2 = 0, 0

	for i in available_levels:
		sum1+=1/(2*eps(i, d, D) - E[nu])

	for mu in range(npair):
		if mu!=nu:
			sum2+=2/(E[mu]-E[nu])

	return 1 - g*sum1 + g*sum2

def equations(E, *data):

	data = g, N, npair, d, D, available_levels

	return [eq(nu, E, g, N, npair, d, D, available_levels) for nu in range(len(E))]

################################################################################

D=1
N=4
d=2*D/N
alpha=0.1

g = alpha*d

npair=2

available_levels = [i for i in range(npair)]
#singly_occupied_levels = 

initE = [2*eps(i, d, D)-0.001 for i in range(N)]


data = (g, N, npair, d, D, available_levels)
Es = fsolve(equations, initE, args=data)

print(Es)
print(sum(Es[:npair]))

Es = fsolve(equations, initE, args=data)

