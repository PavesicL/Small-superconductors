"""
This is a program written to construct and diagonalise the mean field hamiltonian of a small superconductor.

The states are written in the occupancy number representation, as |n_0UP,n_0DOWN,...n_NUP,n_NDOWN>.s
The state vector is a dictionary of the type state = { m : amp }, where |bin(m)> is a basis state and amp is its probability amplitude. As many of the basis state may never be
included in the state, there would be many zeros in the vector of all states. It turns out that for this specific application, the dictionary approach works better.

For N energy levels (each can be occupied by UP and DOWN fermions), the vector has 2^2N elements. The intial state is a state with all DOWN levels occupied. It is represented by a dictionary,
in which the only element is m : 1, where; where m = sum(2^((2i)+1) for i in range(0, N)).
"""

################################################################################

import numpy as np 
import matplotlib.pyplot as plt
from numpy.linalg import norm

from Functions_MFSC_Hamiltonian import *
from Functions_fermionic_multiplication import *


# FUNCTIONS ####################################################################

def construct_initial_state_dict(N):
	"""Initial state |01010101...> is a dictionary, where m is stored with amplitude 1."""

	m=0
	for i in range(0, N):
		m += 2**(2*i)

	state = { m : 1 }

	return state

def d_on_state_dict(n, state, u, v, N):
	"""Calculates the result of the action of the eigenoperator d_n^dag on a given state. Returns a state vector.
	n - which eigenoperator
	state - state vector
	u - vector u, composed of eigenvectors
	v - vector v, composed of eigenvectors
	N - number of energy levels"""

	new_state = {}	#this will be the resulting state

	#iterate over all basis states in a given state vector (actually a dictionary)
	for m in state:
		#iterate over all energy levels
		for i in range(N):

			prefactor_cr, state_cr = fmulti_cr(m, i, 0, N)
			prefactor_an, state_an = fmulti_an(m, i, 1, N)

			#the result of the action of the operator: ±1 * u_ni * state[m] is added to the correct element of the vector new_state 
			if state_cr != None and u[n,i] != 0:

				try:
					new_state[state_cr] += prefactor_cr * u[n, i] * state[m]
				except KeyError:
					new_state[state_cr] = prefactor_cr * u[n, i] * state[m]

			if state_an != None and v[n,i] != 0:
				try:
					new_state[state_an] += prefactor_an * v[n, i] * state[m]
				except KeyError:
					new_state[state_an] = prefactor_an * v[n, i] * state[m]	

	return new_state

def iterative_state_creation(state, u, v, N, save=False, verbosity=True, attemptSpeedup=False):
	"""Starting from the initial state, iteratively applies d_n. Intermediate states can be saved, if save==True. If attemptSpeedup parameter
	is not False but a number, all elements smaller than the number will be disregarded.""" 
	

	listOfStates=[state]

	for n in range(N):

		#print the step numbers during iteration
		if verbosity:
			print("step: "+str(n))

		#delete all elements smaller than attemptSpeedup	
		if attemptSpeedup:
			toDelete=[]
			for i in state:
				if np.abs(state[i]) < attemptSpeedup:
					toDelete.append(i)
	
			for i in toDelete:
				del state[i]

		#calculate a new state		
		state = d_on_state_dict(n, state, u, v, N)	

		#save the state to a list	
		if save:
			listOfStates.append(state)	

	if save:
		return listOfStates
	else:
		return state

def HMF(state, Delta, N):
	"""Computes the result of the MF hamiltonian acting on a given state."""


	#kinetic term: sum_i(eps(i)*(n_i,up + n_i,down))
	kinetic_state = dict_list_sum(
		[dict_prod(eps(i, N), dict_sum(number_op(state, i, 0, N), number_op(state, i, 1, N))) for i in range(N)])

	#interaction term: sum_i( Delta c_iUP^dag c_iDOWN^dag + conj.(Delta) c_iDOWN c_iUP )
	interaction_state = dict_list_sum(
		[dict_sum(dict_prod(Delta, cr(cr(state, i, 1, N), i, 0, N)), dict_prod(np.conj(Delta), an(an(state, i, 0, N), i, 1, N))) for i in range(N)])

	return dict_sum(kinetic_state, interaction_state)

def Hexact(state, N):
	"""Computes the result of the exact Hamiltonian acting on a given state."""

	#kinetic term: sum_i(eps(i)*(n_i,up + n_i,down))
	kinetic_state = dict_list_sum(
		[dict_prod(eps(i, N), dict_sum(number_op(state, i, 0, N), number_op(state, i, 1, N))) for i in range(N)])

	#interaction term: sum_ij(c_iUP^dag c_iDOWN^dag c_jDOWN c_jUP)
	interaction_state = dict_list_sum([
							dict_list_sum([
								cr(cr(an(an(state, j, 1, N), j, 0, N), i, 0, N), i, 1, N) for i in range(N)]) for j in range(N)])
	
	g=1/N
	interaction_state = dict_prod(-g, interaction_state)
	return dict_sum(kinetic_state, interaction_state)

def E_MF(state, Delta, N):
	"""Calculates the expected value of H_MF (mean-field energy) of a given state."""

	g=1/N

	return scalar_prod(state, HMF(state, Delta, N)) + ((Delta**2)/g)

def E_exact(state, N):
	"""Calculates the expected value of H (so-called 'exact' energy) of a given state."""

	return scalar_prod(state, Hexact(state, N))

def occupancy(state):
	"""Calculates the expected occupancy number <N> and the expected spin <S_z> of a given state."""

	szUP = sum([scalar_prod(state, number_op(state, i, 0, N)) for i in range(N)])
	szDOWN = sum([scalar_prod(state, number_op(state, i, 1, N)) for i in range(N)])


	return szUP + szDOWN, (szUP - szDOWN)/2

# PARAMETERS ###################################################################
N = 10
# DO ###########################################################################
plot_both_E=1
plot_occupancy=0
plot_occupancy_mu=0
plot_Sz_mu=0
plot_Delta_mu=0
################################################################################

if plot_both_E:
	#plot MF and 'exact' energy during the iteration

	state = construct_initial_state_dict(N)

	#Diagonalise H, get eigenenergies and vectors u, v
	Delta, energies, u, v = optimizeDelta(N, 0.1, mu=0, precision=1e-6)

	print("Delta: " + str(Delta))
	print("energies")
	print(energies)
	print()

	states = iterative_state_creation(state, u, v, N, save=1)#, attemptSpeedup=1e-5)

	print("MF energy...")
	MFenergyList = []
	for i in range(len(states)):
		print(i)
		MFenergyList.append(E_MF(states[i], Delta, N))

	print("energy...")
	exactenergyList = []
	for i in range(len(states)):
		print(i)
		exactenergyList.append(E_exact(states[i], N))

	print(MFenergyList)
	print(exactenergyList)



	fig = plt.figure(figsize=(10, 5))
		

	ax1 = fig.add_subplot(121)		

	ax1.set_title("N="+str(N))

	ax1.scatter([i for i in range(len(MFenergyList))], exactenergyList)
	ax1.scatter([i for i in range(len(MFenergyList))], MFenergyList)
	ax1.plot([i for i in range(len(MFenergyList))], exactenergyList)
	ax1.plot([i for i in range(len(MFenergyList))], MFenergyList)

	ax1.set_ylabel(r"E")

	ax1.grid()


	ax2 = fig.add_subplot(122)

	ax2.scatter([i for i in range(len(exactenergyList)-1)], [exactenergyList[i+1]-exactenergyList[i] for i in range(N)], label="exact")
	ax2.scatter([i for i in range(len(MFenergyList)-1)], [MFenergyList[i+1]-MFenergyList[i] for i in range(N)], label="MF")
	ax2.plot([i for i in range(len(exactenergyList)-1)], [exactenergyList[i+1]-exactenergyList[i] for i in range(N)])
	ax2.plot([i for i in range(len(MFenergyList)-1)], [MFenergyList[i+1]-MFenergyList[i] for i in range(N)])
	
	print(energies)

	ax2.scatter([i for i in range(N)], energies[:len(energies)//2], label="negative eigenenergies")

	ax2.set_ylabel(r"$\Delta E$")
	ax2.grid()


	ax2.legend()


	plt.tight_layout()
	plt.show()
	#plt.savefig("Slike/energies&differences_n"+str(N)+"_fix.pdf")
	#print("SAVED; N "+str(N))


if plot_occupancy:
	#plot occupanct and S_z during the iteration

	state = construct_initial_state_dict(N)

	#Diagonalise H, get eigenenergies and vectors u, v
	Delta, energies, u, v = optimizeDelta(N, 0.1, mu=0, precision=1e-6)

	print("Delta: " + str(Delta))
	print("energies")
	print(energies)
	print()

	states = iterative_state_creation(state, u, v, N, save=1, attemptSpeedup=1e-5)

	print("occupancy")
	occupancyList, spinList = [], []
	for i in range(len(states)):
		print(i)
		occ = occupancy(states[i])
		occupancyList.append(occ[0])
		spinList.append(occ[1])


	plt.title("N="+str(N))

	plt.scatter([i for i in range(N+1)], occupancyList, label=r"$\langle N \rangle_\uparrow + \langle N \rangle_\downarrow$")
	plt.scatter([i for i in range(N+1)], spinList, label=r"$(\langle N \rangle_\uparrow - \langle N \rangle_\downarrow)/2$")	

	plt.plot(occupancyList)
	plt.plot(spinList)	


	plt.legend()
	plt.grid()

	plt.show()
	#plt.savefig("Slike/occupancy&Sz_n"+str(N)+".pdf")
	plt.close()


if plot_occupancy_mu:
	#plot occupancy dependence on mu

	def N_prediction(mu, N):

		return N + mu*(N-1)


	for N in [7, 8, 9, 10, 11, 12]:
		print(N)	

		occList, spinList = [], []

		DeltaE, factor =  2/(N-1), 2

		mumin, mumax = -factor*DeltaE, factor*DeltaE
		x=np.linspace(mumin, mumax, 15)
		for mu in x:
		
			state = construct_initial_state_dict(N)
			Delta, energies, u, v = optimizeDelta(N, 0.1, mu=mu, precision=1e-6)
			states = iterative_state_creation(state, u, v, N, save=1, attemptSpeedup=1e-5, verbosity=False)
			
			state = states[-1]

			occ = occupancy(state)
			occList.append(occ[0])

		#occupancy
		#plt.scatter(np.linspace(-factor, factor, len(x)), occList, label="N="+str(N))
		#plt.plot(np.linspace(-factor, factor, len(x)), occList)
		
		#prediction
		#plt.plot(np.linspace(-factor, factor, len(x)), [N_prediction(x[i], N) for i in range(len(x))], c="red", linestyle='dashed')

		#occupancy - prediction
		plt.scatter(np.linspace(-factor, factor, len(x)), [occList[i] - N_prediction(x[i], N) for i in range(len(x))], label="N="+str(N))
		plt.plot(np.linspace(-factor, factor, len(x)), [occList[i] - N_prediction(x[i], N) for i in range(len(x))])

	plt.xlabel(r"$\mu [\Delta_\varepsilon]$")
	plt.ylabel(r"$\langle N\rangle$")

	#plt.title("N="+str(N))
	plt.legend()
	
	plt.grid()

	plt.tight_layout(True)
	plt.show()

	#plt.savefig("Slike/occupancy&Sz_n"+str(N)+".pdf")
	plt.close()


if plot_Sz_mu:
	#plot Sz dependence on mu

	
	for N in [10]:
		print(N)	
		spinList1, spinList2 = [], []

		mumin, mumax = -2/N, 2/N
		x=np.linspace(mumin, mumax, 10)
		for mu in x:
		
			state = construct_initial_state_dict(N)

			Delta, energies, u, v = optimizeDelta(N, 0.1, mu=mu, precision=1e-6)

			states = iterative_state_creation(state, u, v, N, save=1, attemptSpeedup=1e-5, verbosity=False)
			
			state1 = states[-1]
			state2 = states[-2]

			occ1 = occupancy(state1)
			occ2 = occupancy(state2)

			spinList1.append(occ1[1])
			spinList2.append(occ2[1])



		#occupancy
		plt.scatter(np.linspace(-2, 2, len(x)), spinList1 , label=r"$\eta_{N}$")
		plt.plot(np.linspace(-2, 2, len(x)), spinList1)
		plt.scatter(np.linspace(-2, 2, len(x)), spinList2 , label=r"$\eta_{N-1}$")
		plt.plot(np.linspace(-2, 2, len(x)), spinList2)


	plt.xlabel(r"$\mu [\Delta_\varepsilon]$")
	#plt.ylabel(r"$\langle N\rangle - N(\mu)$")
	plt.ylabel(r"$\langle S_z\rangle$")

	plt.legend()
	plt.grid()

	plt.tight_layout(True)
	plt.show()

	#plt.savefig("Slike/occupancy&Sz_n"+str(N)+".pdf")
	plt.close()


if plot_Delta_mu:


	for N in [10, 11, 12, 13, 14, 15]:


		DeltaE, factor =  2/(N-1), 7

		mumin, mumax = -factor*DeltaE, factor*DeltaE
		x=np.linspace(mumin, mumax, 200)
		DeltaList = []
		for mu in x:
		
			state = construct_initial_state_dict(N)

			Delta, energies, u, v = optimizeDelta(N, 0.1, mu=mu, precision=1e-6)

			DeltaList.append(Delta)

			#states = iterative_state_creation(state, u, v, N, save=1, attemptSpeedup=1e-5, verbosity=False)
			
			#state = states[-1]

			#occ = occupancy(state)


		#plt.scatter(np.linspace(-factor, factor, len(x)), DeltaList, label="N="+str(N))
		plt.plot(np.linspace(-factor, factor, len(x)), DeltaList, label="N="+str(N))


	plt.xlabel(r"$\mu [\Delta_\varepsilon]$")
	#plt.ylabel(r"$\langle N\rangle - N(\mu)$")
	plt.ylabel(r"$\Delta$")

	plt.legend()
	plt.grid()

	plt.tight_layout(True)
	plt.show()

	#plt.savefig("Slike/occupancy&Sz_n"+str(N)+".pdf")
	plt.close()



