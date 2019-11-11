"""
Functions used for fermionic multiplication. These are written for multiplication in the Nambu basis (c_i,UP^dag, c_i,DOWN), so be careful using them elsewhere!
"""


# BITWISE FUNCTIONS ##################################################

def bitCount(int_type):
	"""Returns the number of bits that are equal to 1 in the binary form of the integer int_type."""
	count = 0
	while(int_type):
		int_type &= int_type - 1

		count += 1
	return(count)

def clearBit(int_type, offset):
	"""Sets the bit at offset to 0."""
	mask = ~(1 << offset)
	return(int_type & mask)

def flipBit(int_type, offset):
	"""Flips the bit at position offset in the integer int_type."""
	mask = 1 << offset
	return(int_type ^ mask)

def testBit(int_type, offset):
    mask = 1 << offset
    return(int_type & mask)

def  countSetBits(n): 
	"""Counts the number of bits that are set to 1 in a given integer."""
	count = 0
	while (n): 
		count += n & 1
		n >>= 1
	return count 

# FERMIONIC MULTIPLICATION ###########################################

def fmulti_cr(m, i, s, N):
	"""Multiplies the state |m> with c_i,s^dag. N is the number of energy levels, and is len(bin(m))//2. Spin: UP = 0, DOWN = 1.
	Returns [±1, integer], where ±1 specifies the sign prefactor, and bin(integer) is the resulting state. If the multiplication is not possible, returns None.
	"""

	new_m = flipBit(m, 2*N - (2*i+1+s))    #flips 2*i-th bit in m

	if new_m > m:    #the operator can act, the result is new_m, with a prefactor ±1
		return prefactor_cr(m, i, s, N), new_m    

	if new_m < m:    #the operator destroys the state; return None (this must not be return 0, as |00...> is also a valid state!)
		return 1, None

def prefactor_cr(m, i, s, N):
	"""Calculates the sign prefactor, obtained when multiplying the state |m> with c_i,s^dag. Spin: UP = 0, DOWN = 1.
	N is the number of energy levels, and is len(bin(m))//2. Takes bin(m), sets all its bit from positions 2N to ith to 0 and counts
	how remaining many bits are equal to 1."""

	new_num = m
	#set bits to zero
	for j in range(0, 2*N - ((2*i)+1+s)):
		new_num = clearBit(new_num, j)   
	
	#count the remaining 1s
	count = bitCount(new_num)
	
	return (-1)**count

def fmulti_an(m, i, s, N):
	"""Multiplies the state |m> with c_i,s. N is the number of energy levels, and is len(bin(m))//2. Spin: UP = 0, DOWN = 1.
	Returns [±1, integer], where ±1 specifies the sign prefactor, and bin(integer) is the resulting state. If the multiplication is not possible, returns None.
	"""

	new_m = flipBit(m, 2*N - ((2*i)+1+s))    #flips 2*i+1-th bit in m

	if new_m < m:	#the operator can act, the result is new_m, with a prefactor ±1
		return prefactor_an(m, i, s, N), new_m    


	if new_m > m:    #the operator destroys the state; return None (this must not be return 0, as |00...> is also a valid state!)
		return 1, None
	
def prefactor_an(m, i, s, N):
	"""Calculates the sign prefactor, obtained when multiplying the state |m> with c_i,s. Spin: UP = 0, DOWN = 1.
	N is the number of energy levels, and is len(bin(m))//2. Takes bin(m), sets all its bit from positions 2N to ith to 0 
	and counts how remaining many bits are equal to 1."""

	new_num = m
	#set bits to zero
	for j in range(0, 2*N - ((2*i)+s)):
		new_num = clearBit(new_num, j)   
	
	#count the remaining 1s
	count = bitCount(new_num)
	
	return (-1)**count

def number_op(state, i, s, N):
	"""Calculates the resut of a number operator acting on a given state (a linear superposition of |m>) at position i with spin s. Spin should be 0 for up and 1 for down."""
	new_state={}

	for m in state:
		if s == 0:
			new_m = flipBit(m, 2*N - (2*i+1))    #flips 2*i-th bit in m
		elif s == 1:
			new_m = flipBit(m, 2*N - ((2*i)+2))    #flips 2*i+1-th bit in m

		if new_m < m:	#occupany in m is 1
			new_state[m] = state[m]

	return new_state		

def cr(state, i, s, N):
	"""Application of c_i,s^dag on a given state (a linear superposition of |m>). Spin should be 0 for up and 1 for down."""
	new_state = {}

	for basis_state in state:
		prefactor_cr, state_cr = fmulti_cr(basis_state, i, s, N)

		if state_cr != None:

			try:
				new_state[state_cr] += prefactor_cr * state[basis_state]
			except KeyError:
				new_state[state_cr] = prefactor_cr * state[basis_state]

	return new_state			

def an(state, i, s, N):
	"""Application of c_i,s on a given state (a linear superposition of |m>). Spin should be 0 for up and 1 for down."""
	new_state = {}

	for basis_state in state:
		prefactor_an, state_an = fmulti_an(basis_state, i, s, N)

		if state_an != None:

			try:
				new_state[state_an] += prefactor_an * state[basis_state]
			except KeyError:
				new_state[state_an] = prefactor_an * state[basis_state]

	return new_state

# UTILITY ######################################################################
# THESE FUNCTIONS ARE USED WHEN THE STATE IS REPRESENTED AS A DICTIONARY - THE KEY IS A STATE, ITS VALUE IS THE PROBABILITY AMPLITUDE.

def dict_sum(a, b):
	"""Calculates a sum of two states, represented by a dictionary."""
	res = a

	for key in b:
		try:
			res[key] += b[key]
		except KeyError:
			res[key] = b[key]
				
	return res

def dict_list_sum(list_of_dicts):
	"""Sums a list of dictionaries."""
	res = {}
	for dictionary in list_of_dicts:
		res=dict_sum(res, dictionary)

	return res

def dict_prod(num, dic):
	"""Multiply every value in the dictionary by a number. Used for multiplying numbers and states."""

	for i in dic:
		dic[i]*=num

	return dic

def scalar_prod(dicta, dictb):
	"""Calculates a scalar product between two states, given as Python dictionaries."""
	res=0
	for a in dicta:
		for b in dictb:
			if a==b:
				res+=dicta[a]*dictb[b]

	return res			


