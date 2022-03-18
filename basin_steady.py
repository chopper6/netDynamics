import basin 
import numpy as np


def calc_size(params, G, SS0=None):
	# calculates the steady basin by updating the inputs for at most num_steps
	# note that params['parallelism'] and ['num_samples'] will be altered during calc_basin_size()
	# pass A0 for mutated or control runs (for example)

	SS = basin.calc_size(params,G,SS0=SS0) # get steady states from random initial conditions
	
	new=True 
	while new:
		SS.shuffle_A0_inputs(params,G)
		SS = basin.calc_size(params, G, SS0=SS) # get steadyStates from input-shuffled attractors of previous steadyStates
		if params['verbose']:
			print('#As =',len(SS.attractors))
		transit_pr, new = SS.map_A_transition_pr() 
		if new:
			# so need a SS_new that will be sim'd
			# and a SS_old that will merge w SS_new, and renormz, then SS.map_A_transition_pr()
			assert(0) # add a function for this
			A0 += [SS.attractors[k].id for k in SS.attractor_order]
			A0 = list(set(A0)) #remove duplicates
			Aweights_new = {A.id:A.size for A in SS.attractors.values()}
			Aweights = {**Aweights_new, **Aweights}
			if params['verbose']: 
				print('\tFound new attractors, rerunning with',len(A0),' A0s.')

	if params['verbose']:
		print('Finished building transition matrix, calculating steady basin.')
	steadyBasin, eigen_vec = calc_markov_chain_SS(params, transit_pr, SS)
	if params['debug']:
		steadyBasin_multpn, basin_vec = steadyBasin_from_transition_matrix(params, SS, transit_pr, 20) 
		assert(np.allclose(eigen_vec,basin_vec))
		#print('multiplication vs eigen:\n',np.round(basin_vec,6),'vs\n', np.round(eigen_vec,6))

	return steadyBasin 


def calc_markov_chain_SS(params, transit_pr, SS):
	eigenvals, eigenvecs = np.linalg.eig(transit_pr)
	if np.sum(eigenvals == 1) > 1:
		print("WARNING: multiple eigenvalues=1 detected")#. Eigenvecs where eigenval=1:\n",eigenvecs[eigenvals==1])
	if np.sum(eigenvals > 1.1) > 0:
		print("\n\n\nERROERRRERRERERRORERRROR: an eigenvalue > 1 detected! Eigenvalues:",eigenvals,'\n\n\n')
	if np.sum(eigenvals == -1) > 0:
		print("\n\nWARNING: periodic steady state detected! eigenvalues:",eigenvals,'\n\n\n')
	SS_eigenvec = eigenvecs[:,np.argmax(eigenvals)] # note that eigen vecs are the columns, not the rows
	SS_eigenvec = SS_eigenvec/np.sum(SS_eigenvec)
	eigen_steadyBasin = basin_matrix_to_SS(params, SS, SS_eigenvec)
	return eigen_steadyBasin, SS_eigenvec


def steadyBasin_from_transition_matrix(params, SS, T, num_steps):
	B = np.array([SS.attractors[k].size for k in SS.attractor_order]) # starting basin size
	for i in range(num_steps):
		if params['verbose']:
			print('at step',i)
		#print('starting from B=',B,'\nT=',T)
		B_next = np.matmul(T,B)
		#print('BxT = ',B_next)
		if params['update_rule']=='sync' and params['debug']: # due to attractor trimming, async methods may not actually sum to 1
			assert(np.isclose(sum(B_next),1))
		if np.allclose(B_next,B):
			if params['verbose']:
				print("\nSteady basin fixed point found.")
			return basin_matrix_to_SS(params, SS, B), B
		B = B_next 

	print("\n\nWARNING: Steady basin fixed point NOT found, returning basin after",num_steps,"steps.\n")
	return basin_matrix_to_SS(params, SS, B), B

def basin_matrix_to_SS(params, SS, B):
	for k in SS.attractors.keys():
		A = SS.attractors[k]
		A.size = B[SS.attractor_order.index(k)]
	return SS

def trim_As(params,SS,thresh=.001):
	assert(0) # haven't look at this in awhile
	# also should renormalizes sizes such that sum=1
	rm_keys = []
	for A in SS.attractors.values():
		if A.size < thresh:
			rm_keys += [A.id] 
	for k in rm_keys:
		del SS.attractors[k]
	if params['map_from_A0']:
		for A in SS.attractors.values():
			rm_A0s = []
			for A0 in A.A0s:
				if A0 in rm_keys:
					rm_A0s += [A0]
			for A0 in rm_A0s:
				del A.A0s[A0]
	SS.order_attractors()
	print('after trim, #As=',len(SS.attractors))