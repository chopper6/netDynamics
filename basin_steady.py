import basin 
import numpy as np
from copy import deepcopy

# TODO: jp can rm ordered attractors stuff in SS class
# TODO: make As a dict instead of a list

def calc_size(params, G, SS0=None):
	# calculates the steady basin by updating the inputs for at most num_steps
	# note that params['parallelism'] and ['num_samples'] will be altered during calc_basin_size()
	# pass A0 for mutated or control runs (for example)

	SS = basin.calc_size(params,G,SS0=SS0) # get steady states from random initial conditions
	
	exact_x0 = True # means rerun with all attractors, rather than just the new ravine attractors
	# running with exact x0 is much slower and doesn't seem to matter in practice
	# issue w/ exact_x0 = False is that a new A may end in an old A, but since old A not in that run (to save time), it is not recorded as a known A
	#	so T[i,j] += As[Aj].A0s[Ai] does not occur where it should
	# either change how As are recorded or save for another time
	# -> need to merge As[Aj].A0s of new & old As

	new, As,A_labels =True, {},[]
	while new:
		SS.shuffle_A0_inputs(params,G)
		SS = basin.calc_size(params, G, SS0=SS) # get steadyStates from input-shuffled attractors of previous steadyStates
		if params['verbose']:
			print('#As =',len(SS.attractors))
		if exact_x0:
			As, A_labels = SS.attractors, [A.id for A in SS.attractors.values()]
			new_As = {k:SS.attractors[k] for k in SS.attractors if k not in A_labels}
		else:
			A_labels_prev = deepcopy(A_labels)
			new_As = {k:SS.attractors[k] for k in SS.attractors if k not in A_labels}
			merge_A0s(As,new_As)
			A_labels += [k for k in SS.attractors.keys() if k not in A_labels]
			#print("\n\nAs_new=\n",As_new.keys(),'\n\nAs=\n',As.keys())

		transit_pr, new_As = map_A_transition_pr(As, A_labels)
		new = len(new_As)>0 
		if new:
			if not exact_x0:
				to_del = []
				for k in SS.attractors:
					if k not in new_As:
						to_del += [k]
				for k in to_del:
					del SS.attractors[k]
			if params['verbose']: 
				print('\tFound new attractors, rerunning with',len(SS.attractors),' A0s.')

	if params['verbose']:
		print('Finished building transition matrix, calculating steady basin.')
	steadyBasin, eigen_vec = calc_markov_chain_SS(params, transit_pr, SS, A_labels)
	if params['debug']:
		steadyBasin_multpn, basin_vec = steadyBasin_from_transition_matrix(params, SS, transit_pr, 20, A_labels) 
		assert(np.allclose(eigen_vec,basin_vec))
		#print('multiplication vs eigen:\n',np.round(basin_vec,6),'vs\n', np.round(eigen_vec,6))

	return steadyBasin 


def map_A_transition_pr(As, A_labels):
	# build transition matrix (a_ij = pr Ai -> Aj)
	# prev_SSs if had to rerun the sim since new A's were found
	T = np.zeros((len(As),len(As)))
	assert(len(As)==len(A_labels))
	new_As = [] # if find any new As need to run sim again
	for i in range(len(As)):
		Ai=A_labels[i]
		found=0
		for j in range(len(As)):
			Aj=A_labels[j]
			if Ai in As[Aj].A0s.keys():
				T[i,j] += As[Aj].A0s[Ai] # Aj's A0s[Ai] should be the # times Ai -> Aj
				found=1
		if not found: # A0 did not contain all attractors, so try again before mapping transition pr
			new_As += [Ai]

	T = np.transpose(T/np.vstack(np.sum(T,axis=1))) # normalize such that sum_j(pr Ai -> Aj) = 1
	return T, new_As


def merge_A0s(As,Anew):
	for k in Anew.keys():
		if k in As.keys():
			for k2 in Anew[k].A0s.keys():
				if k2 in As[k].A0s.keys():
					assert(0) # jp need to increment up 1 but not sure
				else:
					As[k].A0s[k2] = As[k].A0s[k2]
		else:
			As[k] = Anew[k]



def calc_markov_chain_SS(params, transit_pr, SS, A_labels):
	eigenvals, eigenvecs = np.linalg.eig(transit_pr)
	if np.sum(eigenvals == 1) > 1:
		print("WARNING: multiple eigenvalues=1 detected")#. Eigenvecs where eigenval=1:\n",eigenvecs[eigenvals==1])
	if np.sum(eigenvals > 1.1) > 0:
		print("\n\n\nERROERRRERRERERRORERRROR: an eigenvalue > 1 detected! Eigenvalues:",eigenvals,'\n\n\n')
	if np.sum(eigenvals == -1) > 0:
		print("\n\nWARNING: periodic steady state detected! eigenvalues:",eigenvals,'\n\n\n')
	SS_eigenvec = eigenvecs[:,np.argmax(eigenvals)] # note that eigen vecs are the columns, not the rows
	SS_eigenvec = SS_eigenvec/np.sum(SS_eigenvec)
	eigen_steadyBasin = basin_matrix_to_SS(params, SS, SS_eigenvec, A_labels)
	return eigen_steadyBasin, SS_eigenvec


def steadyBasin_from_transition_matrix(params, SS, T, num_steps, A_labels):
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
			return basin_matrix_to_SS(params, SS, B, A_labels), B
		B = B_next 

	print("\n\nWARNING: Steady basin fixed point NOT found, returning basin after",num_steps,"steps.\n")
	return basin_matrix_to_SS(params, SS, B, A_labels), B

def basin_matrix_to_SS(params, SS, B, A_labels):
	for i in range(len(A_labels)):
		A = SS.attractors[A_labels[i]]
		A.size = B[i]
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