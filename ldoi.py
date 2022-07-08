import param, util, logic, net
from copy import deepcopy
import itertools, sys, os, pickle
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# TODO:
# check pinning=False at end of ldoi_bfs()
#	worried the idea is not correct
# rm the 'test' function

FLIP_LMT = 2

def test(G,params,init=[],goi=[]):

	init += get_const_node_inits(G,params)
	#ldoi_solns, negated = ldoi_bfs(G,pinning=1,init=init)
	ldoi_solns = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=init)
	
	if 0:
		print('\nLDOI=')
		for g in goi:
			for k in ldoi_solns:
				if g in k:
					print('LDOI[',k,']=')
					for g2 in ldoi_solns[k]:
						if g2 in params['outputs']:
							print('\t',g2)
			print("\n")
	return ldoi_solns

def convert_solutions(G,ldoi_solns):
	soln_dict = {}
	for i in range(len(ldoi_solns)):
		soln_names = ''
		for j in range(len(ldoi_solns[i])):
			if '+' not in G.nodeNames[i] and '&' not in G.nodeNames[i]: #ignore deep nodes
				if ldoi_solns[i,j]:
					if '+' not in G.nodeNames[j] and '&' not in G.nodeNames[j]: #ignore deep nodes
						soln_names += G.nodeNames[j] + ', '
		if soln_names != '' :
			#print("LDOI(",G.nodeNames[i],') =',soln_names)
			soln_dict[G.nodeNames[i]] = soln_names
		#if negated[i]:
		#	print('\t',G.nodeNames[i],'negates itself')
	return soln_dict


def ldoi_bfs(G, A0=None, pins=None):
	# X is the main array that tracks visited for traditional LDOI
	# corresponding Vexp to G.A_exp should be ordered such that:
	#	 Vexp[0:n] are normal nodes, [n:2n] are negative nodes, and [2n:N] are composite nodes
	
	# optionally pass an attractor A0 that should already be pre-processed such that oscillating nodes have a value of 2

	if pins != None and len(pins)>0:
		assert(A0 is not None) # have not done this yet
		pinned_nodes = cp.array([G.nodeNums[k] for k in pins.keys()] + [G.nodeNums[k]+G.n for k in pins.keys()])
		pinned_vals = cp.array([k for k in pins.values()] + [1-k for k in pins.values()])

	if isinstance(G,net.ParityNet): # TODO: clean this
		n=G.n_neg          # this is confusing af
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		assert(0) # haven't used this is a long time
		n=G.n 
		n_compl = int(G.n/2)
	else:
		assert(0) # LDOI should be on a ParityNet or a DeepNet

	N = G.n_exp
	A = cp.array(G.A_exp, dtype=bool).copy()

	if A0 is not None:
		assert(len(A0)==G.n_exp)  # as in the lng of the regular + parity + expanded nodes
		# A0=build_A0(G,A0)  # should already pre-process this
		#print("transformed A0=",A0)
		X = cp.array([A0 for _ in range(N)],dtype=cp.int8)
		changed = cp.zeros((N,N),dtype=cp.uint8) # vals in [0,1,2]
	else:
		X = cp.zeros((N,N),dtype=bool) # the current sent of nodes being considered
	
	negated = cp.zeros(N,dtype=bool) # if negated[i] then the complement of i is in the LDOI of i
	D = cp.diag(cp.ones(N,dtype=bool))
	D[cp.arange(n,N)]=0 #don't care about source/pinned nodes for complement nodes
	D_compl = D.copy()
	D_compl[:n] = cp.roll(D_compl[:n],n_compl,axis=0)

	max_count = max(cp.sum(A,axis=0))
	if max_count<128: 
		index_dtype = cp.int8
	else: #assuming indeg < 65536/2:
		index_dtype = cp.int16

	num_to_activate = cp.sum(A,axis=0,dtype=index_dtype)
	num_to_activate[:n] = 1 # non composite nodes are OR gates
	counts = cp.tile(num_to_activate,N).reshape(N,N) # num input nodes needed to activate
	zero = cp.zeros((N,N))


	if pins is not None and len(pins) > 0:
		X[:,pinned_nodes] = pinned_vals

	# note that controllers are prioritized over pinned nodes
	X[D] = 1 # cp.fill_diagonal would also work
	if A0 is None:
		#i.e. the descendants of X, including composite nodes iff X covers all their inputs
		X = counts-cp.matmul(X.astype(index_dtype),A.astype(index_dtype))<=0 
		negated = negated | cp.any(X & D_compl,axis=1) 
		X = (~D_compl) & X  
	else:
		X[D_compl] = 0
		#X = init_step_A0(X, changed, index_dtype, counts, A, D, D_compl, n)

	loop,cont=0,True
	while cont: 
		#outpts = cp.array([6,22,42,35])
		#print("\nldoiStep:",X[52,outpts])

		if A0 is None:
			X, negated, cont = step(X, A, D, D_compl, counts, negated, n, n_compl, index_dtype)
		else:
			X, changed, cont = step_A0(G, X, A, D, D_compl, counts, negated, changed, n, n_compl, index_dtype, pins=pins)
			
		loop = loop_debug(N, loop)
	
	if A0 is not None: # can rm later
		debug_A0(X[:n,:n], n, n_compl)
	return X[:n,:n], negated[:n]


def step(X, A, D, D_compl, counts, negated, n, n_compl, index_dtype): 
	# assumes all matrices are boolean (except for counts)!
	for matrix in [X, A, D, D_compl ,negated]: # can rm after a lil
		assert(matrix.dtype==bool)
	
	if 0:	# memoization, currently off to be safe
		X_rolled = X.copy()
		X_rolled[:n] = cp.roll(X_rolled[:n],n_compl,axis=0)
		memoX = cp.matmul((X|D) & cp.logical_not(X_rolled.T), X) 
		# if A has visited B, then add B's visited to A if ~A not in B's visited
		# otherwise must cut all of B's contribution!
	X_next = (counts-cp.matmul(X|D.astype(index_dtype),A.astype(index_dtype))<=0) #| memoX
	
	negated = negated | cp.any(X_next & D_compl,axis=1)
	X_next = cp.logical_not(D_compl) & X_next
	cont = ~ cp.all(X_next == X) 

	return X_next, negated, cont

def loop_debug(N, loop):
	#if loop>N*FLIP_LMT:
	#	print("WARNING: N steps exceeded in LDOI")
	if loop > N**4:
		sys.exit("ERROR: infinite loop in ldoi_bfs!")
	return loop+1

def step_A0(G, X, A, D, D_compl, counts, negated, changed, n, n_compl, index_dtype, pins=None):
	X, changed = X.copy(), changed.copy()
	# quick and dirty way to only update composite, then only regular
	
	x_next_compl, changed_compl, cont_compl = step_A0_half(G, X, A, D, D_compl, counts, negated, changed, n, n_compl, index_dtype,composites=True, pins=pins)
	X[:,n:], changed[:,n:] = x_next_compl[:,n:], changed_compl[:,n:]

	x_next_reg, changed_reg, cont_reg = step_A0_half(G, X, A, D, D_compl, counts, negated, changed, n, n_compl, index_dtype, pins=pins)
	X[:,:n], changed[:,:n] = x_next_reg[:,:n], changed_reg[:,:n]

	#print('s',cp.max(changed_compl), cp.any(cont_compl), cp.max(changed_reg), cp.any(cont_reg))
	cont = cont_reg | cont_compl 
	return X, changed, cont


def step_A0_half(G, X, A, D, D_compl, counts, negated, changed, n, n_compl, index_dtype,composites=False, pins = None):
	if pins != None and len(pins)>0:
		pinned_nodes = cp.array([G.nodeNums[k] for k in pins.keys()] + [G.nodeNums[k]+G.n for k in pins.keys()])
		pinned_vals = cp.array([k for k in pins.values()] + [1-k for k in pins.values()])

	changed=changed.copy()
	X=X.copy() # prob not nec

	xON = X.copy()
	xON[xON==2]=0
	xON = step(xON.astype(bool), A, D, D_compl, counts, negated, n, n_compl, index_dtype)[0] 
	# return step()[0] since only care about the X, not "negated" or "cont"
	xOFF = X.copy()
	xOFF[xOFF==2]=1
	xOFF = step(xOFF.astype(bool), A, D, D_compl, counts, negated, n, n_compl, index_dtype)[0]

	x_next = X.copy() 
	x_next[xON==1]=1
	x_next[xOFF==0]=0  	
	x_next[(~xON) & xOFF]=2

	if pins is not None and len(pins)>0:
		x_next[:,pinned_nodes] = pinned_vals


	# note that controllers are prioritized over pinned nodes	
	x_next[D]=1  
	x_next[D_compl]=0

	if not composites:
		x_next[(x_next!=X) & (changed>=FLIP_LMT)] = 2
		changed[x_next!=X]+=1
		cont = 1-cp.all(x_next[:n,:n]==X[:n,:n])
	else:
		cont = 1-cp.all(x_next[n:,n:]==X[n:,n:])

	assert(cp.all(xOFF | ~xON))
	
	return x_next, changed, cont


def debug_A0(x, n, n_compl):
	#debug, can rm later
	# x should only be the regular nodes (not composites)
	indx2 = cp.argwhere(x==2)
	indx2_compl = indx2.copy() 
	indx2_compl[:,1]= (indx2_compl[:,1]+ n_compl)%n 
	indx1 = cp.argwhere(x==1)
	indx1_compl = indx1.copy() 
	indx1_compl[:,1]= (indx1_compl[:,1]+ n_compl)%n 
	indx0 = cp.argwhere(x==0)
	indx0_compl = indx0.copy() 
	indx0_compl[:,1]= (indx0_compl[:,1]+ n_compl)%n 
	indx2,indx2_compl,indx1,indx1_compl,indx0,indx0_compl = indx2.tolist(),indx2_compl.tolist(),indx1.tolist(),indx1_compl.tolist(),indx0.tolist(),indx0_compl.tolist()
	indx2.sort()
	indx2_compl.sort()
	indx1.sort()
	indx1_compl.sort()
	indx0.sort()
	indx0_compl.sort()
	assert(indx0!=indx1)
	assert(indx0_compl!=indx1_compl)
	assert(indx1!=indx1_compl)
	assert(indx0!=indx0_compl)
	assert(indx2==indx2_compl) # i.e. all rows in indx2 are in indx2_compl
	assert(indx0==indx1_compl)
	assert(indx1==indx0_compl)


def build_A0(G,A0_avg, composites=True):
	assert(isinstance(G,net.ParityNet))
	A0 = cp.ones(A0_avg.shape,dtype=cp.int8)*2 # only need 0,1,2, but int8 is smallest avail

	thresh = .1
	A0[A0_avg < thresh] = 0
	A0[A0_avg > 1-thresh] = 1
	#A0[cp.isclose(A0_avg,0)] = 0 
	#A0[cp.isclose(A0_avg,1)] = 1

	A0not = cp.ones(A0_avg.shape,dtype=cp.int8)*2 
	A0not[A0==1]=0 
	A0not[A0==0]=1

	if composites:
		A0exp = cp.zeros(G.n_exp-2*G.n)
		assert(G.n_exp-2*G.n>=0)
		A0 = cp.hstack([A0,A0not,A0exp])
	else:
		A0 = cp.hstack([A0,A0not])
	return A0 


def ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[]):
	# fixed_nodes should be a list of names such as ['FOXO3','ERK']
	if isinstance(G,net.ParityNet):
		n=G.n_neg 
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		n=G.n 
		n_compl = int(G.n/2)
	else:
		print("ERROR: net is type",type(G), ", instead of ParityNet or DeepNet")
		assert(0) # LDOI should be on a ParityNet or a DeepNet

	all_solns = {k:{} for k in G.nodeNames} #should incld compl names too
	#output_indices = [G.nodeNums[params['outputs'][i]] for i in range(len(params['outputs']))]

	ldoi_fixed = fixed_nodes # TODO clean
	#for pair in fixed_nodes: # this used to be pairs {'name':0} for example
		#indx = G.nodeNums[pair[0]] 
		#if pair[1]==0:
		#	indx += n # i.e. the node's complement
		#ldoi_fixed += [indx]

	k = len(params['inputs'])
	input_indices = [G.nodeNums[params['inputs'][i]] for i in range(k)]
	input_sets = itertools.product([0,1],repeat=k)
	for input_set in input_sets:
		ldoi_inpts = ldoi_fixed.copy()
		for i in range(len(input_set)):
			if input_set[i]==1:
				ldoi_inpts += [input_indices[i]]
			else:
				ldoi_inpts += [input_indices[i] + n_compl] #i.e. its complement

		ldoi_solns, negated = ldoi_bfs(G,pinning=1,init=ldoi_inpts)
		if CUPY:
			ldoi_solns = ldoi_solns.get() #explicitly cast out of cupy
		
		ldoi_solns = convert_solutions(G,ldoi_solns)
		
		# merge diff input sets into one soln set
		input_str = str(input_set)
		for k in ldoi_solns.keys():
			all_solns[k][input_str] = ldoi_solns[k]
	return all_solns

def cut_from_ldoi_over_inputs():
		#avg_sum_ldoi,avg_sum_ldoi_outputs = 0,0
	#avg_num_ldoi_nodes = {k:0 for k in range(n)}
	#avg_num_ldoi_outputs = {k:0 for k in range(n)}

	# .....

	# input loop:
		#avg_sum_ldoi += cp.sum(ldoi_solns)/((n)**2) #normz by max sum poss

		#for i in range(n):
		#	avg_num_ldoi_nodes[i] += cp.sum(ldoi_solns[i])/(n)
		#	for o in output_indices:
		#		if ldoi_solns[i,o]:
		#			avg_num_ldoi_outputs[i] += 1 
		#			avg_sum_ldoi_outputs += 1				
		#		if ldoi_solns[i,o+n_compl]:
		#			avg_num_ldoi_outputs[i] += 1 
		#			avg_sum_ldoi_outputs += 1
		#	avg_num_ldoi_outputs[i] /= len(output_indices)
		#avg_sum_ldoi_outputs /= (len(output_indices)*n)

	#avg_sum_ldoi /= 2**k
	#avg_sum_ldoi_outputs /= 2**k
	#for i in range(n):
	#	avg_num_ldoi_nodes[i] /= 2**k # normz
	#	avg_num_ldoi_outputs[i] /= 2**k # normz

	return {'total':avg_sum_ldoi,'total_onlyOuts':avg_sum_ldoi_outputs, 'node':avg_num_ldoi_nodes,'node_onlyOuts':avg_num_ldoi_outputs}

def get_const_node_inits(G,params):
	#TODO: clean all this nonsense of distinguishing Deep vs ParityNet (in general need to standardize more)
	if isinstance(G,net.ParityNet):
		n=G.n_neg 
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		n=G.n 
		n_compl = int(G.n/2)
	else:
		assert(0) # LDOI should be on a ParityNet or a DeepNet

	init = []
	G.add_self_loops(params) # just in case (TODO clean)
	for nodeName in params['init']:
		if params['init'][nodeName] == 1:
			init += [G.nodeNums[nodeName]]
		elif params['init'][nodeName] == 0:
			init += [(n_compl + G.nodeNums[nodeName]) % n]
		else:
			print("\nERROR: unrecognized value for params['init'][",nodeName,"]:",params['init'][nodeName])
			assert(0) 
	init = list(set(init))
	return init




if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 ldoi.py PARAMS.yaml")

	DEEP=False

	#result = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[])

	if DEEP:
		with open(sys.argv[1],'rb') as f:
			G, params = pickle.load(f)
	else:
		params = param.load(sys.argv[1])
		G = net.ParityNet(params['parity_model_file'],debug=params['debug'])		
	
	visited, negated = ldoi_bfs(G)
	print(visited.astype(int))
	#test(G,params,  init=init,goi=['not p21'])