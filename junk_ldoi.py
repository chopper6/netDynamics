import param, util, logic, net
from copy import deepcopy
import itertools, sys, os, pickle
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# TODO:
#	rm init nonsense
#	see notes under ldoi_bfs() &  ldoi_sizes_over_all_inputs()

#	figure out whne negated should use cp.any() and rm init part (curr if False'd out)
#	due to "rm_contras" and "memoize", negated can be wrong...
#		temp soln is to remove memoize
#	add an assert than regular and complement aren't both ON
#	make it bwds compat with directly calling ldoi or w.e.

# TODO:
# check pinning=False at end of ldoi_bfs()
#	worried the idea is not correct
# rm the 'test' function


def test(G,params,init=[]):
	#ldoi_solns, negated = ldoi_bfs(G,pinning=1,init=init)
	ldoi_solns = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=init)
	return ldoi_solns

def convert_solutions(G,ldoi_solns,negated):	

	# TODO: clean this
	if isinstance(G,net.ParityNet):
		n=G.n_neg 
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		n=G.n 
		n_compl = int(G.n/2)
	else:
		assert(0) # LDOI should be on a ParityNet or a DeepNet


	soln_dict, negated_dict = {},{}
	for i in range(len(ldoi_solns)):
		soln_dict[G.nodeNames[i]] = []
		for j in range(len(ldoi_solns[i])):
			if '+' not in G.nodeNames[i] and '&' not in G.nodeNames[i]: #ignore deep nodes
				if ldoi_solns[i,j]:
					if '+' not in G.nodeNames[j] and '&' not in G.nodeNames[j]: #ignore deep nodes
						soln_dict[G.nodeNames[i]] += [G.nodeNames[j]]
		
		if negated[i]:
			negated_dict[G.nodeNames[i]] = True
		else:
			negated_dict[G.nodeNames[i]] = False

		# below was used with negated as 2d matrix
		#negated_dict[G.nodeNames[i]] = []
		#for j in range(len(negated[i])):
		#	assert(j<=G.n_neg)
		#	if negated[i,j]:
		#		negated_dict[G.nodeNames[i]] += [G.nodeNames[j]] 
	return soln_dict, negated_dict

def shift_to_compl(a,n):
	# gets shifted version of array, while ignoring composites
	return cp.roll(a[:2*n],n,axis=1)

def ldoi_bfs(G,pinning=True,fixed=[],init=[]):
	# A should be adjacency matrix of Gexp
	# and the corresponding Vexp to A should be ordered such that:
	#	 Vexp[0:n] are normal nodes, [n:2n] are negative nodes, and [2n:N] are composite nodes

	# pinning means that sources cannot drive their complement (they are fixed to be ON)
	# without pinning the algorithm will run faster, and it is possible that ~A in LDOI(A)
	
	# TODO: wasting a lot of mem with deepcopy, esp for fixied/init initialization
	#	poss that negated is now wrong
	memoize = False # TODO...

	# TODO: clean this
	if isinstance(G,net.ParityNet):
		n=G.n_neg 
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		n=G.n 
		n_compl = int(G.n/2)
	else:
		assert(0) # LDOI should be on a ParityNet or a DeepNet

	N = G.n_exp
	A = G.A_exp
	A = cp.array(A, dtype=bool).copy()
	
	# Debugging:
	#Alist = A.astype(int).tolist()
	#print("\nLDOI.ldoi_bfs: Aexp=")
	#for row in Alist:
	#	print(row)

	X = cp.zeros((N,N),dtype=bool) # the current sent of nodes being considered
	visited = cp.zeros((N,N),dtype=bool) # if visited[i,j]=1, then j is in the LDOI of i
	negated = cp.zeros(N,dtype=bool) # if negated[i] then the complement of i is in the LDOI of i
	visited_negated = cp.zeros((N,N),dtype=bool) 
	D = cp.diag(cp.ones(N,dtype=bool))

	if len(fixed) > 0:
		fixed = cp.array(fixed)
		D_fixed = deepcopy(D[:n,:n])
		D_fixed[:,fixed] = 1 #assumes complements not in fixed
	if len(init) > 0:
		# TODO sep D for fixed/drivers vs init. Jp for fixed/drivers only just need compl version
		init = cp.array(init)
		D_init = deepcopy(D[:n,:n])
		D_init[:,init] = 1 #assumes complements not in fixed
	if len(fixed)+len(init) > 0:
		# assume drivers override fixed or init
		contra_fixed = D_fixed & shift_to_compl(D[:n,:n],G.n)
		contra_init = D_init & shift_to_compl(D[:n,:n],G.n)
		#print("\ncontra_fixed=\n",contra_fixed.astype(int),'\n\ncontra_init=\n',contra_init.astype(int))
		D_pin = deepcopy(D)
		D_pin[:n,:n] = D_pin[:n,:n] | (D_fixed & ~contra_fixed) 
		D[:n,:n] = D[:n,:n] | (D_fixed & ~contra_fixed) | (D_init & ~contra_init) 
	else:
		D_pin = D

	D[cp.arange(n,N)]=0 #don't care about source nodes for complement nodes
	D_pin[cp.arange(n,N)]=0
	D_compl = D.copy()
	D_compl[:n,:n] = shift_to_compl(D_compl[:n,:n],n_compl)
	D_pin_compl = D_pin.copy()
	D_pin_compl[:n,:n] = shift_to_compl(D_pin_compl[:n,:n],n_compl)

	#print("\nD=\n",D[:n,:n].astype(int),'\n\nD_compl=\n',D_compl[:n,:n].astype(int))

	max_count = max(cp.sum(A,axis=0))
	if max_count<128: 
		index_dtype = cp.int8
	else: #assuming indeg < 65536/2:
		index_dtype = cp.int16

	num_to_activate = cp.sum(A,axis=0,dtype=index_dtype)
	num_to_activate[:n] = 1 # non composite nodes are OR gates
	counts = cp.tile(num_to_activate,N).reshape(N,N) # num input nodes needed to activate
	zero = cp.zeros((N,N))

	X[D] = 1
	X = counts-cp.matmul(X.astype(index_dtype),A.astype(index_dtype))<=0 	
	#i.e. the descendants of X, including composite nodes iff X covers all their inputs
	
	if pinning:
		negated = negated | cp.any(X & D_compl,axis=1) 
		X = cp.logical_not(D_compl) & X  
	X_negated=X
	#print("\nX=\n",X.astype(int),"\n\nD=\n",D.astype(int), '\n\nDcompl=\n',D_compl.astype(int))


	# TODO: make this optional: add D to visited, and D_compl not
	visited = deepcopy(D_pin) 
	visited_negated = deepcopy(visited)
	loop=0
	while cp.any(X) or cp.any(X_negated): 
		#print("\nvisited at loop",loop,"=\n",visited[:n,:n].astype(int))
		visited = visited | X #note that D is included, i.e. source nodes
		visited_negated = visited_negated | X_negated

		# this line was prev not used and just memoized insead:
		if not memoize:
			X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) 
			X_negated = (counts-cp.matmul(visited_negated|D.astype(index_dtype),A.astype(index_dtype))<=0) 

		if pinning:
			if memoize:
				assert(0) # TODO: reimplement
				visited_rolled = visited.copy()
				visited_rolled[:n] = cp.roll(visited_rolled[:n],n_compl,axis=0)
				memoX = cp.matmul((visited|D) & cp.logical_not(visited_rolled.T), visited) 
				# if A has visited B, then add B's visited to A if ~A not in B's visited
				# otherwise must cut all of B's contribution!
				X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) | memoX
			
			X_negated = ~visited_negated & ~D_pin_compl & X_negated # still excluded negated pinned nodes
			negated = negated | cp.any(X & D_compl,axis=1) # TODO update this to handle inits 
			#print("\nX before\n",X.astype(int))
			X = ~visited & ~D_pin_compl & X 
			
		else:
			assert(0) # TODO: reimplement
			if memoize:
				memoX = cp.matmul(visited|D, visited)
				X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) | memoX
			X = ~visited & X

		# debugging:
		loop+=1
		if loop>N:
			print("WARNING: N steps exceeded")
		if loop > N**4:
			sys.exit("ERROR: infinite loop in ldoi_bfs!")

	if not pinning:
		negated = cp.any(visited & D_compl,axis=1) #TODO: check it
	#if subset != []:
	#	visited[:n,:n] = augment_solutions(visited[:n,:n], negated[:n], G, subset,D[:n,:n])

	#print('\n\nldoi visited\n',visited[:n,:n].astype(int),'\n\nnegated\n',visited_negated[:n,:n].astype(int))
	visited[:n,:n] = visited[:n,:n] & ~shift_to_compl(visited_negated[:n,:n],n_compl)
	#print('ldoi visited\n',visited[:n,:n].astype(int))
	#assert(0)

	return visited[:n,:n], negated[:n] 


def compl_index(G, indx):
	assert(not  isinstance(G,net.DeepNet)) # have to match w parity net later
	return (indx+G.n) % G.n_neg

def ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[],subsets={}):
	# TODO: rename subset to init & clean up generally
	#	unlike ldoi_bfs() subset should be indexed by input_set
	# curr: fixed_nodes should be list of ints corresp to the state in the expanded net that is pinned 
	#	can also use as a dict, where indices ae input_sets, but has to match itertool form used here
	# old: fixed_nodes should be a list of names such as ['FOXO3','ERK']

	#print('ldoi_sizes_over_all_inputs() received fixed nodes=',fixed_nodes,'\n\tand subsets=',subsets)
	if isinstance(G,net.ParityNet):
		n=G.n_neg 
		n_compl = G.n 
	elif isinstance(G,net.DeepNet):
		n=G.n 
		n_compl = int(G.n/2)
	else:
		assert(0) # LDOI should be on a ParityNet or a DeepNet

	all_solns, all_negated = {},{}

	k = len(params['inputs'])
	input_indices = [G.nodeNums[params['inputs'][i]] for i in range(k)]
	input_sets = itertools.product([0,1],repeat=k)
	for input_set in input_sets:
		list_of_inits = False # TODO: clean this up
		if isinstance(fixed_nodes,dict):
			if isinstance(fixed_nodes[input_set],list):
				assert(0) # del soon
				list_of_inits = True
			ldoi_inpts = fixed_nodes[input_set] # TODO: prev had copy, if not nec match w else below
		else:
			ldoi_inpts = fixed_nodes.copy() 
		
		for i in range(len(input_set)):
			if input_set[i]==1:
				inpt_val = input_indices[i]
			else:
				inpt_val = input_indices[i] + n_compl #i.e. its complement
			
			# TODO: clean this lists_of_lists shit
			if list_of_inits:
				for i in range(len(ldoi_inpts)):
					ldoi_inpts[i] = cp.append(ldoi_inpts[i],inpt_val)
			else:
				if inpt_val not in ldoi_inpts and compl_index(G,inpt_val) not in ldoi_inpts:
					# TODO: btw should add fn for getting copml
					ldoi_inpts = cp.append(ldoi_inpts,inpt_val)
		
		# jp this is rdn
		#if list_of_inits:
		#	for i in range(len(ldoi_inpts)):				
		#		ldoi_inpts[i] = cp.unique(ldoi_inpts[i])
		#else:
		#	ldoi_inpts = cp.append(ldoi_inpts,inpt_val)

		if subsets != {}:
			subset_init=subsets[input_set]
			subset=[]
			for ele in subset_init:
				if not (ele in ldoi_inpts or compl_index(G,ele) in ldoi_inpts):
					subset += [ele]
		else:
			subset = []
		ldoi_solns, negated = ldoi_bfs(G,pinning=1,fixed=ldoi_inpts,init=subset)
		if CUPY:
			ldoi_solns = ldoi_solns.get() #explicitly cast out of cupy
			negated = negated.get()
		
		ldoi_solns, negated = convert_solutions(G, ldoi_solns, negated)
		
		# merge diff input sets into one soln set
		all_solns[input_set] = {}
		all_negated[input_set] = {}
		for k in ldoi_solns.keys():
			all_solns[input_set][k] = ldoi_solns[k]
			all_negated[input_set][k] = negated[k]
	return all_solns, all_negated


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



def augment_solutions(ldoi_solns, negated, G, subset, diag):
	# merges all solutions with the solutions in subset
	#	unless a solution in subset contradicts
	# makes ldoi_solns, negated as orig NxN matrices, incld composite nodes
	# subset should be the node indices in Gexp
	# LATER: add composite node merging (not that curr only pass visited[:2n,:2n])

	# TODO: unlike visited subset solns should always include self when checking for compl (since assumes that self was present)
	#		also looks like init incorrectly rmd

	# "diag" has the initially pinned values and is added to solution (since pinned)
	#	for ex the driving node (hence diag) and mutation (well this isn't diag)
	ldoi_solns = ldoi_solns | diag

	N = len(ldoi_solns) 

	subset = cp.array(subset)
	subset_solns = ldoi_solns[subset]
	#print("\n\n~~~~Debugging merge_solutions()~~~~\n")
	#print("subset=\n",subset.astype(int),"\n\nvisited=\n",ldoi_solns.astype(int),"\n\nsubset_solns=\n",subset_solns.astype(int))
	#print("\ndtype=",ldoi_solns.dtype,"and",subset_solns.dtype)

	# check dtype of subset_solns
	ldoi_expd = cp.tile(ldoi_solns,len(subset)).reshape(N*len(subset),N)
	#print("shifted subset solns=\n",shift_to_compl(subset_solns,G.n).astype(int),"\n\nldoi_expd=\n",ldoi_expd.astype(int))
	contras = (ldoi_expd & cp.tile(shift_to_compl(subset_solns,G.n),N).reshape(N*len(subset),N))

	contras = cp.any(contras,axis=1) # again, check axis	
	#print("contras=\n",contras.astype(int),"\n\nldoi_expd=\n",ldoi_expd.astype(int))
	subset_solns_expd = cp.vstack([subset_solns]*N)
	#print("subset_solns_expd=\n",subset_solns_expd.astype(int))
	#print('here',subset_solns_expd.shape,(subset_merge).shape)
	subset_solns_expd[contras] = 0
	ldoi_expd = ldoi_expd | subset_solns_expd
	#print("subset_solns_expd2=\n",subset_solns_expd.astype(int))
	#print("ldoi_expd reshaped=\n",ldoi_expd.reshape((N,len(subset),N)).astype(int))
	
	#row_merge = cp.tile(cp.array([i for i in range(N)]),len(subset)).reshape(1,N*len(subset))
	ldoi_soln = cp.any(ldoi_expd.reshape((N,len(subset),N)),axis=1)
	#print("FINAL SOLN=\n",ldoi_soln.astype(int))
	return ldoi_soln 



if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 ldoi.py PARAMS.yaml")

	DEEP=False
	#result = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[])

	if DEEP:
		with open(sys.argv[1],'rb') as f:
			G, params = pickle.load(f)
	elif 0:
		params = param.load(sys.argv[1])
		G = net.ParityNet(params['parity_model_file'],debug=params['debug'])
		for k in params['mutations'].keys():
			mutant = G.nodeNums[k]
			if params['mutations'][k]==0:
				mutant+=G.n	
		visited, negated = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[mutant])
		k='GAB1'
		for inpt_str in visited:
			print("\nINPUT=",inpt_str)
			print('visited[',k,'] :',visited[inpt_str][k])

	else:
		params = param.load(sys.argv[1])
		G = net.ParityNet(params['parity_model_file'],debug=params['debug'])
		init = get_const_node_inits(G,params)
		visited, negated = ldoi_sizes_over_all_inputs(params,G,fixed_nodes=init)
		print("LDOI RESULT = \n",visited)