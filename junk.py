# these are pieces of code that are no longer used, but may be useful later

# old way to build Parity_Net that allows for dynamic parity network construction
# 	but in general horrible idea, since should build parity once and write to file (since build v slow)
class Parity_Net(Net):
	def __init__(self,params,G=None,net_key='model_parity',debug=False):
		negatives = False
		if G is not None:
			print("\nWARNING: building parity network dynamically (not advised unless network is small).\n")
			negatives = True

		super().__init__(net_key=net_key,G=G,negatives=negatives,debug=debug)
		# assumes that negative nodes are already in parity form (i.e. have their logic)
		# note that composite nodes aren't formally added, since only A_exp is actually used

		if G is not None: # instead build from a regular net, so add composite nodes, ect
			G.prepare_for_sim(params) #not best name since this isn't sim... (uses Aexp)
			self.n_exp = self.n_neg # will add this up during build_Aexp()
			# build F for negative nodes
			logic.DNF_via_QuineMcCluskey(params, G,parity=True)
			self.build_Aexp(params)


# exhaustive is not really used, update as needed
	if params['exhaustive']:
		assert(0) #need to update this
		num_samples = (2**(num_nodes-1))
		if params['parallelism'] > num_samples:
			sys.exit("ERROR: parallelism parameter must be < number of samples for exhaustive search (2^n)")
		X0 = itertools.product([0,1], repeat=num_nodes-1) #first index is the always OFF node
		for i in range(int(math.floor((2**(num_nodes-1))/params['parallelism']))):
			x0 = cp.array(list(itertools.islice(X0, params['parallelism'])),dtype=bool)
			x0 = cp.insert(x0, 0, 0, axis=1)
			result  = lap.fixed_point_search(params,x0, clause_mapping, num_nodes)
			add_to_attractors(params, attractors, result)
		for i in range(2**(num_nodes-1) % params['parallelism']): # do the remaining samples sequentially
			x0 = cp.array(list(itertools.islice(X0, 1)),dtype=bool)
			x0 = cp.insert(x0, 0, 0, axis=1)
			result = lap.from_init_val(params,x0, clause_mapping, num_nodes)
			add_to_attractors(params, attractors, result)
		for s in attractors.keys():
			attractors[s]['size'] /= num_samples


# old basin sync 
if params['update_rule'] == 'sync': # later merge and clean these out
		# FIXED POINTS
		if params['verbose']:
			print("Starting fixed point search, using", actual_num_samples, "sample initial points.")
		for i in range(int(actual_num_samples/params['parallelism'])):
			p = .5 #prob a given node is off at start
			x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)
			x0[:,0] = 0 #0th node is the always OFF node
			
			if params['use_phenos']:
				if 'init' in params['phenos'].keys():
					for k in params['phenos']['init']:
						node_indx = node_name_to_num[k]
						x0[:,node_indx] = params['phenos']['init'][k]
				
			if params['verbose'] and i%params['print_lap']==0 and i!=0:
				print("\tAt lap",i)


			result = lap.fixed_point_search(params, x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			cupy_to_numpy(params,result)

			add_to_attractors(params, attractors, result)
			oscil_bin += list(result['state'][result['finished']==False])

		if params['verbose'] and len(oscil_bin) > 0: 
			print('Finished initial run, now finding',len(oscil_bin),'oscillations and remaining fixed points.')


		# FINDING OSCIL
		# run until sure that you are in the oscil
		loop=0
		restart_counter=orig_num_oscils=len(oscil_bin)
		confirmed_oscils = []
		orig_steps_per_lap, orig_fraction_per_lap = params['steps_per_lap'], params['fraction_per_lap']

		while len(oscil_bin) > 0: 
			x0, cutoff = run_oscil_init(params, oscil_bin, restart_counter, loop)
			result = lap.find_oscil_and_fixed_points(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, cutoff, loop)
			
			confirmed_oscils += list(result['state'][result['finished']==True])

		params['steps_per_lap'] = orig_steps_per_lap
		params['fraction_per_lap'] = orig_fraction_per_lap
		if params['debug']:
			assert(orig_num_oscils == len(confirmed_oscils))

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		# todo: redundant w prev step, should clean into a sep fn
		if params['verbose'] and len(confirmed_oscils)>0: 
			print('Finished finding oscillations, now classifying them.')
		oscil_bin = confirmed_oscils
		loop=0
		while len(oscil_bin) > 0: 
			x0, cutoff =run_oscil_init(params, oscil_bin, restart_counter, loop)
			result = lap.categorize_oscil(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, cutoff, loop)
			
			add_to_attractors(params, attractors, result)

		params['steps_per_lap'] = orig_steps_per_lap
		params['fraction_per_lap'] = orig_fraction_per_lap


# FROM LDOI


def ldoi_bfs_big_fat_mess(G,pinning=True,init=[],subset=[]):
	# TODO: rename 'subset' and clean this shit up

	#print('init=',init,'\nsubset=',subset)

	memoize=False # temp soln
	# A should be adjacency matrix of Gexp
	# and the corresponding Vexp to A should be ordered such that:
	#	 Vexp[0:n] are normal nodes, [n:2n] are negative nodes, and [2n:N] are composite nodes

	# pinning means that sources cannot drive their complement (they are fixed to be ON)
	# without pinning the algorithm will run faster, and it is possible that ~A in LDOI(A)
	init = cp.unique(init) # TODO: clean this (ex if [] is passed for init)
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
	visited = cp.zeros((N,N),dtype=bool) 
	# if visited[i,j]=1, then j is in the LDOI of i
	negated = cp.zeros((N,N),dtype=bool) 
	# if negated[i,j] then the complement of j is in the LDOI of i, 
	#	where j=i or some other pinned node (such as from init)
	D = cp.diag(cp.ones(N,dtype=bool))
	
	#memoize=True # temp set to false at beginning of file
	if len(init) > 0:
		if init[0].ndim < 1: # TODO clean this shit
			init = cp.array(init) 
			assert(cp.all(init < N)) # unlike numpy, cupy will wrap if out of bounds
			D[:,init] = 1 
			#assert(cp.all(1-(D & cp.roll(D,n_compl,axis=0)))) # check that initial set is non-contradictory
			#assert(0)
		else:
			memoize=False # if each row has different inits, memoize makes little sense
			for i in range(len(init)):
				init[i] = cp.array(init[i])
				D[i,init[i]] = 1

	# THIS ASSUMES THAT THE DIAG HAS PRECEDENCE OVER INIT
	rm_contras = cp.roll(cp.diag(cp.ones(N,dtype=bool)), n_compl,axis=0)
	D = D & (1-rm_contras)

	D[cp.arange(n,N)]=0 #don't care about source nodes for complement nodes
	
	print("\n\nD=\n",D.astype(int))
	assert(0)
	D_compl = D.copy()
	D_compl[:n] = cp.roll(D_compl[:n],n_compl,axis=1)

	max_count = max(cp.sum(A,axis=0))
	if max_count<128: 
		index_dtype = cp.int8
	else: #assuming indeg < 65536/2:
		index_dtype = cp.int16

	num_to_activate = cp.sum(A,axis=0,dtype=index_dtype)
	num_to_activate[:n] = 1 # non composite nodes are OR gates
	counts = cp.tile(num_to_activate,N).reshape(N,N) # num input nodes needed to activate
	zero = cp.zeros((N,N))

	#X[D] = 1 # wait..why did i use this?
	X = D.copy().astype(bool) #TODO: D should have been bool too...
	if False: # JP THIS IS UNNEC AND CAN RM
		print("Xbefore=\n",X.astype(int))
		print("part1=\n",A.astype(index_dtype),'\npart2=\n',cp.matmul(X.astype(index_dtype),A.astype(index_dtype)))
		X = counts-cp.matmul(X.astype(index_dtype),A.astype(index_dtype))<=0 
		negated = negated | cp.any(X & D_compl,axis=1) # this axis arg could be issue
		X = cp.logical_not(visited) & cp.logical_not(D_compl) & X 
		print("Xafter=\n",X.astype(int))
		# i.e. the descendants of X, including composite nodes iff X covers all their inputs
		
		if pinning:
			negated = negated | (X & D_compl)  # needs to be ANY in certain cases...
			print('negated,X,D_compl')
			print(negated.astype(int),'\n\n\n',negated.astype(int),'\n\n\n',X.astype(int),'\n\n\n',D_compl.astype(int))
			assert(0)
			X = cp.logical_not(D_compl) & X  

	loop=0
	while cp.any(X): 
		visited = visited | X #note that D is included, i.e. source nodes
		if pinning:
			visited_rolled = visited.copy()
			visited_rolled[:n] = cp.roll(visited_rolled[:n],n_compl,axis=0)
		
			X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0)
			if memoize:
				memoX = cp.matmul((visited|D) & cp.logical_not(visited_rolled.T), visited) 
				# if A has visited B, then add B's visited to A if ~A not in B's visited
				# otherwise must cut all of B's contribution!
				X = X | memoX
			negated = negated | (X & D_compl) # needs to be ANY in certain cases...
			X = cp.logical_not(visited) & cp.logical_not(D_compl) & X 
			
		else:
			X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) 
			if memoize:
				memoX = cp.matmul(visited|D, visited)
				X = X | memoX
			X = cp.logical_not(visited) & X

		# debugging:
		loop+=1
		if loop>N:
			print("WARNING: N steps exceeded")
		if loop > N**4:
			sys.exit("ERROR: infinite loop in ldoi_bfs!")

	if not pinning:
		assert(0) #TODO: explicitly debug this before using
		negated = cp.any(visited & D_compl,axis=0) 

	if subset != []:
		visited = augment_solutions(visited[:n,:n], negated[:n,:n], G, subset,D)
	return visited[:n,:n], negated[:n,:n] 


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


# FROM CNTRL

def old():
	#TODO: rm this soon, just for ref
	print('\n\nA FULL:\n',A_intersect)
	ldoi_solns, negates = ldoi.ldoi_sizes_over_all_inputs(params,Gpar,fixed_nodes=A_intersect,negates=True)

	print("\nw full A_inter: solns=\n",ldoi_solns,'\nnegates=\n',negates)
	A_intersect_trimd = {}
	for inpt_set in input_sets:
		A_intersect_trimd[inpt_set] = [[] for _ in range(2*Gpar.n)]
		# should be a faster route
		for c in range(2*Gpar.n):
			cname = Gpar.nodeNames[c]
			for a in A_intersect[inpt_set]:
				a_name = G.nodeNames[a]
				a_name_compl = G.nodeNames[(a+Gpar.n) % (2*Gpar.n)]
				if a_name not in negates[inpt_set][cname] and a_name_compl not in negates[inpt_set][cname]: 
					A_intersect_trimd[inpt_set][c] += [a]
					if inpt_set==(1, 1) and a_name=='x1':
						print("added ",a_name,"to",cname)
					
	print('\n\nA TRIMMED:\n',A_intersect_trimd)
	# c vs cname may become a bit of a headache too...
	# 2nd time fixed_nodes varies by each controller it seems...
	ldoi_solns, negates = ldoi.ldoi_sizes_over_all_inputs(params,Gpar,fixed_nodes=A_intersect_trimd,negates=True)
	
	print("\nw trimmed A_inter: solns=\n",ldoi_solns,'\nnegates=\n',negates)

	return ldoi_solns