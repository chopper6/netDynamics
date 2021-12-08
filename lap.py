# TODO
# need to clean this up, esp sync side
# rename 'state' to 'id'...but i mean everywhere


import itertools, util, sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def step(params, x, G):
	nodes_to_clauses = G.Fmapd['nodes_to_clauses']
	clauses_to_threads = G.Fmapd['clauses_to_threads'] 
	threads_to_nodes = G.Fmapd['threads_to_nodes']
	X = cp.concatenate((x,cp.logical_not(x[:,:G.n])),axis=1)
	# add the not nodes
	# for x (state vector) axis 0 is parallel nets, axis 1 is nodes
	clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
		
	x_next = cp.zeros((G.n),dtype=bool)

	# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
	partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)
	#for j in range(len(clauses_to_threads)): 
	#	# partial truths must be reordered for their corresponding nodes
	#	#print(partial_truths[:,j].shape,threads_to_nodes[j].shape,cp.matmul(partial_truths[:,j],threads_to_nodes[j]).shape)
	#	x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

	x_next = cp.sum(cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(bool)

	# alternative with partial_truths also enclosed in for loop:
	'''  note that these take IDENTICAL time, so likely unfolded by compiler anyway
	for j in range(len(clauses_to_threads)): 
		partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 
		x = x + cp.matmul(partial_truths[:,],threads_to_nodes[j])
	'''
	if util.istrue(params,['PBN','active']):
		flip = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[1-params['PBN']['flip_pr'], params['PBN']['flip_pr']]).astype(bool,copy=False)
		flip[:,0]=0 #always OFF nodes should still be off
		#print(cp.sum((cp.logical_not(flip)*x_next | flip*cp.logical_not(x))[:,1])/100)
		return cp.logical_not(flip)*x_next | flip*cp.logical_not(x)

	return x_next


def transient(params,x0, G, fixed_points_only=False):
	# run network from an initial state 
	# for sync: if fixed_points=True only return 'finished' for fixed points (and move oscils hopefully passed transient point)
			# else fixed_points=False return if oscillated back to starting point 
	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons
	if params['update_rule']=='sync' and  not util.istrue(params,['PBN','active']):
		not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)

	for i in range(params['steps_per_lap']):

		x_next = step(params, x, G)
		
		if params['update_rule']=='sync':
			if not util.istrue(params,['PBN','active']):
				if fixed_points_only:
					not_finished = not_finished & cp.any(cp.logical_xor(x_next,x),axis=1)
				else:
					not_finished = not_finished & cp.any(cp.logical_xor(x_next,x0),axis=1)

				if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
					return exit_sync_transient(x_next, not_finished)
			x = x_next
		elif params['update_rule']=='Gasync': # generalized async, note that this is NOT the same as general async
			p=.5
			which_nodes = cp.random.rand(params['parallelism'], G.n) > p
			x = which_nodes*x_next + (1-which_nodes)*x 
		elif params['update_rule']=='async':
			# WARNING: this is incredibly inefficient (uses an array to update a single element)
			# only use to check against Gasync
			which_nodes = cp.zeros((params['parallelism'], G.n), dtype=bool)
			indx = [cp.arange(params['parallelism']), cp.random.randint(0, high=G.n, size=params['parallelism'])]
			which_nodes[indx]=1
			x = which_nodes*x_next + (1-which_nodes)*x
		else:
			sys.exit("\nERROR 'update_rule' parameter not recognized!\n")

	if params['update_rule']=='sync' and not util.istrue(params,['PBN','active']):
		return exit_sync_transient(x, not_finished)
	#else:
	return x


def exit_sync_transient(x, not_finished):
	return {'finished':cp.logical_not(not_finished), 'state':x, 'avg':x, 'period':cp.logical_not(not_finished),'var':cp.zeros(x.shape)}

def debug_print_outputs(params,G,x,target=[0,0,1,0]):
	k = len(params['inputs'])
	inpt = [G.nodeNums[params['inputs'][i]] for i in range(k)]
	k2 = len(params['outputs'])
	outpt = [G.nodeNums[params['outputs'][i]] for i in range(k2)]
	for row in x:
		if cp.all(cp.logical_not(row[inpt]-cp.array(target))):
			print(target,'->',row[outpt].astype(int))

def categorize_attractor(params,x0, G, calculating_var=False,avg=None):
	# assumes x0 is in the oscil
	# will return a sorted ID for the oscil (max int representation) and the average activity of each node

	# calculating_var is a recursive flag, just for when this function calls itself again (in order to calc_var, if param['calc_var'])
	#	and avg is only passed by this function recursively (not from other modules)

	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons
	ids = cp.array(x0,dtype=bool).copy()

	if params['update_rule'] == 'sync' and not util.istrue(params,['PBN','active']):
		period = cp.array([1 for _ in range(params['parallelism'])],dtype=int)
		not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)

	index_dtype, thread_dtype = get_dtypes(params, G.n)
	simple_ind = cp.array([i for i in range(params['parallelism'])],dtype=thread_dtype) 
	larger = cp.zeros((params['parallelism']),dtype=cp.uint8)
	diff = cp.zeros((params['parallelism'],G.n),dtype=cp.uint8)
	first_diff_col = cp.zeros((params['parallelism']),dtype=index_dtype)

	avg_states = cp.array(x0,dtype=cp.float16)
	#avg_states = cp.zeros(x0.shape,dtype=cp.float16)
	if calculating_var:
		var_states = cp.array(x0,dtype=cp.float16)

	for i in range(params['steps_per_lap']):

		x_next = step(params, x, G)
		if params['update_rule']=='sync':
			x = x_next
			if not util.istrue(params,['PBN','active']):
				not_match = cp.any(cp.logical_xor(x,x0),axis=1)
				period += not_match*not_finished 
				not_finished = cp.logical_and(not_finished, not_match) 

			which_nodes = cp.ones(x.shape)*not_finished[:,cp.newaxis]
		elif params['update_rule']=='Gasync': # generalized async, note that this is NOT the same as general async
			p=.5
			which_nodes = cp.random.rand(params['parallelism'], G.n) > p
			x = which_nodes*x_next + (1-which_nodes)*x 
		elif params['update_rule']=='async':
			# WARNING: this is incredibly inefficient (uses an array to update a single element)
			# and in fact even more efficient, maybe due to the extra for loop?
			# only use to check against Gasync
			which_nodes = cp.zeros((params['parallelism'], G.n), dtype=bool)
			indx = [cp.arange(params['parallelism']), cp.random.randint(0, high=G.n, size=params['parallelism'])]
			which_nodes[indx]=1
			x = which_nodes*x_next + (1-which_nodes)*x
		else:
			sys.exit("\nERROR 'update_rule' parameter not recognized!\n")

		avg_states += which_nodes*x
		if calculating_var:
			var_states += which_nodes*cp.power(x-avg,2)

		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x.astype(cp.int8)
		first_diff_col = ((diff)!=0).argmax(axis=1).astype(index_dtype) #i'm worred that this first goes to 64 then converts to 8..
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 

		if params['update_rule']=='sync' and not util.istrue(params,['PBN','active']) and not util.istrue(params,'calc_var'):
			if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
				return exit_sync_categorize_oscil(x0, ids, not_finished, period, avg_states)

	# using num steps +1 since init state is also added to the avg
	if params['update_rule'] == 'async':
		expected_num_updates = (params['steps_per_lap']+1)/(G.n)
	elif params['update_rule'] == 'Gasync':
		expected_num_updates = (params['steps_per_lap']+1)/2
	else: #sync
		expected_num_updates = params['steps_per_lap']+1
	
	avg_states = avg_states/expected_num_updates
	if params['debug'] and params['update_rule'] == 'sync': #since async and Gasync are based on expected num of updates, they may be slightly off
		assert (cp.all(avg_states <= 1.001))
		assert (cp.all(avg_states >= -.001))
	
	if calculating_var:
		var_states = var_states/(expected_num_updates-1) # -1 for unbiased estim
		assert(expected_num_updates>1) #otherwise variance cannot be well estimated! (if async steps_per_lap should be >> #nodes)
		if params['update_rule']=='sync' and not util.istrue(params,['PBN','active']):
			return exit_sync_categorize_oscil(x0, ids, not_finished, period, avg_states,variance=var_states)
		#else
		return {'state':ids, 'avg':avg_states, 'var': var_states }	

	if util.istrue(params,'calc_var'):
		# recursively call function to actually calculate it
		return categorize_attractor(params,x0, G, calculating_var=True, avg=avg_states)

	if params['update_rule']=='sync' and not util.istrue(params,['PBN','active']):
		return exit_sync_categorize_oscil(x0, ids, not_finished, period, avg_states)
	#else:
	return {'state':ids, 'avg':avg_states}


def exit_sync_categorize_oscil(x0, ids, not_finished, period, avg_states,variance=None):
	avg_states = avg_states/period[:,cp.newaxis]
	exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x0*not_finished[:,cp.newaxis]
	if variance is not None:
		return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period, 'avg':avg_states,'var':variance}	
	#else
	return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period, 'avg':avg_states}	


def get_dtypes(params, num_nodes):
	if num_nodes<256: 
		index_dtype = cp.uint8
	else:
		index_dtype = cp.uint16
	if params['parallelism']<256:
		thread_dtype = cp.uint8
	elif params['parallelism']<65535:
		thread_dtype = cp.uint16
	else:
		thread_dtype = cp.uint32
	return index_dtype, thread_dtype