# TODO
# need to clean this up, esp sync side
# rename 'state' to 'id'...but i mean everywhere


import itertools, util, sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
import numpy as np

def step(params, x, G):

	node_dtype = util.get_node_dtype(params)
	nodes_to_clauses = G.Fmapd['nodes_to_clauses']        
	clauses_to_threads = G.Fmapd['clauses_to_threads'] 
	threads_to_nodes = G.Fmapd['threads_to_nodes']
	X = cp.concatenate((x,cp.logical_not(x[:,:G.n])),axis=1)
	# add the not nodes
	# for x (state vector) axis 0 is parallel nets, axis 1 is nodes

	if not util.istrue(params,['PBN','active']) or not util.istrue(params,['PBN','float']) or params['PBN']['init'] != 'half':
		clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
			
		x_next = cp.zeros((G.n),dtype=node_dtype)

		# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
		partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)
		#for j in range(len(clauses_to_threads)): 
		#	# partial truths must be reordered for their corresponding nodes
		#	#print(partial_truths[:,j].shape,threads_to_nodes[j].shape,cp.matmul(partial_truths[:,j],threads_to_nodes[j]).shape)
		#	x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

		x_next = cp.sum(cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(node_dtype)

		# alternative with partial_truths also enclosed in for loop:
		'''  note that these take IDENTICAL time, so likely unfolded by compiler anyway
		for j in range(len(clauses_to_threads)): 
			partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 
			x = x + cp.matmul(partial_truths[:,],threads_to_nodes[j])
		'''
	else:
		assert(0) # plz debug, may well be broken
		# float version
		# 	basically, on a vector X, AND is product(X), and OR is 1-(product(1-X))
		clauses = cp.prod(X[:,nodes_to_clauses],axis=2)
		x_next = cp.zeros((G.n),dtype=node_dtype)
		partial_truths = 1-cp.prod(1-clauses[:,clauses_to_threads],axis=3)
		#print('partial_truths =',partial_truths )
		#print('before matmul =',cp.swapaxes(partial_truths,0,1),threads_to_nodes)
		#print('after matmul =',cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes))
		x_next = 1-cp.prod(1-cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(node_dtype)
		tol = 1e-5
		assert(cp.all(x_next) < 1.1 and cp.all(x_next) > -0.1)

		x_next[x_next < tol] = 0.0
		x_next[x_next > 1-tol] = 1.0

	if util.istrue(params,['PBN','active']):
		flip = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[1-params['PBN']['flip_pr'], params['PBN']['flip_pr']]).astype(node_dtype,copy=False)
		#if True: #params['PBN']['init'] != 'half':
		return cp.logical_not(flip)*x_next | flip*cp.logical_not(x)
		#else:
		#	assert(0) # seems...broken
		#	return 1 - ((1-cp.logical_not(flip)*x_next) * (1-flip*cp.logical_not(x)))

	return x_next


def transient(params,x0, G, fixed_points_only=False):
	# run network from an initial state 
	# for sync: if fixed_points=True only return 'finished' for fixed points (and move oscils hopefully passed transient point)
			# else fixed_points=False return if oscillated back to starting point 

	node_dtype = util.get_node_dtype(params)
	x = cp.array(x0,dtype=node_dtype).copy()
	x0 = cp.array(x0,dtype=node_dtype).copy() #need for comparisons
	if params['precise_oscils']:
		not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)
	if util.istrue(params,['PBN','active']) and 'inputs' in params.keys():
		input_indx = cp.array(G.input_indices(params))
		input_array = x[:,input_indx].copy()
		# since an input flip should not be permanent, but input_t+1 = input_t

	for i in range(params['steps_per_lap']):
		
		x_next = step(params, x, G)
		
		if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and i%2==0: 
			# reset inputs in case of flip, every other such that input flips still can effect net
			# slight bias: inputs mutate 1/2 as freq, effectively
			x[:,input_indx] = input_array
		
		if params['update_rule']=='sync':
			if params['precise_oscils']:
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

	if params['precise_oscils']:
		#debug_print_outputs(params,G,x)
		#assert(0)
		return exit_sync_transient(x, not_finished)
	#else:
	return x


def exit_sync_transient(x, not_finished):
	return {'finished':cp.logical_not(not_finished), 'state':x, 'avg':x, 'period':cp.logical_not(not_finished)}

def debug_print_outputs(params,G,x,target=None):
	k = len(params['inputs'])
	inpt = [G.nodeNums[params['inputs'][i]] for i in range(k)]
	k2 = len(params['outputs'])
	outpt = [G.nodeNums[params['outputs'][i]] for i in range(k2)]
	if target is not None:
		for row in x:
			if cp.all(cp.logical_not(row[inpt]-cp.array(target))):
				print(target,'->',row[outpt].astype(int))
	else: 
		for row in x:
			print(row[inpt].astype(int),'->',row[outpt].astype(int))


def categorize_attractor(params,x0, G):
	# assumes x0 is in the oscil
	# will return a sorted ID for the oscil (max int representation) and the average activity of each node

	node_dtype = util.get_node_dtype(params)
	x = cp.array(x0,dtype=node_dtype).copy()
	x0 = cp.array(x0,dtype=node_dtype).copy() #need for comparisons
	ids = cp.array(x0,dtype=bool).copy()

	# should be able to clean and rm these or smthg:
	avg_ensemble, var_ensemble, var_time,avg_time, var_x0,avg_x0 = [cp.zeros((1)) for _ in range(6)]  # kinda messy, but just if won't be used
	if util.istrue(params,'track_x0'): # i.e. if calculating lap stats
		x0_ids = np.unique(x0.get(),return_inverse=True,axis=0)[1] # just get index for each row of x0, where if x0_i = x0_j, they have same index
		x0_ids = cp.array(x0_ids)
		num_x0s = int(cp.max(x0_ids))+1
		avg_ensemble, var_ensemble = cp.zeros((num_x0s,G.n),dtype=float),cp.zeros((num_x0s,G.n),dtype=float)
		var_time, avg_time = cp.zeros(x0.shape,dtype=float) , cp.zeros(x0.shape,dtype=float) 
		assert(num_x0s < len(x0)/2) # otherwise really shouldn't be using 'track_x0' param

	if params['precise_oscils']:
		period = cp.ones(params['parallelism'],dtype=int)
		not_finished = cp.ones(params['parallelism'], dtype=bool)

	if util.istrue(params,['PBN','active']) and 'inputs' in params.keys():
		input_indx = cp.array(G.input_indices(params))
		input_array = x[:,input_indx].copy()
		# since an input flip should not be permanent, but input_t+1 = input_t

	index_dtype, thread_dtype = get_dtypes(params, G.n)
	simple_ind = cp.ones(params['parallelism'],dtype=thread_dtype) 
	larger = cp.zeros((params['parallelism']),dtype=cp.uint8)
	diff = cp.zeros((params['parallelism'],G.n),dtype=cp.uint8)
	first_diff_col = cp.zeros((params['parallelism']),dtype=index_dtype)

	avg_states = cp.array(x0,dtype=cp.float16)

	for i in range(params['steps_per_lap']):
		#print("lap.py CyclinD=",x[:,G.nodeNums['CycD']].astype(int),",GSK_3+p15=",cp.logical_xor(x[:,G.nodeNums['GSK_3']],x[:,G.nodeNums['p15']]).astype(int))
		
		x_next = step(params, x, G)
		if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and i%2==0: 
			# reset inputs in case of flip, every other such that input flips still can effect net
			# slight bias: inputs mutate 1/2 as freq, effectively
			x[:,input_indx] = input_array

		if params['update_rule']=='sync':
			x = x_next
			if params['precise_oscils']:
				not_match = cp.any(cp.logical_xor(x,x0),axis=1)
				period += not_match*not_finished 
				not_finished = cp.logical_and(not_finished, not_match) 
				which_nodes = cp.ones(x.shape)*not_finished[:,cp.newaxis]
			else:
				which_nodes = cp.ones(x.shape)
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

		avg_states += which_nodes*x  # TODO: why only update if node changes?
		if util.istrue(params,'track_x0'):
			#avg_ensemble += cp.mean(x,axis=0)
			#var_ensemble += cp.var(x, axis=0)
			for i in range(num_x0s):
				subx = x[x0_ids==i]
				avg_ensemble[i] += cp.mean(subx,axis=0)
				var_ensemble[i] += cp.var(subx,axis=0,ddof=0) 
					# TODO: should be ddof=1 for unbiased sample, but then divides by 0 sometimes...
			#var_t += cp.power(x,2) # note that actually calculating <x^2> and will - <x>^2 at the end
			avg_time += x

		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x.astype(cp.int8)
		first_diff_col = ((diff)!=0).argmax(axis=1).astype(index_dtype) #i'm worred that this first goes to 64 then converts to 8..
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 

		if params['precise_oscils']:
			if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']: 
				return exit_sync_categorize_oscil(params, x0, ids, not_finished, period, avg_states)

	# using num steps +1 since init state is also added to the avg
	if params['update_rule'] == 'async':
		expected_num_updates = (params['steps_per_lap'])/(G.n) # apparently this is normalized elsewhere
	elif params['update_rule'] == 'Gasync':
		expected_num_updates = (params['steps_per_lap'])/2 # apparently this is normalized elsewhere
	elif util.istrue(params,['PBN','active']) or params['map_from_A0'] or util.istrue(params,'skips_precise_oscils'):
		expected_num_updates =params['steps_per_lap']
	else: #sync
		if params['precise_oscils']:
			expected_num_updates = period[:,cp.newaxis].astype(float)
			assert(not util.istrue(params,['PBN','active'])) #otherwise need to add normzn jp
		else:
			expected_num_updates = params['steps_per_lap']

	if util.istrue(params,'track_x0'):
		# average over time steps
		#print('before avg over t:',avg_ensemble)
		avg_x0 = cp.copy(avg_time)
		avg_time /= params['steps_per_lap'] 

		# average over x0
		avg_ensemble = cp.sum(avg_ensemble,axis=0) / params['steps_per_lap']  #*expected_num_updates)
		var_ensemble = cp.mean(var_ensemble,axis=0) / params['steps_per_lap']  #*expected_num_updates)

		#var_t = cp.mean(var_t - cp.power(avg_ensemble,2),axis=0) # note that avg_ensemble is really just average in general
		
		# relies on fact that xi is boolean (so xi^2=xi):
		#	and since avg_total has not yet been avg'd over instances of x_i, works as avg_time
		var_time = avg_time - cp.power(avg_time,2) # this gives variance for all x0
		var_time = cp.mean(var_time, axis=0) # now avg variance for each node
		assert(params['steps_per_lap']>1) # else shouldn't be measuring temporal variance
		#var_time *= params['steps_per_lap'] / (params['steps_per_lap']-1) # to make it unbiased sample stat

		avg_x0_total=0
		var_x0 = cp.zeros(G.n)
		for i in range(num_x0s):
			w_x0=len(avg_x0[x0_ids==i])/params['num_samples']
			avg_x0_i = cp.mean(avg_x0[x0_ids==i],axis=0)/params['steps_per_lap']
			avg_x0_total += avg_x0_i*w_x0
			var_x0 += (avg_x0_i - cp.power(avg_x0_i,2))*w_x0
			# TODO: add back unbiased thing once comfortable (shouldn't make big diff)
			#n = len(avg_x0)*params['steps_per_lap']
			#var_x0 *= n/(n-1) #make it unbiased
		#var_x0 /= num_x0s 

		avg_x0 = avg_x0_total #cp.mean(avg_x0,axis=0)/params['steps_per_lap']

	avg_states = avg_states/expected_num_updates

	if params['precise_oscils']:
		return exit_sync_categorize_oscil(params, x0, ids, not_finished, period, avg_states)
	#else:
	return {'state':ids, 'avg':avg_states,'avg_total':avg_x0, 'var_ensemble':var_ensemble,'var_time':var_time, 'var_x0':var_x0}


def exit_sync_categorize_oscil(params,x0, ids, not_finished, period, avg_states):
	avg_states = avg_states/period[:,cp.newaxis].astype(float)
	exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x0*not_finished[:,cp.newaxis]
	if params['debug'] and params['update_rule'] == 'sync': #since async and Gasync are based on expected num of updates, they may be slightly off
		assert (cp.all(avg_states <= 1.001))
		assert (cp.all(avg_states >= -.001))

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