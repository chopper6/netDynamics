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
	# add the not nodes
	# for x (state vector) axis 0 is parallel nets, axis 1 is nodes
	if not util.istrue(params,['PBN','active']) or not util.istrue(params,['PBN','float']):
		X = cp.concatenate((x,cp.logical_not(x[:,:G.n])),axis=1)
		clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies

		# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
		partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)

		#x_next = cp.zeros((G.n),dtype=node_dtype)
		#for j in range(len(clauses_to_threads)): 
		#	# partial truths must be reordered for their corresponding nodes
		#	#print(partial_truths[:,j].shape,threads_to_nodes[j].shape,cp.matmul(partial_truths[:,j],threads_to_nodes[j]).shape)
		#	x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

		x_next = cp.sum(cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(node_dtype)
		#print('lap.step next=',x_next.astype(int))
		# alternative with partial_truths also enclosed in for loop:
		'''  note that these take IDENTICAL time, so likely unfolded by compiler anyway
		for j in range(len(clauses_to_threads)): 
			partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 
			x = x + cp.matmul(partial_truths[:,],threads_to_nodes[j])
		'''	
		if util.istrue(params,['PBN','active']): # already know that float=False
			flip = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[1-params['PBN']['flip_pr'], params['PBN']['flip_pr']]).astype(node_dtype,copy=False)

			x_next = ((~flip) & x_next) | (flip & (~x_next))		

	elif 1:
		# PBN float version1

		X = cp.concatenate((x,1-x),axis=1) # append NOT x to the states
		#print('\n\n\nlap.step():')
		#G.print_matrix_names(X)
		clauses = cp.prod(X[:,nodes_to_clauses],axis=2)
		clauses *= G.Fmapd['clauses_multiplier']
		partial_truths = cp.sum(clauses[:,clauses_to_threads],axis=3)
		x_next = cp.sum(cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(node_dtype)
		
		flip_pr = params['PBN']['flip_pr']
		x_next = x_next*(1-flip_pr) + (1-x_next)*flip_pr

		
		x_next[:,0] = 0 # make sure the OFF nodes are not flipped
		
		tol=1e-5
		assert(cp.all(x_next<1+tol) & cp.all(x_next>0-tol))

	else:
		# PBN float version2

		X = cp.concatenate((x,1-x),axis=1) # append NOT x to the states
		print('\n\nlap.step():')
		G.print_matrix_names(X)
		clauses = cp.prod(X[:,nodes_to_clauses],axis=2)
		print('clauzez =',clauses[0])
		partial_truths = cp.prod(1-clauses[:,clauses_to_threads],axis=3)
		print('partial_truths =',partial_truths[0])
		print('before matmul =',cp.swapaxes(partial_truths,0,1)[0])
		print('after matmul =',1-cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes)[0])
		x_next = 1-cp.prod(cp.matmul(cp.swapaxes(partial_truths,0,1),threads_to_nodes),axis=0).astype(node_dtype)
		# jp matmul fucks it up since involves an addition 

		print("x_next before flip=")#[:20,:20])
		G.print_matrix_names(x_next)
		assert(0)
		flip_pr = params['PBN']['flip_pr']
		x_next = x_next*(1-flip_pr) + (1-x_next)*flip_pr

		# make sure the OFF nodes are not flipped
		x_next[:,0] = 0
		
		tol=1e-5

		#print("x_next over=",x_next[x_next>1+tol])
		#print("x_next under=",np.where(x_next<0-tol))
		assert(cp.all(x_next<1+tol) & cp.all(x_next>0-tol))

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
	
	if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and len(params['inputs'])>0:
		input_indx = cp.array(G.input_indices(params))
		input_array = x[:,input_indx].copy()
		# since an input flip should not be permanent, but input_t+1 = input_t

	for i in range(params['steps_per_lap']):
		
		x_next = step(params, x, G)
		
		if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and i%2==0 and len(params['inputs'])>0: 
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

	if params['precise_oscils']:
		period = cp.ones(params['parallelism'],dtype=int)
		not_finished = cp.ones(params['parallelism'], dtype=bool)

	if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and len(params['inputs'])>0:
		input_indx = cp.array(G.input_indices(params))
		input_array = x[:,input_indx].copy()
		# since an input flip should not be permanent, but input_t+1 = input_t

	index_dtype, thread_dtype = get_dtypes(params, G.n)
	simple_ind = cp.ones(params['parallelism'],dtype=thread_dtype) 
	larger = cp.zeros((params['parallelism']),dtype=cp.uint8)
	diff = cp.zeros((params['parallelism'],G.n),dtype=cp.uint8)
	first_diff_col = cp.zeros((params['parallelism']),dtype=index_dtype)
	
	if util.istrue(params,'var_window'):
		assert(params['var_window']<=params['steps_per_lap']) # else can't calc it
		# dtype should actually be bool for deterministic but will typ use with PBN
		windowed_x = cp.zeros((params['var_window'],params['parallelism'],G.n),dtype=float) 
		windowed_var_total = cp.zeros(G.n,dtype=float) # for each node, will find the var over the window and sum each time step 

	if not G.PBN:
		avg_states = cp.array(x0,dtype=int)
	else:
		avg_states = cp.array(x0,dtype=float)

	for i in range(params['steps_per_lap']):
		x_next = step(params, x, G)
		if util.istrue(params,['PBN','active']) and 'inputs' in params.keys() and i%2==0 and len(params['inputs'])>0: 
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
			if util.istrue(params,['PBN','float_update']):
				x = p*x_next + (1-p)*x
			else:
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

		avg_states += x 

		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x.astype(cp.int8)
		first_diff_col = ((diff)!=0).argmax(axis=1).astype(index_dtype) #i'm worred that this first goes to 64 then converts to 8..
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 
	
		if util.istrue(params,'var_window'):
			windowed_x[i%params['var_window']]= x 
			if i>= params['var_window']:
				window_avg = cp.mean(windowed_x,axis=0)
				windowed_var_total += cp.mean(window_avg - cp.power(window_avg,2),axis=0)

		if params['precise_oscils']:
			if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
				avg_states = avg_states/(i+2) #+1 for init, +1 since #laps = i+1
				return exit_sync_categorize_oscil(params, x0, ids, not_finished, period, avg_states)

	#avg_total = cp.sum(avg_states,axis=0)/(params['num_samples']*(params['steps_per_lap']+1))
	avg_states = avg_states /(params['steps_per_lap']+1)

	# note that var ignores var btwn diff instances (since some of their var is due to diff in x0)
	var_total = cp.mean(avg_states - cp.power(avg_states,2),axis=0)
	avg_total = cp.mean(avg_states,axis=0)
	if util.istrue(params,'var_window'):
		windowed_var=windowed_var_total/(params['steps_per_lap']+1-params['var_window'])
		windowed_var=cp.mean(windowed_var,axis=0)
	else:
		windowed_var=cp.array(0)

	if params['precise_oscils']:
		return exit_sync_categorize_oscil(params, x0, ids, not_finished, period, avg_states)
	#else:
	return {'state':ids, 'avg':avg_states,'avg_total':avg_total,'windowed_var':windowed_var,'var_total':var_total}


def exit_sync_categorize_oscil(params,x0, ids, not_finished, period, avg_states):
	# avg already divided by # steps before this
	#avg_states = avg_states/period[:,cp.newaxis].astype(float)
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