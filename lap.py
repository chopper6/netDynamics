# TODO
# need to clean this up, esp sync side
# rename 'state' to 'id'...but i mean everywhere


import itertools, util, sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def step(params, x, num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
	# add the not nodes
	# for x (state vector) axis 0 is parallel nets, axis 1 is nodes
	clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
		
	x = cp.zeros((num_nodes),dtype=bool)

	# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
	partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)
			
	for j in range(len(clauses_to_threads)):
		# partial truths must be reordered for their corresponding nodes
		x = x + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

	# alternative with partial_truths also enclosed in for loop:
	'''  note that these take IDENTICAL time, so likely unfolded by compiler anyway
	for j in range(len(clauses_to_threads)): 
		partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 
		x = x + cp.matmul(partial_truths[:,],threads_to_nodes[j])
	'''
	return x

def fixed_point_search(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	# use to find fixed points and pass transients such that end within an oscillation (if not a fixed point)

	# overview of DNF approach:
		#	all: are all elements of the clause correct? if so, clause is TRUE
		#	any: are any of the clauses correct? if so, node is ON		
		# note that CNF would be cp.all(cp.any(...))
		# also note that only the positive nodes are computed (then negatives are concatenated)

	x = cp.array(x0)
	
	for i in range(params['steps_per_lap']):
		x_next = step(params, x,num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes)

		if params['lap_stop_rule'] == 'steady': # i doubt frequently checking for steady state is efficient
			# if x + x_next = 0 (ie nxor is true) for all nodes, then at a steady state
			if cp.sum(cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1))/params['parallelism'] >= params['fraction_per_lap']:
				return {'finished':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next, 'avg':x_next, 'period':cp.ones(params['parallelism'])}
			
		elif params['lap_stop_rule']: # i.e. if not "False" otherwise unrecognized
			sys.exit("ERROR: unrecognized argument for 'lap_stop_rule'")

		x = x_next

	# do once more but keep x_next sep to check for steady state
	X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
	clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
	x_next = cp.zeros((num_nodes))
	partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)		
	for j in range(len(clauses_to_threads)):
		x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

	return {'finished':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next, 'avg':x_next, 'period':cp.ones(params['parallelism'])}



def find_oscil_and_fixed_points(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	# run network from an initial state are see if return to it (ie reach an oscil)

	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons
	not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)
	
	for i in range(params['steps_per_lap']):

		x_next = step(params, x,num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes)

		#not_match = cp.any(cp.logical_xor(x_next,x0),axis=1) #this is without checking for fixed points still
		not_match = cp.logical_and(cp.any(cp.logical_xor(x_next,x0),axis=1), cp.any(cp.logical_xor(x_next,x),axis=1))
		not_finished = cp.logical_and(not_finished, not_match) 
		x=x_next

		if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
			return {'finished':cp.logical_not(not_finished), 'state':x,'period':cp.ones(params['parallelism']), 'avg':x}
	# since only add fixed points to the attractors, assume period=1
	return {'finished':cp.logical_not(not_finished), 'state':x,'period':cp.ones(params['parallelism']), 'avg':x}


def categorize_oscil(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	# assumes x0 is in the oscil
	# will return a sorted ID for the oscil (max int representation) and the period

	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons
	ids = cp.array(x0,dtype=bool).copy()
	period = cp.array([1 for _ in range(params['parallelism'])],dtype=int)
	not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)

	index_dtype, thread_dtype = get_dtypes(params, num_nodes)

	simple_ind = cp.array([i for i in range(params['parallelism'])],dtype=thread_dtype) 

	larger = cp.zeros((params['parallelism']),dtype=cp.uint8)
	diff = cp.zeros((params['parallelism'],num_nodes),dtype=cp.uint8)
	first_diff_col = cp.zeros((params['parallelism']),dtype=index_dtype)

	avg_states = cp.array(x0,dtype=cp.float16)

	for i in range(params['steps_per_lap']):

		x = step(params, x,num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes)

		not_match = cp.any(cp.logical_xor(x,x0),axis=1)
		period += not_match*not_finished 
		not_finished = cp.logical_and(not_finished, not_match) 

		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x.astype(cp.int8)
		first_diff_col = ((diff)!=0).argmax(axis=1).astype(index_dtype) #i'm worred that this first goes to 64 then converts to 8..
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 

		avg_states = not_finished[:,cp.newaxis]*x + avg_states

		if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
			avg_states = avg_states/period[:,cp.newaxis]
			exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x0*not_finished[:,cp.newaxis]
			return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period, 'avg':avg_states}
			

	# if return to and pick up where left off: avg_states = not_finished[:,cp.newaxis]*avg_states + cp.logical_not(not_finished)[:,cp.newaxis]*avg_states/period
	avg_states = avg_states/period[:,cp.newaxis]
	exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x0*not_finished[:,cp.newaxis]
	return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period, 'avg':avg_states}



def async_transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons

	for i in range(params['steps_per_lap']):

		x_next = step(params, x,num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes)
		
		if params['update_rule']=='sync':
			x = x_next
			assert(False) #why is async_transient() function being called with sync??
		elif params['update_rule']=='Gasync': # generalized async, note that this is NOT the same as general async
			p=.5
			which_nodes = cp.random.rand(params['parallelism'], num_nodes) > p
			x = which_nodes*x_next + (1-which_nodes)*x 
		elif params['update_rule']=='async':
			# WARNING: this is incredibly inefficient (uses an array to update a single element)
			# only use to check against Gasync
			which_nodes = cp.zeros((params['parallelism'], num_nodes), dtype=bool)
			indx = [cp.arange(params['parallelism']), cp.random.randint(0, high=num_nodes, size=params['parallelism'])]
			which_nodes[indx]=1
			x = which_nodes*x_next + (1-which_nodes)*x
		else:
			sys.exit("\nERROR 'update_rule' parameter not recognized!\n")

	return x

def async_categorize_attractor(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes):
	# assumes x0 is in the oscil
	# will return a sorted ID for the oscil (max int representation) and the average activity of each node

	x = cp.array(x0,dtype=bool).copy()
	x0 = cp.array(x0,dtype=bool).copy() #need for comparisons
	ids = cp.array(x0,dtype=bool).copy()

	index_dtype, thread_dtype = get_dtypes(params, num_nodes)
	simple_ind = cp.array([i for i in range(params['parallelism'])],dtype=thread_dtype) 
	larger = cp.zeros((params['parallelism']),dtype=cp.uint8)
	diff = cp.zeros((params['parallelism'],num_nodes),dtype=cp.uint8)
	first_diff_col = cp.zeros((params['parallelism']),dtype=index_dtype)

	avg_states = cp.array(x0,dtype=cp.float16)

	for i in range(params['steps_per_lap']):

		x_next = step(params, x,num_nodes, nodes_to_clauses, clauses_to_threads, threads_to_nodes)
		if params['update_rule']=='sync':
			x = x_next
			assert(False) #why is async_transient() function being called with sync??
		elif params['update_rule']=='Gasync': # generalized async, note that this is NOT the same as general async
			p=.5
			which_nodes = cp.random.rand(params['parallelism'], num_nodes) > p
			x = which_nodes*x_next + (1-which_nodes)*x 
		elif params['update_rule']=='async':
			# WARNING: this is incredibly inefficient (uses an array to update a single element)
			# and in fact even more efficient, maybe due to the extra for loop?
			# only use to check against Gasync
			which_nodes = cp.zeros((params['parallelism'], num_nodes), dtype=bool)
			indx = [cp.arange(params['parallelism']), cp.random.randint(0, high=num_nodes, size=params['parallelism'])]
			which_nodes[indx]=1
			x = which_nodes*x_next + (1-which_nodes)*x
		else:
			sys.exit("\nERROR 'update_rule' parameter not recognized!\n")

		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x.astype(cp.int8)
		first_diff_col = ((diff)!=0).argmax(axis=1).astype(index_dtype) #i'm worred that this first goes to 64 then converts to 8..
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 

		avg_states += avg_states

	avg_states = avg_states/params['steps_per_lap']
	return {'state':ids, 'avg':avg_states}

			


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