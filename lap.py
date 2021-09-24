import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def fixed_point_search(params,x0, clause_mapping, num_nodes):
	# use to find fixed points and pass transients such that end within an oscillation (if not a fixed point)

	# overview of DNF approach:
		#	all: are all elements of the clause correct? if so, clause is TRUE
		#	any: are any of the clauses correct? if so, node is ON		
		# note that CNF would be cp.all(cp.any(...))
		# also note that only the positive nodes are computed (then negatives are concatenated)

	x = cp.array(x0)
	nodes_to_clauses = cp.array(clause_mapping['nodes_to_clauses'])
	clauses_to_threads = cp.array(clause_mapping['clauses_to_threads'])
	threads_to_nodes = cp.array(clause_mapping['threads_to_nodes'])
	
	for i in range(params['steps_per_lap']):

		X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
		# add the not nodes
		# for x (state vector) axis 0 is parallel nets, axis 1 is nodes
		clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
			
		if not params['lap_stop_rule']:
			x = cp.zeros((num_nodes))

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

		elif params['lap_stop_rule'] == 'steady': # i doubt frequently checking for steady state is efficient
			x_next = cp.zeros((num_nodes))

			# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
			partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)
					
			for j in range(len(clauses_to_threads)):
				# partial truths must be reordered for their corresponding nodes
				x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

			# if x + x_next = 0 (ie nxor is true) for all nodes, then at a steady state
			if cp.sum(cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1))/params['parallelism'] >= params['fraction_per_lap']:
				return {'finished':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next}
			
			x = x_next 

		else:
			sys.exit("ERROR: unrecognized argument for 'lap_stop_rule'")

	# do once more but keep x_next sep to check for steady state
	X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
	clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
	x_next = cp.zeros((num_nodes))
	partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)		
	for j in range(len(clauses_to_threads)):
		x_next = x_next + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

	return {'finished':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next}



def from_somewhere_in_oscil(params,x0, clause_mapping, num_nodes):
	# run network from an initial state are see if return to it (ie reach an oscil)
	# will return a sorted ID for the oscil (max int representation) and the period

	x = cp.array(x0).copy()
	x0 = cp.array(x0).copy() #need for comparisons
	period = cp.array([0 for _ in range(params['parallelism'])],dtype=int)
	not_finished = cp.array([1 for _ in range(params['parallelism'])],dtype=bool)

	if num_nodes<256: 
		index_dtype = cp.uint8
	else:
		index_dtype = cp.uint16
	simple_ind = cp.array([i for i in range(len(x0))],dtype=index_dtype) 
	ids = x0.copy()

	nodes_to_clauses = cp.array(clause_mapping['nodes_to_clauses'])
	clauses_to_threads = cp.array(clause_mapping['clauses_to_threads'])
	threads_to_nodes = cp.array(clause_mapping['threads_to_nodes'])
	
	for i in range(params['steps_per_lap']):

		X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
		clauses = cp.all(X[:,nodes_to_clauses],axis=2) 

		x = cp.zeros((num_nodes))

		partial_truths = cp.any(clauses[:,clauses_to_threads],axis=3)
					
		for j in range(len(clauses_to_threads)):
			x = x + cp.matmul(partial_truths[:,j],threads_to_nodes[j])

		not_match = cp.any(cp.logical_xor(x,x0),axis=1)
		period += not_match*not_finished 
		not_finished = cp.logical_and(not_finished, not_match) 
		
		# next few lines are just to replace ids with current states that are "larger", where larger is defined by int representation (i.e. first bit that is different)
		diff = ids-x
		first_diff_col = ((diff)!=0).argmax(axis=1)
		larger = diff[[simple_ind,first_diff_col]] #note that numpy/cupy handles this as taking all elements indexed at [simple_ind[i],first_diff_col[i]]   
		ids = (larger==-1)[:,cp.newaxis]*x + (larger!=-1)[:,cp.newaxis]*ids # if x is 'larger' than current id, then replace id with it 
		
		if cp.sum(cp.logical_not(not_finished)/params['parallelism']) >= params['fraction_per_lap']:
			exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x*not_finished[:,cp.newaxis]
			return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period}
			

	exit_states = ids*cp.logical_not(not_finished)[:,cp.newaxis]+x*not_finished[:,cp.newaxis]
	return {'finished':cp.logical_not(not_finished), 'state':exit_states,'period':period}
