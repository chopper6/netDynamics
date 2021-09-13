import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def from_init_val(params,x0, clause_mapping, num_nodes):

	x = cp.array(x0)
	nodes_to_clauses = cp.array(clause_mapping['nodes_to_clauses'])
	clauses_to_threads = cp.array(clause_mapping['clauses_to_threads'])
	threads_to_nodes = cp.array(clause_mapping['threads_to_nodes'])
	
	for i in range(params['steps_per_lap']):

		X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
		# add the not nodes
		# for x (state vector) axis 0 is parallel nets, axis 1 is nodes

		# because of how clause_index is structured, the next line only produces output for all nodes, but not their negatives
		# however, it requires the negative nodes as separate inputs
		
		if params['lap_stop_rule'] != 'steady':


			clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies

			x = cp.zeros((num_nodes))
			for j in range(len(clauses_to_threads)): # TODO: unroll this for loop (and compare speeds)
				# partial truths are sufficient for a node to be on (ie they are some but not all clauses)
				partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 

				# partial truths must be reordered for their corresponding nodes
				x = x + cp.matmul(partial_truths[:,],threads_to_nodes[j])

				#	all: are all elements of the clause correct? if so, clause is TRUE
				#	any: are any of the clauses correct? if so, node is ON
				
				# note that CNF would be cp.all(cp.any(...))

		else: # i doubt frequently checking for steady state is efficient
			assert(0) #TODO update this (and rm unless about as fast) 
			x_next = cp.any(cp.all(X[clause_index],axis=3),axis=2) 
			# if x + x_next = 0 (ie nxor is true) for all nodes, then at a steady state
			if cp.sum(cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1)) >= params['percent_steady']:
				return {'steady_state':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next}
			x = x_next 


	# run once more, this time copy to see if in SS. This is redundant is lap_stop_rule==steady
	# TODO clean this part (poss sep "step" fn)
	X = cp.concatenate((x,cp.logical_not(x[:,:num_nodes])),axis=1)
	clauses = cp.all(X[:,nodes_to_clauses],axis=2) #this is gonna to a bitch to change when clause_index has mult, diff copies
	x_next = cp.zeros((num_nodes))
	for j in range(len(clauses_to_threads)): 
		partial_truths = cp.any(clauses[:,clauses_to_threads[j]],axis=2) 
		x_next = x_next + cp.matmul(partial_truths[:,],threads_to_nodes[j])

	return {'steady_state':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next}


