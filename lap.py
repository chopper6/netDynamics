import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def from_init_val(params,x0, clause_mapping, num_nodes):
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
			if cp.sum(cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1))/params['parallelism'] >= params['percent_steady']:
				steps_taken = i
				return {'steady_state':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next, 'steps_taken': steps_taken}
			
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

	return {'steady_state':cp.all(cp.logical_not(cp.logical_xor(x,x_next)),axis=1), 'state':x_next, 'steps_taken':params['steps_per_lap']}


