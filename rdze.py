def randomize_experiment(param_file):
	reps = 2
	num_swaps = 8
	mut_dist_thresh, cnt_dist_thresh = .2,.2
	net_stats, node_stats = {1:{},2:{}}, {}

	params = parse.params(param_file)
	params['save_fig'] = True 
	F, V = parse.get_logic(params)
	n2=len(V['#2name'])-2 #inclds composite nodes but not 0 node

	result = exhaustive(params, F, V, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
	net_stats[1]['original'] = result['network_stats1']
	net_stats[2]['original'] = result['network_stats2']
	node_stats = result['node_stats']
	node_stat_labels = result['node_stat_labels']

	parse.apply_mutations(params,F)
	F_mapd, A = parse.get_clause_mapping(params, F, V)
	
	#canal_score, total_score = features.calc_canalizing(F)

	# deg-preserving logic scrambles
	for r in range(reps): 
		F_copy, A_copy, V_copy = deepcopy(F), deepcopy(A),deepcopy(V)
		for _ in range(num_swaps):
			success=False
			while not success:
				node = V['#2name'][rd.randint(1,int(len(V['#2name'])/2)-1)] #skip that 0 node, and only do pos nodes
				success = swap_TT_row(F_copy, node, 1, num_attempts=20) # 1 means only change 1 row of TT
		if params['debug']:
			# check is degree preserving
			assert(np.all(np.sum(A_copy,axis=0)== np.sum(A,axis=0)) and np.all(np.sum(A_copy,axis=1)== np.sum(A,axis=1)))
		result = exhaustive(params, F_copy, V_copy, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
		net_stats[1]['logicScramble_' + str(r)] = result['network_stats1']
		net_stats[2]['logicScramble_' + str(r)] = result['network_stats2']
		for k in node_stats:
			node_stats[k] += result['node_stats'][k]

	# deg-preserving edge swaps
	for r in range(reps): 
		F_copy, A_copy, V_copy = deepcopy(F), deepcopy(A),deepcopy(V)
		swap_edges(F_copy, A_copy, V_copy,num_swaps*10,no_repeat_edges=True) #modifies F and A in place 
		if params['debug']:
			# check is degree preserving
			assert(np.all(np.sum(A_copy,axis=0)== np.sum(A,axis=0)) and np.all(np.sum(A_copy,axis=1)== np.sum(A,axis=1)))
		result  = exhaustive(params, F_copy, V_copy, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
		net_stats[1]['edgeSwap_' + str(r)] = result['network_stats1']
		net_stats[2]['edgeSwap_' + str(r)] = result['network_stats2']
		for k in node_stats:
			node_stats[k] += result['node_stats'][k]
			

	pickle_file = params['output_dir'] + '/stats.pickle' 
	stats = {'node_stats':node_stats,'net_stats':net_stats,'node_stat_labels':node_stat_labels}
	print("Pickling a file to: ", pickle_file)
	pickle.dump(stats, open( pickle_file, "wb" ))
	plot.control_exper_bar(params, net_stats) 
	plot.control_exper_scatter(params, node_stats, node_stat_labels)


def swap_edges(F,A,V,repeats,no_repeat_edges=True):
	# pick 2 random edges, then pick source or targets, then swap their variables
	# note that logic is preserved, albiet with different sources
	# modifies F and A in place
	for r in range(repeats):
		E1 = get_rd_edge(A)
		try_again = True
		loop=0
		while try_again : #ensure they are diff edges
			E2 = get_rd_edge(A)
			try_again = np.all(E1==E2)
			if no_repeat_edges and (A[E1[0],E2[1]]!=0 or A[E2[0],E1[1]]!=0): 
				#TODO: should repeat edges check the new or the old A?? curr using updated A
				try_again = True
			loop+=1
			assert(loop<10**5)

		source1, source2, target1, target2 = V['#2name'][E1[0]], V['#2name'][E2[0]], V['#2name'][E1[1]], V['#2name'][E2[1]]

		swap_literals(F[target1],source1,source2)
		swap_literals(F[target2],source2,source1)
		
		# rm old, add new edges (assumes unsigned)
		A[E1[0],E1[1]]=0
		A[E2[0],E2[1]]=0	
		A[E1[0],E2[1]]=1
		A[E2[0],E1[1]]=1


def swap_TT_row(F, node, num_swaps, num_attempts=100):
	# returns True if swap was successful, False if ran out of attempts
	# note that a swap with all desired constraints may not be possible

	# WARNING: assumes bnet all over the place here
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node = parse.get_file_format('bnet')
	
	attempt=0
	while True:
		inputs_rev, inputs_fwd =  get_inputs_to_node(F[node], not_str)
		int_clauses = []
		for clause in F[node]:
			int_clause = 0
			for ele in clause:
				if not_str not in ele: 
					int_clause += 2**inputs_fwd[ele]
			int_clauses += [int_clause]
		n = len(inputs_rev)

		for _ in range(num_swaps):
			loop=0
			while True:
				new_term = rd.randint(0,2**n-1)
				if new_term not in int_clauses:
					del int_clauses[rd.randint(0,len(int_clauses)-1)]
					int_clauses += [new_term]
					break
				if loop>num_attempts:
					return False
				loop+=1

		reduced_fn = logic.run_qm(int_clauses, n, 'bnet', inputs_rev)

		parsed_clauses = reverse_parse_reduced_clauses(reduced_fn,node)
		inputs_rev_orig = inputs_rev.copy()
		inputs_rev, inputs_fwd =  get_inputs_to_node(parsed_clauses, not_str)
		if len(inputs_rev) == n: #i.e. check that maintain same # of functional inputs (can lose due to redundancy)
			F[node] = parsed_clauses
			return True

		assert(len(inputs_rev) <= n)

		if attempt > num_attempts:
			return False
		attempt+=1


def get_inputs_to_node(clauses, not_str):
	# fwd is name2#, rev is #toname
	inputs_rev, inputs_fwd = [],{}
	for clause in clauses:
		for ele in clause:
			ele_ = str(ele).replace(not_str,'')
			if ele_ not in inputs_rev:
				inputs_fwd[ele_] = len(inputs_rev)
				inputs_rev += [ele_]
	return inputs_rev, inputs_fwd

def reverse_parse_reduced_clauses(reduced_fn, node_name):
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node = parse.get_file_format('bnet')
	clauses = reduced_fn.split(clause_split)
	parsed_clauses = []
	for clause in clauses:
		this_clause=[]
		for symbol in strip_from_clause:
			clause = clause.replace(symbol,'')
		literals = clause.split(literal_split)
		for j in range(len(literals)):
			literal_name = literals[j]
			for symbol in strip_from_node:
				literal_name = literal_name.replace(symbol,'')
			this_clause += [literal_name]
		parsed_clauses += [this_clause]
	return parsed_clauses


def swap_literals(logic_fn,lit1,lit2):
	# swaps lit1 for lit2 in the logical function
	for clause in logic_fn: 
		for i in range(len(clause)):
			if clause[i] == lit1:
				clause[i] = lit2



def get_rd_edge(A):
	sign=0
	loop=0
	while sign==0:
		assert(loop<10**10) 
		E = np.random.randint(len(A), size=2)
		sign = A[E[0],E[1]]
		loop+=1
	return E



if __name__ == "__main__":
	randomize_experiment(sys.argv[1]) 