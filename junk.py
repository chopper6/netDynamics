# these are pieces of code that are no longer used, but may be useful later

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