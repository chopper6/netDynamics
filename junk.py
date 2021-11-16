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
