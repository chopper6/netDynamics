import itertools, util, math, warnings, sys
import lap
CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


def calc_basin_size(params, clause_mapping, node_mapping):
	#num_nodes is not counting the negative copies
	node_name_to_num = node_mapping['name_to_num']
	node_num_to_name = node_mapping['num_to_name']
	num_nodes = int(len(node_num_to_name)/2) #i.e. excluding the negative copies
	if params['debug']:
		assert(len(node_num_to_name)%2==0)

	attractors = {'oscillates':{'size':0}} 
	#oscillates means anything not a fixed point
	#size is later normalized sT sum(size)=1
	
	if params['exhaustive']:
		num_samples = (2**(num_nodes-1))
		if params['parallelism'] > num_samples:
			sys.exit("ERROR: parallelism parameter must be < number of samples for exhaustive search (2^n)")
		X0 = itertools.product([0,1], repeat=num_nodes-1) #first index is the always OFF node
		for i in range(int(math.floor((2**(num_nodes-1))/params['parallelism']))):
			x0 = cp.array(list(itertools.islice(X0, params['parallelism'])),dtype=bool)
			x0 = cp.insert(x0, 0, 0, axis=1)
			result  = lap.from_init_val(params,x0, clause_mapping, num_nodes)
			add_to_attractors(params, attractors, result)
		for i in range(2**(num_nodes-1) % params['parallelism']): # do the remaining samples sequentially
			x0 = cp.array(list(itertools.islice(X0, 1)),dtype=bool)
			x0 = cp.insert(x0, 0, 0, axis=1)
			result = lap.from_init_val(params,x0, clause_mapping, num_nodes)
			add_to_attractors(params, attractors, result)
		for s in attractors.keys():
			attractors[s]['size'] /= num_samples

	else:
		steps_taken, samples_processed = cp.zeros((params['parallelism'])), 0
		#clause_index = util.copy_to_larger_dim(clause_index,params['parallelism']) # for later

		p = .5 #prob a given node is off at start
		x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)
		x0[:,0] = 0 #0th node is the always OFF node
		
		if not params['num_samples'] and not params['max_laps']:
			sys.exit("\nERROR: Simulation exit condition required, 'num_samples' or 'max_laps' must not be > 0.") 
		
		i=0
		while True:
			if params['verbose'] and i%params['print_lap']==0:
				print("At lap",i,"with",samples_processed,"samples processed.")


			result = lap.from_init_val(params,x0, clause_mapping, num_nodes)
			if params['cupy']: #convert cupy to numpy (this param is added in parse.py)
				result['steady_state'] = result['steady_state'].get()
				result['state'] = result['state'].get()

			steps_taken += result['steps_taken'] #i.e. counting how many times failed 
			add_to_attractors(params, attractors, result,steps_taken=steps_taken)

			steps_taken[result['steady_state']==True]=0 #reset samples that have reached attractor already
			samples_processed += cp.sum(result['steady_state']) + cp.sum(steps_taken >= params['max_steps_per_sample'])
			# must be after the above line to avoid double counting those that have finished, but before the below line
			steps_taken[steps_taken >= params['max_steps_per_sample']]=0 #give up on samples that have already run too many laps without reaching steady state
			

			if params['num_samples'] and samples_processed >= params['num_samples']:
				break

			if params['max_laps'] and i>=params['max_laps']:
				if params['num_samples']:
					print("\nWARNING: exiting since reached 'max_laps', all samples have not have ben run\n")
				break

			x0 = result['state']

			# add new initial conditions to entries that finished
			for j in range(len(steps_taken)):
				if steps_taken[j]==0: #done or given up, so refill w new sample
					x0[j]= cp.random.choice(a=[0,1], size=(num_nodes), p=[p, 1-p]).astype(bool,copy=False)
			i+=1


		if params['debug']:
			assert(sum([attractors[i]['size'] for i in attractors.keys()])==samples_processed)
		
		for s in attractors.keys():
			attractors[s]['size'] /= samples_processed 

		if samples_processed == 0:
			sys.exit("ERROR: 0 samples finished, try increasing how long each sample is run.")
	if params['use_phenos']:
		map_to_phenos(params, attractors, node_name_to_num)
	return attractors


def map_to_phenos(params, attractors, node_name_to_num):
	# might be a cleaner place to put this
	# want to return which phenos corresp to which attractors 

	outputs = [node_name_to_num[params['phenos']['outputs'][i]] for i in range(len(params['phenos']['outputs']))]
	for k in attractors.keys():
		if k=='oscillates':
			attractors[k]['pheno'] ='oscillates'
		else:
			attractors[k]['pheno'] = ''
			for i in range(len(outputs)):
				attractors[k]['pheno']+=k[outputs[i]-1] # -1 since attractors don't include 0 always OFF node


def add_to_attractors(params, attractors, result, steps_taken=None):
	for i in range(len(result['state'])):
		if result['steady_state'][i]:
			state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			if state not in attractors.keys():
				attractors[state] = {}
				attractors[state]['size']=1
			else:
				attractors[state]['size']+=1
		elif steps_taken is None or steps_taken[i] >= params['max_steps_per_sample']: #if given up on finding a steady state for this sample
			attractors['oscillates']['size'] += 1


def format_state_str(x):
	label=''
	for ele in x:
		if ele == True:
			label+='1'
		else:
			label+='0'
	return label
