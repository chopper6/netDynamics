import itertools, util, math, warnings, sys
import runNet
CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


def calc_basin_size(params, clause_mapping, node_mapping, num_nodes):
	#num_nodes is not counting the negative copies
	node_name_to_num = node_mapping['node_name_to_num']
	node_num_to_name = node_mapping['node_num_to_name']

	# LATER: clause index might change btwn laps of runNet (ex dyn topo, PBNs)

	attractors = {'oscillates':{'size':0}} #oscillates means anything not a fixed point
		#size is normalized sT sum(size)=1
	
	if params['exhaustive']:
		assert(0) #have not updated this for some time
		# TODO: add parallelism, in fact need to check exhaustive WITHIN laps loop
		#	maybe add file runLaps, sep from runNet -> runInstance?

		X0 = [0]+itertools.product([0,1], repeat=num_nodes-1) #first index is the always OFF node
		for x0 in X0:
			x0 = cp.array(x0,dtype=bool)
			result = runNet.from_init_val(params,clause_index,x0,num_neg_nodes)
			add_to_attractors(params, attractors, result)
		for s in attractors.keys():
			attractors[s]['size'] /= math.pow(2,num_nodes)

	else:
		attempt_count, samples_processed = cp.zeros((params['parallelism'])), 0
		#clause_index = util.copy_to_larger_dim(clause_index,params['parallelism'])
		p = .5 #prob a given node is off at start
		x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)
		x0[:,0] = 0 #0th node is the always OFF node
		
		for i in range(params['max_laps']):

			result = runNet.from_init_val(params,x0, clause_mapping, num_nodes)

			if params['cupy']: #convert cupy to numpy (this param is added in parse.py)
				result['steady_state'] = result['steady_state'].get()
				result['state'] = result['state'].get()

			attempt_count += 1-result['steady_state'] #i.e. counting how many times failed 
			samples_processed += cp.sum(result['steady_state']) + cp.sum(attempt_count>params['max_lap_reruns'])
			add_to_attractors(params, attractors, result,attempt_count)
			attempt_count[attempt_count > params['max_lap_reruns']]=0 #give up on samples that have already run too many laps without reaching steady state
			
			if samples_processed >= params['num_samples']:
				break

			if i==params['max_laps']-1:
				warnings.warn("\nFew samples than desired run, since reached 'max_laps' early.\n")

			x0 = result['state']

			# add new initial conditions to entries that finished
			for j in range(len(attempt_count)):
				if attempt_count[j]==0: #done or given up, so refill w new sample
					x0[j]= cp.random.choice(a=[0,1], size=(num_nodes), p=[p, 1-p]).astype(bool,copy=False)

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
				attractors[k]['pheno']+=k[outputs[i]]

	#sort attractors by the phenotypes
	# if state string > |'oscillated'|, throws error. Besides I'm not sure this line does what I want...
	#attractors_sorted_by_pheno = sorted(attractors.keys(), key=lambda tup: [tup[i] for i in outputs])


def add_to_attractors(params,attractors, result, attempt_count):
	for i in range(len(result['state'])): 
		if result['steady_state'][i]:
			state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			if state not in attractors.keys():
				attractors[state] = {}
				attractors[state]['size']=1
			else:
				attractors[state]['size']+=1
		elif attempt_count[i] > params['max_lap_reruns']: #if given up on finding a steady state for this sample
			attractors['oscillates']['size'] += 1


def format_state_str(x):
	label=''
	for ele in x:
		if ele == True:
			label+='1'
		else:
			label+='0'
	return label
