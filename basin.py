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

	attractors = {} #size is later normalized sT sum(size)=1
	oscil_bin = [] #put all the samples that don't reach a steady state here
	actual_num_samples = math.ceil(params['num_samples']/params['parallelism'])*params['parallelism']
	if actual_num_samples != params['num_samples']:
		print('WARNING: Due to parallelism, actual number of samples = ',actual_num_samples)
	
	if params['verbose']:
		print("Starting initial laps to find fixed points, using", actual_num_samples, "initial points.")
	for i in range(int(actual_num_samples/params['parallelism'])):
		p = .5 #prob a given node is off at start
		x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)
		x0[:,0] = 0 #0th node is the always OFF node
		
		if params['use_phenos']:
			if 'statics' in params['phenos'].keys():
				for k in params['phenos']['statics']:
					node_indx = node_name_to_num[k]
					x0[:,node_indx] = params['phenos']['statics'][k]
			
		if params['verbose'] and i%params['print_lap']==0 and i!=0:
			print("At lap",i)


		result = lap.fixed_point_search(params, x0, clause_mapping, num_nodes)
		if params['cupy']: #convert cupy to numpy (this param is added in parse.py)
			result['finished'] = result['finished'].get()
			result['state'] = result['state'].get()

		add_to_attractors(params, attractors, result)
		oscil_bin += list(result['state'][result['finished']==False])

	if params['verbose']: 
		print('Finished initial laps, now specifying',len(oscil_bin),'oscillations.')


	loop=0
	restart_counter=len(oscil_bin)
	orig_steps_per_lap, orig_fraction_per_lap = params['steps_per_lap'], params['fraction_per_lap']
	cutoff=None
	while len(oscil_bin) > 0: 
		if params['verbose'] and loop%params['print_lap']==0 and loop!=0:
			print("At lap",loop,"with",len(oscil_bin),"samples remaining.")

		if restart_counter<0:
			params['steps_per_lap'] = int(params['steps_per_lap']*params['steps_per_lap_gain'])
			restart_counter = len(oscil_bin)
		restart_counter -= params['parallelism'] # decrement by # samples will run

		if len(oscil_bin)*params['fraction_per_lap'] < params['parallelism']:
			params['fraction_per_lap'] = 1 #require all threads finish to allow long oscils to finish
		if len(oscil_bin) < params['parallelism']:
			cutoff = len(oscil_bin)
			oscil_bin += [oscil_bin[-1] for i in range(params['parallelism']-len(oscil_bin))]
		x0 = oscil_bin[:params['parallelism']]
		del oscil_bin[:params['parallelism']]

		if params['debug']:
			assert(cp.all(cp.logical_not(cp.array(x0)[:,0]))) #always OFF node should still be off after running awhile

		result = lap.from_somewhere_in_oscil(params,x0, clause_mapping, num_nodes)

		if params['cupy']: #convert cupy to numpy (this param is added in parse.py)
			result['finished'] = result['finished'].get()
			result['state'] = result['state'].get()
			result['period'] = result['period'].get() # for now not doing anything with period info
		if cutoff is not None:
			result['finished'] = result['finished'][:cutoff]
			result['state'] = result['state'][:cutoff]
			result['period'] = result['period'][:cutoff]

		add_to_attractors(params, attractors, result)
		oscil_bin += list(result['state'][result['finished']==False])

		if params['debug'] and loop>10**6:
			sys.exit("\nERROR: infinite loop inferred in basin.py\n")

		loop+=1

	params['steps_per_lap'] = orig_steps_per_lap
	params['fraction_per_lap'] = orig_fraction_per_lap

	for s in attractors.keys():
		attractors[s]['size'] /= actual_num_samples

	if params['use_phenos']:
		map_to_phenos(params, attractors, node_name_to_num)

	if params['verbose']:
		print('Finished with',len(attractors),'attractors.')
	return attractors


def map_to_phenos(params, attractors, node_name_to_num):
	# might be a cleaner place to put this
	# want to return which phenos corresp to which attractors 

	outputs = [node_name_to_num[params['phenos']['outputs'][i]] for i in range(len(params['phenos']['outputs']))]
	for k in attractors.keys():
		attractors[k]['pheno'] = ''
		for i in range(len(outputs)):
			attractors[k]['pheno']+=k[outputs[i]-1] # -1 since attractors don't include 0 always OFF node


def add_to_attractors(params, attractors, result):
	for i in range(len(result['state'])):
		if result['finished'][i]:
			state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			if state not in attractors.keys():
				attractors[state] = {}
				attractors[state]['size']=1
			else:
				attractors[state]['size']+=1


def format_state_str(x):
	label=''
	for ele in x:
		if ele == True:
			label+='1'
		else:
			label+='0'
	return label
