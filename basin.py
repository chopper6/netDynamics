import itertools, util, math, warnings, sys
import lap
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


def calc_basin_size(params, clause_mapping, node_mapping):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils

	#num_nodes is not counting the negative copies
	node_name_to_num = node_mapping['name_to_num']
	node_num_to_name = node_mapping['num_to_name']

	# cast into cupy:
	nodes_to_clauses = cp.array(clause_mapping['nodes_to_clauses'])
	clauses_to_threads = cp.array(clause_mapping['clauses_to_threads']) 
	threads_to_nodes = cp.array(clause_mapping['threads_to_nodes'])

	num_nodes = int(len(node_num_to_name)/2) #i.e. excluding the negative copies
	if params['debug']:
		assert(len(node_num_to_name)%2==0)

	attractors = {} #size is later normalized sT sum(size)=1
	oscil_bin = [] #put all the samples that don't reach a steady state here
	actual_num_samples = math.ceil(params['num_samples']/params['parallelism'])*params['parallelism']
	if actual_num_samples != params['num_samples']:
		print('WARNING: Due to parallelism, actual number of samples = ',actual_num_samples)
	
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
			if attractors[k]['avg'][outputs[i]-1] > params['phenos']['output_thresholds'][i]:
				# -1 since attractors don't include 0 always OFF node
				attractors[k]['pheno']+='1'
			else:
				attractors[k]['pheno']+='0'
			# old version without thresh:
			#attractors[k]['pheno']+=k[outputs[i]-1] # -1 since attractors don't include 0 always OFF node



def add_to_attractors(params, attractors, result):
	for i in range(len(result['state'])):
		if result['finished'][i]:
			state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			if state not in attractors.keys():
				attractors[state] = {'state':state}
				attractors[state]['size']=1
				attractors[state]['period'] = result['period'][i]
				attractors[state]['avg'] = result['avg'][i][1:] #again skip 0th node
			else:
				attractors[state]['size']+=1

def format_state_str(x):
	label=''
	for ele in x:
		if ele == True:
			label+='1'
		elif ele == False:
			label+='0'
		else:
			label+=str(ele) #to handle int strs
	return label


def run_oscil_init(params, oscil_bin, restart_counter, loop):
	if params['verbose'] and loop%params['print_lap']==0 and loop!=0:
			print("\tAt lap",loop,"with",len(oscil_bin),"samples remaining.")

	if restart_counter<0:
		params['steps_per_lap'] = int(params['steps_per_lap']*params['steps_per_lap_gain'])
		restart_counter = len(oscil_bin)
	restart_counter -= params['parallelism'] # decrement by # samples will run

	if len(oscil_bin)*params['fraction_per_lap'] < params['parallelism']:
		params['fraction_per_lap'] = 1 #require all threads finish to allow long oscils to finish
	
	cutoff=None
	if len(oscil_bin) < params['parallelism']:
		cutoff = len(oscil_bin)
		oscil_bin += [oscil_bin[-1] for i in range(params['parallelism']-len(oscil_bin))]
	x0 = oscil_bin[:params['parallelism']]
	del oscil_bin[:params['parallelism']]

	if params['debug']:
		assert(cp.all(cp.logical_not(cp.array(x0)[:,0]))) #always OFF node should still be off after running awhile

	return x0, cutoff

def run_oscils_extract(params, result, oscil_bin, cutoff, loop):
	if cutoff is not None:
		for s in ['finished','state','period','avg']:
			if s in result.keys():
				result[s] = result[s][:cutoff]

	oscil_bin += list(result['state'][result['finished']==False])
	if params['debug'] and loop>10**6:
		sys.exit("\nERROR: infinite loop inferred in basin.py\n")

	loop+=1

	return result, loop


def cupy_to_numpy(params,result):
	# if using cupy, not extracting these from GPU will lead to SIGNIFICANT slow down
	if params['cupy']:
		for k in result.keys():
			result[k]=result[k].get()