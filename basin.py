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
	oscil_bin = [] #put all the samples that are unfinished oscillators
	confirmed_oscils = [] #put all samples that have oscillated back to their initial state 
	actual_num_samples = math.ceil(params['num_samples']/params['parallelism'])*params['parallelism']
	if actual_num_samples != params['num_samples']:
		print('WARNING: Due to parallelism, actual number of samples = ',actual_num_samples)
	
	if params['update_rule'] == 'sync' and not params['PBN']['active']:

		# FIXED POINTS & EASY OSCILS
		if params['verbose']:
			print("Starting fixed point search, using", actual_num_samples, "sample initial points.")
		for i in range(int(actual_num_samples/params['parallelism'])):
			x0 = get_init_sample(params, node_name_to_num, num_nodes)

			# with fixed_points_only = True, will return finished only for fixed points
			# and hopefully move oscillations past their transient phase
			result = lap.transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, fixed_points_only=True)

			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, None, 0)
			add_to_attractors(params, attractors, result) 
			oscil_bin += list(result['state'][result['period']!=1])

		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin, attractors, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=True)

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		if params['verbose'] and confirmed_oscils != []: 
			print('Finished finding oscillations, now classifying them.')
		sync_run_oscils(params, confirmed_oscils, attractors, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=False)


	elif params['update_rule'] in ['async','Gasync'] or params['PBN']['active']: 
		for i in range(int(actual_num_samples/params['parallelism'])):
			x0 = get_init_sample(params, node_name_to_num, num_nodes)

			x_in_attractor = lap.transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			result = lap.categorize_attractor(params,x_in_attractor, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)

			cupy_to_numpy(params,result)

			add_to_attractors(params, attractors, result)

		for A in attractors.keys():
			
			attractors[A]['totalAvg']=attractors[A]['avg'].copy()
			attractors[A]['totalVar']=attractors[A]['var'].copy()
			attractors[A]['avg']/=attractors[A]['size']
			attractors[A]['var']/=attractors[A]['size']

	else:
		sys.exit("ERROR: unrecognized parameter for 'update_rule'")



	for s in attractors.keys():
		attractors[s]['size'] /= actual_num_samples

	if params['use_phenos']:
		map_to_phenos(params, attractors, node_name_to_num)

	if params['verbose']:
		print('Finished with',len(attractors),'attractors.')

	return attractors, attr2pheno(params, attractors)


def get_init_sample(params, node_name_to_num, num_nodes):
	p = .5 #prob a given node is off at start
	x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)
	x0[:,0] = 0 #0th node is the always OFF node
	
	if params['use_phenos']:
		if 'init' in params['phenos'].keys():
			for k in params['phenos']['init']:
				node_indx = node_name_to_num[k]
				x0[:,node_indx] = params['phenos']['init'][k]

	return x0

def sync_run_oscils(params, oscil_bin, attractors, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=False):
	if oscil_bin == [] or oscil_bin is None:
		return
	restart_counter=orig_num_oscils=len(oscil_bin)
	orig_steps_per_lap, orig_fraction_per_lap = params['steps_per_lap'], params['fraction_per_lap']	
	loop=0
	confirmed_oscils = [] #only used for transient
	while len(oscil_bin) > 0: 
		x0, cutoff, restart_counter = run_oscil_init(params, oscil_bin, restart_counter, loop)
		if transient:
			result = lap.transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, fixed_points_only=False)
		else:
			result = lap.categorize_attractor(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
		cupy_to_numpy(params,result)
		result, loop = run_oscils_extract(params, result, oscil_bin, cutoff, loop)
		
		if transient:
			confirmed_oscils += list(result['state'][result['finished']==True])
		else:
			add_to_attractors(params, attractors, result)

	params['steps_per_lap'] = orig_steps_per_lap
	params['fraction_per_lap'] = orig_fraction_per_lap

	if params['debug'] and transient:
		assert(orig_num_oscils == len(confirmed_oscils))
		return confirmed_oscils


def attr2pheno(params, attractors):
	if not params['use_phenos']:
		return None
	phenos = {}
	for k in attractors:
		A = attractors[k]
		if A['pheno'] not in phenos.keys():
			phenos[A['pheno']] = {'attractors':{k:A},'size':A['size']}
		else:
			phenos[A['pheno']]['size'] += A['size']
			phenos[A['pheno']]['attractors'][k] = A
	return phenos


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
	# only fps means only add the fixed points, i.e. where period = 1
	for i in range(len(result['state'])):
		if params['update_rule'] == 'sync' and not params['PBN']['active']:
			if result['finished'][i]:
				state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
				if state not in attractors.keys():
					attractors[state] = {'state':state}
					attractors[state]['size']=1
					attractors[state]['period'] = result['period'][i] #again skip 0th node
					for k in ['avg','var']:
						attractors[state][k] = result[k][i][1:] #again skip 0th node
				else:
					attractors[state]['size']+=1
		else:
			state = format_state_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			if state not in attractors.keys():
				attractors[state] = {'state':state}
				attractors[state]['size']=1
				attractors[state]['avg'] = result['avg'][i][1:] #again skip 0th node
				attractors[state]['var'] = result['var'][i][1:] 
			else:
				attractors[state]['size']+=1
				attractors[state]['avg'] += result['avg'][i][1:] 
				attractors[state]['var'] += result['var'][i][1:] 


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

	return x0, cutoff, restart_counter

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