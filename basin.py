import itertools, util, math, warnings, sys
import lap
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# note that size of a phenotype or attract refers to its basin size

class Attractor:
	def __init__(self, params, V, attractor_id, period, avg, var):
		self.id = attractor_id
		self.size = 1
		if params['update_rule'] == 'sync' and not params['PBN']['active']:
			self.period = period
		self.avg = avg
		self.var = var
		if params['use_phenos']:
			self.map_to_pheno(params,V) 

	def map_to_pheno(self, params, V):
		outputs = [V['name2#'][params['outputs'][i]] for i in range(len(params['outputs']))]

		if 'inputs' in params.keys():
			inputs = [V['name2#'][params['inputs'][i]] for i in range(len(params['inputs']))]

		self.phenotype = ''
		if 'inputs' in params.keys() and params['use_inputs_in_pheno']:
			assert(not params['PBN']['active']) # TODO: how to def input thresh?
			for i in range(len(inputs)):
				if self.avg[inputs[i]-1] > 0: 
					# -1 since attractors don't include 0 always OFF node
					self.phenotype +='1'
				else:
					self.phenotype +='0'	
			self.phenotype +='|'
		for i in range(len(outputs)): 
			if self.avg[outputs[i]-1] > params['output_thresholds'][i]: 
				# -1 since attractors don't include 0 always OFF node
				self.phenotype +='1'
			else:
				self.phenotype +='0'		 


class Phenotype:
	def __init__(self, attractors, size):
		self.attractors = attractors
		self.size = size

		
class SteadyStates:
	# collects sets of attractors and phenotypes

	def __init__(self, params,V):
		self.attractors = {} 
		self.phenotypes = {}
		self.params = params
		self.V = V

	def add(self, attractor_id, period, avg, var): # id should be unique to each attractor
		if attractor_id not in self.attractors.keys():
			self.attractors[attractor_id] = Attractor(self.params, self.V, attractor_id, period, avg, var)
		else:
			self.attractors[attractor_id].size += 1
			if self.params['update_rule'] != 'sync' or self.params['PBN']['active']:
				self.attractors[attractor_id].avg += avg
				self.attractors[attractor_id].var += var


	def add_attractors(self, result):
		for i in range(len(result['state'])):
			if self.params['update_rule'] != 'sync' or self.params['PBN']['active']:
				finished=True
				period=None
			else:
				finished = result['finished'][i]
				period = result['period'][i]

			attractor_id = format_id_str(result['state'][i][1:]) #skip 0th node, which is the always OFF node
			self.add(attractor_id, period, result['avg'][i][1:], result['var'][i][1:]) #again skip 0th node for avg n var

	def normalize_attractors(self, actual_num_samples):		
		if self.params['update_rule'] in ['async','Gasync'] or self.params['PBN']['active']:
			for s in self.attractors:
				A = self.attractors[s] 
				A.totalAvg = A.avg.copy() #TODO see if this causes err, if so, move to attrs
				A.totalVar = A.var.copy() 
				A.avg/=A.size
				A.var/=A.size

		for s in self.attractors:
			self.attractors[s].size /= actual_num_samples


	def build_phenos(self):
		for k in self.attractors:
			A = self.attractors[k]
			if A.phenotype not in self.phenotypes.keys():
				self.phenotypes[A.phenotype] = Phenotype({k:A},A.size)
			else:
				self.phenotypes[A.phenotype].size += A.size 
				self.phenotypes[A.phenotype].attractors[k] = A


#############################################################################################

def calc_basin_size(params, clause_mapping, V):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils

	#num_nodes is not counting the negative copies
	node_name_to_num = V['name2#']
	node_num_to_name = V['#2name']

	# cast into cupy:
	nodes_to_clauses = cp.array(clause_mapping['nodes_to_clauses'])
	clauses_to_threads = cp.array(clause_mapping['clauses_to_threads']) 
	threads_to_nodes = cp.array(clause_mapping['threads_to_nodes'])

	num_nodes = int(len(node_num_to_name)/2) #i.e. excluding the negative copies
	if params['debug']:
		assert(len(node_num_to_name)%2==0)

	steadyStates = SteadyStates(params, V) 

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
			x0 = get_init_sample(params, node_name_to_num, num_nodes, V)

			# with fixed_points_only = True, will return finished only for fixed points
			# and hopefully move oscillations past their transient phase
			result = lap.transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, fixed_points_only=True)

			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, None, 0)
			steadyStates.add_attractors(result)
			#oscil_bin += list(result['state'][result['period']!=1]) # del this line, added to oscil in run_oscil_extract() jp

		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin, steadyStates, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=True)

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		if params['verbose'] and confirmed_oscils != []: 
			print('Finished finding oscillations, now classifying them.')
		sync_run_oscils(params, confirmed_oscils, steadyStates, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=False)

	elif params['update_rule'] in ['async','Gasync'] or params['PBN']['active']: 
		for i in range(int(actual_num_samples/params['parallelism'])):
			x0 = get_init_sample(params, node_name_to_num, num_nodes, V)

			x_in_attractor = lap.transient(params,x0, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			result = lap.categorize_attractor(params,x_in_attractor, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes)
			cupy_to_numpy(params,result)
			steadyStates.add_attractors(result)

	else:
		sys.exit("ERROR: unrecognized parameter for 'update_rule'")

	steadyStates.normalize_attractors(actual_num_samples)

	if params['use_phenos']:
		steadyStates.build_phenos()

	return steadyStates


def get_init_sample(params, node_name_to_num, num_nodes, V):
	p = .5 #prob a given node is off at start
	x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],num_nodes), p=[p, 1-p]).astype(bool,copy=False)

	
	if 'inputs' in params.keys():
		k = len(params['inputs'])
		actual_num_samples = math.floor(params['parallelism']/(2**k))*2**k
		print('basin: actual num samples:', actual_num_samples)
		input_indices = [V['name2#'][params['inputs'][i]] for i in range(k)]
		input_sets = itertools.product([0,1],repeat=k)
		i=0
		for input_set in input_sets:
			x0[i*actual_num_samples/(2**k):(i+1)*actual_num_samples/(2**k)][input_indices] = input_set
			i+=1
		assert(i==2**k)

	x0[:,0] = 0 #0th node is the always OFF node
	
	if 'init' in params.keys():
		for k in params['init']:
			node_indx = node_name_to_num[k]
			x0[:,node_indx] = params['init'][k]

	return x0

def sync_run_oscils(params, oscil_bin, steadyStates, num_nodes,nodes_to_clauses, clauses_to_threads, threads_to_nodes, transient=False):
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
			steadyStates.add_attractors(result)

	params['steps_per_lap'] = orig_steps_per_lap
	params['fraction_per_lap'] = orig_fraction_per_lap

	if params['debug'] and transient:
		assert(orig_num_oscils == len(confirmed_oscils))
		return confirmed_oscils


def format_id_str(x):
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