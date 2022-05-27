# Organizes dynamics of networks into steady states, including attractors and phenotypes
# and calculates their basin sizes by exhaustive simulation

import itertools, util, math, sys
import lap, plot, param, basin_steady, sandbox
from net import Net
from copy import deepcopy
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# anything going into lap.py (i.e. x0) should be cp, but other matrices should be np
import numpy as np

# note that size of a phenotype or attractor refers to its basin size


#########################################################################################################

def main(param_file):
	# standard simulation: measure the basin size of the steady states and plot them

	params, G = init(param_file)
	steadyStates = measure(params, G)
	#total_avg = [round(s,3) for s in steadyStates.stats['total_avg']]
	#print('basin, total avg=\n',total_avg)#[1:])
	plot.pie(params, steadyStates,G)
	

def measure(params, G, SS0=None):
	# just measure the basin size of steady states

	if not util.istrue(params,'steady_basin'):
		steadyStates = calc_size(params,G,SS0=SS0)
	else:
		assert(not params['PBN']['active']) # haven't implemented steady basin with PBN
		steadyStates = basin_steady.calc_size(params, G,SS0=SS0)
		# currently assuming that exact ratio of A0 for steady basin is irrelv (only transition pr's matter)
		#	this is true if there is only one stationary distribution for the resulting markov chain
	#print("#attractors =",len(steadyStates.attractors))
	return steadyStates


def sequential(param_file,make_plot=True):
	# runs the healthy network, then applies mutations in params and runs the network again with those mutations
	
	params = param.load(param_file)
	mutations = params['mutations'].copy() 
	params['mutations'] = {} # first run without the mutations
	G = Net(params)
	SS_healthy = measure(params, G)
	params['output_img'] = params['output_img'].replace('.png','_healthy.png')
	if make_plot:
		plot.pie(params, SS_healthy,G)

	params['mutations'] = mutations
	G.prepare(params) 
	SS_mutated = measure(params, G, SS0=SS_healthy)
	params['output_img'] = params['output_img'].replace('_healthy.png','_mutated.png')
	if make_plot:
		plot.pie(params, SS_mutated,G)

	if 'controllers' in params.keys() and len(params['controllers']) > 0:
		params['mutations'] = {}
		for k in mutations:
			if k not in params['controllers'].keys(): # since controller takes precedence
				params['mutations'][k] = mutations[k]
		for k in params['controllers']:
			params['mutations'][k] = params['controllers'][k]
		G.prepare(params) 
		SS_controlled = measure(params, G, SS0=SS_mutated)
		params['output_img'] = params['output_img'].replace('_mutated.png','_controlled.png')
		if make_plot:
			plot.pie(params, SS_controlled,G)
		return SS_healthy, SS_mutated, SS_controlled 

	else:
		return SS_healthy, SS_mutated


def init(param_file):
	params = param.load(param_file)
	G = Net(params)
	return params, G

#########################################################################################################

class Attractor:
	def __init__(self, params, G, attr_data):
		# attr_data mandatory keys: id, state, size, avg
		# 			 optional keys: period, A0

		self.id = attr_data['id']
		self.state = attr_data['state'] # this is ONE set in the attractor (even if the attractor actually oscilaltes)
		self.size = attr_data['size']

		self.avg = attr_data['avg']
		self.totalAvg = attr_data['avg'].copy() #won't be normalized so is really a sum

		if 'period' in attr_data:
			self.period = attr_data['period']
		if 'A0' in attr_data:
			self.A0s={}
			for a0 in attr_data['A0']:
				if a0 not in self.A0s:
					self.A0s[a0]=1 
				else:
					self.A0s[a0]+=1

		if util.istrue(params,'use_phenos'):
			self.map_to_pheno(params,G) 

	def map_to_pheno(self, params, G):
		outputs = G.output_indices()

		if 'inputs' in params.keys():
			inputs = G.input_indices()

		self.phenotype = ''
		if 'inputs' in params.keys() and params['use_inputs_in_pheno']:
			thresh=.5
			for i in range(len(inputs)):
				if self.avg[inputs[i]] > thresh: 
					self.phenotype +='1'
				else:
					self.phenotype +='0'	
			self.phenotype +='|'

		for i in range(len(outputs)): 
			assert(len(outputs)==len(params['output_thresholds'])) # otherwise number of outputs != number of output thresholds!
			if float(self.avg[outputs[i]]) > params['output_thresholds'][i]: 
				self.phenotype +='1'
			else:
				self.phenotype +='0'
	
	def __str__(self):		 
		return str(self.id) # note that id is only ONE state in the attractor

class Phenotype:
	def __init__(self, attractors, size):
		self.attractors = attractors # {}
		self.size = size
		self.inputs = None 
		self.outputs = None 

	def __str__(self):
		return str(self.size)

		
class SteadyStates:
	# collects sets of attractors and phenotypes

	def __init__(self, params,G):
		self.attractors = {} 
		self.attractor_order = [] # attractor ids in order 
		self.phenotypes = {}
		stat_names = ['total_avg','windowed_var','total_var']
		self.stats = {k:np.zeros(G.n, dtype=float) for k in stat_names}
		self.params = params
		self.G = G

		# see build_A0()
		self.A0 = None
		self.A0_source = None
		self.Aweights = None

	def phenos_str(self):
		s=''
		for k in self.phenotypes:
			if s!='':
				s+=', '
			s+=k+':'+str(self.phenotypes[k].size)
		return s

	def add_attractors(self, result, A0s=None):

		if self.params['map_from_A0']:
			assert(A0s is not None) # i could be wrong here \:
			assert(len(result['state'])==len(A0s))

		assert(type(result['state']) is np.ndarray) 

		if self.params['precise_oscils']:
			finished_indx = (result['finished']==True)
			for k in result:
				result[k] = result[k][finished_indx]

		unique, indx, inverse, counts = np.unique(result['state'],axis=0,return_index=True,return_inverse=True, return_counts=True)
		# usage:
		# 	unique, indx, inverse, counts = np.unique(x)
		# 	x[indx] == unique
		# 	unique[inverse] = x
		attractor_ids = [format_id_str(unique[i]) for i in range(len(unique))]		

		avgs = np.zeros((len(unique),self.G.n)) # this used to not be done for sync, unclear why tho
		if self.params['map_from_A0']:
			mapped_A0s = [[] for _ in range(len(unique))]
		for i in range(len(inverse)):
			avgs[inverse[i]] += result['avg'][i]
			if self.params['map_from_A0']:
				mapped_A0s[inverse[i]]+=[format_id_str(A0s[i])]
		for i in range(len(avgs)):
			avgs[i]/=counts[i]
	
		for i in range(len(unique)):
			attr_data = {'id':attractor_ids[i], 'state':unique[i], 'size':counts[i], 'avg':avgs[i]}
			if 'period' in result:
				attr_data['period'] = result['period'][indx][i]
			if self.params['map_from_A0']:	
				attr_data['A0'] = mapped_A0s[i]
			self.attractors[attractor_ids[i]] = Attractor(self.params, self.G, attr_data)


	def normalize_attractors(self):		
		if self.params['update_rule'] in ['async','Gasync'] or util.istrue(self.params,['PBN','active']) or util.istrue(self.params,['skips_precise_oscils']):
			for s in self.attractors:
				A = self.attractors[s] 
				A.totalAvg = A.avg.copy() 
				A.avg/=A.size 

		for s in self.attractors:
			self.attractors[s].size /= self.params['num_samples']


	def build_phenos(self):
		for k in self.attractors:
			A = self.attractors[k]
			if A.phenotype not in self.phenotypes.keys():
				self.phenotypes[A.phenotype] = Phenotype({k:A},A.size)
				if '|' in A.phenotype:
					parts = A.phenotype.split('|')
					inpts, outpts = parts[0],parts[1]
					self.phenotypes[A.phenotype].inputs = inpts
					self.phenotypes[A.phenotype].outputs = outpts 
			else:
				self.phenotypes[A.phenotype].size += A.size 
				self.phenotypes[A.phenotype].attractors[k] = A


	def update_stats(self, result):
		self.stats['total_avg'] += result['avg_total']
		self.stats['total_var'] += result['var_total']
		self.stats['windowed_var'] += result['windowed_var']
		self.stats['avg_std_in_time'] = result['avg_std_in_time']
		self.stats['std_btwn_threads'] = result['std_btwn_threads']

	def normalize_stats(self):
		reps = int(self.params['num_samples']/self.params['parallelism'])
		if reps>1:
			assert(0) # can rm, just be aware that avg is tech wrong (rather it is an avg of an avg)
			for k in self.stats.keys():
				self.stats[k] /= reps

	def renormz_by_A0(self,SS0):
		# normalize the size of each attractor by the weight of its initial attractor
		total=0
		#assert(self.params['update_rule']=='sync') # else check normzn before using, but should be ok
		#print('basin:',self.attractors.keys(),SS0.Aweights.keys())

		#print('SS0:', [(x[:4],round(SS0.Aweights[x],3)) for x in SS0.Aweights.keys()])
		#print('\nnewSS:', [(x[:4],round(self.Aweights[x],3)) for x in self.Aweights.keys()])
		#assert(0) 
		Aweights = SS0.Aweights
		assert(math.isclose(sum([A.size for A in self.attractors.values()]),1))
		#assert(math.isclose(sum([A for A in Aweights.values()]),1)) # may not sum to 1 if for ex inputs where shuffled
		for A in self.attractors.values():
			assert(A.size <= 1)
			incoming_weight=0
			for k in A.A0s:
				incoming_weight += Aweights[k] #*A.A0s[k] .
			A.size = incoming_weight
			total += A.size 

		if not math.isclose(total,1):
			print('\nbasin.renormz_by_A0 is not correct! total=',total) # curr occurs w A.A0s[k]>1
			assert(0)


	def build_A0(self, SS0=None):
		self.attractor_order = list(self.attractors.keys())
		A_ids = [self.attractors[k].id for k in self.attractor_order]# attractors in ORDERED list, like def within SSs
		self.A0 = cp.array([[int(a) for a in A] for A in A_ids])
		self.A0_source = self.A0.copy()
		if SS0 is not None:
			self.renormz_by_A0(SS0)
		# in case A.size is altered by SS0, calculate Aweights after any renormz
		self.Aweights = {A.id:A.size for A in self.attractors.values()}
		assert(math.isclose(1,sum(self.Aweights.values())))

	def shuffle_A0_inputs(self,params,G):
		# only called from steady basin (maybe move it there?)
		# prob could be more effic with some more np
		# normalizes Aweights accordingly
		x0=[]
		input_indices = G.input_indices()
		input_sets = G.get_input_sets()
		ref_A0=[] # later used to map which x0 come from which A0
		for A in self.A0:
			ref_A0 += [[int(a) for a in A] for i in range(len(input_sets))]
			for input_set in input_sets:
				state = [int(a) for a in A]
				for k in range(len(input_indices)):
					state[input_indices[k]] = input_set[k]
				x0+=[state]
		params['num_samples']=params['parallelism']=len(x0)
		x0=cp.array(x0)
		for k in params['init']:
			indx = G.nodeNums[k]
			x0[:,indx] = params['init'][k]
		assert(len(x0)==len(ref_A0))
		self.A0 = cp.array(x0) 
		self.A0_source = cp.array(ref_A0)
		self.Aweights = {k:self.Aweights[k]/len(input_sets) for k in self.Aweights.keys()}

	def __str__(self):
		s="Attractors of SS=\n"
		s+= str([A.avg for A in self.attractors.values()])
		return s

##################################### One Basin #############################################################

def calc_size(params, G, SS0=None):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils
	# SS0 is an optional initial steady state to start from (typically from a previous run)

	steadyStates = SteadyStates(params, G) 

	params['map_from_A0'] = False
	if SS0 is not None:
		# poss rename x0/A0 to make more clear (they are only diff if input shuffle occured)
		x0, A0, Aweights = SS0.A0, SS0.A0_source, SS0.Aweights
		x0 = apply_setting_to_x0(params, G, x0)

		params['map_from_A0'] = True
		if not (len(x0)==params['num_samples']==params['parallelism']): # or change implementation
			params['num_samples']=params['parallelism']=len(x0)
			if params['verbose']:
				print('\nWARNING: basin.py changing parallelism and num_samples parameters to match SS0')

	params['precise_oscils']= (params['update_rule'] == 'sync' and not params['map_from_A0'] and not util.istrue(params,['PBN','active']) and not util.istrue(params,['skips_precise_oscils']))
	# using from A0 with sync causes complications like when to end oscil (since x0 is no longer nec in oscil)

	if params['precise_oscils']:
		oscil_bin = [] #put all the samples that are unfinished oscillators
		confirmed_oscils = [] #put all samples that have oscillated back to their initial state 
		
		# FIXED POINTS & EASY OSCILS
		for i in range(int(params['num_samples']/params['parallelism'])):
			if not params['map_from_A0']:
				x0 = get_init_sample(params, G)
			# with fixed_points_only = True, will return finished only for fixed points
			# and help move oscillations past their transient phase
			result = lap.transient(params,x0, G, fixed_points_only=True)
			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, None, 0)
			steadyStates.add_attractors(result)

		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin,steadyStates, G,transient=True)

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		sync_run_oscils(params, confirmed_oscils, steadyStates, G, transient=False)

	else: # async, Gasync, stoch, starting from A0,...increasingly becoming the main method due to simplicity

		for i in range(int(params['num_samples']/params['parallelism'])):
			if not params['map_from_A0']:
				x0 = get_init_sample(params, G)
				A0=None
			if util.istrue(params,['PBN','active']) and util.istrue(params,['PBN','float']):
				x0 = recast_Fmapd_and_x0(params, G, x0)
			x_in_attractor = lap.transient(params, x0, G)
			result = lap.categorize_attractor(params, x_in_attractor, G)
			cupy_to_numpy(params,result)
			steadyStates.update_stats(result)
			steadyStates.add_attractors(result, A0s=A0)
		steadyStates.normalize_stats()

	steadyStates.normalize_attractors()
	steadyStates.build_A0(SS0) #also renormalizes if SS0!=None
	if util.istrue(params,'use_phenos'):
		steadyStates.build_phenos()

	return steadyStates


################################# SYNC OSCILS SHENANIGANS ########################################

def sync_run_oscils(params, oscil_bin, steadyStates, G, transient=False):
	if oscil_bin == [] or oscil_bin is None:
		return
	restart_counter=orig_num_oscils=len(oscil_bin)
	orig_steps_per_lap, orig_fraction_per_lap = params['steps_per_lap'], params['fraction_per_lap']	
	loop=0
	confirmed_oscils = [] #only used for transient
	while len(oscil_bin) > 0:
		x0, cutoff, restart_counter = run_oscil_init(params, oscil_bin, restart_counter, loop)
		if transient:
			result = lap.transient(params,x0, G, fixed_points_only=False)
		else:
			result = lap.categorize_attractor(params,x0, G)
		cupy_to_numpy(params,result)
		result, loop = run_oscils_extract(params, result, oscil_bin, cutoff, loop)
		
		if transient:
			confirmed_oscils += list(result['state'][result['finished']==True])
		else:
			steadyStates.add_attractors(result) 

	params['steps_per_lap'] = orig_steps_per_lap
	params['fraction_per_lap'] = orig_fraction_per_lap

	if transient:
		if params['debug']:
			assert(orig_num_oscils == len(confirmed_oscils))
		return confirmed_oscils


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


################################# MISC ########################################

def get_init_sample(params, G):
	
	p = .5 #prob a given node is off at start
	x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[p, 1-p]).astype(bool,copy=False)
	
	if 'init' in params.keys():
		for k in params['init']:
			node_indx = G.nodeNums[k]
			x0[:,node_indx] = params['init'][k]

	if 'inputs' in params.keys():
		input_indices = G.input_indices()
		input_sets = G.get_input_sets()
		i=0
		for input_set in input_sets:
			x0[int(i*params['parallelism']/(2**len(params['inputs']))):int((i+1)*params['parallelism']/(2**len(params['inputs']))),input_indices] = cp.array(input_set)
			i+=1
		assert(i==2**len(params['inputs']))

	#print('temp in basin:')
	#x0[0] = cp.array([0,0,1,0,1,0,1])
	return x0

def format_id_str(x):
	return ''.join(map(str,x.astype(int)))

def cupy_to_numpy(params,result):
	# if using cupy, not extracting these from GPU will lead to SIGNIFICANT slow down
	if params['cupy']:
		for k in result.keys():
			result[k]=result[k].get()

def recast_Fmapd_and_x0(params, G,x0):
	# for PBN!
	# adds ON and OFF nodes at indices n and n+1
	# and reshapes arrays accordingly
	# also need to init them properly...maybe move to after x0 is picked?
	# and while we're adding ON and OFF nodes, also use for input nodes (rather than a self-loop)
	#	or maybe just for all self-loop nodes? even if not considered input

	# TODO: this should be only for floats actually, same for changes in net.py

	OFF_indx=0
	ON_indx=G.n
	x0[:,0] = 0 # make sure OFF node is off 

	# usually to make a sq matrix, use repeats of clauses, nodes ect since 1*1=1, 0*0=0 ect
	# but this doesn't hold for PBN, so need to use ON/OFF nodes instead
	#print('\nbefore:\n',G.Fmapd['threads_to_nodes'].astype(int))
	G.Fmapd['nodes_to_clauses'] = replace_row_duplicates(G.Fmapd['nodes_to_clauses'],ON_indx,1) # used in an AND gate, so turn duplicates ON
	for i in range(len(G.Fmapd['clauses_to_threads'])): # laaazy, should fix replace_row_duplicates() instead
		G.Fmapd['clauses_to_threads'][i] = replace_row_duplicates(G.Fmapd['clauses_to_threads'][i],OFF_indx,1) # used in an OR gate, so turn duplicates OFF
	
	# jp SHOULDN'T change threads_to_nodes, but will see
	#for i in range(len(G.Fmapd['threads_to_nodes'])):
	#	G.Fmapd['threads_to_nodes'][i] = replace_row_duplicates(G.Fmapd['threads_to_nodes'][i],OFF_indx,1) # used in an OR gate, so turn duplicates OFF
	#print('\nafter:\n',G.Fmapd['threads_to_nodes'].astype(int))
	
	return x0

def replace_row_duplicates(a,val,axis):
	unique = cp.sort(a,axis=axis)
	duplicates = unique[:,  1:] == unique[:, :-1]
	unique[:, 1:][duplicates] = val 
	return unique


def apply_setting_to_x0(params, G, x0):
	# in case params have changed, for example sequential run, need to apply mutations to x0
	for k in params['mutations']:
		x0[:,G.nodeNums[k]] = params['mutations'][k]
	for k in params['init']:
		x0[:,G.nodeNums[k]] = params['init'][k]

	return x0 

#############################################################################################


if __name__ == "__main__":
	if len(sys.argv) not in [2,3]:
		sys.exit("Usage: python3 basin.py PARAMS.yaml [seq]")

	if len(sys.argv) == 3: 
		if sys.argv[2] != 'seq':
			sys.exit("Only optional arg is 'seq'. Usage: python3 basin.py PARAMS.yaml [seq]")
		else:
			sequential(sys.argv[1])

	else:
		main(sys.argv[1])