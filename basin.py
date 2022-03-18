import itertools, util, math, sys
import lap, plot, param, basin_steady, sandbox
from net import Net
from copy import deepcopy
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# anything going into lap.py (i.e. x0) should be cp, but other matrices should be np
import numpy as np


# note that size of a phenotype or attractor refers to its basin size

# shuffling attractors for steady basin runs out of mem (Grieco, Gasync, 10^5 samples)
#		could implement diff num_samples vs parallelism and/or check if any of the A0 only diff w.r.t inputs (and merge them if so)

# TODO: clean up how var is handled

#########################################################################################################

def main(param_file):

	params, G = init(param_file)
	if params['PBN']['active']:
		print("\nCustom PBN run\n")
		steadyStates = sandbox.test_PBN(params, G)
	else:
		steadyStates = measure(params, G)
	plot.pie(params, steadyStates,G)
	

def measure(params, G, A0=None, Aweights=None):
	if not util.istrue(params,'steady_basin'):
		steadyStates = calc_basin_size(params,G,A0=A0,Aweights=Aweights)
	else:
		# curr assuming that exact ratio of A0 for steady basin is irrelv (only transition pr's matter)
		steadyStates = basin_steady.calc_basin_size(params, G,A0=A0,Aweights=Aweights)
	#print("#attractors =",len(steadyStates.attractors))
	return steadyStates


#########################################################################################################

class Attractor:
	def __init__(self, params, G, attractor_id, state, period, avg, var, A0):
		self.id = attractor_id
		self.state = state
		self.size = 1
		if params['precise_oscils']:
			self.period = period
		self.avg = avg
		self.var = var
		self.totalAvg = avg #won't be normalized so is really a sum
		self.totalVar = var 

		if A0 is not None:
			self.A0s = {format_id_str(A0):1}
		if util.istrue(params,'use_phenos'):
			self.map_to_pheno(params,G) 

	def map_to_pheno(self, params, G):
		outputs = [G.nodeNums[params['outputs'][i]] for i in range(len(params['outputs']))]

		if 'inputs' in params.keys():
			inputs = G.input_indices(params)

		self.phenotype = ''
		if 'inputs' in params.keys() and params['use_inputs_in_pheno']:
			thresh=0
			if util.istrue(params,['PBN','active']):
				thresh=.5
			for i in range(len(inputs)):
				if self.avg[inputs[i]] > thresh: 
					self.phenotype +='1'
				else:
					self.phenotype +='0'	
			self.phenotype +='|'
		for i in range(len(outputs)): 
			if float(self.avg[outputs[i]]) > params['output_thresholds'][i]: 
				self.phenotype +='1'
			else:
				self.phenotype +='0'		 


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
		stat_names = ['total_avg','ensemble_var','temporal_var','total_var']
		self.stats = {k:np.zeros(G.n, dtype=float) for k in stat_names}
		self.params = params
		self.G = G
		self.A0_counts = {} # counts number of each A0 used, later useful for normalization

		# see build_A0()
		self.A0 = None
		self.A0_source =None
		self.Aweights = None

	def phenos_str(self):
		s=''
		for k in self.phenotypes:
			if s!='':
				s+=', '
			s+=k+':'+str(self.phenotypes[k].size)
		return s

	def add(self, attractor_id, state, period, avg, A0=None,var_t=None): # id should be unique to each attractor
		if attractor_id not in self.attractors.keys():
			self.attractors[attractor_id] = Attractor(self.params, self.G, attractor_id, state, period, avg, var_t, A0)
		else:
			self.attractors[attractor_id].size += 1

			if A0 is not None:
				Astr = format_id_str(A0)
				if Astr in self.attractors[attractor_id].A0s.keys():
					self.attractors[attractor_id].A0s[Astr] += 1 
				else:
					self.attractors[attractor_id].A0s[Astr] = 1 
				if Astr in self.A0_counts.keys():
					self.A0_counts[Astr] += 1 
				else:
					self.A0_counts[Astr] = 1


			if self.params['update_rule'] != 'sync' or util.istrue(self.params,['PBN','active']):
				self.attractors[attractor_id].avg += avg
				if var_t is not None:
					self.attractors[attractor_id].var_t += var_t



	def add_attractors(self, params, result, A0s):
		# map_from_A0 indicates that transition pr from A_0 -> A_f should be measured
		# TODO clean this mess
		# also i suspect this is a bottleneck of sorts

		if params['map_from_A0']:
			#print('while adding A:',len(result['state']),len(A0))
			assert(len(result['state'])==len(A0s))
		for i in range(len(result['state'])):
			if not params['precise_oscils']:
				finished=True
				period=None
			else:
				finished = result['finished'][i]
				period = result['period'][i]

			if finished:
				attractor_id = format_id_str(result['state'][i]) 

				if not params['map_from_A0']:
					a0=None
				else:
					a0 = A0s[i]

				# having 2 fns to add attr is confusing too
				self.add(attractor_id, result['state'][i], period, result['avg'][i], a0)

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

	def map_A_transition_pr(self):
		# build transition matrix (a_ij = pr Ai -> Aj)
		# indexed according to self.attractor_order
		n = len(self.attractor_order)
		T = np.zeros((n,n))
		for i in range(n):
			Ai = self.attractor_order[i]
			found=0
			for j in range(n):
				Aj = self.attractor_order[j]
				if Ai in self.attractors[Aj].A0s.keys():
					T[i,j] = self.attractors[Aj].A0s[Ai] # Aj's A0s[Ai] should be the # times Ai -> Aj
					found=1
			if not found: # A0 did not contain all attractors, so try again before mapping transition pr
				return None, True

		T = np.transpose(T/np.vstack(np.sum(T,axis=1))) # normalize such that sum_j(pr Ai -> Aj) = 1
		return T, False


	def update_stats(self, result):
		self.stats['total_avg'] += result['avg_total']
		self.stats['ensemble_var'] += result['var_ensemble']
		self.stats['temporal_var'] += result['var_time']
		self.stats['total_var'] += result['var_x0']

	def normalize_stats(self):
		reps = int(self.params['num_samples']/self.params['parallelism'])
		if reps>1:
			assert(0) # can rm, just be aware that avg is tech wrong (rather it is an avg of an avg)
			for k in self.stats.keys():
				self.stats[k] /= reps

	def adjust_Aweights(self,A0,Aweights):
		# normalize the size of each attractor by the weight of its initial attractor
		total=0
		assert(self.params['update_rule']=='sync') # else check normzn before using
		
		assert(math.isclose(sum([A.size for A in self.attractors.values()]),1))
		assert(math.isclose(sum([A for A in Aweights.values()]),1))
		for A in self.attractors.values():
			assert(A.size <= 1)
			incoming_weight=0
			for k in A.A0s:
				assert(A.A0s[k]==1) # should be okay if not, just double check
				incoming_weight += Aweights[k]*A.A0s[k]/self.A0_counts[k]
				print('normz:',self.A0_counts[k])
				assert(0) # not fixed yet
			#print("\tAsize adds prod Asize*len(attractors),incoming_weight=",A.size,len(self.attractors),incoming_weight)
			total += A.size 


		if not (total < 1.1 and total > .9): # should really by isclose()
			print('total=',total)
			assert(0)

		total2=0
		for A in self.attractors.values():
			A.size/=total
			total2+=A.size 
		total=total2 
		if not (math.isclose(total,1)):
			print("\nWARNING: basin.adjust_Aweights do not sum to 1!\n") 
		

	def build_A0(self):
		self.attractor_order = list(self.attractors.keys())
		self.A0_source = [self.attractors[k].id for k in self.attractor_order]# attractors in ORDERED list, like def within SSs
		self.A0 = cp.array([[int(a) for a in A] for A in self.A0_source])
		self.Aweights = {A.id:A.size for A in self.attractors.values()}

	def shuffle_A0_inputs(self,params,G):
		# called within calc_basin_size if A0s are passed
		# prob could be more effic with some more np
		x0=[]
		input_sets = G.get_input_sets(params)
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
		ref_A0 = np.array(ref_A0)
		#ref_A0 = cp.tile(cp.array(ref_A0),int(len(x0)/len(A0))).reshape(x0.shape)
		assert(len(x0)==len(ref_A0))
		self.A0 = cp.array(A0) 
		self.A0_source = ref_A0

##################################### One Basin #############################################################

def calc_basin_size(params, G, SS0=None):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils
	# SS0 is an optional initial steady state to start from (typically from a previous run)

	steadyStates = SteadyStates(params, G) 

	params['map_from_A0'] = False
	if SS0 is not None:
		x0, A0, Aweights = SS0.A0, SS0.A0_source, SS0.Aweights
		params['map_from_A0'] = True
		assert(len(x0)==params['num_samples']==params['parallelism']) # otherwise need to change implementation

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
			steadyStates.add_attractors(params, result, None)

		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin,steadyStates, G,transient=True)

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		sync_run_oscils(params, confirmed_oscils, steadyStates, G, transient=False)

	else: # async, Gasync, stoch, starting from A0,...increasingly becoming the main method

		for i in range(int(params['num_samples']/params['parallelism'])):
			if not params['map_from_A0']:
				x0 = get_init_sample(params, G)
			x_in_attractor = lap.transient(params, x0, G)
			result = lap.categorize_attractor(params, x_in_attractor, G)
			cupy_to_numpy(params,result)
			steadyStates.update_stats(result)
			steadyStates.add_attractors(params,result, A0)

	steadyStates.normalize_stats()
	steadyStates.normalize_attractors()
	steadyStates.build_A0_and_weights(SS0) #also renormalizes if SS0!=None
	assert(0) # left of here
	if SS0 is not None:
		steadyStates.adjust_Aweights(A0,Aweights)

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
			steadyStates.add_attractors(params, result,None) 

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
	
	if util.istrue(params,['PBN','active']) and util.istrue(params,['PBN','float']):
		assert(0) #plz explicitly debug the float stuff
		x0 = .5*cp.ones((params['parallelism'],G.n)).astype(float,copy=False)
	else:
		p = .5 #prob a given node is off at start
		x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[p, 1-p]).astype(bool,copy=False)
		
	if 'init' in params.keys():
		for k in params['init']:
			node_indx = G.nodeNums[k]
			x0[:,node_indx] = params['init'][k]

	if 'inputs' in params.keys():
		input_sets = G.get_input_sets(params)
		i=0
		for input_set in input_sets:
			x0[int(i*params['parallelism']/(2**k)):int((i+1)*params['parallelism']/(2**k)),input_indices] = cp.array(input_set)
			i+=1
		assert(i==2**k)

	return x0

def format_id_str(x):
	return ''.join(map(str,x.astype(int)))

def cupy_to_numpy(params,result):
	# if using cupy, not extracting these from GPU will lead to SIGNIFICANT slow down
	if params['cupy']:
		for k in result.keys():
			result[k]=result[k].get()



#############################################################################################


if __name__ == "__main__":
	if len(sys.argv) not in [2,3]:
		sys.exit("Usage: python3 basin.py PARAMS.yaml")
	
	if len(sys.argv) == 3:
		if sys.argv[2] != 'repeat':
			print("Unknown 2nd param...",sys.argv[2])
		repeat_test(sys.argv[1])
	else:
		main(sys.argv[1])