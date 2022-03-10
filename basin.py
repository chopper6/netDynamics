import itertools, util, math, sys
import lap, plot, param
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
		steadyStates = test_PBN(params, G)
	else:
		steadyStates = measure(params, G)
	plot.pie(params, steadyStates,G)
	

def measure(params, G, A0=None, Aweights=None):
	if not util.istrue(params,'steady_basin'):
		steadyStates = calc_basin_size(params,G,A0=A0,Aweights=Aweights)
	else:
		# curr assuming that exact ratio of A0 for steady basin is irrelv (only transition pr's matter)
		steadyStates = calc_steady_basin(params, G,A0=A0,Aweights=Aweights)
	#print("#attractors =",len(steadyStates.attractors))
	return steadyStates


def test_PBN(params, G):
	mult=1
	orig_num_samples = params['num_samples']
	SS = calc_basin_size(params,G)
	A0, Aweights =  get_A0_and_weights(SS)
	print("Stoch #A =",np.array(A0).shape)
	A0 = np.repeat(A0,mult)
	params['PBN']['active'] = params['PBN']['flip_pr'] = 0
	params['parallelism'] = params['num_samples'] = len(A0)
	SS = calc_basin_size(params,G,A0=A0)
	A0, Aweights =  get_A0_and_weights(SS)
	A0 = np.array(A0)
	print("Stoch after det transient =",np.array(A0).shape)

	params['parallelism'] = params['num_samples'] = orig_num_samples
	SS_det = calc_basin_size(params,G)
	A0, Aweights =  get_A0_and_weights(SS_det)
	print("Det #A =",np.array(A0).shape)
	#A0 = np.vstack([A0 for _ in range(mult)])
	#print(np.array(A0).shape)
	params['parallelism'] = params['num_samples'] = len(A0)
	SS_det = calc_basin_size(params,G,A0=A0)
	A0, Aweights =  get_A0_and_weights(SS_det)
	A0 = np.array(A0)
	print("Det after det transient =",np.array(A0).shape)

	return SS



def repeat_test(param_file):
	print("\nRUNNING REPEAT TEST\n")
	params, G = init(param_file)
	repeats=20

	if not params['steady_basin']:
		base = calc_basin_size(params,G).attractors
	else:
		base = calc_steady_basin(params, G).attractors
	As=[base]
	max_dist=0
	for k in range(repeats):
		print('starting repeat #',k+1,'/',repeats)
		if not params['steady_basin']:
			repeat = calc_basin_size(params,G).attractors
		else:
			repeat=calc_steady_basin(params, G).attractors
		for A in As:
			d = dist(A,repeat)
			#print('\nDISTANCE =',d,'\n')
			max_dist = max(d,max_dist)
		As += [repeat]
	print("\n\nMax pairwise distance among",repeats,"repeats =",max_dist,'\n\n')


def dist(A1,A2):
	d1=0
	for k in A1:
		if k in A2:
			d1+=abs(A1[k].size-A2[k].size)
		else:
			d1+=A1[k].size 
	d2=0
	for k in A2:
		if k in A1:
			d2+=abs(A1[k].size-A2[k].size)
		else:
			d2+=A2[k].size

	return (d1+d2)/2


#######################################################################################################

def init(param_file):
	params = param.load(param_file)
	G = Net(model_file=params['model_file'],debug=params['debug'])
	G.prepare_for_sim(params)
	return params, G

def temp_print_periodic(SS, G,params):
	for k in SS.attractors:
		A=SS.attractors[k]
		#print("\nnext A:")
		for i in range(len(A.avg)):
			inputs = params['inputs']#['EGFR_stimulus', 'TGFBR_stimulus', 'FGFR3_stimulus', 'DNA_damage']
			input_state = [A.avg[G.nodeNums[inpt]] for inpt in inputs]
			if A.avg[i] not in [0,1] and G.nodeNames[i] in params['outputs']:
			#if input_state == [0,0,0,0]:
				print('inputs=',input_state,':',G.nodeNames[i], 'avg=', A.avg[i])

def debug_print(params,G,steadyStates):
	k2 = len(params['outputs'])
	import numpy as np 
	outpt = np.array([G.nodeNums[params['outputs'][i]] for i in range(k2)])
	for k in steadyStates.attractors:
		A =  steadyStates.attractors[k]
		inpts = A.phenotype.split('|')[0]
		if inpts == '0000':
			print('with input 0000: ','A.size, A.period,A.avg[outpt]\n',A.size, A.period,A.avg[outpt])
	for k in steadyStates.phenotypes:
		P =  steadyStates.phenotypes[k]	
		print('pheno',k,'=',P)

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

				# can rm
				#if util.istrue(self.params,'calc_var'):
				#	A.totalVar = A.var.copy() 
				#	A.var/=A.size 

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


	def order_attractors(self):
		self.attractor_order = list(self.attractors.keys())

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
		#print('\n\nA0ws=',sum([Aweights[k] for k in Aweights]))
		#print('Asize sum=',sum([self.attractors[k].size for k in self.attractors]))
		used = {k:0 for k in Aweights}
		for A in self.attractors.values():
			assert(A.size <= 1)
			incoming_weight=0
			for k in A.A0s: 
				used[k]=1
				incoming_weight += Aweights[k]*A.A0s[k] # weight*# of such A0s that reach this A
			#print("\tAsize adds prod Asize*len(attractors),incoming_weight=",A.size,len(self.attractors),incoming_weight)
			A.size *= len(self.attractors)*incoming_weight
			total += A.size 
		#print('total=',total)
		#assert(total < 1.1 and total > .9) 

		total2=0
		for A in self.attractors.values():
			A.size/=total
			total2+=A.size 
		total=total2 
		if not (math.isclose(total,1)):
			print("\nWARNING: basin.adjust_Aweights do not sum to 1!\n") 
		

##################################### One Basin #############################################################

def calc_basin_size(params, G,A0=None,input_shuffle=False,Aweights=None):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils

	steadyStates = SteadyStates(params, G) 

	params['map_from_A0'] = False
	if A0 is not None:
		params['map_from_A0'] = True
		x0=A0
		if input_shuffle:
			x0, A0 = shuffle_x0_inputs(params, G, A0)
		else:
			x0 = convert_A0_to_x0(params, A0)
		assert(len(x0)==params['num_samples']==params['parallelism']) # otherwise need to change implementation

	params['precise_oscils']= (params['update_rule'] == 'sync' and not params['map_from_A0'] and not util.istrue(params,['PBN','active']) and not util.istrue(params,['skips_precise_oscils']))
	# using from A0 with sync causes complications like when to end oscil (since x0 is no longer nec in oscil)

	if params['precise_oscils']:
		oscil_bin = [] #put all the samples that are unfinished oscillators
		confirmed_oscils = [] #put all samples that have oscillated back to their initial state 
		
		# FIXED POINTS & EASY OSCILS
		if params['verbose']:
			print("Starting fixed point search, using", params['num_samples'], "sample initial points.")
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
		if params['verbose'] and confirmed_oscils != []: 
			print('Finished finding oscillations, now classifying them.')

		sync_run_oscils(params, confirmed_oscils, steadyStates, G, transient=False)

	else: # async, Gasync, stoch, starting from A0,...

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
	steadyStates.order_attractors()
	if Aweights is not None and A0 is not None:
		steadyStates.adjust_Aweights(A0,Aweights)

	# TODO: clean this, for ex, calls order_attractos again
	if params['update_rule'] != 'sync' and params['async_trim']:
		trim_As(params,steadyStates,thresh=params['async_trim'])

	if util.istrue(params,'use_phenos'):
		steadyStates.build_phenos()
	return steadyStates


def get_init_sample(params, G):
	
	if 0:
		print("basin.get_init_sample() temp")
		# each x0 over each A
		p = .5 #prob a given node is off at start
		num_inputs = len(params['inputs'])
		size = int(params['parallelism']/(2**num_inputs))
		x0 = cp.random.choice(a=[0,1], size=(size,G.n), p=[p, 1-p]).astype(bool,copy=False)
		x0 = cp.vstack([x0 for _ in range(2**num_inputs)]) # should be a nicer way to do this natively
		# tile or repeat(axis=0) goes a,a,b,b,c,c...but need a,b,c,a,b,c,....

	else:
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
		k = len(params['inputs'])
		input_indices = [G.nodeNums[params['inputs'][i]] for i in range(k)]
		input_sets = itertools.product([0,1],repeat=k)
		i=0
		for input_set in input_sets:
			x0[int(i*params['parallelism']/(2**k)):int((i+1)*params['parallelism']/(2**k)),input_indices] = cp.array(input_set)
			i+=1
		assert(i==2**k)

	return x0


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


def format_id_str(x):
	# TODO: this is slower than it should be
	label=''
	for i in range(len(x)):
		ele=x[i]
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

#################################### STEADY BASIN #####################################################

def calc_steady_basin(params, G, A0=None, Aweights=None):
	# calculates the steady basin by updating the inputs for at most num_steps
	# note that params['parallelism'] and ['num_samples'] will be altered during calc_basin_size()
	# pass A0 for mutated or control runs (for example)

	# Aweights are likely unnec, since SS is usually indepd of init, but puting here anyhow

	SS = calc_basin_size(params,G,A0=A0,Aweights=Aweights,input_shuffle=False) # get steady states from random initial conditions
	A0, Aweights =  get_A0_and_weights(SS)

	if params['debug']:
		assert(len(SS.attractors)==len(SS.attractor_order))
	
	new=True 
	while new:
		#print('basin 1st attractor=',A0[0],'\nlng of A0=',len(A0))
		SS = calc_basin_size(params, G, A0=A0,Aweights=Aweights, input_shuffle=True) # get steady states from input-shuffled attractors
		assert(len(SS.attractors)==len(SS.attractor_order))
		#debug_ghost_A(SS)
		if params['verbose']:
			print('#As =',len(SS.attractors))
		transit_pr, new = SS.map_A_transition_pr() 
		if new:
			A0 += [SS.attractors[k].id for k in SS.attractor_order]
			A0 = list(set(A0)) #remove duplicates
			Aweights_new = {A.id:A.size for A in SS.attractors.values()}
			Aweights = {**Aweights_new, **Aweights}
			if params['verbose']: 
				print('\tFound new attractors, rerunning with',len(A0),' A0s.')

	if params['verbose']:
		print('Finished building transition matrix, calculating steady basin.')
	steadyBasin, eigen_vec = calc_markov_chain_SS(params, transit_pr, SS)
	if params['debug']:
		steadyBasin_multpn, basin_vec = steadyBasin_from_transition_matrix(params, SS, transit_pr, 20) 
		assert(np.allclose(eigen_vec,basin_vec))
		#print('multiplication vs eigen:\n',np.round(basin_vec,6),'vs\n', np.round(eigen_vec,6))

	return steadyBasin 


def get_A0_and_weights(SS):
	# poss use np.array(A0)
	A0 = [SS.attractors[k].id for k in SS.attractor_order]# attractors in ORDERED list, like def within SSs
	Aweights = {A.id:A.size for A in SS.attractors.values()}
	return A0, Aweights

def calc_markov_chain_SS(params, transit_pr, SS):
	# could add an assert that there are not 2 eigenvals = 0 (implies depd on init)

	eigenvals, eigenvecs = np.linalg.eig(transit_pr)
	if np.sum(eigenvals == 1) > 1:
		print("WARNING: multiple eigenvalues=1 detected")#. Eigenvecs where eigenval=1:\n",eigenvecs[eigenvals==1])
	if np.sum(eigenvals > 1.1) > 0:
		print("\n\n\nERROERRRERRERERRORERRROR: an eigenvalue > 1 detected! Eigenvalues:",eigenvals,'\n\n\n')
	if np.sum(eigenvals == -1) > 0:
		print("\n\nWARNING: periodic steady state detected! eigenvalues:",eigenvals,'\n\n\n')
	SS_eigenvec = eigenvecs[:,np.argmax(eigenvals)] # note that eigen vecs are the columns, not the rows
	SS_eigenvec = SS_eigenvec/np.sum(SS_eigenvec)
	eigen_steadyBasin = basin_matrix_to_SS(params, SS, SS_eigenvec)
	return eigen_steadyBasin, SS_eigenvec

def trim_As(params,SS,thresh=.001):
	# not that size is NOT renormalized
	rm_keys = []
	for A in SS.attractors.values():
		if A.size < thresh:
			rm_keys += [A.id] 
	for k in rm_keys:
		del SS.attractors[k]
	if params['map_from_A0']:
		for A in SS.attractors.values():
			rm_A0s = []
			for A0 in A.A0s:
				if A0 in rm_keys:
					rm_A0s += [A0]
			for A0 in rm_A0s:
				del A.A0s[A0]
	SS.order_attractors()
	print('after trim, #As=',len(SS.attractors))


def steadyBasin_from_transition_matrix(params, SS, T, num_steps):
	B = np.array([SS.attractors[k].size for k in SS.attractor_order]) # starting basin size
	for i in range(num_steps):
		if params['verbose']:
			print('at step',i)
		#print('starting from B=',B,'\nT=',T)
		B_next = np.matmul(T,B)
		#print('BxT = ',B_next)
		if params['update_rule']=='sync' and params['debug']: # due to attractor trimming, async methods may not actually sum to 1
			assert(np.isclose(sum(B_next),1))
		if np.allclose(B_next,B):
			if params['verbose']:
				print("\nSteady basin fixed point found.")
			return basin_matrix_to_SS(params, SS, B), B
		B = B_next 

	print("\n\nWARNING: Steady basin fixed point NOT found, returning basin after",num_steps,"steps.\n")
	return basin_matrix_to_SS(params, SS, B), B

def debug_ghost_A(steadyStates):
	SS = steadyStates
	for target in steadyStates.attractor_order:
		assert(len(steadyStates.attractors[target].A0s)>0)
		#for source in steadyStates.attractors[target].A0s:
		#	assert(source in steadyStates.attractor_order) 
	n = len(SS.attractor_order)
	print('curr As:')
	for i in range(n):
		print(SS.attractor_order[i])
	print('~~~\n')
	for i in range(n):
		Ai = SS.attractor_order[i]
		found=0
		for j in range(n):
			Aj = SS.attractor_order[j]
			if Ai in SS.attractors[Aj].A0s.keys():
				found=1
		if not found:
			print(Ai)
			assert(0)
		else:
			print('pass')

def basin_matrix_to_SS(params, SS, B):
	for k in SS.attractors.keys():
		A = SS.attractors[k]
		A.size = B[SS.attractor_order.index(k)]
	return SS


def shuffle_x0_inputs(params, G, A0):
	# called within calc_basin_size if A0s are passed
	# prob could be more effic with some more np
	x0=[]
	input_indices = G.input_indices(params)
	input_sets = list(itertools.product([0,1],repeat=len(params['inputs'])))
	ref_A0=[] # later used to map which x0 come from which A0
	for A in A0:
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
	return x0, ref_A0

def convert_A0_to_x0(params, A0):
	return cp.array([[int(a) for a in A] for A in A0])


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