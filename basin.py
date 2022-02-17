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
	if not util.istrue(params,['steady_basin']):
		steadyStates = calc_basin_size(params,G)
	else:
		steadyStates = calc_steady_basin(params, G, params['steady_basin_max_steps'])
	plot.pie(params, steadyStates,G)
	
	#print('basin.main: phenos=',steadyStates.phenos_str())
	#print('basin.main: #A=',len(steadyStates.attractors),'n=',G.n)
	#temp_print_periodic(steadyStates, G,params)

		
def repeat_test(param_file):
	print("\nRUNNING REPEAT TEST\n")
	params, G = init(param_file)
	repeats=20

	if not params['steady_basin']:
		base = calc_basin_size(params,G).attractors
	else:
		base=calc_steady_basin(params, G, params['steady_basin_max_steps']).attractors
	As=[base]
	max_dist=0
	for k in range(repeats):
		print('starting repeat #',k+1,'/',repeats)
		if not params['steady_basin']:
			repeat = calc_basin_size(params,G).attractors
		else:
			repeat=calc_steady_basin(params, G, params['steady_basin_max_steps']).attractors
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
	def __init__(self, params, G, attractor_id, period, avg, var, A0):
		self.id = attractor_id
		self.size = 1
		if params['update_rule'] == 'sync' and not util.istrue(params,['PBN','active']) and not params['map_from_A0']:
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
			assert(not util.istrue(params,['PBN','active'])) # TODO: how to def input thresh? or non stoch inputs
			for i in range(len(inputs)):
				if self.avg[inputs[i]] > 0: 
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
		self.params = params
		self.G = G

	def phenos_str(self):
		s=''
		for k in self.phenotypes:
			if s!='':
				s+=', '
			s+=k+':'+str(self.phenotypes[k].size)
		return s

	def add(self, attractor_id, period, avg, A0=None,var=None): # id should be unique to each attractor
		if attractor_id not in self.attractors.keys():
			self.attractors[attractor_id] = Attractor(self.params, self.G, attractor_id, period, avg, var, A0)
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
				if var is not None:
					self.attractors[attractor_id].var += var



	def add_attractors(self, result, A0s, map_from_A0):
		# map_from_A0 indicates that transition pr from A_0 -> A_f should be measured

		if map_from_A0:
			#print('while adding A:',len(result['state']),len(A0))
			assert(len(result['state'])==len(A0s))
		for i in range(len(result['state'])):
			if self.params['update_rule'] != 'sync' or util.istrue(self.params,['PBN','active']) or map_from_A0:
				finished=True
				period=None
			else:
				finished = result['finished'][i]
				period = result['period'][i]

			if finished:
				attractor_id = format_id_str(result['state'][i]) 

				if not map_from_A0:
					a0=None
				else:
					a0 = A0s[i]

				# TODO: clean up how var is handled
				if 'var' in result.keys():
					self.add(attractor_id, period, result['avg'][i], a0, result['var'][i]) 
				else:
					self.add(attractor_id, period, result['avg'][i], a0)

	def normalize_attractors(self):		
		if self.params['update_rule'] in ['async','Gasync'] or util.istrue(self.params,['PBN','active']):
			for s in self.attractors:
				A = self.attractors[s] 
				A.totalAvg = A.avg.copy() 
				A.avg/=A.size 
				if util.istrue(self.params,'calc_var'):
					A.totalVar = A.var.copy() 
					A.var/=A.size 

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

		T = T/np.vstack(np.sum(T,axis=1)) # normalize such that sum_j(pr Ai -> Aj) = 1
		return T, False


	def order_attractors(self):
		self.attractor_order = list(self.attractors.keys())
		

##################################### One Basin #############################################################

def calc_basin_size(params, G,A0=None,input_shuffle=False):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils

	steadyStates = SteadyStates(params, G) 

	params['map_from_A0'] = False
	if A0 is not None:
		params['map_from_A0'] = True
		x0=A0
		if input_shuffle:
			x0, A0 = shuffle_x0_inputs(params, G, A0)
		assert(len(x0)==params['num_samples']==params['parallelism']) # otherwise need to change implementation

	if params['update_rule'] == 'sync' and not util.istrue(params,['PBN','active']) and not params['map_from_A0']:
		oscil_bin = [] #put all the samples that are unfinished oscillators
		confirmed_oscils = [] #put all samples that have oscillated back to their initial state 
		
		# FIXED POINTS & EASY OSCILS
		if params['verbose']:
			print("Starting fixed point search, using", params['num_samples'], "sample initial points.")
		for i in range(int(params['num_samples']/params['parallelism'])):
			if not params['map_from_A0']: # redundant, since assumes not map_from_A0, see above
				x0 = get_init_sample(params, G)
			# with fixed_points_only = True, will return finished only for fixed points
			# and help move oscillations past their transient phase
			result = lap.transient(params,x0, G, fixed_points_only=True)
			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, None, 0)
			steadyStates.add_attractors(result, None, False) # not map from A0 (todo: clean)


		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin,steadyStates, G,transient=True)


		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		if params['verbose'] and confirmed_oscils != []: 
			print('Finished finding oscillations, now classifying them.')

		sync_run_oscils(params, confirmed_oscils, steadyStates, G, transient=False)


	elif params['update_rule'] in ['async','Gasync'] or util.istrue(params,['PBN','active']) or params['map_from_A0']: 
		for i in range(int(params['num_samples']/params['parallelism'])):
			if not params['map_from_A0']:
				x0 = get_init_sample(params, G)
			x_in_attractor = lap.transient(params, x0, G)
			result = lap.categorize_attractor(params, x_in_attractor, G)
			cupy_to_numpy(params,result)
			steadyStates.add_attractors(result, A0, params['map_from_A0'])

	else:
		sys.exit("ERROR: unrecognized parameter for 'update_rule'")

	steadyStates.normalize_attractors()
	steadyStates.order_attractors()

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
		if util.istrue(params,['PBN','active']) and params['PBN']['init'] == 'half':
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
			steadyStates.add_attractors(result,None,False) # assume not mapping from A0

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

def calc_steady_basin(params, G, num_steps, A0=None):
	# calculates the steady basin by updating the inputs for at most num_steps
	# note that params['parallelism'] and ['num_samples'] will be altered during calc_basin_size()
	# pass A0 for mutated or control runs (for example)
	assert(A0 is None) # plz explicitly debug A0!=None case before using
	SS = calc_basin_size(params,G,A0=A0,input_shuffle=False) # get steady states from random initial conditions
	if params['debug']:
		assert(len(SS.attractors)==len(SS.attractor_order))
	
	new=True 
	while new:
		A0 = [SS.attractors[k].id for k in SS.attractor_order]# attractors in ORDERED list, like def within SSs
		#print('basin 1st attractor=',A0[0],'\nlng of A0=',len(A0))
		SS = calc_basin_size(params, G, A0=A0, input_shuffle=True) # get steady states from input-shuffled attractors
		assert(len(SS.attractors)==len(SS.attractor_order))
		#debug_ghost_A(SS)
		print('#As =',len(SS.attractors))
		transit_pr, new = SS.map_A_transition_pr() 
		if new:
			print('Found new attractors, rerunning.')
	print('Finished building transition matrix, calculating steady basin.')
	steadyBasin = steadyBasin_from_transition_matrix(params, SS, transit_pr, num_steps) # finish this fn
	return steadyBasin 

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
		print('at step',i)
		#print('starting from B=',B,'\nT=',T)
		B_next = np.matmul(B,T)
		#print('BxT = ',B_next)
		if params['update_rule']=='sync' and params['debug']: # due to attractor trimming, async methods may not actually sum to 1
			assert(np.isclose(sum(B_next),1))
		if np.allclose(B_next,B):
			print("\nSteady basin fixed point found.")
			return basin_matrix_to_SS(params, SS, B)
		B = B_next 

	print("\n\nWARNING: Steady basin fixed point NOT found, returning basin after",num_steps,"steps.\n")
	return basin_matrix_to_SS(params, SS, B)

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

	return x0, ref_A0

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