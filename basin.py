import itertools, util, math, sys
import lap, parse, plot
from net import Net
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# note that size of a phenotype or attract refers to its basin size

#########################################################################################################

def main(param_file):
	params = parse.params(param_file)
	G = Net(params)
	steadyStates = find_steadyStates(params,G)
	#print('basin.main: phenos=',steadyStates.phenos_str())
	plot.pie(params, steadyStates,G)

def find_steadyStates(params,G): 
	G.build_Fmapd_and_A(params)
	steadyStates = calc_basin_size(params,G)
	return steadyStates

#########################################################################################################

class Attractor:
	def __init__(self, params, G, attractor_id, period, avg, var):
		self.id = attractor_id
		self.size = 1
		if params['update_rule'] == 'sync' and not util.istrue(params,['PBN','active']):
			self.period = period
		self.avg = avg
		self.var = var
		self.totalAvg = avg #won't be normalized so is really a sum
		self.totalVar = var 
		if util.istrue(params,'use_phenos'):
			self.map_to_pheno(params,G) 

	def map_to_pheno(self, params, G):
		outputs = [G.nodeNums[params['outputs'][i]] for i in range(len(params['outputs']))]

		if 'inputs' in params.keys():
			inputs = [G.nodeNums[params['inputs'][i]] for i in range(len(params['inputs']))]

		self.phenotype = ''
		if 'inputs' in params.keys() and params['use_inputs_in_pheno']:
			assert(not util.istrue(params,['PBN','active'])) # TODO: how to def input thresh?
			for i in range(len(inputs)):
				if self.avg[inputs[i]] > 0: 
					self.phenotype +='1'
				else:
					self.phenotype +='0'	
			self.phenotype +='|'
		for i in range(len(outputs)): 
			if self.avg[outputs[i]] > params['output_thresholds'][i]: 
				self.phenotype +='1'
			else:
				self.phenotype +='0'		 


class Phenotype:
	def __init__(self, attractors, size):
		self.attractors = attractors
		self.size = size
		self.inputs = None 
		self.outputs = None 

	def __str__(self):
		return str(self.size)

		
class SteadyStates:
	# collects sets of attractors and phenotypes

	def __init__(self, params,G):
		self.attractors = {} 
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

	def add(self, attractor_id, period, avg, var=None): # id should be unique to each attractor
		if attractor_id not in self.attractors.keys():
			self.attractors[attractor_id] = Attractor(self.params, self.G, attractor_id, period, avg, var)
		else:
			self.attractors[attractor_id].size += 1
			if self.params['update_rule'] != 'sync' or util.istrue(self.params,['PBN','active']):
				self.attractors[attractor_id].avg += avg
				if var is not None:
					self.attractors[attractor_id].var += var


	def add_attractors(self, result):
		for i in range(len(result['state'])):
			if self.params['update_rule'] != 'sync' or util.istrue(self.params,['PBN','active']):
				finished=True
				period=None
			else:
				finished = result['finished'][i]
				period = result['period'][i]

			if finished:
				attractor_id = format_id_str(result['state'][i]) 
				# TODO: clean up how var is handled
				if 'var' in result.keys():
					self.add(attractor_id, period, result['avg'][i], result['var'][i]) 
				else:
					self.add(attractor_id, period, result['avg'][i])

	def normalize_attractors(self):		
		if self.params['update_rule'] in ['async','Gasync'] or util.istrue(self.params,['PBN','active']):
			for s in self.attractors:
				A = self.attractors[s] 
				A.totalAvg = A.avg.copy() #TODO see if this causes err, if so, move to attrs
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


##########################################################################################################

def calc_basin_size(params, G):
	# overview: run 1 to find fixed points, 2 to make sure in oscil, run 3 to categorize oscils

	steadyStates = SteadyStates(params, G) 

	oscil_bin = [] #put all the samples that are unfinished oscillators
	confirmed_oscils = [] #put all samples that have oscillated back to their initial state 
	
	if params['update_rule'] == 'sync' and not util.istrue(params,['PBN','active']):

		# FIXED POINTS & EASY OSCILS
		if params['verbose']:
			print("Starting fixed point search, using", params['num_samples'], "sample initial points.")
		for i in range(int(params['num_samples']/params['parallelism'])):
			x0 = get_init_sample(params, G)
			# with fixed_points_only = True, will return finished only for fixed points
			# and help move oscillations past their transient phase
			result = lap.transient(params,x0, G, fixed_points_only=True)
			cupy_to_numpy(params,result)
			result, loop = run_oscils_extract(params, result, oscil_bin, None, 0)
			steadyStates.add_attractors(result)

		# TRANSIENT OSCILS
		# run until sure that sample is in the oscil
		confirmed_oscils = sync_run_oscils(params, oscil_bin, steadyStates, G, transient=True)

		# CLASSIFYING OSCILS
		# calculate period, avg on state, ect
		if params['verbose'] and confirmed_oscils != []: 
			print('Finished finding oscillations, now classifying them.')

		sync_run_oscils(params, confirmed_oscils, steadyStates, G, transient=False)


	elif params['update_rule'] in ['async','Gasync'] or util.istrue(params,['PBN','active']): 
		for i in range(int(params['num_samples']/params['parallelism'])):
			x0 = get_init_sample(params, G)

			x_in_attractor = lap.transient(params,x0, G)
			result = lap.categorize_attractor(params,x_in_attractor, G)
			cupy_to_numpy(params,result)
			steadyStates.add_attractors(result)

	else:
		sys.exit("ERROR: unrecognized parameter for 'update_rule'")

	steadyStates.normalize_attractors()

	if util.istrue(params,'use_phenos'):
		steadyStates.build_phenos()

	return steadyStates


def get_init_sample(params, G):
	p = .5 #prob a given node is off at start
	x0 = cp.random.choice(a=[0,1], size=(params['parallelism'],G.n), p=[p, 1-p]).astype(bool,copy=False)
	
	if 'inputs' in params.keys():
		k = len(params['inputs'])
		input_indices = [G.nodeNums[params['inputs'][i]] for i in range(k)]
		input_sets = itertools.product([0,1],repeat=k)
		i=0
		for input_set in input_sets:
			x0[int(i*params['parallelism']/(2**k)):int((i+1)*params['parallelism']/(2**k)),input_indices] = cp.array(input_set)
			i+=1
		assert(i==2**k)

	if 'init' in params.keys():
		for k in params['init']:
			node_indx = G.nodeNums[k]
			x0[:,node_indx] = params['init'][k]

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


##################################################################################


if __name__ == "__main__":
	if len(sys.argv) not in [2]:
		sys.exit("Usage: python3 basin.py PARAMS.yaml")
	
	main(sys.argv[1])