import ldoi, parse, plot, logic, basin, features
import sys, math, pickle
import numpy as np
import random as rd
from copy import deepcopy
from net import Net

CONTROL_PARAMS = {
	'mut_thresh':.8, 
	'cnt_thresh':.5, 
	'max_mutator_size':2, 
	'max_control_size':1,
	'norm':1
}

class Perturbations:
	def __init__(self, params, G,phenosWT,mut_thresh, cnt_thresh,max_control_size,norm):
		self.mutators, self.solutions, self.controllers = [],{},[]
		self.mut_thresh = mut_thresh
		self.mut_greedy_thresh = mut_thresh/10 #temp solution
		self.cnt_thresh = cnt_thresh 
		self.filtered_nodes = [node.name for node in G.regularNodes if node.name not in params['outputs'] and node.name not in params['inputs']]

		self.num_solutions=0
		self.phenosWT = phenosWT
		self.max_control_size = max_control_size

		self.num_inputs = len(params['inputs'])
		if norm in ['max','inf']:
			self.norm = np.inf
		else:
			self.norm = float(norm)

	def __str__(self):
		s= '\n# of mutators = ' + str(len(self.mutators)) + '\n# of controllers = ' + str(len(self.controllers)) + '\nAll solutions:' 
		for k in self.solutions.keys():
			s+= '\n' + str(k) + ' reversed by ' + str(self.solutions[k])
		return s

	def add_solution(self,mutator,solution):
		mutator = str(mutator['mutants'])
		if mutator in self.solutions.keys():
			assert(solution not in self.solutions[mutator])
			if self.max_control_size==1:
				self.solutions[mutator] += solution
			else: # need to distinguish sets of solutions
				self.solutions[mutator] += [solution]
		else:
			if self.max_control_size==1:
				self.solutions[mutator] = solution 
			else: 
				self.solutions[mutator] = [solution] 
		self.num_solutions += 1

class NetMetrics:
	def __init__(self, params, phenosWT,G):
		self.G=G
		self.fragility, self.reversibility = 0,0
		
		# TODO: how to clean node level features these features? add back later
		#mut_pts, cnt_pts, ldoi_mut, ldoi_out_mut, ldoi_cnt, ldoi_out_cnt = [[] for _ in range(6)]
		#cnt_scores = {C:[] for C in nodes}

		corr_nodes, corr_avg = features.check_corr(params, G, phenosWT)
		self.corr = corr_avg

		ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,G,fixed_nodes=[])
		self.ldoi_total = ldoi_stats['total']
		self.ldoi_total_outs = ldoi_stats['total_onlyOuts']
		# ldoi returns other stuff for indiv nodes, but ignoring for now

	def __str__(self):
		s = '#mutators=' + str(metrics.num_mutators)
		s+= '\n#solutions=' + str(metrics.num_solutions) 
		s+= '\nmutation mag = ' + str(metrics.fragility)
		s+= '\ncontrol mag = ' + str(metrics.reversibility)
		return s

	def finalize(self,perturbs):
		# adds last few metrics and normalizes
		self.num_mutators = len(perturbs.mutators)
		m=self.num_mutators
		self.num_controllers = len(perturbs.controllers)
		self.percent_mutators = self.num_mutators/self.G.n_neg 
		self.percent_solutions = self.num_solutions/max(m*G.n_neg,1) 
		self.percent_controllers = self.num_controllers/self.G.n_neg  

		self.fragility /= self.G.n_neg
		self.reversibility /= max(m,1) 

	
def exhaustive(params, G, mut_thresh, cnt_thresh, norm=1,max_mutator_size=2, max_control_size = 1, measure=False):

	assert('setting_file' in params.keys()) # this control module is based on phenotypes, so a setting file is mandatory

	# complexity is roughly O(s*n^m+1) where s = simulation time, n=#nodes, m=max control set size
	# assumes cannot alter pheno nodes, nor input nodes

	# if measure is true, several net metrics will be measured, such as LDOI

	# these are all assumptions of the control run
	params['verbose']=0 
	params['use_inputs_in_pheno']=True 
	params['use_phenos']=True

	phenosWT = mutate_and_sim(params, G).phenotypes
	perturbs = Perturbations(params,G,phenosWT,mut_thresh, cnt_thresh, max_control_size,norm)
	orig_mutated = deepcopy(params['mutations'])
	if measure:
		metrics = NetMetrics(params, phenosWT, G)
	else:
		metrics = None

	attempt_mutate(params,G,perturbs, metrics,[],0, max_mutator_size)
	
	for mutator in perturbs.mutators:
		params_orig = deepcopy(params)
		print('Trying to fix:',mutator['mutants'],'from dist',mutator['dist'])
		for mutant in mutator['mutants']:
			params['mutations'][mutant[0]] = mutant[1]
		attempt_control(params,G, perturbs, metrics, mutator, [], max_control_size)
		params = params_orig
	if measure:
		metrics.finalize(perturbs)
		print(metrics)
		return metrics, perturbs

	print(perturbs)
	return perturbs


def mutate_and_sim(params, G_orig):
	G = deepcopy(G_orig)
	G.apply_mutations(params) 
	steadyStates = basin.find_steadyStates(params, G)
	return steadyStates

####################################################################################################

#TODO there seems to be some bug here for later sets of mutators (mutn dist way too high)
def attempt_mutate(params, G, perturbs, metrics, curr_mutators, curr_mut_dist, depth,verbose=True):
	if depth == 0:
		return 

	curr_mutators_orig, curr_mut_dist_orig = deepcopy(curr_mutators), curr_mut_dist
	orig_mutations = deepcopy(params['mutations'])
	for M in perturbs.filtered_nodes:
		if verbose:
			print('Testing node',M)
		for b in [0,1]:
			if (M,b) not in curr_mutators:

				curr_mutators, curr_mut_dist = deepcopy(curr_mutators_orig), curr_mut_dist_orig
				params['mutations'] = deepcopy(orig_mutations)
				params['mutations'][M] = b
				phenosM = mutate_and_sim(params, G).phenotypes
				mut_dist = diff(perturbs.phenosWT,phenosM,perturbs.num_inputs,norm=perturbs.norm)
				if metrics is not None:
					metrics.fragility += mut_dist # TODO: this should only be for highest depth. ANd btw rename depth if it counts down

				if mut_dist  > perturbs.mut_thresh:
					curr_mutators += [(M,b)]
					perturbs.mutators += [{'mutants':curr_mutators,'dist':mut_dist}]
					print("\nAdded mutator:",curr_mutators,'at dist',round(mut_dist,3))
					#print([(k,phenosM[k].size) for k in phenosM.keys()])
				elif mut_dist > curr_mut_dist + perturbs.mut_greedy_thresh: 
					curr_mutators += [(M,b)]
					attempt_mutate(params, G, perturbs, metrics, curr_mutators, mut_dist, depth-1,verbose=False)
					# recurse attempt mutate at 1 lower depth


def attempt_control(params,G, perturbs, metrics, mutator, control_set, depth):
	# TODO: add similar greedy approach, where if doesn't fix enough, don't continue to recurse
	# note that mutator and control_set are not yet in perturbs since they are actively build/altered by this function 
	if depth == 0:
		return 
	
	reversibility = 0 # TODO: fix for recursion (not sure how to weight pr of larger sets)

	# WARNING: ldoi stats won't work with larger depth
	#ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,F,V,fixed_nodes=[mutator])
	
	mut_dist = mutator['dist'] # TODO what if mult mutators?

	for C in perturbs.filtered_nodes:
		orig_mutations = deepcopy(params['mutations'])
		for b in [0,1]:
			if (C,b) not in mutator['mutants'] and (C,b) not in control_set:
				params['mutations'][C] = b
				phenosC = mutate_and_sim(params, G).phenotypes

				#change = 1-diff(phenosWT,phenosC)/mut_dist
				cnt_dist = diff(perturbs.phenosWT,phenosC,perturbs.num_inputs,norm=perturbs.norm)
				reversibility += max(0,mut_dist-cnt_dist)
				if cnt_dist < mut_dist - perturbs.cnt_thresh: 
					phenosW = perturbs.phenosWT
					perturbs.add_solution(mutator,control_set+[(C,b)])
					if (C,b) not in perturbs.controllers:
						perturbs.controllers +=[(C,b)]
				else:
					# recurse using this node as a control + another (until reach max_depth)
					# note that recursed revblty is currently not used
					attempt_control(params,G, perturbs, metrics, mutator,control_set+[(C,b)], depth-1)

				params['mutations'] = deepcopy(orig_mutations) # reset (in case orig mutator was tried as a control node)

	if metrics is not None:
		metrics.reversibility +=  reversibility/(2*len(perturbs.filtered_nodes)-1) # -1 since will not use the orig mutation for control (but will try flipping it)


def diff(P1, P2, num_inputs, norm=1):
	P1_basins, P2_basins = [],[]
	for io in P1:
		P1_basins += [P1[io].size]
		if io not in P2:
			P2_basins += [0]
		else:
			P2_basins += [P2[io].size]
	for io in P2:
		if io not in P1:
			# i.e only if skipped before
			P2_basins += [P2[io].size]
			P1_basins += [0]
	
	P1_basins, P2_basins = np.array(P1_basins), np.array(P2_basins)

	N = 2**num_inputs
	if False: # TODO temp
		normz_factor=2
	elif norm in ['inf','max',np.inf]:
		norm = np.inf 
		normz_factor=1/N # this is 1/ max_poss
	else:
		normz_factor=math.pow(2*N*math.pow(1/N,norm),1/norm) # note that as norm -> inf, normz_factor -> 1/N
	result= np.linalg.norm(P1_basins-P2_basins,ord=norm)
	result/=normz_factor
	if not (result >= -.01 and result <= 1.01): # may not hold for norms that are norm max or 1, not sure how to normalize them
		print("\nWARNING: difference =",result,", not in [0,1]!") # this really should not occur! If it does, check if an issue of accuracy
		#print("P1 basins:",P1_basins,"\nP2_bsins:",P2_basins,'\n\n')
	#print('\ndist=',result,'\nfrom',P1_basins,'\tvs',P2_basins)
	return result


def tune_dist(param_file, reps):
	# finds dist thresh such that max dist btwn 2 runs of the same net < dist thresh/2
	# TODO: add z-distrib, with enough samples it can be approx as such. Then return suggested val sT p(Z>val) <= given alpha

	assert(0) #TODO: update this
	params = parse.params(param_file)
	F, F_mapd, A, V  = parse.net(params)
	max_dist = 0
	params['verbose']=False #i assume
	phenos_start = basin.calc_basin_size(params,F_mapd,V).phenotypes 
	phenos1 = phenos2 = phenos_start
	for r in range(1,reps):
		if r % (reps/10) == 0:
			print("at run #",r)
		phenos = basin.calc_basin_size(params,F_mapd,V).phenotypes 
		if diff(phenos1,phenos) > max_dist and diff(phenos1,phenos) > diff(phenos2,phenos):
			max_dist = diff(phenos1,phenos)
			phenos2 = phenos 
		elif diff(phenos2,phenos) > max_dist:
			max_dist = diff(phenos2,phenos)
			phenos1 = phenos 

	return max_dist*10

if __name__ == "__main__":
	if len(sys.argv) < 3:
		sys.exit("Usage: python3 control.py PARAMS.yaml [exh | tune]")
	if sys.argv[2] == 'tune':
		dist_thresh = tune_dist(sys.argv[1], 100)
		print("suggested distance threshold =",dist_thresh) 
	elif sys.argv[2] == 'exh':
		params = parse.params(sys.argv[1])
		G = Net(params)
		exhaustive(params, G, CONTROL_PARAMS['mut_thresh'], CONTROL_PARAMS['cnt_thresh'], max_mutator_size=CONTROL_PARAMS['max_mutator_size'], max_control_size = CONTROL_PARAMS['max_control_size'],norm=CONTROL_PARAMS['norm'])