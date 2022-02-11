import ldoi, param, plot, logic, basin, features, net
import sys, math, pickle, itertools
import numpy as np
from operator import itemgetter
import random as rd
from copy import deepcopy
from net import Net

# run time is approximately O(t*n^(c+m))
#	where c=max_control_size,m=max_mutator_size,t=simulation time
CONTROL_PARAMS = {
	'mut_thresh':.06, 	# minimum distance threshold to be considered a mutant 
	'cnt_thresh':.06, 	# min distance less than the mutation distance to be considered a controller
	'max_mutator_size':1,  
	'max_control_size':1,
	'norm':1,			# the norm used to measure distance. Use an integer or 'max'
	'verbose':True,	# toggle how much is printed to the console

	'shuffle_inputs':1, # only works if from_attractors is true
	'from_attractors': 1,
	'apply_majority':0
}

# should_check() if not generalized beyond mutator/control sizes of 2. 
# Using larger sizes will result in some redundant simulations (but will still be correct)

class Perturbations:
	def __init__(self, params, G,SS_WT,mut_thresh, cnt_thresh,max_control_size,norm):
		self.mutators, self.solutions, self.controllers,self.irreversible,self.self_reversible = [],{},[],[],[]
		# irreversible are mutators without a solution

		self.mut_thresh = mut_thresh
		self.mut_greedy_thresh = 0 #mut_thresh/10 # can greedily reduce recursion by setting this > 0
		self.cnt_thresh = cnt_thresh 

		# inputs, inits, and outputs are assumed not to be valid target
		self.filtered_nodes = [node.name for node in G.nodes if node.name not in params['outputs'] and node.name not in params['inputs']]# and node.name not in params['init'].keys()]

		self.num_solutions=0
		self.SS_WT = SS_WT
		self.x0 = None # clean, but idea is to save attractors to start from
		self.phenosWT = SS_WT.phenotypes
		self.max_control_size = max_control_size

		self.num_inputs = len(params['inputs'])
		if norm in ['max','inf']:
			self.norm = np.inf
		else:
			self.norm = float(norm)

	def __str__(self,solutions=False):
		s= '\n# of mutators = ' + str(len(self.mutators))
		s+= '\n# of controllers = ' + str(len(self.controllers)) 
		s+= '\n# irreversible mutators = ' + str(len(self.irreversible))
		s+= '\nIrreversible mutators: ' + str(self.irreversible)
		s+= '\n# self reversible mutators: ' + str(len(self.self_reversible))
		if solutions:
			s+= '\n\nAll solutions:' 
			for k in self.solutions.keys():
				s+= '\n' + str(k) + ' at dist ' + str(round(self.solutions[k]['mut_dist'],3)) + ' reversed by ' 
				i=0
				for soln in self.solutions[k]['controllers']:
					s+='\n\t'+ str(soln) + '\tto dist ' +str(round(self.solutions[k]['cnt_dists'][i],3))
					i+=1


		return s

	def add_solution(self,mutator,solution,cnt_dist):
		mutants = str(mutator['mutants'])
		if mutants in self.solutions.keys():
			assert(solution not in self.solutions[mutants]['controllers'])
			self.solutions[mutants]['controllers'] += [solution]
			self.solutions[mutants]['cnt_dists'] += [cnt_dist]
		else:
			self.solutions[mutants] = {}
			self.solutions[mutants]['controllers'] = [solution] 
			self.solutions[mutants]['mut_dist'] = mutator['dist']
			self.solutions[mutants]['cnt_dists'] = [cnt_dist]
		self.num_solutions += 1

	def calc_irrev_mutators(self):

		for mutator in self.mutators:
			mutants = str(mutator['mutants'])
			if mutants not in self.solutions.keys():
				self.irreversible += [mutants]

	def calc_self_reversible(self):
		for mutator in self.mutators:
			assert(len(mutator['mutants'])==1) # otherwise not yet implemented
			compl = deepcopy(mutator['mutants'])
			compl[0] = (compl[0][0],(compl[0][1]+1) % 2)
			if str(mutator['mutants']) in self.solutions.keys():
				if compl in self.solutions[str(mutator['mutants'])]['controllers']:
					self.self_reversible += [mutator['mutants']]

class NetMetrics:
	def __init__(self, params, phenosWT,G):
		self.G=G
		self.fragility, self.reversibility = 0,0
		
		# TODO: how to clean node level features these features? add back later
		# mut_pts, cnt_pts, ldoi_mut, ldoi_out_mut, ldoi_cnt, ldoi_out_cnt = [[] for _ in range(6)]
		# cnt_scores = {C:[] for C in nodes}

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
		s+= '\n'
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

	
def exhaustive(params, G, mut_thresh, cnt_thresh, norm=1,max_mutator_size=1, max_control_size = 1, measure=False):

	if 'mutations' in params.keys() and len(params['mutations'])!=0:
		print('WARNING: control search starts with existing mutations! Double check the settings file.\n')
	assert('setting_file' in params.keys()) # this control module is based on phenotypes, so a setting file is mandatory

	# complexity is roughly O(s*n^m+1) where s = simulation time, n=#nodes, m=max control set size
	# assumes cannot alter pheno nodes, nor input nodes

	# if measure is true, several net metrics will be measured, such as LDOI

	# these are all assumptions of the control run
	params['verbose']=0 
	params['use_inputs_in_pheno']=True 
	params['use_phenos']=True

	# TODO: clean this double call! i.e. shouldn't have to build perturbs and then remodify
	SS_WT = mutate_and_sim(params, G,apply_majority=CONTROL_PARAMS['apply_majority'])
	perturbs = Perturbations(params,G,SS_WT,mut_thresh, cnt_thresh, max_control_size,norm)
	if CONTROL_PARAMS['from_attractors']:
		perturbs.x0 = basin.get_x0_from_SS(params, perturbs.SS_WT,G,shuffle_inputs=CONTROL_PARAMS['shuffle_inputs'])
		if CONTROL_PARAMS['shuffle_inputs']:
			SS_WT = mutate_and_sim(params, G,x0=perturbs.x0,apply_majority=CONTROL_PARAMS['apply_majority'])
			perturbs.x0 = basin.get_x0_from_SS(params, perturbs.SS_WT,G,shuffle_inputs=False)
	perturbs.SS_WT = SS_WT 
	perturbs.phenosWT = SS_WT.phenotypes


	orig_mutated = deepcopy(params['mutations'])
	if measure:
		metrics = NetMetrics(params, phenosWT, G)
	else:
		metrics = None

	print("\nSearching for mutators...")
	# returns WT phenotypes of LDOI just so don't have to recompute
	ldoi_WT = attempt_mutate(params,G,perturbs, metrics,[],0, max_mutator_size)
	sim_WT = extract_sim_WT(params, [perturbs.SS_WT.attractors[k] for k in perturbs.SS_WT.attractors.keys()])
	# TODO: add sim_WT as alt for controller eval

	print("\nSearching for controllers...")

	ldoi_eval = {'true positive':0,'true negative':0,'false positive':0,'false negative':0}
	for mutator in perturbs.mutators:
		params_orig = deepcopy(params)
		if CONTROL_PARAMS['verbose']:
			print('Trying to fix:',mutator['mutants'],'from dist',round(mutator['dist'],3))
		for mutant in mutator['mutants']:
			params['mutations'][mutant[0]] = mutant[1]
		attempt_control(params,deepcopy(G), perturbs, metrics, mutator, [], max_control_size,ldoi_WT,sim_WT,ldoi_eval)
		params = params_orig

	ldoi_precision, ldoi_recall = calc_precision_recall(ldoi_eval)
	print("\nControl LDOI precision =",ldoi_precision,", recall =",ldoi_recall,'\n')
	perturbs.calc_irrev_mutators()
	perturbs.calc_self_reversible()

	if measure:
		metrics.finalize(perturbs)
		print(metrics)
		return metrics, perturbs

	print(perturbs)
	return perturbs


def mutate_and_sim(params_orig, G_orig,x0=None,apply_majority=False):
	G, params = deepcopy(G_orig), deepcopy(params_orig)
	if x0 is not None:
		x0=deepcopy(x0)
		params['num_samples'] = params['parallelism'] = len(x0) 
		# TODO: don't do this, above
		# 	mix up is due to basin.py mutating and then building x0. here opposite
		for mutant in params['mutations']:
			indx = G.nodeNums[mutant]
			x0[:,indx] = params['mutations'][mutant]
	G.prepare_for_sim(params)
	steadyStates = basin.calc_basin_size(params,G,x0=x0)
	if apply_majority:
		apply_majority_A(params, G, steadyStates)
		for A in steadyStates.attractors.values():
			print('size=',A.size)
	return steadyStates

####################################################################################################
				
def attempt_mutate(params_orig, G, perturbs, metrics, curr_mutators_orig, curr_mut_dist, depth, verbose=True):
	if depth == 0:
		return 

	params = deepcopy(params_orig)
	curr_mut_dist_orig = curr_mut_dist
	orig_mutations = deepcopy(params['mutations'])
	go_further = []

	Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])
	mutant_ldoi = ldoi.test(Gpar,params,init=[]) # TODO: rename ldoi fn someday
	WT = extract_ldoi_WT(params, G.not_string, mutant_ldoi)
	#print('attempt_mutate, WT=',WT)
	ldoi_eval = {'true positive':0,'true negative':0,'false positive':0,'false negative':0}
	# in M loop: check if mutator is predicted by LDOI (i.e. != WT)
	# track LDOI false positive, false negative, true positive, true negative
	# test, then do again with controllers, where init=[mutator]

	for M in perturbs.filtered_nodes:
		#if verbose:
		#	print('Testing node',M)
		for b in [0,1]:
			curr_mutators = deepcopy(curr_mutators_orig)
			if should_check(M, b, curr_mutators, perturbs.mutators,mutants=True):
				curr_mutators, curr_mut_dist = deepcopy(curr_mutators_orig), curr_mut_dist_orig
				params['mutations'] = deepcopy(orig_mutations)
				params['mutations'][M] = b
				SS_M = mutate_and_sim(params, G,x0=perturbs.x0,apply_majority=CONTROL_PARAMS['apply_majority'])
				if CONTROL_PARAMS['shuffle_inputs']:
					mut_x0 = basin.get_x0_from_SS(params,SS_M,G,shuffle_inputs=CONTROL_PARAMS['shuffle_inputs'])
					SS_M = mutate_and_sim(params, G,x0=mut_x0,apply_majority=CONTROL_PARAMS['apply_majority'])
					
				phenosM = SS_M.phenotypes

				ldoi_predicted_diff = ldoi_diff(params, G.not_string, mutant_ldoi, (M,b), WT)
				#print('ldoi_predicted_diff=',ldoi_predicted_diff)

				mut_dist = diff(perturbs.phenosWT,phenosM,perturbs.num_inputs,norm=perturbs.norm)
				if metrics is not None:
					metrics.fragility += mut_dist # TODO: this should only be for highest depth. ANd btw rename depth if it counts down

				if mut_dist > perturbs.mut_thresh:
					curr_mutators += [(M,b)]
					mut_x0 = basin.get_x0_from_SS(params,SS_M,G,shuffle_inputs=False)
					perturbs.mutators += [{'mutants':curr_mutators,'dist':mut_dist,'x0':mut_x0,'ldoi_diff':ldoi_predicted_diff}]
					if CONTROL_PARAMS['verbose']:
						print("Added mutator:",curr_mutators,'at dist',round(mut_dist,3))
					#print([(k,phenosM[k].size) for k in phenosM.keys()])
					if ldoi_predicted_diff == 0:
						ldoi_eval['false negative'] += 1
					else:
						ldoi_eval['true positive'] += 1
				#elif mut_dist > curr_mut_dist + perturbs.mut_greedy_thresh: 
				#	assert(0) # not currently using greedy thresh
				#	curr_mutators += [(M,b)]
				#	go_further += [{'curr_mutators':deepcopy(curr_mutators),'mut_dist':mut_dist,'params':deepcopy(params)}]
				else:
					if ldoi_predicted_diff == 0:
						ldoi_eval['true negative'] += 1
					else:
						ldoi_eval['false positive'] += 1

	if depth>1:
		for ele in go_further:
			if CONTROL_PARAMS['verbose']:
				print('Recursing search for mutators with',ele['curr_mutators'])
			attempt_mutate(ele['params'], G, perturbs, metrics, ele['curr_mutators'], ele['mut_dist'], depth-1,verbose=False)
			# recurse attempt mutate at 1 lower depth

	ldoi_precision, ldoi_recall = calc_precision_recall(ldoi_eval)
	print("\nMutant LDOI precision =",ldoi_precision,", recall =",ldoi_recall,'\n')
	return WT

def attempt_control(params,G, perturbs, metrics, mutator, control_set_orig, depth, ldoi_WT, sim_WT, ldoi_eval):
	# TODO: add similar greedy approach, where if doesn't fix enough, don't continue to recurse
	# note that mutator and control_set are not yet in perturbs since they are actively build/altered by this function 
	if depth == 0:
		return 
	
	reversibility = 0 # TODO: fix for recursion (not sure how to weight pr of larger sets)

	# WARNING: ldoi stats won't work with larger depth
	#ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,F,V,fixed_nodes=[mutator])
	
	mut_dist = mutator['dist'] # TODO what if mult mutators?
	go_further = []

	Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])

	assert(len(mutator['mutants'])==1) # assuming singleton mutators
	this_mutant = mutator['mutants'][0]
	if this_mutant[1] == 1:
		init_ind = Gpar.nodeNums[this_mutant[0]]
	elif this_mutant[1] == 0:
		init_ind = Gpar.nodeNums[Gpar.not_string + this_mutant[0]]
	else:
		assert(0)
	
	cntr_ldoi = ldoi.test(Gpar,params,init=[init_ind]) # TODO: rename ldoi fn someday

	for C in perturbs.filtered_nodes:
		orig_mutations = deepcopy(params['mutations'])
		for b in [0,1]:
			control_set = deepcopy(control_set_orig)
			prev_groups=[]
			if str(mutator['mutants']) in perturbs.solutions.keys():
				prev_groups= perturbs.solutions[str(mutator['mutants'])]['controllers']
			if should_check(C, b, control_set, prev_groups):
				params['mutations'][C] = b
				if CONTROL_PARAMS['from_attractors']:
					x0 = mutator['x0']
				else:
					x0 = None
				SS_C = mutate_and_sim(params,G,x0=x0,apply_majority=CONTROL_PARAMS['apply_majority'])

				if CONTROL_PARAMS['shuffle_inputs']:
					x0 = basin.get_x0_from_SS(params,SS_C,G,shuffle_inputs=CONTROL_PARAMS['shuffle_inputs'])
					SS_C = mutate_and_sim(params, G,x0=x0,apply_majority=CONTROL_PARAMS['apply_majority'])

				phenosC = SS_C.phenotypes

				#change = 1-diff(phenosWT,phenosC)/mut_dist
				cnt_dist = diff(perturbs.phenosWT,phenosC,perturbs.num_inputs,norm=perturbs.norm)
				reversibility += max(0,mut_dist-cnt_dist)

				ldoi_predicted_diff = ldoi_diff(params, G.not_string, cntr_ldoi, (C,b), ldoi_WT, sim_WT=sim_WT)

				#print("\t\tC =",C,': C_diff=',ldoi_predicted_diff, 'm_diff=', mutator['ldoi_diff'])
				if cnt_dist < mut_dist - perturbs.cnt_thresh: 
					phenosW = perturbs.phenosWT
					if CONTROL_PARAMS['verbose']:
						print("\tControl with",control_set+[(C,b)],'reduces to dist',round(cnt_dist,3))
					perturbs.add_solution(mutator,control_set+[(C,b)],cnt_dist)
					if (C,b) not in perturbs.controllers:
						perturbs.controllers +=[(C,b)]

					if ldoi_predicted_diff < mutator['ldoi_diff']:
						ldoi_eval['true positive'] += 1
					else:
						ldoi_eval['false negative'] += 1
				else:					
					if ldoi_predicted_diff < mutator['ldoi_diff']:
						ldoi_eval['false positive'] += 1
					else:
						ldoi_eval['true negative'] += 1
					# recurse using this node as a control + another (until reach max_depth)
					# note that recursed revblty is currently not used
					go_further += [{'control_set':deepcopy(control_set+[(C,b)]),'params':deepcopy(params)}]

				params['mutations'] = deepcopy(orig_mutations) # reset (in case orig mutator was tried as a control node)

	if metrics is not None:
		metrics.reversibility +=  reversibility/(2*len(perturbs.filtered_nodes)-1) # -1 since will not use the orig mutation for control (but will try flipping it)

	if depth>1:
		for ele in go_further:
			if CONTROL_PARAMS['verbose']:
				print("Recursing control search with",ele['control_set'])
			attempt_control(ele['params'],G, perturbs, metrics, mutator,ele['control_set'], depth-1)


def extract_ldoi_WT(params, not_str, ldoi_solns):
	# this is all very messy, but to find a solution that doesn't have any mutations (i.e. WT),
	#	take one where the mutator is an input, whose value is the same as the input condition 

	inpt_sets = ldoi_solns[params['inputs'][0]].keys()
	WT = {}
	for inpt in inpt_sets:
		if '(1' in inpt:
			WT[inpt] = extract_ldoi_outputs(params, not_str, ldoi_solns[params['inputs'][0]][inpt])
		else:
			WT[inpt] = extract_ldoi_outputs(params, not_str, ldoi_solns[not_str + params['inputs'][0]][inpt])
	return WT

def extract_sim_WT(params, As):
	assert(isinstance(As[0],basin.Attractor))
	# again very sloppy approach, if decide to keep it, clean it
	# uses majority attractor for each input condition

	input_indices = [G.nodeNums[params['inputs'][i]] for i in range(len(params['inputs']))]
	output_indices = [G.nodeNums[params['outputs'][i]] for i in range(len(params['outputs']))]
	input_sets = list(itertools.product([0,1],repeat=len(params['inputs'])))
	
	WT = {}
	max_sizes = {str(inpt):0 for inpt in input_sets}
	for inpt in input_sets:
		for A in As:
			if A.size > max_sizes[str(inpt)]:
				incld=True
				for k in range(len(params['inputs'])):
					if int(A.id[input_indices[k]]) != int(inpt[k]): # shuold tech be avg
						incld=False
						break
				if incld: 
					max_sizes[inpt] = A.size
					WT[str(inpt)] = []
					for output in output_indices:
						WT[str(inpt)] += [int(A.id[output])]
	return WT

def ldoi_diff(params, not_str, ldoi_solns, target, WT,sim_WT=None):
	# checks if WT and target predict the same pheno

	# this is all very messy, but to find a solution that doesn't have any mutations (i.e. WT),
	#	take one where the mutator is an input, whose value is the same as the input condition

	# target is (name, value) of the mutator or controller whose ldoi you want to extract 

	if target[1]==0:
		name = not_str + target[0]
	elif target[1]==1:
		name = target[0]
	else:
		assert(0)
	inpt_sets = ldoi_solns[params['inputs'][0]].keys()
	extract = {}

	diff=0
	for inpt in inpt_sets:
		extract[inpt] = extract_ldoi_outputs(params, not_str, ldoi_solns[name][inpt])
		if extract[inpt] != WT[inpt]:
			if sim_WT is None or sim_WT[str(inpt)] != extract[inpt]: # i.e. a controller can match EITHER the ldoi_WT or sim_WT
				diff+=1
	return diff

def extract_ldoi_outputs(params, not_str, ldoi_soln): 
	barcode = []
	for output in params['outputs']:
		if output in ldoi_soln:
			barcode +=[1] # driven on
		elif not_str + output in ldoi_soln:
			barcode +=[0] # driven off
		else:
			barcode +=[2] # not driven
	return barcode


def calc_precision_recall(ldoi_eval):
	if (ldoi_eval['true positive'] + ldoi_eval['false positive']) == 0:
		precision = 'NA'
	else: 
		precision = ldoi_eval['true positive']/(ldoi_eval['true positive'] + ldoi_eval['false positive'])

	if (ldoi_eval['true positive'] + ldoi_eval['false negative']) == 0:
		recall = 'NA'
	else: 
		recall = ldoi_eval['true positive']/(ldoi_eval['true positive'] + ldoi_eval['false negative'])
	return precision, recall


def should_check(nodeName, b, curr_group, prev_groups, mutants=False):
	# TODO: clean the mutants switch
	if mutants:
		prev_groups = [prev_groups[i]['mutants'] for i in range(len(prev_groups))]

	# returns if should check adding this node to the existing group
	in_prev_recursion = (nodeName,0) in curr_group or (nodeName,1) in curr_group

	# this ensures don't double check a set of (M1,M2) and then (M2,M1)
	in_alpha_order = (curr_group==[] or curr_group[-1][0]<nodeName)

	# TODO: generalize beyond k=2
	in_prev_group = [(nodeName,b)] in prev_groups 
	return not in_prev_recursion and in_alpha_order and not in_prev_group


def apply_majority_A(params, G, SS):
	# prob should move this to basin
	assert(0) # this part was all messed up
	input_sets, inpts = get_input_sets(params, G)

	for inpt_set in input_sets:
		max_size=0
		for A in SS.attractors.values():
			match=True
			for k in range(len(inpts)):
				if A.id[inpts[k]] != inpt_set[k]:
					match=False
					break
			if match:
				if A.size > max_size:
					max_size = A.size 
					A.size=1
				else:
					A.size = 0 

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
	if norm in ['inf','max',np.inf]:
		norm = np.inf 
		normz_factor=1/N # this is 1/ max_poss
	else:
		normz_factor=math.pow(2*N*math.pow(1/N,norm),1/norm) # note that as norm -> inf, normz_factor -> 1/N
	result= np.linalg.norm(P1_basins-P2_basins,ord=norm)
	result/=normz_factor
	if not (result >= -.01 and result <= 1.01): # may not hold for norms that are norm max or 1, not sure how to normalize them
		print("\nWARNING: difference =",result,", not in [0,1]!") # this really should not occur! If it does, check if an issue of accuracy
		print("P1 basins:",P1_basins,"\nP2_bsins:",P2_basins,'\n\n')
	return result


def tune_dist(param_file, reps):
	# finds dist thresh such that max dist btwn 2 runs of the same net < dist thresh/2
	# TODO: add z-distrib, with enough samples it can be approx as such. Then return suggested val sT p(Z>val) <= given alpha

	assert(0) #TODO: update and debug this function before using 

	params = param.params(param_file)
	F, F_mapd, A, V  = param.net(params)
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


###########################################################################################

# TODO:
#	- add option to run ldoi w/o memoization where each row is diff (2nd run of ldoi will use with diff pinned As)
#		- and change intersection_LDOI() accordingly
#	- add a fn to sim from all attrs, to confirm anything this predicts (and later use for exh)
#	- explicitly debug unmemoized LDOI w/ more complicated toy model

# LATER:
#	standardize a whole lot, esp going in and out of ldoi (num or names ect)
#	handle wider cases such as no inputs
# 	add warning if curr mutations in settings file

def intersect_control(param_file, mutator):
	# curr singleton mutator & singleton controller
	# later: generalize to mult mutators, mult controllers
	#			return an order list of top k controllers
	params, G = basin.init(param_file)

	G.prepare_for_sim(params) # jp this is nec
	SS_WT = basin.calc_basin_size(params,G) # only need WT for which inputs corresp to which outputs
	target_IO = io_pairs(params,SS_WT, G)
	#print("target IO =",target_IO)

	params['mutations'][mutator[0]] = mutator[1] 
	G.prepare_for_sim(params)
	SS_M = basin.calc_basin_size(params,G)

	A_intersect = intersection_attractor(params, SS_M, G)
	#print("intersect_control: A_intersect=", A_intersect)
	ldoi_result = intersection_LDOI(params, A_intersect, G)
	#print("ldoi final result=",ldoi_result)
	controller_scores = score_controller_ldois(params,ldoi_result,target_IO, G)
	top=10
	top_controllers = dict(sorted(controller_scores.items(), key = itemgetter(1), reverse = True)[:top])
	print("\ntop controller scores=",top_controllers)
	
	return controller_scores

def io_pairs(params, SS, G):
	# take dominant output for each input set
	# returns dict IO = {inputSet: corresp output}
	IO = {}
	outpts = np.array([G.nodeNums[params['outputs'][i]] for i in range(len(params['outputs']))])
	input_sets,inpts = get_input_sets(params, G)
	largest = {inpt_set:0 for inpt_set in input_sets}
	for inpt_set in input_sets:
		for A in SS.attractors.values():
			A.id = np.array(list(map(int, A.id))) # str to numpy
			if np.all(A.id[inpts] == inpt_set):
				if A.size > largest[inpt_set]:
					largest[inpt_set] = A.size
					outpt = A.id[outpts]
					outpt[outpt > params['output_thresholds']] = 1
					outpt_names = []
					for o in range(len(outpt)):
						if outpt[o] == 0:
							outpt_names += [G.not_string + params['outputs'][o]]
						else:
							outpt_names += [params['outputs'][o]]
					IO[inpt_set] = outpt_names
	return IO


def intersection_attractor(params, SS_M, G,thresh=1):
	# for each input set, find the intersection of all attractors
	# oscils are not included
	# returns dict {inputSet: attractor}
	input_sets, inpts = get_input_sets(params, G)
	A_inter = {inpt_set:np.array([1 for i in range(2*G.n)],dtype=bool) for inpt_set in input_sets}

	for inpt_set in input_sets:
		for A in SS_M.attractors.values():
			A.id = np.array(list(map(int, A.id))) # str to numpy
			if np.all(A.id[inpts] == inpt_set):

				# for async and Gasync these might be slightly off...should fix
				assert(params['update_rule']=='sync')
				assert(np.all(A.avg<=1) and np.all(A.avg>=0))
				fixed_states = np.concatenate((A.avg,1-A.avg)) 
				# since looking at nodes on Gexp, only look for 1, not 0
				#	and remove all floats, which indicate oscils
				fixed_states[fixed_states < thresh] = 0 
				# both a node and its compl should not be on
				assert(np.all(1-(fixed_states.astype(bool) & np.roll(fixed_states,G.n).astype(bool)))) 
				A_inter[inpt_set] = np.logical_and(A_inter[inpt_set], fixed_states).astype(int)
				
		A_inter[inpt_set] = np.where(A_inter[inpt_set] == 1)[0]
	return A_inter 


def intersection_LDOI(params, A_intersect, G):
	# run LDOI over each input set, using a controller x the intersection_attr
	# rm each node in the intersection_attr if it negates itself (controller spc)
	# run LDOI again with this trimmed intersection_attr (x controller again)
	# return 2D dict {controller: {input_set: canalized_outputs}}

	Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])
	input_sets, inpts = get_input_sets(params, G)
	mutants = get_mutant_indices(params, G)

	#print('\n\nA FULL:\n',A_intersect)
	ldoi_solns, negates = ldoi.ldoi_sizes_over_all_inputs(params,Gpar,fixed_nodes=mutants,subsets=A_intersect)
	return ldoi_solns

def get_mutant_indices(params, G):
	assert(len(params['mutations'])>0) 
	inds = []
	for k in params['mutations']:
		if params['mutations'][k] == 0:
			inds+= [G.nodeNums[G.not_string + k]]
		elif params['mutations'][k] == 1:
			inds+= [G.nodeNums[k]]
		else:
			assert(0) # mutations should always be 0 or 1
	return inds


def score_controller_ldois(params, ldoi_result,target,G):
	# %correct ios
	scores = {}
	input_sets,inpts = get_input_sets(params, G)
	for i in range(G.n):
		for b in [0,1]:
			if b==0:
				name = G.nodeNames[i] + '=0'
				indx = i+G.n
				assert(indx <= G.n*2)
			else:
				name = G.nodeNames[i] + '=1' 
				indx = i
			scores[name]=0
			for inpt_set in input_sets: 
				match=True 
				for output_val in target[inpt_set]:
					if G.nodeNames[indx] not in ldoi_result[inpt_set] or output_val not in ldoi_result[inpt_set][G.nodeNames[indx]]:
						match=False 
						break
				if match:
					scores[name]+=1
			scores[name]/=len(input_sets)
	return scores

def get_input_sets(params, G):
	# been forgetting to use this \:
	inpts = np.array([G.nodeNums[params['inputs'][i]] for i in range(len(params['inputs']))])
	input_sets = itertools.product([0,1],repeat=len(inpts))
	return list(input_sets), inpts # assume inputs are a managable size

###########################################################################################

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 control.py PARAMS.yaml [exh, intersect]")
	
	if sys.argv[2] == 'exh':
		params = param.load(sys.argv[1])
		G = Net(model_file=params['model_file'],debug=params['debug'])
		assert(CONTROL_PARAMS['max_mutator_size']==1 and CONTROL_PARAMS['max_control_size']==1) # debug required before using larger sets

		perturbs = exhaustive(params, G, CONTROL_PARAMS['mut_thresh'], CONTROL_PARAMS['cnt_thresh'], max_mutator_size=CONTROL_PARAMS['max_mutator_size'], max_control_size = CONTROL_PARAMS['max_control_size'],norm=CONTROL_PARAMS['norm'])
		print('\n using these control params:',CONTROL_PARAMS)
	
	elif sys.argv[2] == 'intersect': 
		mutator=('Myc',1)   #('GAB1',0) #('d',0) #('Cytoc_APAF1',0)
		intersect_control(sys.argv[1], mutator)

	else:
		assert(0) # unrecognized 3rd arg