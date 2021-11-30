import ldoi, parse, plot, logic, basin, features
import sys, math, pickle
import numpy as np
import random as rd
from copy import deepcopy
from net import Net


class Perturbations:
	def __init__(self, params, G,mut_thresh, cnt_thresh,max_control_size,phenosWT):
		self.mutators, self.solutions, self.controllers = [],[],[]
		self.mut_thresh = mut_thresh
		self.cnt_thresh = cnt_thresh 
		self.filtered_nodes = [node.name for node in G.regularNodes if node not in params['outputs'] and node not in params['inputs']]
		self.phenosWT = phenosWT
		self.max_control_size = max_control_size

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


	def finalize(self,perturbs):
		# adds last few metrics and normalizes
		self.num_mutators = len(perturbs.mutators)
		m=self.num_mutators
		self.num_solutions = len(perturbs.solutions)
		self.num_controllers = len(perturbs.controllers)
		self.percent_mutators = self.num_mutators/self.G.n_neg 
		self.percent_solutions = self.num_solutions/max(m*G.n_neg,1) 
		self.percent_controllers = self.num_controllers/self.G.n_neg  

		self.fragility /= self.G.n_neg
		self.reversibility /= max(m,1) 

	
def exhaustive(params, G, mut_thresh, cnt_thresh, max_control_size = 1):

	# complexity is roughly O(s*n^m+1) where s = simulation time, n=#nodes, m=max control set size
	# assumes cannot alter pheno nodes, nor input nodes

	params['verbose']=0 #i assume
	params['use_inputs_in_pheno']=True #also assume

	phenosWT = mutate_and_sim(params, G).phenotypes
	metrics = NetMetrics(params, phenosWT, G)
	perturbs = Perturbations(params,G,phenosWT,mut_thresh,max_control_size, cnt_thresh)
	orig_mutated = deepcopy(params['mutations'])

	for M in perturbs.filtered_nodes:
		print('Testing node',M)
		for b in [0,1]:
			params['mutations'][M] = b
			phenosM = mutate_and_sim(params, G).phenotypes

			mut_dist = diff(phenosWT,phenosM)
			metrics.fragility += mut_dist

			if mut_dist  > mut_thresh:
				perturbs.mutators += [(M,b)]

				revblty = try_and_fix(params,G, metrics, perturbs, (M,b), mut_dist, [], depth)
				metrics.reversibility += reversibility

			params['mutations'] = deepcopy(orig_mutated) #reset
	
	metrics.finalize(perturbs)

	# TODO: change to a default print(netMetrics) function
	print('\n#mutators=',metrics.num_mutators)
	print('#solutions=',metrics.num_solutions)
	print('mutation mag = ', metrics.fragility, ', control mag = ', metrics.reversibility)
	print('\nall solutions:',perturbs.solutions)

	return metrics, perturbs


def mutate_and_sim(params, G_orig):
	G = deepcopy(G_orig)
	G.apply_mutations(params) 
	steadyStates = basin.find_steadyStates(params, G)
	return steadyStates

####################################################################################################

def try_and_fix(params,G, metrics, perturbs, mutator, mut_dist, control_set, depth):
	# note that mutator and control_set are not yet in perturbs since they are actively build/altered by this function 
	if depth == 0:
		return revblty
	
	reversibility = 0 # TODO: fix for recursion (not sure how to weight pr of larger sets)

	# WARNING: ldoi stats won't work with larger depth
	#ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,F,V,fixed_nodes=[mutator])

	for C in perturbs.filtered_nodes:
		orig_mutations = deepcopy(params['mutations'])
		for b in [0,1]:
			if (C,b) != mutator and (C,b) not in control_set:
				params['mutations'][C] = b
				phenosC = mutate_and_sim(params, G).phenotypes

				#change = 1-diff(phenosWT,phenosC)/mut_dist
				control_dist = diff(perturbs.phenosWT,phenosC)
				reversibility += max(0,mut_dist-control_dist)
				if control_dist < mut_dist - perturbs.control_thresh:  
					#print('\t',C,'=',b,'cnt_dist=',round(control_dist,3),'mut_dist=', round(mut_dist,3),'vs orig mutation',mutator[0],'=',mutator[1])
					pertubs.solutions += [{'mutation':mutator,'control':control_set+[(C,b)]}]
					if (C,b) not in perturbs.controllers:
						perturbs.controllers +=[(C,b)]
				else:
					# recurse using this node as a control + another (until reach max_depth)
					# note that recursed revblty is currently not used
					revblty = try_and_fix(params,G, metrics, perturbs, mutator, mut_dist, control_set+[(C,b)], depth-1)

				params['mutations'] = deepcopy(orig_mutations) # reset (in case orig mutator was tried as a control node)

	return reversibility/(2*len(perturbs.filtered_nodes)-1) # -1 since will not use the orig mutation for control (but will try flipping it)


def diff(P1, P2, norm=1):
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
	if norm in ['inf','max']:
		norm = np.inf 

	result= np.linalg.norm(P1_basins-P2_basins,ord=norm)
	if norm == 1:
		result/=2 # will double count any overlap
		# not sure how to normalize this for other norms!
	if not (result >= -.01 and result <= 1.01): # may not hold for norms that are norm max or 1, not sure how to normalize them
		print("\nWARNING: difference =",result,", not in [0,1]!") # this really should not occur! If it does, check if an issue of accuracy
		#print("P1 basins:",P1_basins,"\nP2_bsins:",P2_basins,'\n\n')
	return result


def tune_dist(param_file, reps):
	# finds dist thresh such that max dist btwn 2 runs of the same net < dist thresh/2

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
	if len(sys.argv) < 2:
		sys.exit("Usage: python3 control.py PARAMS.yaml [exh | tune]")
	if sys.argv[2] == 'exh':
		if len(sys.argv) != 5:
			sys.exit("Usage for exhaustive run: python3 control.py PARAMS.yaml exh mutation_threshold control_threshold")
		params = parse.params(sys.argv[1])
		G = Net(params)
		mut_thresh, cnt_thresh = float(sys.argv[3]), float(sys.argv[4])
		exhaustive(params, G, mut_thresh, cnt_thresh, max_control_size = 1)
	elif sys.argv[2] == 'tune':
		dist_thresh = tune_dist(sys.argv[1], 100)
		print("suggested distance threshold =",dist_thresh) 
	else:
		assert(0)