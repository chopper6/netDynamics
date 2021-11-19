import main, ldoi, parse, plot
import sys, math, pickle
import numpy as np
import random as rd
from copy import deepcopy

# TODO
# test based DEBUG with more than just main
#	why tf does it speed up as it goes thru nodes....?
# so far don't see any C1 nodes for bladder cancer (that completely fix all inputs)...likely an err somewhere
# return net_file to main params, model param rename to 'setting' (optional), and sep folders for params, nets, and settings
# parse other model
# add treatment(s), LDOI, corr, features
# sep randomize into new file
# auto divide # runs by input, and have a toggle to evenly try all inputs [jp need now for speed reasons]
#	then clean IOs, sep toggle from phenos
#	poss deeper than just at 'multi', since 'main' runs would likely also want to use

# TODO before thurs
# check over main points/ double check test-based
# give toy example of what I'm measuring and rd process

# DO LATER UNLESS NEC NOW
# zero node is nasty af sometimes
#		isn't a self loop and an init condition sufficient?
#		and auto add it to 'init' in model file 
# logic.DNF_cell_collective: handle edge cases like 1 clause (jp) and constants
# clean the shitstorm that is multi.py
# just apply_mutations from parse instead of parsing network from scratch each time main calc attractors is called
# add time stamped directory or something
# forcing parallelism is bad for accessibility (basin/parse.py)

def randomize_experiment(param_file):
	reps = 2
	num_swaps = 10
	dist_thresh = .01
	labels = ['vanilla'] + ['random_' + str(r) for r in range(reps)]
	stats = {label:{} for label in labels}

	params = parse.params(param_file)
	F, V = parse.get_logic(params)
	n2=len(V['#2name'])-2 #inclds composite nodes but not 0 node

	mutators, mutant_mag, controllers, control_mag = exhaustive(params, F, V, dist_thresh=dist_thresh)
	stats['vanilla'] = {'%mutators':len(mutators)/n2, 'control_per_mutant':len(controllers)/max(len(mutators),1),'mutator_mag':mutant_mag, 'control_mag':control_mag}
	
	for r in range(reps):
		parse.apply_mutations(params,F)
		F_mapd, A = parse.get_clause_mapping(params, F, V) 
		if params['debug']:
			A_copy = A.copy()
		swap_edges(F,A,V,num_swaps,no_repeat_edges=True) #modifies F and A in place 
		if params['debug']:
			# check is degre preserving
			assert(np.all(np.sum(A_copy,axis=0)== np.sum(A,axis=0)) and np.all(np.sum(A_copy,axis=1)== np.sum(A,axis=1)))
		mutators, mutant_mag, controllers, control_mag = exhaustive(params, F, V, dist_thresh=dist_thresh)
		stats['random_' + str(r)] = {'%mutators':len(mutators)/n2, 'control_per_mutant':len(controllers)/max(len(mutators),1),'mutator_mag':mutant_mag, 'control_mag':control_mag}
	
	pickle_file = params['output_dir'] + '/stats.pickle' 
	print("Pickling a file to: ", pickle_file)
	pickle.dump(stats, open( pickle_file, "wb" ))
	plot.control_exper_bar(params, stats) 


def swap_edges(F,A,V,repeats,no_repeat_edges=True):
	# pick 2 random edges, then pick source or targets, then swap their variables
	# note that logic is preserved, albiet with different sources
	# modifies F and A in place
	for r in range(repeats):
		E1 = get_rd_edge(A)
		try_again = True
		loop=0
		while try_again : #ensure they are diff edges
			E2 = get_rd_edge(A)
			try_again = np.all(E1==E2)
			if no_repeat_edges and (A[E1[0],E2[1]]!=0 or A[E2[0],E1[1]]!=0): 
				#TODO: should repeat edges check the new or the old A?? curr using updated A
				try_again = True
			loop+=1
			assert(loop<10**5)

		source1, source2, target1, target2 = V['#2name'][E1[0]], V['#2name'][E2[0]], V['#2name'][E1[1]], V['#2name'][E2[1]]

		swap_literals(F[target1],source1,source2)
		swap_literals(F[target2],source2,source1)
		
		# rm old, add new edges (assumes unsigned)
		A[E1[0],E1[1]]=0
		A[E2[0],E2[1]]=0	
		A[E1[0],E2[1]]=1
		A[E2[0],E1[1]]=1


def swap_literals(logic_fn,lit1,lit2):
	# swaps lit1 for lit2 in the logical function
	for clause in logic_fn: 
		for i in range(len(clause)):
			if clause[i] == lit1:
				clause[i] = lit2



def get_rd_edge(A):
	sign=0
	loop=0
	while sign==0:
		assert(loop<10**10) 
		E = np.random.randint(len(A), size=2)
		sign = A[E[0],E[1]]
		loop+=1
	return E



def exhaustive(params, F, V,max_control_size = 1, dist_thresh=.2):
	# assumes cannot alter pheno nodes, nor input nodes
	# complexity is O(s*n^m+1*2^i) where s is the simulation time of a single input and i is the number of inputs, and m is the max control set size

	# TODO: clean implementation of dist thresh
		# should be some stat signif val, i.e. if run same run twice with 95% conf should not exceed dist
	# another issue is mult hypoth testing, since will by trying many of these (could retry those that work or smthg)

	n = len(V['#2name'])
	nodes = V['#2name'][1:int(n/2)] #excluding 0 node and negative nodes (TODO cleaner?)
	nodes = [node for node in nodes if node not in params['outputs'] and node not in params['inputs']]

	params['verbose']=0 #i assume

	steadyStates = main.find_attractors_prebuilt(params, F, V).phenotypes
	orig_mutated = deepcopy(params['mutations'])
	mutators, solutions = [],[]
	mutant_mag, control_mag = 0,0

	for M in nodes:
		print('Testing node',M)
		for b in [0,1]:
			params['mutations'][M] = b
			phenosM = main.find_attractors_prebuilt(params, F, V).phenotypes 

			mut_dist = diff(phenosWT,phenosM)

			if mut_dist  > dist_thresh:
				#print([(k,phenosM[k]['size']) for k in phenosM])
				mutators += [(M,b)]
				mutant_mag += mut_dist

				solutions, control_mag = try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, [], control_mag, (M,b), max_control_size,dist_thresh)

				#corrs = check_corr(params, phenosWT, nodes, V['name2#']) # unfinished...
				
			params['mutations'] = deepcopy(orig_mutated) #reset

	mutant_mag /= len(nodes)-1 #incld negative nodes but not the 0 node 
	control_mag /= len(nodes)-1 
	#A,n,N,V = parse.expanded_net(params['net_file'])
	#ldoi_solns, negated = ldoi.ldoi_bfs(A,n,N,pinning=1)
	print('\n#mutators=',len(mutators))
	print('\n#solutions=',len(solutions))
	print('\nmutation mag = ', mutant_mag, ', control mag = ', control_mag)
	print(solutions[:10])
	return mutators, mutant_mag, solutions, control_mag



####################################################################################################

def check_corr(params, nodes, phenosWT):
	assert(0) #TODO: left off here
	for C in nodes:
		for inpt in phenosWT:
			x=1



def try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, control_set, control_mag, mutator, depth,dist_thresh): 
	if depth == 0:
		return solutions, control_mag

	for C in nodes:
		orig_mutations = deepcopy(params['mutations'])
		for b in [0,1]:
			if (C,b) != mutator and (C,b) not in control_set:
				params['mutations'][C] = b
				phenosC = main.find_attractors_prebuilt(params, F, V).phenotypes

				change = 1-diff(phenosWT,phenosC)/mut_dist
				#print([(k,phenosC[k]['size']) for k in phenosC])
				if change > dist_thresh:
					control_mag += change
					solutions += [{'mutation':mutator,'control':control_set+[(C,b)]}]
				else:
					# recurse using this node as a control + another (until reach max_depth)
					solutions, control_mag = try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, control_set+[(C,b)], control_mag, mutator, depth-1,dist_thresh) 

				params['mutations'] = deepcopy(orig_mutations) # reset (in case orig mutator was tried as a control node)

	return solutions, control_mag


def diff(P1, P2, norm='max'):
	P1_basins, P2_basins = [],[]
	for io in P1:
		P1_basins += [P1[io]['size']]
		if io not in P2:
			P2_basins += [0]
		else:
			P2_basins += [P2[io]['size']]
	for io in P2:
		if io not in P1:
			# i.e only if skipped before
			P2_basins += [P2[io]['size']]
			P1_basins += [0]
	
	P1_basins, P2_basins = np.array(P1_basins), np.array(P2_basins)
	if norm in ['inf','max']:
		norm = np.inf 

	return np.linalg.norm(P1_basins-P2_basins,ord=norm)



if __name__ == "__main__":
	if sys.argv[2] == 'exh':
		params = parse.params(sys.argv[1])
		F, V = parse.get_logic(params)
		exhaustive(params, F, V)
	elif sys.argv[2] == 'rd':
		randomize_experiment(sys.argv[1]) 
	else:
		assert(0)