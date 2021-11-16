import main, ldoi, multi, parse
import sys, math
import numpy as np

# TODO
# auto exclude all inputs and output, via an adjacency matrix maybe. Clause_mapping is too opaque jp...
#	also that zero node is nasty af sometimes
# distinguish C1 and C2 when return? jpp at least for now
# clean pheno file goddamn
# add treatment(s), LDOI, corr, features

def reverse_randomize(param_file):
	reps = 3
	labels = ['vanilla'] + ['random_' + str(r) for r in range(reps)]
	stats = {label:{} for label in labels}
	mutators, mutant_mag, controllers, control_mag = exhaustive(param_file,max_control_size=1)
	stats['vanilla'] = {'|mutators|':len(mutators), '|controllers|':len(controllers),'mutator_mag':mutant_mag, 'control_mag':control_mag}
	for r in range(reps):
		param_file2 = randomize_deg_preserve(param_file)
		mutators, mutant_mag, controllers, control_mag = exhaustive(param_file2,max_control_size=1)
		stats['random_' + str(r)] = {'|mutators|':len(mutators), '|controllers|':len(controllers),'mutator_mag':mutant_mag, 'control_mag':control_mag}
	plot.something(stats) #bars jp

def randomize_deg_preserve(param_file):
	return
	# should have an easier way to do this than write to file and reparse...right?
	# many times: pick 2 rd edges, and flip either in or out nodes x1, x2
	#	target nodes replaces every instance of x1 with x2 and vice versa


def exhaustive(param_file,max_control_size = 1):
	# i assume cannot alter pheno nodes, nor input nodes, and control of the mutated node is allowed
	# complexity is O(s*n^m+1*2^i) where s is the simulation time of a single input and i is the number of inputs, and m is the max control set size
	
	excluded = ['n1','p1','p2'] # TODO: auto exclude all inputs and output, via an adjacency matrix maybe. Clause_mapping is too opaque jp...
	params = parse.params(param_file)
	clause_mapping, node_mapping = parse.net(params)
	n = len(node_mapping['num_to_name'])
	nodes = node_mapping['num_to_name'][1:int(n/2)] #excluding 0 node and negative nodes (TODO cleaner?)
	nodes = [node for node in nodes if node not in excluded]

	params['verbose']=0 #i assume

	phenosWT = multi.input_product_sim(params)
	orig_mutated = params['phenos']['mutations']
	mutators, solution = [],[]
	mutant_mag, control_mag = 0,0

	for M in nodes:
		print('Testing node',M)
		for b in [0,1]:
			params['phenos']['mutations'][M] = b
			phenosM = multi.input_product_sim(params) 

			change = diff(phenosWT,phenosM)

			if change < 1:
				mutators += [(M,b)]
				mutant_mag += 1-change

				solutions, control_mag = try_and_fix(params, nodes, solutions, control_mag, phenosWT, [], (M,b), max_control_size)

				#corrs = check_corr(params, phenosWT, nodes, node_mapping['name_to_num']) # unfinished...
				
			params['phenos']['mutations'] = orig_mutated #reset

	mutant_mag /= len(nodes-1) #incld negative nodes but not the 0 node 
	control_mag /= len(nodes-1) 
	#A,n,N,node_mapping = parse.expanded_net(params['net_file'])
	#ldoi_solns, negated = ldoi.ldoi_bfs(A,n,N,pinning=1)
	print('mutators=',mutators)
	print('solutions=',solutions)
	return mutators, mutant_mag, solutions, control_mag



####################################################################################################

def check_corr(params, nodes, phenosWT):
	assert(0) #TODO: left off here
	for C in nodes:
		for inpt in phenosWT:
			x=1



def try_and_fix(params, nodes, solutions, phenosWT, control_set, control_mag, mutator, depth, max_depth): # TODO update this
	if depth == 0:
		return solutions, control_mag

	for C in nodes:
		orig_mutations = params['phenos']['mutations']
		for b in [0,1]:
			params['phenos']['mutations'][C] = b
			phenosC = multi.input_product_sim(params) 

			change = diff(phenosWT,phenosC):
			if change < 1:
				control_mag += 1-change
				solutions += [{'mutation':mutator,'control':control_set+[(C,b)]}]
			else:
				# recurse using this node as a control + another (until reach max_depth)
				solutions, control_mag = try_and_fix(params, nodes, solutions, phenosWT, control_set+[(C,b2)], mutator, depth-1) 

			params['phenos']['mutations'] = orig_mutations # reset (in case orig mutator was tried as a control node)

	return solutions, control_mag


def diff(P1, P2, norm=1):
	P1_basins, P2_basins = [],[]
	for i in P1:
		for o in P1[i]:
			P1_basins += P1[i][o]
			if i not in P2 or o not in P2[i]:
				P2_basins += [0]
			else:
				P2_basins += P2[i][o]
	for i in P2:
		for o in P2[i]:
			if i not in P1 or o not in P1[i]:
				# i.e only if skipped before
				P2_basins += P2[i][o]
				P1_basins += [0]
	
	P1_basins, P2_basins = np.array(P1_basins), np.array(P2_basins)
	if norm in ['inf','max']:
		norm = np.inf 

	return np.linalg.norm(P1_basins-P2_basins,ord=norm)



if __name__ == "__main__":
	exhaustive(sys.argv[1])
