import main, ldoi, parse
import sys, math

def exhaustive(param_file):
	# i assume cannot alter pheno nodes, nor input nodes, and control of the mutated node is allowed
	# complexity is O(s*n^m+1*2^i) where s is the simulation time of a single input and i is the number of inputs, and m is the max control set size

	max_control_size = 2
	sep_inputs = False
	excluded = ['n1','p1','p2']
	params = parse.params(param_file)
	clause_mapping, node_mapping = parse.net(params)
	n = len(node_mapping['num_to_name'])
	nodes = node_mapping['num_to_name'][1:int(n/2)] #excluding 0 node and negative nodes (TODO cleaner?)
	nodes = [node for node in nodes if node not in excluded]

	params['verbose']=0 #i assume

	phenosWT = input_product_sim(params,sep=sep_inputs)

	orig_mutated = params['phenos']['mutations']
	mutators = []
	if sep_inputs:
		solutions = {0:[],1:[],'both':[]}
	else:
		solutions = []

	for M in nodes:
		print('Testing node',M)
		for b in [0,1]: #TODO itertools
			params['phenos']['mutations'][M] = b
			phenosM = input_product_sim(params,sep=sep_inputs) 

			if sep_inputs:
				changed=False
				for b2 in [0,1]: #TODO itertools
					changed = changed | diff(phenosWT[b2],phenosM[b2])
			else:
				changed = diff(phenosWT, phenosM)

			if changed:
				mutators += [(M,b)]

				solutions = try_and_fix(params, nodes, solutions, phenosWT, [], (M,b), 1, max_control_size,sep=sep_inputs)

				if sep_inputs:
					corrs = check_corr(params, phenosWT, nodes, node_mapping['name_to_num'])
				
			params['phenos']['mutations'] = orig_mutated #reset

	#A,n,N,node_mapping = parse.expanded_net(params['net_file'])
	#ldoi_solns, negated = ldoi.ldoi_bfs(A,n,N,pinning=1)
	print('mutators=',mutators)
	print('solutions=',solutions)



####################################################################################################

def check_corr(params, nodes, phenosWT):
	assert(0) #TODO: left off here
	for C in nodes:
		for inpt in phenosWT:
			x=1



def try_and_fix(params, nodes, solutions, phenosWT, control_set, mutator, depth, max_depth,sep=False):
	if depth > max_depth:
		return solutions

	for C in nodes:
		for b2 in [0,1]:
			params['phenos']['mutations'][C] = b2
			phenosC = input_product_sim(params,sep=sep) 

			if not sep:
				if not diff(phenosWT, phenosC):
					solutions += [{'mutation': mutator,'control':control_set+[(C,b2)]}]
					print("Found solution to",mutator,":",control_set+[(C,b2)])
			else:
				both=True
				for b in [0,1]:
					if not diff(phenosWT[b],phenosC[b]):
						solutions[b] += [{'mutation':(M,b),'control':control_set+[(C,b2)]}]
					else:
						both=False
				if both:
					solutions['both'] += [{'mutation':(M,b),'control':control_set+[(C,b2)]}]
			
			solutions = try_and_fix(params, nodes, solutions, phenosWT, control_set+[(C,b2)], mutator, depth+1, max_depth,sep=sep) 



def diff(P1, P2, err=0.1):
	for k in P1:
		if k not in P2:
			return True
		else:
			if not math.isclose(P1[k]['size'], P2[k]['size'], abs_tol=err):
				return True
	return False


# should prob put this function in a diff file
def input_product_sim(params,sep=False):
	allPhenos={}
	assert(len(params['phenos']['init'])) #TODO change to itertools product for > 1. Also changed input string in io_phenos()
	for inpt in params['phenos']['init']:
		for b in [0,1]: 
			params['phenos']['init'][inpt] = b
			attractors, phenos, node_mapping = main.find_attractors(params)

			if not sep:
				merge_phenos(allPhenos,phenos)
			else:
				io_phenos(allPhenos,phenos,b)
	if not sep:
		normz_phenos(allPhenos, 2**len(params['phenos']['init']))
	return allPhenos

def io_phenos(IOs, new_phenos, input_str):
	assert(input_str not in IOs.keys())
	IOs[input_str] = {}
	for k in new_phenos:
		assert(k not in IOs[input_str].keys())
		IOs[input_str][k] = new_phenos[k]

def merge_phenos(A,B):
	# takes 2 dicts of phenos & merges B into A
	for k in B.keys():
		if k in A.keys():
			A[k]['size'] += B[k]['size']
			for k2 in B[k]['attractors'].keys():
				if k2 in A[k]['attractors'].keys():
					A[k]['attractors'][k2]['size'] += B[k]['attractors'][k2]['size']
				else:
					A[k]['attractors'][k2] = B[k]['attractors'][k2]
		else:
			A[k] = B[k]

def normz_phenos(A,l):
	for k in A.keys():
		A[k]['size'] /= l

if __name__ == "__main__":
	exhaustive(sys.argv[1])
