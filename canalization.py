import lap, basin, util, param, net, ldoi, control
import sys
from copy import deepcopy
import numpy as np

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# TODO
# ikey/okey of an attr should be handled in basin.py (ie standardized)
#		similar to id, maybe just put it in attr as soon as make it
#		but end up using it for the intersection A so i dunno
# major pheno should maybe be in basin? or make a new module?
# compare #steps reqd to async

# what was i talking about here? : apply mutations by directly pinning in attractor intersection
# and first avg, then use a threshold BEFORE making intersection A


def contextual_canalization(Gpar, A0, pins=None):
	# for now just calls a modified ldoi
	# eventually want to implement without the expanded network form
	soln, negated = ldoi.ldoi_bfs(Gpar,A0=A0,pins=pins)
	if CUPY:
		soln = soln.get()
	return soln

def usage_example(param_file):
	# example of how to use contextual canalization with a specific starting attractor

	params = param.load(param_file)
	Gpar = net.ParityNet(params) # contextual canalization uses the parity net form
	
	A0 = np.random.choice([0,1,2],size=Gpar.n) # set the state of each node, in this case randomly
	A0 = ldoi.build_A0(Gpar,A0) # convert to A0 on expanded network
	
	soln = contextual_canalization(Gpar, A0, pins=params['mutations']) 
	# note that pins should use a dictionary of {'node_name':pinned_value}

	# soln returns a matrix, where each row are the fixed nodes from a different controller
	# this can be changed into the node names as follows
	fixed_nodes = {}
	for i in range(len(soln)): # for each controller
		fixed = np.where(soln[i]==1)[0] # get the indices of fixed nodes
		control_node = Gpar.nodeNames[i]
		fixed_nodes[control_node] = Gpar.node_vector_to_names(fixed) # convert those indices into names

	return fixed_nodes


def contextual_canalization_control(param_file, num_candidates=4, use_mutated_net=False, detailed_return=False):
	# majority control approach using contextual canalization
	# returns all nodes sorted by their scores (best first), and their corresponding scores
	# num_candidates is just the number that are printed to console

	params = param.load(param_file)
	mutations = params['mutations'].copy()  # {'p53_PTEN':0} #'PTEN':0}
	params['mutations'] = {} # first run without the mutations
	G = net.Net(params)
	Gpar = net.ParityNet(params)		
	assert(G.n==Gpar.n) #otherwise parity file doesn't match the regular model file

	# run healthy network
	SS_healthy = basin.measure(params, G)
	healthy_dom_pheno = build_dominant_pheno(G, SS_healthy)

	if use_mutated_net: # by default, the intersection attractor does not use the mutated network
		# run mutated network
		params['mutations'] = mutations
		G.prepare(params) 
		SS_mutant = basin.measure(params, G, SS0=SS_healthy)

		# note that control skips mutated network and just pins the mutated nodes during contextual canalization
		A_intersect = build_intersection_attractors(params, Gpar, SS_mutant)
	else:
		A_intersect = build_intersection_attractors(params, Gpar, SS_healthy)

	scores = np.zeros((G.n*2)) 
	# one score per poss controller: 2* num nodes, since for each x_i use x_i=1 and x_i=0

	if detailed_return:
		#input_set_dict = {str(Gpar.input_names_from_vector(s)):[] for s in Gpar.get_input_sets()}
		input_set_dict = {s:[] for s in Gpar.get_input_sets()}
		fixed_nodes = {k:input_set_dict.copy() for k in Gpar.nodeNames} 

	# find controllers by running contextual canalization on A_intersect of each input condition
	input_ind = G.input_indices()
	output_ind = G.output_indices()
	for A0 in A_intersect.values(): # one per input condition
		canal_soln = contextual_canalization(Gpar, A0, pins=params['mutations'])

		ikey = str([int(x) for x in A0[input_ind]])
		#inames =  str(Gpar.input_names_from_vector(A0[input_ind]))
		inames =  str(ikey)
		major_okey = healthy_dom_pheno[ikey]
		canal_okeys = canal_soln[:,output_ind]

		scores += np.all((canal_okeys==major_okey),axis=1).reshape(G.n*2)
	
		if detailed_return:
			node_inds = np.array([i for i in range(Gpar.n_neg)])
			for i in range(len(canal_soln)):
				fixed = np.where(canal_soln[i]==1)[0]
				fixed_nodes[Gpar.nodeNames[i]][inames] = Gpar.node_vector_to_names(fixed)

	scores /= len(A_intersect.keys())  # normalize by number of input sets
	top_nodes = [x for _, x in sorted(zip(scores, Gpar.nodeNames))]
	top_scores = sorted(scores)
	top_nodes.reverse()
	top_scores.reverse()
	print("Top nodes=",top_nodes[:num_candidates],"\nwith scores=",top_scores[:num_candidates])


	if detailed_return:
		return top_nodes, top_scores, fixed_nodes, A_intersect
	else:
		return top_nodes, top_scores


def build_dominant_pheno(G, SS):
	# returns dict {ikey:dominant_okey}
	input_ind = G.input_indices()
	output_ind = G.output_indices()
	count = {}
	okeys = {}
	for A in SS.attractors.values():
		ikey = str([int(x) for x in A.avg[input_ind]])
		okey = [int(x) for x in A.avg[output_ind]]
		okeys[str(okey)] = np.array(okey)
		if ikey not in count:
			count[ikey] = {}
		if str(okey) not in count[ikey]:
			count[ikey][str(okey)] = A.size
		else:
			count[ikey][str(okey)] += A.size
	dominant = {}
	for ikey in count:
		max_val = 0
		for okey in count[ikey]:
			if count[ikey][str(okey)] > max_val:
				max_val = count[ikey][str(okey)]
				dominant[ikey] = okeys[str(okey)]

	return dominant 

def build_intersection_attractors(params, G, SS):
	assert(isinstance(G,net.ParityNet)) # can change this, but careful 
	inpt_ind = G.input_indices()
	input_orgnzd = {} #{str(k):[] for k in G.get_input_sets(params)}
	for A in SS.attractors.values():
		input_key = str([int(x) for x in A.avg[inpt_ind]])
		if input_key not in input_orgnzd.keys():
			input_orgnzd[input_key] = []
		A_certain = ldoi.build_A0(G,A.avg)
		input_orgnzd[input_key] += [A_certain]

	A_intersect = {}
	for k in input_orgnzd:
		if CUPY:
			stacked = np.array([input_orgnzd[k][i].get() for i in range(len(input_orgnzd[k]))])
		else:
			stacked = np.array([input_orgnzd[k][i] for i in range(len(input_orgnzd[k]))])
		# for each node (col), if all attrs (rows) are not the same, then all equal 2
		isSame = np.array([np.all(np.equal(stacked[:,i],stacked[0,i])) for i in range(len(stacked[0]))]) # should be a better numpy way...
		A_intersect[k] = input_orgnzd[k][0]
		A_intersect[k][isSame==False] = 2 
	
	return A_intersect

def apply_inits_to_intersect(params,G,A_intersect):
	if 'init' in params.keys():
		init_nodes = [G.nodeNums[k] for k in params['init']]
		init_vals = [params['init'][G.nodeNames[k]] for k in init_nodes]

		# then add the complements
		init_vals += [1-params['init'][G.nodeNames[k]] for k in init_nodes]
		init_nodes += [G.nodeNums[k]+G.n for k in params['init']]

		init_vals = np.array(init_vals)
		init_nodes = np.array(init_nodes)

		if len(init_nodes) > 0:
			for k in A_intersect:
				A_intersect[k][init_nodes] = init_vals

def apply_pins(X,n):
	# where n is number of nodes (i.e. 1/2 height of X)
	assert(len(X)==n*2)
	cp.fill_diagonal(X,1)
	X_rolled = cp.roll(X,n,axis=0)
	cp.fill_diagonal(X_rolled,0)
	X = cp.roll(X_rolled,n,axis=0) 
	return X


if __name__ == "__main__":
	if len(sys.argv) not in [2]:
		sys.exit("Usage: python3 canalization.py PARAMS.yaml")
	
	top_nodes, top_scores = contextual_canalization_control(sys.argv[1])