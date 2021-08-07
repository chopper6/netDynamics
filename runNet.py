import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def from_init_val(params, clause_index, x, num_neg_nodes):
	# x begins as init state is repeatedly updated
	for i in range(params['max_iters_per_sample']):
		
		X = cp.concatenate((x,cp.logical_not(x[:num_neg_nodes])))
		# because of how clause_index is structured, the next line only produces output for all nodes, but not their negatives
		# however, it requires the negative nodes as separate inputs

		x_next = cp.any(cp.all(X[clause_index],axis=2),axis=1)
		# THIS IS THE MAIN LINE OF ALGO
		#	all: are all elements of the clause correct? if so, clause is TRUE
		#	any: are any of the clauses correct? if so, node is ON
		# note that CNF would just be np.all(np.any(...))

		if cp.array_equal(x,x_next):
			return {'steady_state':True, 'state':format_state_str(x_next)}
		x = x_next
	return {'steady_state':False, 'state':format_state_str(x_next)}



def format_state_str(x):
	label=''
	for ele in x:
		if ele == True:
			label+='1'
		else:
			label+='0'
	return label
