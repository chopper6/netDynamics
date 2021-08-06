import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def from_init_val(params, clause_index, x):
	# x begins as init state is repeatedly updated
	for i in range(params['max_iters_per_sample']):

		x_next = cp.any(cp.all(x[clause_index],axis=2),axis=1)
		# THIS IS THE MAIN LINE OF ALGO
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
