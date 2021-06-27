import itertools, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def from_init_val(params, adj, x):
	# x begins as init state is repeatedly updated
	if CUPY:
		x = cp.array(x, dtype=cp.float16) #else adj x are diff types -> err

	for i in range(params['max_iters_per_sample']):
		#print('dot:',adj,'.',x,'=',cp.dot(adj,x))
		x_next = cp.isclose(1,cp.dot(adj,x),rtol=1e-05, atol=1e-08)
		#isclose used to avoid possible rounding errors
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
