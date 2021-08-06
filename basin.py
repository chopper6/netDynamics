import itertools, util, math
import runNet
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


def calc_basin_size(params, clause_index):
	n = clause_index.shape[0]
	steady_states = {'oscil':0} #oscil means anything not a fixed point
	if params['exhaustive']:
		X0 = itertools.product([0,1], repeat=n)
		for x0 in X0:
			x0 = cp.array(x0,dtype=bool)
			result = runNet.from_init_val(params,clause_index,x0)
			add_to_steady_states(steady_states, result)
		for s in steady_states.keys():
			steady_states[s] /= math.pow(2,n)

	else:
		p = .5 #prob a given node is off at start
		for i in range(params['num_samples']):
			x0 = cp.random.choice(a=[0,1], size=(n), p=[p, 1-p]).astype(bool)
			result = runNet.from_init_val(params,clause_index,x0)
			add_to_steady_states(steady_states, result)
		for s in steady_states.keys():
			steady_states[s] /= params['num_samples']

	return steady_states


def add_to_steady_states(steady_states, result):
	if result['steady_state']:
		if result['state'] not in steady_states.keys():
			steady_states[result['state']]=1
		else:
			steady_states[result['state']]+=1
	else:
		stead_states['oscil'] += 1