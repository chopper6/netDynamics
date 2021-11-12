import main, parse, plot
import sys, os, itertools
from copy import deepcopy


def variance_test(param_file):
	# TODO: fix avg & var calc, organize io/visualize, test net generator
	# poss bar chart for avg_A [var(x1)], grouping PP, PN, NN's together 
	# later need variable noise lvls
	reps=1
	params = parse.params(param_file)
	params['net_file'] = 'input/temp_net.txt'
	params['verbose'] = False

	stats = {'avg':[],'var':[]}
	groups = ['PP_and','NN_and','PN_and','PP_or','NN_or','PN_or','P','N']
	feats = {'noiseless':{g:deepcopy(stats) for g in groups},'noisy':{g:deepcopy(stats) for g in groups}}

	for noise in [False, True]:
		params['PBN']['active'] = noise
		if noise:
			noise_str='noisy'
		else:
			noise_str = 'noiseless'

		settings = itertools.product([0,1],repeat=5) 
		for s in settings:
			if s[1:] not in [(0,0,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),(1,1,0,1)]: #remove symmetries over the middle node's plane
				generate_coupled_FBL(params['net_file'], s[0], s[1], s[2], s[3], s[4])
				loopType, gate = coupled_loop_type(s[0],s[1], s[2], s[3], s[4])
				calc_stats(params, reps, noise_str, loopType, gate, feats)
		for s in [[0,1],[1,1],[0,0],[1,0]]:
			generate_FBL(params['net_file'], s[0], s[1])
			loopType = loop_type(s[0],s[1])
			calc_stats(params, reps, noise_str, loopType, None, feats)

	plot.probably_bar(params,feats)

	os.remove('input/temp_net.txt')

def calc_stats(params, reps, noise_str, loopType, gate, feats):
	avg_avg_var = 0
	avg_avg_avg = 0

	if params['update_rule'] == 'sync' and not params['PBN']['active']:
		var_str = 'var' #since not averaged anyway
		avg_str = 'avg'
	else:
		var_str = 'totalVar' #make sure to get the non-averaged form
		avg_str = 'totalAvg'

	for r in range(reps):
		attractors, phenos, node_mapping = main.find_attractors(params)
		middle_node = node_mapping['name_to_num']['x1']
		avg_variance=0 # sum_nodes sum_As var_node_in_A  / #nodes #As
		avg_avg = 0

		for k in attractors.keys():
			avg_variance += attractors[k][var_str][middle_node]
			avg_avg += attractors[k][avg_str][middle_node]

		if not params['update_rule'] == 'sync' or params['PBN']['active']: 
			normz = params['num_samples'] 
		else:
			normz = len(attractors) #since sync w/o noise only takes 1 'sample', since is deterministic 

		avg_avg_var += avg_variance / normz
		avg_avg_avg  += avg_avg / normz

		if params['debug'] and params['update_rule'] == 'sync':
			assert(0 <= avg_avg_avg <= 1)

		if False: #this is for var of all nodes
			for k in attractors.keys():
				avg_variance += sum(attractors[k][var_str])
				avg_avg += sum(attractors[k][avg_str])
			num_nodes = len(attractors[k][var_str])
			avg_avg_var += avg_variance / (params['num_samples'] * num_nodes )
			avg_avg_avg  += avg_avg / (params['num_samples'] * num_nodes )

	avg_avg_var/=reps
	avg_avg_avg/=reps

	if gate is None:
		feats[noise_str][loopType]['avg'] += [avg_avg_avg]
		feats[noise_str][loopType]['var'] += [avg_avg_var]	
	else:
		feats[noise_str][loopType+gate]['avg'] += [avg_avg_avg]
		feats[noise_str][loopType+gate]['var'] += [avg_avg_var]

def generate_coupled_FBL(net_file, AND, E21, E31, E12, E13): 
	#E's are for edges, so E21=0 -> x2 -| x1
	# AND is if middle node is an AND of the other two nodes (else uses OR)
	with open(net_file,'w') as file:
		file.write('DNFsymbolic\n')
		if AND:
			joiner = '&'
		else:
			joiner = ' '
		file.write('x1\t'+sign(E21)+'x2'+joiner+sign(E31)+'x3\n')
		file.write('x2\t'+sign(E12)+'x1\n')
		file.write('x3\t'+sign(E13)+'x1')

def coupled_loop_type(logic, E21, E31, E12, E13):
	if logic:
		gate = '_and'
	else:
		gate = '_or'
	sign2loop = (E21+E12)%2 #where 0 is P, 1 is N
	sign3loop = (E31+E13)%2
	if sign3loop==sign2loop==1:
		return 'NN',gate
	elif sign3loop==sign2loop==0:
		return 'PP',gate
	else:
		return 'PN',gate

def sign(b):
	if b:
		return ''
	else:
		return '-'

def generate_FBL(net_file, E21, E12): 
	with open(net_file,'w') as file:
		file.write('DNFsymbolic\n')
		file.write('x1\t'+sign(E21)+'x2\n')
		file.write('x2\t'+sign(E12)+'x1')

def loop_type(E21, E12):
	if E12==E21:
		return 'P'
	else:
		return 'N'


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 probably.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	variance_test(sys.argv[1])