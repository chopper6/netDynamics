import basin, param, plot
from net import Net
import sys, os, itertools
from copy import deepcopy

# LATER:
# try longer loops, see if more robust
#	curr doing all combos, but should really isolate certain prop into sep imgs
#	mult bars per net type may also not be nec (or just show 1x)
# add confidence intervals, after several runs
# larger AND/OR gates (ie less rev'ble)
# pickling and pickle-ploting
# within vs btwn As
# consider var of more than just the pivot node?
# poss build longer regular FBLs

def variance_test(param_file):
	params = param.load(param_file)
	print("Simulating loops with flip probability=",params['PBN']['flip_pr'],', for',params['loopy_reps'],'repeats.')
	params['model_file'] = 'models/temp_net.txt'
	params['verbose'] = False
	params['inputs'],params['outputs'],params['init']={},{},{} #override
	params['temporal_var'] = True
	params['skips_precise_oscils'] = True
	reps=params['loopy_reps']

	stats = {'average':[],'ensemble variance':[], 'temporal variance':[], 'variance':[]}
	groups = ['P','PP_and','PP_or','PP_xor','PN_and','PN_or','PN_xor','NN_and','NN_or','NN_xor','N']
	groups += ['PP_and_long','PP_or_long','PP_xor_long','PN_and_long','PN_or_long','PN_xor_long','NN_and_long','NN_or_long','NN_xor_long']
	feats = {'no noise':{g:deepcopy(stats) for g in groups},'noisy':{g:deepcopy(stats) for g in groups}}

	for noise in [False, True]:
		print("Starting with noise",noise)
		params['PBN']['active'] = noise
		if noise:
			noise_str='noisy'
		else:
			noise_str = 'no noise'

		settings = list(itertools.product([0,1],repeat=4)) 

		for logic in ['AND','OR','XOR']:
			for s in settings:
				if s not in [(0,0,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),(1,1,0,1)]: #remove symmetries over the middle node's plane
					generate_coupled_FBL(params['model_file'], logic, s[0], s[1], s[2], s[3])
					loopType, gate = coupled_loop_type(logic, s[0], s[1], s[2], s[3],longLoop=False)
					calc_stats_v2(params, reps, noise_str, loopType, gate, feats)

					generate_long_coupled_FBL(params['model_file'], logic, s[0], s[1], s[2], s[3], num_per_loop=4)
					loopType, gate = coupled_loop_type(logic, s[0], s[1], s[2], s[3],longLoop=True)
					calc_stats_v2(params, reps, noise_str, loopType, gate, feats)
			print("\tFinished coupled FBL with",logic)
		for s in [[0,1],[1,1],[0,0],[1,0]]:
			generate_FBL(params['model_file'], s[0], s[1])
			loopType = loop_type(s[0],s[1])
			calc_stats_v2(params, reps, noise_str, loopType, None, feats)
		print("\tFinished regular FBLs")

	plot.probably_bar(params,feats)

	os.remove('models/temp_net.txt')


def calc_stats_v2(params, reps, noise_str, loopType, gate, feats):
	avg_var = 0
	avg_avg = 0
	avg_var_time = 0
	avg_var_total = 0

	G = Net(model_file=params['model_file'],debug=params['debug'])
	G.prepare_for_sim(params)

	for r in range(reps):
		#SS = basin.calc_basin_size(params,G,x0=None)
		SS = basin.calc_basin_size(params,G)
		middle_node = G.nodeNums['x1']
		avg_var += SS.stats['ensemble_var'][middle_node] # sum_nodes sum_As var_node_in_A  / #nodes #As
		avg_avg += SS.stats['total_avg'][middle_node]
		avg_var_time += SS.stats['temporal_var'][middle_node]
		avg_var_total += SS.stats['total_var'][middle_node]

	avg_var/=reps
	avg_avg/=reps
	avg_var_time /=reps
	avg_var_total /=reps

	if params['debug'] and params['update_rule']=='sync':
		#print(avg_avg,avg_var,avg_var_time,avg_var_total )
		assert(-.0001 <= avg_avg <= 1.0001)
		assert(-.0001 <= avg_var <= 1.0001) # know that max var is .25 for bool          # TODO: fix ensemble var!
		assert(-.0001 <= avg_var_time <= 1.0001)
		assert(-.0001 <= avg_var_total <= 1.0001)

	if gate is None:
		loop_str = loopType 
	else:
		loop_str = loopType+gate
	feats[noise_str][loop_str]['average'] += [avg_avg]
	feats[noise_str][loop_str]['ensemble variance'] += [avg_var]	
	feats[noise_str][loop_str]['temporal variance'] += [avg_var_time]
	feats[noise_str][loop_str]['variance'] += [avg_var_total]	


def generate_coupled_FBL(net_file, logic, E21, E31, E12, E13): 
	#E's are for edges, so E21=0 -> x2 -| x1
	# AND is if middle node is an AND of the other two nodes (else uses OR)
	with open(net_file,'w') as file:
		file.write('DNFsymbolic\n')
		if logic == 'AND':
			joiner = '&'
			file.write('x1\t'+sign(E21)+'x2'+joiner+sign(E31)+'x3\n')
		elif logic == 'OR':
			joiner = ' '
			file.write('x1\t'+sign(E21)+'x2'+joiner+sign(E31)+'x3\n')
		elif logic == 'XOR':
			# here
			file.write('x1\t'+sign((E21+1)%2)+'x2&'+sign(E31)+'x3 ' +sign(E21)+'x2&'+sign((E31+1)%2)+'x3\n')
		else:
			assert(0) #unrecognzd fn
		file.write('x2\t'+sign(E12)+'x1\n')
		file.write('x3\t'+sign(E13)+'x1')


def generate_long_coupled_FBL(net_file, logic, E21, E31, E12, E13, num_per_loop=8): 
	#E's are for edges, so E21=0 -> x2 -| x1
	# AND is if middle node is an AND of the other two nodes (else uses OR)
	with open(net_file,'w') as file:
		file.write('DNFsymbolic\n')
		if logic == 'AND':
			joiner = '&'
			file.write('x1\t'+sign(E21)+'x'+str(num_per_loop+1)+joiner+sign(E31)+'x'+str(2*num_per_loop+1)+'\n')
		elif logic == 'OR':
			joiner = ' '
			file.write('x1\t'+sign(E21)+'x'+str(num_per_loop+1)+joiner+sign(E31)+'x'+str(2*num_per_loop+1)+'\n')
		elif logic == 'XOR':
			# here
			file.write('x1\t'+sign((E21+1)%2)+'x'+str(num_per_loop+1)+'&'+sign(E31)+'x'+str(2*num_per_loop+1)+' ' +sign(E21)+'x'+str(num_per_loop+1)+'&'+sign((E31+1)%2)+'x'+str(2*num_per_loop+1)+'\n')
		else:
			assert(0) #unrecognzd fn
		for i in range(2,num_per_loop+1):
			file.write('x' + str(i+1) + '\t'+'x'+str(i)+'\n')
			file.write('x' + str(num_per_loop+i+1) + '\t'+'x'+str(num_per_loop+i)+'\n')
		# x_1 is still the middle node (for backwards compat)
		file.write('x2\t'+sign(E12)+'x1\n')
		file.write('x'+str(num_per_loop+2)+'\t'+sign(E13)+'x1')


def coupled_loop_type(logic, E21, E31, E12, E13,longLoop=False):
	if logic=='AND':
		gate = '_and'
	elif logic=='OR':
		gate = '_or'
	elif logic=='XOR':
		gate = '_xor'
	else:
		assert(0)
	if longLoop:
		gate += '_long'
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
	
	variance_test(sys.argv[1])
	print("Done.")