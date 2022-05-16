# currently used for testing sensitivity of small motifs to noise

import basin, param, plot, util
from net import Net
import sys, os, itertools, pickle
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


	reps=params['loopy_reps']
	all_loops = 0 
	longloops=0
	xorr = 0

	print("Simulating loops with flip probability=",params['PBN']['flip_pr'])#,', for',params['loopy_reps'],'repeats.')
	params['model_file'] = 'models/temp_net.txt'
	params['verbose'] = False
	params['inputs'],params['outputs'],params['init']={},{},{} #override
	params['skips_precise_oscils'] = True
	assert(params['var_window'])
	assert(not params['PBN']['float'])

	stats = {'average':[],'slow variance':[],'fast variance':[]}
	#groups = ['P','N']
	groups = ['P','PP_and','PP_or','PN_and','PN_or','NN_and','NN_or','N']
	logics = ['AND','OR']
	if all_loops:
		logics = ['AND','OR','XOR']
		groups = ['P','PP_and','PP_or','PN_and','PN_or','NN_and','NN_or','N']
		groups += ['PP_and_long','PN_and_long','NN_and_long','PP_or_long','PN_or_long','NN_or_long']
		groups += ['PP_xor','PN_xor','NN_xor','PP_xor_long','PN_xor_long','NN_xor_long']
	elif longloops:
		logics = ['AND']
		groups = ['PP_and','PN_and','NN_and']
		groups += ['PP_and_long','PN_and_long','NN_and_long']
	elif xorr:
		logics = ['AND','XOR']
		groups = ['P','PP_and','PP_xor','PN_and','PN_xor','NN_and','NN_xor','N']
	elif 0: # xor and longloops
		logics = ['AND','XOR']
		groups = ['PP_xor','PN_xor','NN_xor']
		groups += ['PP_xor_long','PN_xor_long','NN_xor_long']

	
	feats = {'no noise':{g:deepcopy(stats) for g in groups},'noisy':{g:deepcopy(stats) for g in groups}}

	for noise in [False, True]:
		print("Starting with noise",noise)
		params['PBN']['active'] = noise
		if noise:
			noise_str='noisy'
		else:
			noise_str = 'no noise'

		settings = list(itertools.product([0,1],repeat=4)) 

		for logic in logics:
			for s in settings:
				if s not in [(0,0,0,1),(0,1,0,0),(0,1,0,1),(0,1,1,0),(0,1,1,1),(1,1,0,1)]: #remove symmetries over the middle node's plane
					generate_coupled_FBL(params['model_file'], logic, s[0], s[1], s[2], s[3])
					loopType, gate = coupled_loop_type(logic, s[0], s[1], s[2], s[3],longLoop=False)
					calc_stats_v2(params, reps, noise_str, loopType, gate, feats)

					if longloops:
						generate_long_coupled_FBL(params['model_file'], logic, s[0], s[1], s[2], s[3], num_per_loop=2)
						loopType, gate = coupled_loop_type(logic, s[0], s[1], s[2], s[3],longLoop=True)
						calc_stats_v2(params, reps, noise_str, loopType, gate, feats)
			print("\tFinished coupled FBL with",logic)
		if not longloops or all_loops:
			for s in [[0,1],[1,1],[0,0],[1,0]]:
				generate_FBL(params['model_file'], s[0], s[1])
				loopType = loop_type(s[0],s[1])
				calc_stats_v2(params, reps, noise_str, loopType, None, feats)
			print("\tFinished regular FBLs")

	pickle_file = (params['output_dir'] +params['output_img']).replace('.png','.pickle')
	with open(pickle_file,'wb') as f:
		pickle.dump({'params':params,'feats':feats},f)

	plot.probably_bar(params,feats)

	os.remove('models/temp_net.txt')

def from_pickle(param_file):
	params = param.load(param_file)
	pickle_file = (params['output_dir'] +params['output_img']).replace('.png','.pickle')
	with open(pickle_file,'rb') as f:
		data = pickle.load(f)
	plot.probably_bar(data['params'],data['feats'])


def calc_stats_v2(params, reps, noise_str, loopType, gate, feats):
	avg_stats = {s:0 for s in ['total_avg','total_var','windowed_var']}

	G = Net(params)

	for r in range(reps):
		SS = basin.measure(params,G)
		middle_node = G.nodeNums['x1']
		for s in avg_stats.keys():
			avg_stats[s] += SS.stats[s][middle_node]

	for s in avg_stats.keys():
		avg_stats[s]/=reps

	if params['debug'] and params['update_rule']=='sync':
		#print(avg_avg,avg_var,avg_var_time,avg_var_total )
		assert(-.0001 <= avg_stats['total_avg'] <= 1.0001)
		assert(-.0001 <= avg_stats['total_var'] <= 0.2501) # know that max var is .25 for bool
		assert(-.0001 <= avg_stats['windowed_var'] <= 0.2501) 

	if gate is None:
		loop_str = loopType 
	else:
		loop_str = loopType+gate
	feats[noise_str][loop_str]['average'] += [avg_stats['total_avg']]
	feats[noise_str][loop_str]['fast variance'] += [avg_stats['windowed_var']]
	feats[noise_str][loop_str]['slow variance'] += [avg_stats['total_var'] - avg_stats['windowed_var']]	


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


def generate_long_coupled_FBL(net_file, logic, E21, E31, E12, E13, num_per_loop=4): 
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
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 probably.py PARAMS.yaml [run|pickle]")
	
	if sys.argv[2]=='pickle':
		from_pickle(sys.argv[1])
	elif sys.argv[2]=='run':
		variance_test(sys.argv[1])
	else:
		print("Unrecognized 2nd argument:",sys.argv[2])
		sys.exit()
	print("Done.")