# for making a mess
# curr mainly with PBNs

import basin, util, param, plot
import sys, pickle
from copy import deepcopy
import numpy as np
from net import Net




############################   PBN   #################################################

def x0_variance(param_file):
	reps = 1 # just use 1 now

	params = param.load(param_file)
	params['PBN']['active'] = True
	params['PBN']['float'] = True
	if params['update_rule'] == 'Gasync':
		params['PBN']['float_update'] = True # this uses PBN representation of Gasync/async
	G = Net(params)

	std_devs = []
	flip_prs = [0,.001, .01, .1, .5]
	avg_std_in_time, labels = [],[]
	
	for flip_pr in flip_prs:
		params['PBN']['flip_pr'] = flip_pr

		print("starting flip pr", flip_pr)
		node_avgs = []
		for r in range(reps):
			steadyStates = basin.measure(params, G)
			#node_avgs += [steadyStates.stats['total_avg']]
			#avg_std_in_time += [steadyStates.stats['avg_std_in_time']]
			#labels += ['flip chance ' + str(flip_pr) + ' repeat ' + str(r)]

		node_std_btwn_runs = np.std(node_avgs,axis=0)					
		# TODO: worry that this is handling multiple copies wrong...want to compare floats of SINGLE theads

		#std_devs += ['{:0.3e}'.format(np.mean(node_std_btwn_runs),3)]
		std_devs += ['{:0.3e}'.format(steadyStates.stats['std_btwn_threads'],3)] # disad is don't have worst case distance but avg
		avg_std_in_time += [steadyStates.stats['avg_std_in_time']]
		labels += ['flip chance ' + str(flip_pr)]

	print('for flip pr=',flip_prs,'\tstd devs=',std_devs)

	plot.dev_in_time_lines(avg_std_in_time, labels)


def test_PBN(params, G):
	# TODO: add an optional 'mult' to multiply the number of A0s, async might also use it

	orig_num_samples = params['num_samples']
	SS = basin.calc_size(params,G)
	print("Stoch #A =",len(SS.attractors))
	params['PBN']['active'] = params['PBN']['flip_pr'] = 0
	#params['parallelism'] = params['num_samples'] = len(SS.A0)
	SS = basin.calc_size(params,G,SS0=SS)
	print("Stoch after det transient =",len(SS.attractors))

	params['parallelism'] = params['num_samples'] = orig_num_samples # i don't really like that params changes internally
	SS_det = basin.calc_size(params,G)
	print("Det #A =",len(SS_det.attractors))
	SS_det = basin.calc_size(params,G,SS0=SS_det)
	print("Det after det transient =",len(SS_det.attractors))

	return SS


def stoch_then_det(params_orig, G):
	# TODO: add an optional 'mult' to multiply the number of A0s, async might also use it

	#print("\nRUNNING STOCHASTIC, THEN DETERMINISTIC\n")
	params=deepcopy(params_orig)
	params['PBN']['active'] = 1
	#assert(params['PBN']['flip_pr']>0)
	G.prepare(params)
	SS = basin.measure(params,G)
	#params['PBN']['flip_pr'] = 0
	G.prepare(params)
	SS = basin.measure(params,G,SS0=SS)
	return SS

def just_stoch(params_orig,G):
	params=deepcopy(params_orig)
	#assert(params['PBN']['flip_pr']>0)
	G.prepare(params)
	SS = basin.measure(params,G)
	return SS

def just_det(params_orig,G):
	params=deepcopy(params_orig)
	params['PBN']['flip_pr'] = 0
	#assert(params['PBN']['flip_pr']>0)
	G.prepare(params)
	SS = basin.measure(params,G)
	return SS

def double_det(params_orig,G):
	# just to be comparable to stoch_then_det
	# somehow var is higher than a singular run..jp bug
	params=deepcopy(params_orig)
	params['PBN']['flip_pr'] = 0
	G.prepare(params)
	SS = basin.measure(params,G)
	#print('\n1:',[A.id for A in SS.attractors.values()])
	#print('\t',[A.size for A in SS.attractors.values()])
	SS = basin.measure(params,G,SS0=SS)
	#print('2:',[A.id for A in SS.attractors.values()])
	#print('\t',[A.size for A in SS.attractors.values()])
	return SS

def seq_ending_test(param_file):
	print("\nRUNNING SEQ ENDING TEST\n")
	params, G = basin.init(param_file) # why is init in basin anyway?
	assert(params['PBN']['active']) # required such that init loaded net is PBN form
	repeats=10**1

	if params['PBN']['active']:
		base=stoch_then_det(params, G).attractors
	else:
		base = double_det(params,G).attractors
	As=[base]
	max_dist=0
	for k in range(repeats):
		if (k+1)%int(repeats/4)==0:
			print('starting repeat #',k+1,'/',repeats)
		if params['PBN']['active']:
			repeat=stoch_then_det(params, G).attractors
		else:
			repeat = double_det(params,G).attractors
		for A in As:
			d = dist(A,repeat)
			#print('\nDISTANCE =',d,'\n')
			max_dist = max(d,max_dist)
		As += [repeat]
	print("\n\nMax pairwise distance among",repeats,"repeats =",max_dist,'\n\n')


def ending_comparison(params,G,verbose=True, repeats=1):
	if verbose:
		print("\nRUNNING SEQ ENDING COMPARISON TEST\n")
	params['PBN']['active'] = 1 # required such that init loaded net is PBN form
	params['skips_precise_oscils'] = True

	#params['PBN']['float'] = False	# distance is based on phenotypes after thresholding...which might be wrong

	outputs = G.output_indices()
	avg_dist=np.zeros(len(outputs))
	for k in range(repeats):
		if (k+1)%int(repeats/1)==0 and verbose:
			print('starting repeat #',k+1,'/',repeats)
		if 1: #new version, seems better
			det = just_det(params,G).stats['total_avg']
			noisy=just_stoch(params,G).stats['total_avg']
			#print('det=',det[outputs],'vs noisy=',noisy[outputs])
			avg_dist += np.abs(det[outputs]-noisy[outputs]) #avg dist btwn outputs
		elif 0: #odl version
			det = just_det(params,G).phenotypes
			noisy=just_stoch(params,G).phenotypes
			noisyPrint, detPrint ={k:noisy[k].size for k in noisy},{k:det[k].size for k in det}
			#print('\n\nnoisy=',noisyPrint,'\n\ndet=',detPrint)
			avg_dist += dist(noisy,det)
		else: # real old version
			noisy=stoch_then_det(params, G).phenotypes
			det = double_det(params,G).phenotypes
			# then computed dist
	avg_dist/=repeats
	if verbose:
		print("\n\nAvg distance between stoch and det of",params['outputs'] ,"=",np.round(avg_dist,5),'\n\n')
	return np.round(avg_dist,5)


def mutatation_seach(param_file):
	# compares the phenotypes of deterministic and probabilistic networks over many different mutations

	params_orig, G = basin.init(param_file)
	params_orig['PBN']['active'] = 1 # required such that init loaded net is PBN form
	repeats = 1
	dist = {}
	for node in G.nodes:
		if node.name != 'OFF':
			for b in [0,1]:
				#print('setting', node.name,'=',b)
				params=deepcopy(params_orig)
				params['mutations'] = {node.name:b}
				dist[node.name+'='+str(b)] = ending_comparison(params,G,verbose=False, repeats=repeats) 
				print(node.name+'='+str(b),':',dist[node.name+'='+str(b)])
				# note that ending_comparison() internally calls G.prepare() to apply mutations
	
	#print('final dists=')
	#for k in dist:
	#	print(k,':',dist[k])
	with open(params['output_dir']+'stoch_mutant_dist.pickle','wb') as f:
		pickle.dump({'dist':dist, 'params':params_orig},f)

def plot_stoch_mutant_dist():
	with open('./output/stoch06/grieco_fast_stoch_mutant_dist.pickle','rb') as f:
		pickled = pickle.load(f)
	dist, params = pickled['dist'], pickled['params']
	dist_vals = [list(d)[0] for d in list(dist.values())]
	plot.stoch_mutant_dist(params, dist_vals)


def dist(A1,A2):
	d1=0
	for k in A1:
		if k in A2:
			d1+=abs(A1[k].size-A2[k].size)
		else:
			d1+=A1[k].size 
	d2=0
	for k in A2:
		#if k in A1:
		#	d2+=abs(A1[k].size-A2[k].size)
		if k not in A1:
			d2+=A2[k].size

	return (d1+d2)/2

#######################################################################################################

if __name__ == "__main__":
	if len(sys.argv) not in [2,3]:
		sys.exit("Usage: python3 sandbox.py PARAMS.yaml [runtype]")
	
	if len(sys.argv) == 3:
		if sys.argv[2] == 'end':
			params, G = basin.init(sys.argv[1]) # why is init in basin anyway?
			ending_comparison(params, G)
		elif sys.argv[2] == 'x0':
			x0_variance(sys.argv[1])
		elif sys.argv[2] == 'mutants':
			mutatation_seach(sys.argv[1])
		elif sys.argv[2] == 'plot':
			plot_stoch_mutant_dist()
		else:
			print("Unknown 2nd param...",sys.argv[2])
	else:
		print("I dunno what the default should be yet \\:")