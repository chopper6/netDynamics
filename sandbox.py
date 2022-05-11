# for making a mess
# curr mainly with PBNs

import basin, util, param
import sys
from copy import deepcopy
import numpy as np
from net import Net




############################   PBN   #################################################

def x0_stoch_vs_det(param_file):
	# also want to check scaling w.r.t # samples and # steps too
	# then create 3 images
	reps = 20


	params = param.load(param_file)
	params['track_x0'] = True 
	params['PBN']['active'] = True
	params['PBN']['float'] = True
	PBN = util.istrue(params,['PBN','active']) and util.istrue(params,['PBN','float'])
	G = Net(model_file=params['model_file'],debug=params['debug'],PBN=PBN)
	G.prepare_for_sim(params)

	std_devs = []
	flip_prs = [0, .001]#, .01, .1, .5]
	
	for flip_pr in flip_prs:
		params['PBN']['flip_pr'] = flip_pr

		node_avgs = []
		for r in range(reps):
			steadyStates = basin.measure(params, G)
			node_avgs += [steadyStates.stats['total_avg']]

		node_std_btwn_runs = np.std(node_avgs,axis=0)
		std_devs += [np.mean(node_std_btwn_runs)]

	print('for flip pr=',flip_prs,'\tvariances=',std_devs)




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
	SS = basin.calc_size(params,G)
	params['PBN']['active'] = params['PBN']['flip_pr'] = 0
	SS = basin.calc_size(params,G,SS0=SS)
	return SS

def double_det(params_orig,G):
	# just to be comparable to stoch_then_det
	# somehow var is higher than a singular run..jp bug
	params=deepcopy(params_orig)
	SS = basin.calc_size(params,G)
	#print('\n1:',[A.id for A in SS.attractors.values()])
	#print('\t',[A.size for A in SS.attractors.values()])
	SS = basin.calc_size(params,G,SS0=SS)
	#print('2:',[A.id for A in SS.attractors.values()])
	#print('\t',[A.size for A in SS.attractors.values()])
	return SS

def repeat_test(param_file):
	print("\nRUNNING REPEAT TEST\n")
	params, G = basin.init(param_file) # why is init in basin anyway?
	repeats=10**1

	if params['PBN']['active']:
		base=stoch_then_det(params, G).attractors
	else:
		base = double_det(params,G).attractors
	As=[base]
	max_dist=0
	for k in range(repeats):
		if (k+1)%int(repeats/10)==0:
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

# haven't used these in awhile

def temp_print_periodic(SS, G,params):
	for k in SS.attractors:
		A=SS.attractors[k]
		#print("\nnext A:")
		for i in range(len(A.avg)):
			inputs = params['inputs']#['EGFR_stimulus', 'TGFBR_stimulus', 'FGFR3_stimulus', 'DNA_damage']
			input_state = [A.avg[G.nodeNums[inpt]] for inpt in inputs]
			if A.avg[i] not in [0,1] and G.nodeNames[i] in params['outputs']:
			#if input_state == [0,0,0,0]:
				print('inputs=',input_state,':',G.nodeNames[i], 'avg=', A.avg[i])

def debug_print(params,G,steadyStates):
	k2 = len(params['outputs'])
	import numpy as np 
	outpt = np.array([G.nodeNums[params['outputs'][i]] for i in range(k2)])
	for k in steadyStates.attractors:
		A =  steadyStates.attractors[k]
		inpts = A.phenotype.split('|')[0]
		if inpts == '0000':
			print('with input 0000: ','A.size, A.period,A.avg[outpt]\n',A.size, A.period,A.avg[outpt])
	for k in steadyStates.phenotypes:
		P =  steadyStates.phenotypes[k]	
		print('pheno',k,'=',P)


#######################################################################################################

if __name__ == "__main__":
	if len(sys.argv) not in [2,3]:
		sys.exit("Usage: python3 sandbox.py PARAMS.yaml [runtype]")
	
	if len(sys.argv) == 3:
		if sys.argv[2] == 'repeat':
			repeat_test(sys.argv[1])
		elif sys.argv[2] == 'x0':
			x0_stoch_vs_det(sys.argv[1])
		else:
			print("Unknown 2nd param...",sys.argv[2])
	else:
		print("I dunno what the default should be yet \\:")