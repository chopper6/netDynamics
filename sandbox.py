# for making a mess
# curr mainly with PBNs


def test_PBN(params, G):
	Ps = {}
	mult=1 # if change this, Aweight normalization also needs to change?
	orig_num_samples = params['num_samples']
	SS = calc_basin_size(params,G)
	A0, Aweights =  get_A0_and_weights(SS)
	print("Stoch #A =",np.array(A0).shape)
	A0 = np.repeat(A0,mult)
	params['PBN']['active'] = params['PBN']['flip_pr'] = 0
	params['parallelism'] = params['num_samples'] = len(A0)
	SS = calc_basin_size(params,G,A0=A0)
	A0, Aweights =  get_A0_and_weights(SS)
	A0 = np.array(A0)
	print("Stoch after det transient =",np.array(A0).shape)

	params['parallelism'] = params['num_samples'] = orig_num_samples
	SS_det = calc_basin_size(params,G)
	A0, Aweights =  get_A0_and_weights(SS_det)
	print("Det #A =",np.array(A0).shape)
	#A0 = np.vstack([A0 for _ in range(mult)])
	#print(np.array(A0).shape)
	params['parallelism'] = params['num_samples'] = len(A0)
	SS_det = calc_basin_size(params,G,A0=A0)
	A0, Aweights =  get_A0_and_weights(SS_det)
	A0 = np.array(A0)
	print("Det after det transient =",np.array(A0).shape)

	return SS


def stoch_then_det(params_orig, G):
	#print("\nRUNNING STOCHASTIC, THEN DETERMINISTIC\n")

	mult=1
	# don't care about Aweights? also jp Aweights reqds some debugging...

	params = deepcopy(params_orig)
	SS = calc_basin_size(params,G)
	A0, Aweights =  get_A0_and_weights(SS)
	A0 = np.repeat(A0,mult)
	params['PBN']['active'] = params['PBN']['flip_pr'] = 0
	params['parallelism'] = params['num_samples'] = len(A0)
	SS = calc_basin_size(params,G,A0=A0,Aweights=Aweights)
	return SS


def repeat_test(param_file):
	print("\nRUNNING REPEAT TEST\n")
	params, G = init(param_file)
	repeats=20

	if params['PBN']['active']:
		base=stoch_then_det(params, G).attractors
	else:
		base = measure(params,G).attractors
	As=[base]
	max_dist=0
	for k in range(repeats):
		print('starting repeat #',k+1,'/',repeats)
		if params['PBN']['active']:
			repeat=stoch_then_det(params, G).attractors
		else:
			repeat = measure(params,G).attractors
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

def init(param_file):
	params = param.load(param_file)
	G = Net(model_file=params['model_file'],debug=params['debug'])
	G.prepare_for_sim(params)
	return params, G

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
