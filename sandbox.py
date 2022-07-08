# for making a mess
# curr mainly with PBNs

import basin, util, param, plot
import sys, pickle, os
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
		std_devs += ['{:0.3e}'.format(steadyStates.stats['var_threads'],3)] # disad is don't have worst case distance but avg
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
	assert(params['PBN']['flip_pr']>0 and params['PBN']['active'] == 1)
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
	assert(params['var_window']>0)

	# very much in need of a clean, confusing af
	# some of these stats are never used/passed on
	# some are normalized before repeats, some after, and multiple repeats doesn't even work
	# mutatation_seach() will further normalize some of the stats

	#params['PBN']['float'] = False	# distance is based on phenotypes after thresholding...which might be wrong

	outputs = G.output_indices()
	avg_dist=np.zeros(len(outputs))
	avg_thread_var=0

	avg_thread_var_det=0
	temporal_var_det, temporal_var_stoch  = 0,0
	cancer_det, cancer_noisy = 0,0

	fast_var_all_det, slow_var_all_det, fast_var_all_stoch, slow_var_all_stoch = 0,0,0,0
	fast_var_outputs_det, slow_var_outputs_det, fast_var_outputs_stoch, slow_var_outputs_stoch = 0,0,0,0
	
	# TODO put these all stats into a dict and auto pass keys instead lol

	for k in range(repeats):
		if (k+1)%int(repeats/1)==0 and verbose:
			print('starting repeat #',k+1,'/',repeats)
		if 1: 
			det_stats = just_det(params,G).stats
			det_iavg = det_stats['input_sep']['total_avg']
			noisy_stats = just_stoch(params,G).stats
			noisy_iavg = noisy_stats['input_sep']['total_avg']
			
			thread_var, thread_var_det, dist=0,0,0
			avg_slow_var_all_det, avg_slow_var_all_stoch, avg_slow_var_outputs_det, avg_slow_var_outputs_stoch = 0,0,0,0
			num_input_states = len(params['input_state_indices'])

			# average over inputs first
			for i in range(num_input_states):
				thread_var += noisy_stats['input_sep']['var_threads'][i]
				thread_var_det += det_stats['input_sep']['var_threads'][i]
				dist += np.abs(det_iavg[i][outputs]-noisy_iavg[i][outputs])

				fast_var_all_det += np.mean(det_stats['windowed_var_input_split']) #'input_sep']['total_var'][i]) 
				fast_var_all_stoch += np.mean(noisy_stats['windowed_var_input_split'])
				fast_var_outputs_det += np.mean(det_stats['windowed_var_input_split'][outputs])
				fast_var_outputs_stoch += np.mean(noisy_stats['windowed_var_input_split'][outputs])

				slow_var_all_det += np.mean(det_stats['input_sep']['slow_var'][i])
				slow_var_all_stoch += np.mean(noisy_stats['input_sep']['slow_var'][i])
				slow_var_outputs_det += np.mean(det_stats['input_sep']['slow_var'][i][outputs])
				slow_var_outputs_stoch += np.mean(noisy_stats['input_sep']['slow_var'][i][outputs])

			avg_thread_var += thread_var[outputs] / num_input_states
			dist /= num_input_states
			avg_thread_var_det += thread_var_det[outputs] / num_input_states

			# next 2 lines just for debug plz dont use
			#avg_thread_var += np.mean(det_stats['var_threads']) # just to check w grieco only
			#avg_thread_var_det += np.mean(det_stats['var_threads'][outputs])
			# either 1) std vs var matters (would be v weird), 2) input split matters, 3) didn't use [outputs] before


			fast_var_all_det /= num_input_states
			fast_var_all_stoch /= num_input_states
			fast_var_outputs_det /= num_input_states
			fast_var_outputs_stoch /= num_input_states
			slow_var_all_det /= num_input_states
			slow_var_all_stoch /= num_input_states
			slow_var_outputs_det /= num_input_states
			slow_var_outputs_stoch /= num_input_states

			temporal_var_det += np.mean(det_stats['avg_std_in_time_outputs'])
			temporal_var_stoch += np.mean(noisy_stats['avg_std_in_time_outputs'])
			cancer_det += eval_cancerousness(params, det_stats['total_avg'][outputs])
			cancer_noisy += eval_cancerousness(params, noisy_stats['total_avg'][outputs])

		elif 0: 
			# look both btwn threads in noise with: 'var_threads'
			# and the input-sep diff btwn det and noisy avgs of output nodes
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

	assert(repeats==1) # else normalize all metrics by # repeats

	result = {
		'dist':dist, 'thread_var_det':thread_var_det, 'thread_var_stoch':thread_var, 
		'cancer_det':cancer_det, 'cancer_noisy':cancer_noisy, 'temporal_var_det':temporal_var_det,'temporal_var_stoch':temporal_var_stoch,
		'fast_var_all_det':fast_var_all_det, 'slow_var_all_det':slow_var_all_det, 'fast_var_all_stoch':fast_var_all_stoch, 'slow_var_all_stoch':slow_var_all_stoch,
		'fast_var_outputs_det':fast_var_outputs_det, 'slow_var_outputs_det':slow_var_outputs_det, 'fast_var_outputs_stoch':fast_var_outputs_stoch, 'slow_var_outputs_stoch':slow_var_outputs_stoch,
		}
	for k in result.keys():
		result[k] = np.round(result[k],5)
	return result


def mutatation_seach(param_file):
	# compares the phenotypes of deterministic and probabilistic networks over many different mutations

	params_orig, G = basin.init(param_file)
	params_orig['PBN']['active'] = 1 # required such that init loaded net is PBN form
	repeats = 1
	dist, thread_var_det, thread_var_stoch = {}, {}, {} # two metrics of effect of mutation on noise
	cancerousness_det, cancerousness_noisy = {},{}  # metric for effect of mutation on cancerousness, depending det/noisy model
	temporal_var_det, temporal_var_stoch = {},{}
	slowfast_keys = ['fast_var_all_det', 'slow_var_all_det', 'fast_var_all_stoch', 'slow_var_all_stoch', 'fast_var_outputs_det', 'slow_var_outputs_det', 'fast_var_outputs_stoch', 'slow_var_outputs_stoch']
	slowfast = {k:{} for k in slowfast_keys}

	baseline_result = ending_comparison(params_orig,G,verbose=False, repeats=repeats) 
	dists, thread_vars_det, thread_vars_stoch, baseline_cancer_det, baseline_cancer_noisy = baseline_result['dist'], baseline_result['thread_var_det'], baseline_result['thread_var_stoch'], baseline_result['cancer_det'], baseline_result['cancer_noisy']
	temporal_vars_det, temporal_vars_stoch = baseline_result['temporal_var_det'], baseline_result['temporal_var_stoch']
	baseline_dist = np.round(np.mean(dists),5)
	baseline_thread_var_det = np.round(np.mean(thread_vars_det),5)
	baseline_thread_var_stoch = np.round(np.mean(thread_vars_stoch),5)
	baseline_temporal_vars_det = np.round(np.mean(temporal_vars_det),5)
	baseline_temporal_vars_stoch = np.round(np.mean(temporal_vars_stoch),5)
	
	print('\nBASELINE dist:',baseline_dist,'\tthr stddev:',baseline_thread_var_det,'\n\tcancrness det:',baseline_cancer_det,'\tcancrness_noisy:',baseline_cancer_noisy)
	for node in G.nodes:
		if node.name != 'OFF':
			for b in [0,1]:
				#print('setting', node.name,'=',b)
				params=deepcopy(params_orig)
				params['mutations'] = {node.name:b}
				result = ending_comparison(params,G,verbose=False, repeats=repeats) 
				dists, thread_vars_det, thread_vars_stoch, cancer_det, cancer_noisy, temporal_vars_det, temporal_vars_stoch = result['dist'], result['thread_var_det'], result['thread_var_stoch'], result['cancer_det'], result['cancer_noisy'], result['temporal_var_det'], result['temporal_var_stoch']
				dist[node.name+'='+str(b)] = np.round(np.mean(dists) - baseline_dist,5)
				thread_var_det[node.name+'='+str(b)] = np.round(np.mean(thread_vars_det) - baseline_thread_var_det,5)
				thread_var_stoch[node.name+'='+str(b)] = np.round(np.mean(thread_vars_stoch) - baseline_thread_var_stoch,5)
				cancerousness_det[node.name+'='+str(b)] = np.round(cancer_det - baseline_cancer_det,5)
				cancerousness_noisy[node.name+'='+str(b)] = np.round(cancer_noisy - baseline_cancer_noisy,5)
				print('\n',node.name+'='+str(b),'dist:',dist[node.name+'='+str(b)],'\tthr stddev:',thread_var_det[node.name+'='+str(b)],'\n    cancerDet:',cancerousness_det[node.name+'='+str(b)],'\tcancerStoch:',cancerousness_noisy[node.name+'='+str(b)])
				#print("unsubtrct cancer det, stoch =",np.round(np.mean(cancer_det),5),',',np.round(np.mean(cancer_noisy),5) )
				# note that ending_comparison() internally calls G.prepare() to apply mutations

				temporal_var_det[node.name+'='+str(b)] = np.round(np.mean(temporal_vars_det)-baseline_temporal_vars_det,5)
				temporal_var_stoch[node.name+'='+str(b)] = np.round(np.mean(temporal_vars_stoch)-baseline_temporal_vars_stoch,5)

				for k in slowfast_keys:
					slowfast[k][node.name+'='+str(b)] = np.round(result[k]-baseline_result[k],5)
	
	#print('final dists=')
	#for k in dist:
	#	print(k,':',dist[k])
	img_name = params['output_img'].replace('.png','')
	pickle_file = params['output_dir']+'stoch_mutant_dist_' + img_name + '.pickle'
	i=1
	while os.path.exists(pickle_file):
		pickle_file = params['output_dir']+'stoch_mutant_dist_' + img_name + '_' + str(i) + '.pickle'
		i+=1
	with open(pickle_file,'wb') as f:
		data = {'detstoch_dist':dist, 'params':params_orig,'thread_var_det':thread_var_det,'thread_var_stoch':thread_var_stoch,'cancer_det':cancerousness_det, 'cancer_noisy':cancerousness_noisy, 'temporal_var_det':temporal_var_det, 'temporal_var_stoch':temporal_var_stoch}
		for k in slowfast_keys:
			data[k] = slowfast[k]
		pickle.dump(data,f)


def top_down_motifs(param_file_grieco, param_file_fumia):
	files = ['./output/noisy07/finals/stoch_mutant_dist_grieco_sync.pickle'] #'./output/noisy07/stoch_mutant_dist_grieco_12.pickle']#,'./output/noisy07/stoch_mutant_dist_fumia_5.pickle']
	names = ['grieco','fumia']
	for i in range(len(files)):
		with open(files[i], 'rb') as f:
			pickled = pickle.load(f)

		selector_key = 'slow_var_all_stoch' #'slow_var_outputs_stoch'
		
		for k in pickled['params']['outputs']:
			for b in [0,1]:
				del pickled[selector_key][k+'='+str(b)]
		for k in pickled['params']['inputs']:
			for b in [0,1]:
				del pickled[selector_key][k+'='+str(b)]
		driver = max(pickled[selector_key], key=pickled[selector_key].get)
		damper = min(pickled[selector_key], key=pickled[selector_key].get)

		# then test with and without mutation
		if names[i]=='grieco':
			param_file = param_file_grieco
		else:
			param_file = param_file_fumia
		params, G = basin.init(param_file)
		assert(params['PBN']['active'])
		assert(not params['PBN']['float'])

		driver_node, driver_state = driver.split('=')
		damper_node, damper_state = damper.split('=')

		#	maybe all stats should have a per-node and sep-node version?

		WT_slowvar_split = np.array(basin.measure(params, G).stats['windowed_var_input_split_sep']) #['input_sep']['slow_var'])

		params['mutations'] = {driver_node:driver_state}
		G.prepare(params)
		M_driver_slowvar_split = np.array(basin.measure(params, G).stats['windowed_var_input_split_sep']) #['input_sep']['slow_var'])
		M_driver_slowvar = np.max(M_driver_slowvar_split - WT_slowvar_split,axis=0) # avg over inputs, keep nodes sep

		print("\nDRIVER INDUCED SLOW VAR for", names[i],"using",driver_node,'=',driver_state,":")
		for j in range(G.n):
			if 1 : #M_driver_slowvar[j] > .01:
				print(G.nodeNames[j],':',np.round(M_driver_slowvar[j],5))

		params['mutations'] = {damper_node:damper_state}
		G.prepare(params)
		M_damper_slowvar_split = np.array(basin.measure(params, G).stats['input_sep']['slow_var'])
		M_damper_slowvar = np.max(M_damper_slowvar_split - WT_slowvar_split,axis=0)

		print("\nDAMPER INDUCED SLOW VAR for", names[i],"using",damper_node,'=',damper_state,":")
		for j in range(G.n):
			if M_damper_slowvar[j] < -.001:
				print(G.nodeNames[j],':',np.round(M_damper_slowvar[j],5))


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

def eval_cancerousness(params, output_avgs):
	if params['model_file'] == 'models/grieco.bnet' or params['model_file'] == 'models/fumia.txt':
		#return output_avgs[1]/(output_avgs[1] + output_avgs[0]) 
		return output_avgs[1] - output_avgs[0] 
	else:
		return 0

#######################################################################################################

def plot_stoch_mutant_dist():
	files = ['./output/noisy07/fastvar_corrected/stoch_mutant_dist_grieco_Gasync.pickle','./output/noisy07/fastvar_corrected/stoch_mutant_dist_fumia_Gasync.pickle']
	names = ['grieco','fumia']
	for i in range(len(files)):
		with open(files[i], 'rb') as f:
			pickled = pickle.load(f)
		#for k in pickled:
		#	if k not in ['params']:
		#		print(k)
		#		pickled[k] = [list(d)[0] for d in list(pickled[k].values())]
		#dist, thread_var, cancer_det, cancer_stoch, params = pickled['dist'], pickled['params']
		#dist_vals = [list(d)[0] for d in list(dist.values())]
		plot.stoch_mutant_dist(pickled['params'],names[i], pickled)


#######################################################################################################

if __name__ == "__main__":
	if len(sys.argv) not in [2,3,4]:
		sys.exit("Usage: python3 sandbox.py PARAMS.yaml [runtype]")
	
	if len(sys.argv) in [3,4]:
		if sys.argv[2] == 'end':
			params, G = basin.init(sys.argv[1]) # why is init in basin anyway?
			ending_comparison(params, G)
		elif sys.argv[2] == 'x0':
			x0_variance(sys.argv[1])
		elif sys.argv[2] == 'mutants':
			mutatation_seach(sys.argv[1])
		elif sys.argv[2] == 'motifs':
			top_down_motifs(sys.argv[1], sys.argv[3])
		elif sys.argv[2] == 'plot':
			plot_stoch_mutant_dist()
		else:
			print("Unknown 2nd param...",sys.argv[2])
	else:
		print("I dunno what the default should be yet \\:")