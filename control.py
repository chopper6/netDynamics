import main, ldoi, parse, plot, logic,basin,features
import sys, math, pickle
import numpy as np
import random as rd
from copy import deepcopy

# TODO
# test based DEBUG with more than just main
#	why tf does it speed up as it goes thru nodes....?
#	seems steady now, at least when no mutation found
# return net_file to main params, model param rename to 'setting' (optional), and sep folders for params, nets, and settings
# try toy net again & mammCC
# add treatment(s), LDOI, corr, features
# sep randomize into new file
# making deepcopies everywhere is clearly not an efficient way to go

# TODO before thurs
# check over main points/ double check test-based
# give toy example of what I'm measuring and rd process
# premeet on wed

# DO LATER UNLESS NEC NOW
# expanded net and regular parse make V diff..exp does not incld 0 and 1
# clean the mess here
# quick alterations to expanded net, sT ldoi can be run on it after mutations, ect
# passing around the encoding has become messy af, along with figuring out the node list
# zero node is nasty af sometimes
#		isn't a self loop and an init condition sufficient?
#		and auto add it to 'init' in model file 
#		a complication is always off clause (but can find a better soln)
# logic.DNF_cell_collective: handle edge cases like 1 clause (jp) and constants
# clean the shitstorm that is multi.py
# just apply_mutations from parse instead of parsing network from scratch each time main calc attractors is called
# add time stamped directory or something
# forcing parallelism is bad for accessibility (basin/parse.py)
# pick io order for phenos (curr i|o)
#	poss only match |o part for colors of pie chart


def exhaustive(params, G, max_control_size = 1, mut_dist_thresh=.2,cnt_dist_thresh=.2):

	# assumes cannot alter pheno nodes, nor input nodes
	# complexity is O(s*n^m+1*2^i) where s is the simulation time of a single input and i is the number of inputs, and m is the max control set size

	nodes = [node for node in G.nodeNames if node not in params['outputs'] and node not in params['inputs']]

	params['verbose']=0 #i assume
	params['use_inputs_in_pheno']=True #also assume

	phenosWT = main.find_attractors_prebuilt(params, F, V).phenotypes
	orig_mutated = deepcopy(params['mutations'])
	mutators, solutions = [],[]
	mutant_mag, control_mag = 0,0

	mut_pts, cnt_pts, ldoi_mut, ldoi_out_mut, ldoi_cnt, ldoi_out_cnt = [[] for _ in range(6)]
	cnt_scores = {C:[] for C in nodes}
	H_cond = features.check_corr(params, V, phenosWT)

	ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,F,V,fixed_nodes=[])
	# returns {'total','total_onlyOuts', 'node','node_onlyOuts'}

	mut_ldoi_total, mut_ldoi_total_out = 0,0

	for M in nodes:
		print('Testing node',M)
		mut_dist_sum = 0
		for b in [0,1]:
			params['mutations'][M] = b
			phenosM = main.find_attractors_prebuilt(params, F, V).phenotypes 

			mut_dist = diff(phenosWT,phenosM)
			mut_dist_sum += mut_dist
			mutant_mag += mut_dist

			if mut_dist  > mut_dist_thresh:
				#print([(k,phenosM[k]['size']) for k in phenosM])
				mutators += [(M,b)]

				solutions, control_mag_, mutated_ldoi_stats = try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, [], control_mag, (M,b), max_control_size,cnt_dist_thresh,cnt_scores,ldoi_cnt, ldoi_out_cnt,[])
				control_mag += control_mag_
				#mut_ldoi_total += mutated_ldoi_stats['total']
				#mut_ldoi_total_out += mutated_ldoi_stats['total_onlyOuts']
				#print(control_mag, control_mag_/(2*len(nodes)))

			params['mutations'] = deepcopy(orig_mutated) #reset

			indx = V['name2#'][M]-1
			if b==0:
				indx += int(n/2)-1 #ie the complement index
			ldoi_mut += [[mut_dist,ldoi_stats['node'][indx]]]
			ldoi_out_mut += [[mut_dist,ldoi_stats['node_onlyOuts'][indx]]]
		
		if H_cond[M] is not None:
			mut_pts += [[mut_dist_sum/2,H_cond[M]]]

	mutant_mag /= 2*len(nodes)
	control_mag /= max(len(mutators),1)
	#mut_ldoi_total /= len(mutators)
	#mut_ldoi_total_out /= len(mutators)

	for C in nodes:
		if H_cond[C] is not None:
			cnt_pts += [[sum(cnt_scores[C]),H_cond[C]]]


	print('\n#mutators=',len(mutators))
	print('\n#solutions=',len(solutions))
	print('\nmutation mag = ', mutant_mag, ', control mag = ', control_mag)
	print('\nall solutions:',solutions)

	H_cond_avg = 0
	for value in H_cond.values():
		if value is not None:
			H_cond_avg += value 
	H_cond_avg/= len(nodes)
	n2=len(V['#2name'])-2
	net_stats1 = {'%mutators':len(mutators)/n2,'mutability':mutant_mag, '%controllers':len(solutions)/max(len(mutators)*(n2-1),1),'reversibility':control_mag}
	net_stats2= {'|ldoi|':ldoi_stats['total'],'|ldoi_outputs|':ldoi_stats['total_onlyOuts'],'correlation':H_cond_avg}#,\
	 #'|ldoi| mutated':mut_ldoi_total, '|ldoi_outputs| mutated':mut_ldoi_total_out}
	
	node_stats = {'mut_pts':mut_pts,'cnt_pts':cnt_pts,'ldoi_mut':ldoi_mut,'ldoi_out_mut':ldoi_out_mut,'ldoi_cnt':ldoi_cnt,'ldoi_out_cnt':ldoi_out_cnt}
	
	# stupidly i did these in reverse [x,y] order to how i add points lol
	node_stat_labels = {'mut_pts':['entropy','mutation magnitude'],'cnt_pts':['entropy','control score'], \
	 'ldoi_mut':['|ldoi|','mutation magnitude'],'ldoi_out_mut':['|ldoi only outputs|','mutation magnitude'], \
	 'ldoi_cnt':['|ldoi|','control score'],'ldoi_out_cnt':['|ldoi only outputs|','control score']}
	return {'network_stats1':net_stats1,'network_stats2':net_stats2, 'node_stats':node_stats,'node_stat_labels':node_stat_labels}



####################################################################################################

def try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, control_set, control_mag, mutator, depth,control_dist_thresh,cnt_scores,ldoi_cnt, ldoi_out_cnt,ldoi_stats): 
	if depth == 0:
		return solutions, 0
	
	control_mag = 0 #so shouldn't pass as param, also not recursing w it properly

	# WARNING: ldoi stats won't work with larger depth
	#ldoi_stats = ldoi.ldoi_sizes_over_all_inputs(params,F,V,fixed_nodes=[mutator])

	for C in nodes:
		orig_mutations = deepcopy(params['mutations'])
		for b in [0,1]:
			if (C,b) != mutator and (C,b) not in control_set:
				params['mutations'][C] = b
				phenosC = main.find_attractors_prebuilt(params, F, V).phenotypes

				#change = 1-diff(phenosWT,phenosC)/mut_dist
				control_dist = diff(phenosWT,phenosC)
				#print('\twith',C,b,'control dist =',control_dist)
				#print([(k,phenosC[k]['size']) for k in phenosC])
				cnt_scores[C] += [max(0,mut_dist-control_dist)]
				control_mag += max(0,mut_dist-control_dist)
				if control_dist < mut_dist - control_dist_thresh:  
					print('\t',C,'=',b,'cnt_dist=',round(control_dist,3),'mut_dist=', round(mut_dist,3),'vs orig mutation',mutator[0],'=',mutator[1])

					solutions += [{'mutation':mutator,'control':control_set+[(C,b)]}]
				else:
					# recurse using this node as a control + another (until reach max_depth)
					solutions, control_mag_ = try_and_fix(params,F, V, nodes, solutions, phenosWT, mut_dist, control_set+[(C,b)], control_mag, mutator, depth-1,control_dist_thresh,cnt_scores,ldoi_cnt, ldoi_out_cnt,ldoi_stats) 


				params['mutations'] = deepcopy(orig_mutations) # reset (in case orig mutator was tried as a control node)
			
				indx = V['name2#'][C]-1
				if b==0:
					indx += int(len(V['name2#'])/2)-1 # all this mf off by ones due to the mf 0 node
				#ldoi_cnt += [[max(0,1-control_dist/mut_dist),ldoi_stats['node'][indx]]]
				#ldoi_out_cnt += [[max(0,1-control_dist/mut_dist),ldoi_stats['node_onlyOuts'][indx]]]

	return solutions, control_mag/(2*len(nodes)-1), ldoi_stats


def diff(P1, P2, norm=1):
	P1_basins, P2_basins = [],[]
	for io in P1:
		P1_basins += [P1[io].size]
		if io not in P2:
			P2_basins += [0]
		else:
			P2_basins += [P2[io].size]
	for io in P2:
		if io not in P1:
			# i.e only if skipped before
			P2_basins += [P2[io].size]
			P1_basins += [0]
	
	P1_basins, P2_basins = np.array(P1_basins), np.array(P2_basins)
	if norm in ['inf','max']:
		norm = np.inf 

	result= np.linalg.norm(P1_basins-P2_basins,ord=norm)
	if norm == 1:
		result/=2 # will double count any overlap
		# not sure how to normalize this for other norms!
	if not (result >= -.01 and result <= 1.01): # may not hold for norms that are norm max or 1, not sure how to normalize them
		print("\nWARNING: difference =",result,", not in [0,1]!") # this really should not occur! If it does, check if an issue of accuracy
		#print("P1 basins:",P1_basins,"\nP2_bsins:",P2_basins,'\n\n')
	return result


def tune_dist(param_file, reps):
	# finds dist thresh such that max dist btwn 2 runs of the same net < dist thresh/2

	params = parse.params(param_file)
	F, F_mapd, A, V  = parse.net(params)
	max_dist = 0
	params['verbose']=False #i assume
	phenos_start = basin.calc_basin_size(params,F_mapd,V).phenotypes 
	phenos1 = phenos2 = phenos_start
	for r in range(1,reps):
		if r % (reps/10) == 0:
			print("at run #",r)
		phenos = basin.calc_basin_size(params,F_mapd,V).phenotypes 
		if diff(phenos1,phenos) > max_dist and diff(phenos1,phenos) > diff(phenos2,phenos):
			max_dist = diff(phenos1,phenos)
			phenos2 = phenos 
		elif diff(phenos2,phenos) > max_dist:
			max_dist = diff(phenos2,phenos)
			phenos1 = phenos 

	return max_dist*10

if __name__ == "__main__":
	if sys.argv[2] == 'exh':
		params = parse.params(sys.argv[1])
		F, V = parse.get_logic(params)
		exhaustive(params, F, V,mut_dist_thresh=.1, cnt_dist_thresh=.1)
	elif sys.argv[2] == 'tune':
		dist_thresh = tune_dist(sys.argv[1], 100)
		print("suggested distance threshold =",dist_thresh) 
	else:
		assert(0)