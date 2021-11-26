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

def randomize_experiment(param_file):
	reps = 2
	num_swaps = 8
	mut_dist_thresh, cnt_dist_thresh = .2,.2
	net_stats, node_stats = {1:{},2:{}}, {}

	params = parse.params(param_file)
	params['save_fig'] = True 
	F, V = parse.get_logic(params)
	n2=len(V['#2name'])-2 #inclds composite nodes but not 0 node

	result = exhaustive(params, F, V, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
	net_stats[1]['original'] = result['network_stats1']
	net_stats[2]['original'] = result['network_stats2']
	node_stats = result['node_stats']
	node_stat_labels = result['node_stat_labels']

	parse.apply_mutations(params,F)
	F_mapd, A = parse.get_clause_mapping(params, F, V)
	
	#canal_score, total_score = features.calc_canalizing(F)

	# deg-preserving logic scrambles
	for r in range(reps): 
		F_copy, A_copy, V_copy = deepcopy(F), deepcopy(A),deepcopy(V)
		for _ in range(num_swaps):
			success=False
			while not success:
				node = V['#2name'][rd.randint(1,int(len(V['#2name'])/2)-1)] #skip that 0 node, and only do pos nodes
				success = swap_TT_row(F_copy, node, 1, num_attempts=20) # 1 means only change 1 row of TT
		if params['debug']:
			# check is degree preserving
			assert(np.all(np.sum(A_copy,axis=0)== np.sum(A,axis=0)) and np.all(np.sum(A_copy,axis=1)== np.sum(A,axis=1)))
		result = exhaustive(params, F_copy, V_copy, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
		net_stats[1]['logicScramble_' + str(r)] = result['network_stats1']
		net_stats[2]['logicScramble_' + str(r)] = result['network_stats2']
		for k in node_stats:
			node_stats[k] += result['node_stats'][k]

	# deg-preserving edge swaps
	for r in range(reps): 
		F_copy, A_copy, V_copy = deepcopy(F), deepcopy(A),deepcopy(V)
		swap_edges(F_copy, A_copy, V_copy,num_swaps*10,no_repeat_edges=True) #modifies F and A in place 
		if params['debug']:
			# check is degree preserving
			assert(np.all(np.sum(A_copy,axis=0)== np.sum(A,axis=0)) and np.all(np.sum(A_copy,axis=1)== np.sum(A,axis=1)))
		result  = exhaustive(params, F_copy, V_copy, mut_dist_thresh=mut_dist_thresh, cnt_dist_thresh=cnt_dist_thresh)
		net_stats[1]['edgeSwap_' + str(r)] = result['network_stats1']
		net_stats[2]['edgeSwap_' + str(r)] = result['network_stats2']
		for k in node_stats:
			node_stats[k] += result['node_stats'][k]
			

	pickle_file = params['output_dir'] + '/stats.pickle' 
	stats = {'node_stats':node_stats,'net_stats':net_stats,'node_stat_labels':node_stat_labels}
	print("Pickling a file to: ", pickle_file)
	pickle.dump(stats, open( pickle_file, "wb" ))
	plot.control_exper_bar(params, net_stats) 
	plot.control_exper_scatter(params, node_stats, node_stat_labels)


def swap_edges(F,A,V,repeats,no_repeat_edges=True):
	# pick 2 random edges, then pick source or targets, then swap their variables
	# note that logic is preserved, albiet with different sources
	# modifies F and A in place
	for r in range(repeats):
		E1 = get_rd_edge(A)
		try_again = True
		loop=0
		while try_again : #ensure they are diff edges
			E2 = get_rd_edge(A)
			try_again = np.all(E1==E2)
			if no_repeat_edges and (A[E1[0],E2[1]]!=0 or A[E2[0],E1[1]]!=0): 
				#TODO: should repeat edges check the new or the old A?? curr using updated A
				try_again = True
			loop+=1
			assert(loop<10**5)

		source1, source2, target1, target2 = V['#2name'][E1[0]], V['#2name'][E2[0]], V['#2name'][E1[1]], V['#2name'][E2[1]]

		swap_literals(F[target1],source1,source2)
		swap_literals(F[target2],source2,source1)
		
		# rm old, add new edges (assumes unsigned)
		A[E1[0],E1[1]]=0
		A[E2[0],E2[1]]=0	
		A[E1[0],E2[1]]=1
		A[E2[0],E1[1]]=1


def swap_TT_row(F, node, num_swaps, num_attempts=100):
	# returns True if swap was successful, False if ran out of attempts
	# note that a swap with all desired constraints may not be possible

	# WARNING: assumes bnet all over the place here
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node = parse.get_file_format('bnet')
	
	attempt=0
	while True:
		inputs_rev, inputs_fwd =  get_inputs_to_node(F[node], not_str)
		int_clauses = []
		for clause in F[node]:
			int_clause = 0
			for ele in clause:
				if not_str not in ele: 
					int_clause += 2**inputs_fwd[ele]
			int_clauses += [int_clause]
		n = len(inputs_rev)

		for _ in range(num_swaps):
			loop=0
			while True:
				new_term = rd.randint(0,2**n-1)
				if new_term not in int_clauses:
					del int_clauses[rd.randint(0,len(int_clauses)-1)]
					int_clauses += [new_term]
					break
				if loop>num_attempts:
					return False
				loop+=1

		reduced_fn = logic.run_qm(int_clauses, n, 'bnet', inputs_rev)

		parsed_clauses = reverse_parse_reduced_clauses(reduced_fn,node)
		inputs_rev_orig = inputs_rev.copy()
		inputs_rev, inputs_fwd =  get_inputs_to_node(parsed_clauses, not_str)
		if len(inputs_rev) == n: #i.e. check that maintain same # of functional inputs (can lose due to redundancy)
			F[node] = parsed_clauses
			return True

		assert(len(inputs_rev) <= n)

		if attempt > num_attempts:
			return False
		attempt+=1

def get_inputs_to_node(clauses, not_str):
	# fwd is name2#, rev is #toname
	inputs_rev, inputs_fwd = [],{}
	for clause in clauses:
		for ele in clause:
			ele_ = str(ele).replace(not_str,'')
			if ele_ not in inputs_rev:
				inputs_fwd[ele_] = len(inputs_rev)
				inputs_rev += [ele_]
	return inputs_rev, inputs_fwd

def reverse_parse_reduced_clauses(reduced_fn, node_name):
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node = parse.get_file_format('bnet')
	clauses = reduced_fn.split(clause_split)
	parsed_clauses = []
	for clause in clauses:
		this_clause=[]
		for symbol in strip_from_clause:
			clause = clause.replace(symbol,'')
		literals = clause.split(literal_split)
		for j in range(len(literals)):
			literal_name = literals[j]
			for symbol in strip_from_node:
				literal_name = literal_name.replace(symbol,'')
			this_clause += [literal_name]
		parsed_clauses += [this_clause]
	return parsed_clauses


def swap_literals(logic_fn,lit1,lit2):
	# swaps lit1 for lit2 in the logical function
	for clause in logic_fn: 
		for i in range(len(clause)):
			if clause[i] == lit1:
				clause[i] = lit2



def get_rd_edge(A):
	sign=0
	loop=0
	while sign==0:
		assert(loop<10**10) 
		E = np.random.randint(len(A), size=2)
		sign = A[E[0],E[1]]
		loop+=1
	return E

###############################################################################################################

def exhaustive(params, F, V,max_control_size = 1, mut_dist_thresh=.2,cnt_dist_thresh=.2):

	# assumes cannot alter pheno nodes, nor input nodes
	# complexity is O(s*n^m+1*2^i) where s is the simulation time of a single input and i is the number of inputs, and m is the max control set size

	n = len(V['#2name'])
	nodes = V['#2name'][1:int(n/2)] #excluding 0 node and negative nodes (TODO cleaner?)
	nodes = [node for node in nodes if node not in params['outputs'] and node not in params['inputs']]

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
	elif sys.argv[2] == 'rd':
		randomize_experiment(sys.argv[1]) 
	elif sys.argv[2] == 'tune':
		dist_thresh = tune_dist(sys.argv[1], 100)
		print("suggested distance threshold =",dist_thresh) 
	else:
		assert(0)