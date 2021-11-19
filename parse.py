import os, sys, yaml, util, math

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


def params(param_file):

	if not os.path.isfile(param_file):
		sys.exit("Can't find parameter file: " + str(param_file)) 
	
	with open(param_file,'r') as f:
		params = yaml.load(f,Loader=yaml.SafeLoader)

	# yaml doesn't maintain json's 10e5 syntax, so here is support for scientific notation. Syntax: 10^5
	for k in params.keys():
		if isinstance(params[k], str) and '^' in params[k]:
			parts = params[k].split('^')
			params[k] = int(parts[0])**int(parts[1])

	params['parallelism'] = int(max(params['parallelism'],1)) #1 is actually sequential, i.e. run 1 at a time

	CUPY, cp = util.import_cp_or_np(try_cupy=1) #test import
	params['cupy'] = CUPY

	if not os.path.isfile(params['model_file']):
		sys.exit("Can't find model_file: " + params['model_file'] + ', check path in parameter file.')
	if os.path.splitext(params['model_file'])[-1].lower() != '.yaml':
		sys.exit("model_file must be yaml format.")
	
	with open(params['model_file'],'r') as f:
		model = yaml.load(f,Loader=yaml.SafeLoader)

	if params['debug']:
		shared_items = {k: params[k] for k in params if k in model}
		assert(len(shared_items)==0) #should not have overlaping keys between params and model files
	params = {**model, **params} # in python3.9 can use params_model | params, but may require some people to update python

	return params


def net(params):

	F, V = get_logic(params)
	# F is the logic function, and V are the nodes (specifically how node names map to numbers)

	apply_mutations(params,F)
	F_mapd, A = get_clause_mapping(params, F, V) 
	 
	return F, F_mapd, A, V


def get_logic(params):
	# returns F, V
	# F: the logic of the network, where F={}, F[node_name] = [clause1,clause2,...] and each clause=[literal1,literal2,...]
	#	DNF so F[node] = clause1 OR clause2 ..., whereas clause = literal1 AND literal2 ...
	# V['#2name'] = [] that maps node number's to their original names
	# V['name2#'] = {} that maps node's name to their number

	net_file = params['net_file']
	# net file should be in DNF, see README for specifications

	node_name_to_num, node_num_to_name = {},[]
	node_name_to_num['0']=0 # always OFF node is first
	node_num_to_name += ['0'] 

	F = {'0':[[0]]}

	max_literals, max_clauses, num_clauses, n = 1, 1, 1,1 #all start at 1 due to the always OFF node (and always off clause)

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
	with open(net_file,'r') as file:
		extension = net_file.split('.')
		if extension[-1] == 'bnet':
			format_name='bnet'
		else:
			format_name = file.readline().replace('\n','')
		node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)

		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if loop > 1000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 

			line = line.strip().split(node_fn_split)
			node_name = line[0].strip()
			for symbol in strip_from_node:
				node_name = node_name.replace(symbol,'')
			if params['debug']:
				assert(node_name not in node_name_to_num.keys())
			node_name_to_num[node_name] = n
			node_num_to_name += [node_name]
			F[node_name] =  []
			n+=1

			clauses = line[1].split(clause_split)
			num_clauses += len(clauses)
			max_clauses = max(max_clauses, len(clauses))
			for clause in clauses:
				this_clause=[]
				max_literals = max(max_literals, len(clause.split(literal_split)))
				for symbol in strip_from_clause:
					clause = clause.replace(symbol,'')
				literals = clause.split(literal_split)
				for j in range(len(literals)):
					literal_name = literals[j]
					for symbol in strip_from_node:
						literal_name = literal_name.replace(symbol,'')
					this_clause += [literal_name]
				F[node_name] += [this_clause]

			loop += 1

	# put the negative nodes as 2nd half of node array
	node_name_to_num['1'] = n #add the always ON node (i.e. not 0)
	node_num_to_name += ['1']
	node_num=n+1
	for i in range(1,n):
		node_name_to_num[not_str + node_num_to_name[i]] = node_num
		node_num_to_name += [not_str + node_num_to_name[i]]
		node_num+=1

	assert(len(node_name_to_num)==len(node_num_to_name)==n*2) #sanity check
	
	V = {'#2name':node_num_to_name, 'name2#':node_name_to_num}
	
	return F, V


def apply_mutations(params,F):
	if 'mutations' in params.keys():
		for f in F:
			if f in params['mutations']:
				F[f] = [str(params['mutations'][f])] #should be '0' or '1' which refer to the always OFF and ON nodes
				if params['debug']:
					assert(F[f][0] in ['0','1'])


def get_clause_mapping(params, F, V): 

	n = len(F)
	if params['debug']:
		assert(n==len(V['#2name'])/2)

	# building clauses_to_threadsex, i.e. the index for the function of each node
	if n<256: 
		index_dtype = cp.uint8
	elif n<65536:
		index_dtype = cp.uint16
	else:
		index_dtype = cp.uint32

	num_clauses, max_literals, max_clauses = 0,0,0
	for node in F:
		num_clauses += len(F[node])
		max_clauses = max(max_clauses, len(F[node]))
		for clause in F[node]:
			max_literals = max(max_literals, len(clause))

	node_num_to_name = V['#2name']
	node_name_to_num = V['name2#']

	# 0th clause is an always false clause
	nodes_to_clauses = cp.zeros((num_clauses,max_literals),dtype=index_dtype) # the literals in each clause
	nodes_clause = {i:[] for i in range(n)}
	curr_clause = 0

	# ADJACENCY MATRIX, unsigned
	A = cp.zeros((n,n),dtype=bool)
	
	# go thru F to parse the clauses
	nodes_clause[0] += [curr_clause]
	A[0,0]=1
	curr_clause += 1

	for l in range(1,n): #(1,n) due to the 0 node
		node_num=l 
		node_name = node_num_to_name[l]
		clauses = F[node_name]
		for i in range(len(clauses)):
			clause = clauses[i]
			clause_fn = []
			for j in range(len(clause)):
				literal_node = node_name_to_num[clause[j]]
				clause_fn += [literal_node]
				if literal_node >= n: #i.e. is a negative node
					A[node_num, literal_node-n] = 1
				else:
					A[node_num, literal_node] = 1
			for j in range(len(clause), max_literals): # filling to make sq matrix
				clause_fn += [literal_node]

			nodes_to_clauses[curr_clause] = clause_fn
			nodes_clause[node_num] += [curr_clause]
			curr_clause += 1


	# bluid clause index with which to compress the clauses -> nodes
	#	nodes_to_clauses: computes which nodes are inputs (literals) of which clauses
	#	clauses_to_threads: maps which clauses will be processed by which threads (and #threads = #nodes)
	#	threads_to_nodes: maps which threads are functions of which nodes
	clauses_to_threads = []
	threads_to_nodes = [] 
	m=min(params['clause_bin_size'],max_clauses) #later make this a param, will be max clauses compressed per thread

	i=0
	while sum([len(nodes_clause[i2]) for i2 in range(n)]) > 0:
		# ie when have mapped all clauses

		# after each clauses_to_threads[i], need to add an index for where the new (compressed) clauses will go
		this_set=[[] for _ in range(n)]
		threads_to_nodes += [cp.zeros((n,n),dtype=bool)]
		sorted_keys = sorted(nodes_clause, key=lambda k: len(nodes_clause[k]), reverse=True)
		nodes_clause = {k:nodes_clause[k] for k in sorted_keys}
		node_indx = 0

		for j in range(n):
			take_from_node = sorted_keys[node_indx]
			threads_to_nodes[i][j,take_from_node]=1
			if sum([len(nodes_clause[i2]) for i2 in range(n)]) > 0:	
				if len(nodes_clause[take_from_node]) >= m:
					this_set[j] = nodes_clause[take_from_node][:m]
					if len(nodes_clause[take_from_node]) == m:
						node_indx += 1
					del nodes_clause[take_from_node][:m]
				else:
					top = len(nodes_clause[take_from_node])
					this_set[j] = nodes_clause[take_from_node][:top]
					rem = m-(top)
					this_set[j] += [0 for _ in range(rem)] #use a false clause to make matrix square (assuming DNF)
					del nodes_clause[take_from_node][:top]
					node_indx += 1
			else: #finished, just need to filll up the array
				this_set[j] = [0 for _ in range(len(this_set[j-1]))] #alt could copy this_set[j-1]

		clauses_to_threads += [this_set]
		
		i+=1
		if i>1000000:
			sys.exit("ERROR: infinite loop likely in parse.net()")
	
	if params['parallelism']<256:
		thread_dtype = cp.uint8
	elif params['parallelism']<65535:
		thread_dtype = cp.uint16
	else:
		thread_dtype = cp.uint32
	clauses_to_threads = cp.array(clauses_to_threads,dtype=thread_dtype)

	clause_mapping = {'nodes_to_clauses':nodes_to_clauses, 'clauses_to_threads':clauses_to_threads, 'threads_to_nodes':threads_to_nodes}
	
	catch_errs(params,  clause_mapping, V)
	return clause_mapping, A


def sequences(seq_file_pos, seq_file_neg):
	seqs = []
	with open(seq_file_pos,'r') as file:
		
		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if loop > 1000000:
				sys.exit("Hit an infinite loop, unless file is monstrously huge") 

			seq = []
			was_tab=True
			word=''
			for c in line:
				if c=='\n':
					if word == '':
						seq+=['']
					else:
						seq+=[(word,1)]
					seqs+=[seq]
					break
				elif c=='\t':
					if was_tab:
						seq+=['']
					else:
						if word == '':
							seq+=['']
						else:
							seq+=[(word,1)]
					word=''
				else:
					was_tab=False
					if c in ['\\','/','-']:
						word+='_'
					else:
						word+=c



			loop += 1


	with open(seq_file_neg,'r') as file:
		
		i = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if i > 1000000:
				sys.exit("Hit an infinite loop, unless file is monstrously huge") 

			was_tab=True
			word=''
			j=0
			for c in line:
				if c=='\n':
					if word != '':
						assert(seqs[i][j]=='')
						seqs[i][j]=(word,0)
					break
				elif c=='\t':
					if not was_tab and word != '':
						assert(seqs[i][j]=='')
						seqs[i][j]=(word,0)
					word=''
					j+=1
				else:
					was_tab=False
					if c in ['\\','/','-']:
						word+='_'
					else:
						word+=c
			i += 1

	return seqs


def expanded_net(net_file):
	# assumes that net_file is already in expanded form, to avoid calling qm every time
	# use reduce.py ahead of time to do so
	# net file should already be in DNF and include negative nodes

	node_name_to_num, node_num_to_name = {},[]
	n=num_clauses=0

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
	# go through net_file 2 times
	# go thru 1st time to get nodes & composite nodes
	with open(net_file,'r') as file:
		extension = net_file.split('.')
		if extension[-1] == 'bnet':
			format_name='bnet'
		else:
			format_name = file.readline().replace('\n','')
		node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)

		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break

			line = line.strip().split(node_fn_split)
			node_name = line[0].strip()
			for symbol in strip_from_node:
				node_name = node_name.replace(symbol,'')
			
			assert(node_name not in node_name_to_num.keys())
			node_name_to_num[node_name] = n
			node_num_to_name += [node_name]
			n+=1

			clauses = line[1].split(clause_split)
			for clause in clauses:
				literals = clause.split(literal_split)
				if len(literals)>1:
					num_clauses += 1

	N = num_clauses+n
	A = cp.zeros((N,N))
	assert(n%2==0) #should be even since half are regular and half are complement nodes

	assert(len(node_name_to_num)==len(node_num_to_name)==n) #sanity check


	# building clauses_to_threadsex, i.e. the index for the function of each node
	if n<256: 
		index_dtype = cp.uint8
	elif n<65536:
		index_dtype = cp.uint16
	else:
		index_dtype = cp.uint32

	composite_counter = 0
	
	# go thru 2nd time to build composite nodes and A
	with open(net_file,'r') as file:
		if format_name!='bnet':
			line = file.readline() #again skip 1st line, the encoding

		for _ in range(n): 

			line = file.readline() 
			line = line.strip().split(node_fn_split)
			node = node_name_to_num[line[0].strip()]

			clauses = [splited.strip() for splited in line[1].split(clause_split)]

			# fill A with A[2*n+composite_counter,node] = 1 for each composite node
			# and A[i,2*n+composite_counter]=1 for each input i to the composite node

			for i in range(len(clauses)):
				clause = clauses[i]
				clause_fn = []
				for symbol in strip_from_clause:
					clause = clause.replace(symbol,'')
	
				literals = clause.split(literal_split)

				if len(literals)>1:

					A[n+composite_counter,node]=1
					for j in range(len(literals)):
						literal_name = literals[j]
						for symbol in strip_from_node:
							literal_name = literal_name.replace(symbol,'')
						literal_node = node_name_to_num[literal_name]
						
						A[literal_node,n+composite_counter]=1
					composite_counter+=1
				else:
					literal_name = literals[0]
					for symbol in strip_from_node:
						literal_name = literal_name.replace(symbol,'')
					literal_node = node_name_to_num[literal_name]
					
					A[literal_node,node]=1


	V = {'#2name':node_num_to_name, 'name2#':node_name_to_num}
	if params['debug']:
		assert(n%2==0) #half should be negative nodes
	return A,int(n/2),N,V




def catch_errs(params, clause_mapping, V):
	if params['debug']:
		assert(len(V['name2#'])==len(V['#2name']))
		assert(len(V['name2#'])%2==0) #since 1/2 should be the negative copies

		if 'mutations' in params.keys():
			for k in params['mutations']:
				assert(k in V['name2#'].keys())

	if 'inputs' in params.keys():
		k = len(params['inputs'])
		actual_num_samples = math.floor(params['parallelism']/(2**k))*2**k

		assert(actual_num_samples>0) # possible to have too low parallelism for the number of inputs used (min 2^k required)
		
		if actual_num_samples!=params['num_samples']:
			print("\nWARNING: only", str(actual_num_samples),"used to maintain even ratio of input samples.\n")


def get_file_format(format_name):
	recognized_formats = ['DNFwords','DNFsymbolic','bnet']

	if format_name not in recognized_formats:
		sys.exit("ERROR: first line of network file is the format, which must be one of" + str(recognized_formats))
	
	if format_name == 'DNFsymbolic':
		node_fn_split = '\t'
		clause_split = ' '
		literal_split = '&'
		not_str = '-'
		strip_from_clause = []
		strip_from_node = []
	elif format_name == 'DNFwords':
		node_fn_split = '*= '
		clause_split = ' or '
		literal_split = ' and '
		not_str = 'not '
		strip_from_clause = ['(',')']
		strip_from_node = ['(',')']
	elif format_name == 'bnet':
		node_fn_split = ',\t'
		clause_split = ' | '
		literal_split = ' & '
		not_str = '!'
		strip_from_clause = ['(',')']
		strip_from_node = []

	return node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node




if __name__ == "__main__": # just for debugging purposes
	print("Debugging parse.py")

	sequences('input/efSeq_pos.txt', 'input/efSeq_neg.txt')