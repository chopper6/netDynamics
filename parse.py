import os, sys, yaml, util

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


def params(param_file):

	if not os.path.isfile(param_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
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

	if params['use_phenos'] and 'pheno_file' in params.keys() and params['pheno_file'] is not None:
		if not os.path.isfile(params['pheno_file']):
			sys.exit("Can't find phenotype file: " + params['pheno_file'] + ', check pheno_file parameter.')
		if os.path.splitext(params['pheno_file'])[-1].lower() != '.yaml':
			sys.exit("Phenotype file must be yaml format, check pheno_file parameter.")
		
		with open(params['pheno_file'],'r') as f:
			params['phenos'] = yaml.load(f,Loader=yaml.SafeLoader)


	return params


def net(params):
	net_file = params['net_file']
	# net file should be in DNF
	# separate node from its function by tab

	node_name_to_num, node_num_to_name = {},[]
	node_name_to_num['0']=0 # always OFF node is first
	node_num_to_name += ['0'] 

	max_literals, max_clauses, num_clauses, n = 1, 1, 1,1 #all start at 1 due to the always OFF node (and always off clause)

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
	# go through net_file 2 times
	# go thru 1st time to get nodes & max_literals:
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
			n+=1

			clauses = line[1].split(clause_split)
			num_clauses += len(clauses)
			max_clauses = max(max_clauses, len(clauses))
			for clause in clauses:
				max_literals = max(max_literals, len(clause.split(literal_split)))

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


	# building clauses_to_threadsex, i.e. the index for the function of each node
	if n<256: 
		index_dtype = cp.uint8
	elif n<65536:
		index_dtype = cp.uint16
	else:
		index_dtype = cp.uint32

	# 0th clause is an always false clause
	nodes_to_clauses = cp.zeros((num_clauses,max_literals),dtype=index_dtype) # the literals in each clause
	nodes_clause = {i:[] for i in range(n)}
	curr_clause = 0
	
	# go thru 2nd time to parse the clauses:
	with open(net_file,'r') as file:
		if format_name!='bnet':
			line = file.readline() #again skip 1st line, the encoding
		# first clause is for the 0 node (always false)
		nodes_clause[0] += [curr_clause]
		curr_clause += 1

		for _ in range(n-1): 

			line = file.readline() 
			line = line.strip().split(node_fn_split)
			node = node_name_to_num[line[0].strip()]
			# if using mutations and this node is mutated:
			if params['use_phenos'] and line[0] in params['phenos']['mutations']:
				clauses = [str(params['phenos']['mutations'][line[0]])] #should be '0' or '1' which refer to the always OFF and ON nodes

				if params['debug']:
					assert(clauses == ['0'] or clauses ==['1'])

			else:
				clauses = [splited.strip() for splited in line[1].split(clause_split)]

			for i in range(len(clauses)):
				clause = clauses[i]
				clause_fn = []
				for symbol in strip_from_clause:
					clause = clause.replace(symbol,'')
				literals = clause.split(literal_split)
				for j in range(len(literals)):
					literal_name = literals[j]
					for symbol in strip_from_node:
						literal_name = literal_name.replace(symbol,'')
					literal_node = node_name_to_num[literal_name]
					clause_fn += [literal_node]
				for j in range(len(literals), max_literals):
					clause_fn += [literal_node]

				nodes_to_clauses[curr_clause] = clause_fn
				nodes_clause[node] += [curr_clause]
				curr_clause += 1

	# TODO: clean this and add explanation, curr v opaque
	# TODO: redo with several powers of m (i.e. several log bins of max clauses)
	# bluid clause index with which to compress the clauses -> nodes
	clauses_to_threads = []
	threads_to_nodes = [] 
	m=min(params['clause_bin_size'],max_clauses) #later make this a param, will be max clauses compressed per thread

	i=0
	while sum([len(nodes_clause[i]) for i in range(n)]) > 0:
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
			if sum([len(nodes_clause[i]) for i in range(n)]) > 0:	
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

	#print('done',nodes_to_clauses)
	#print('\n\n',clauses_to_threads)

	clause_mapping = {'nodes_to_clauses':nodes_to_clauses, 'clauses_to_threads':clauses_to_threads, 'threads_to_nodes':threads_to_nodes}
	node_mapping = {'num_to_name':node_num_to_name, 'name_to_num':node_name_to_num}

	catch_errs(params,  clause_mapping, node_mapping)
	return clause_mapping, node_mapping



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


	node_mapping = {'num_to_name':node_num_to_name, 'name_to_num':node_name_to_num}
	return A,int(n/2),N,node_mapping



def catch_errs(params, clause_mapping, node_mapping):
	if params['debug']:
		assert(len(node_mapping['name_to_num'])==len(node_mapping['num_to_name']))
		assert(len(node_mapping['name_to_num'])%2==0) #since 1/2 should be the negative copies

		if params['use_phenos'] and 'mutations' in params['phenos'].keys():
			for k in params['phenos']['mutations']:
				assert(k in node_mapping['name_to_num'].keys())


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