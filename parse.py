import os, sys, yaml, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


def params(param_file):
	with open(param_file,'r') as f:
		params = yaml.load(f,Loader=yaml.FullLoader)
	return params


def net(net_file):
	# net file should be in DNF
	# separate node from its function by tab
	# each clause should be separated by a space
	# each element of clause separated by &
	# - means element of clause is negative
	# for example: X = (A and B) or (not C) --> X	A&B -C
	# each node should only have 1 function, ie #lines = #nodes

	node_name_to_num, node_num_to_name, neg_nodes = {},{},[]
	max_clauses, max_literals, n = 1, 1, 0

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
	# go through net_file 3 times. Lazy, but parse only gets called once so ok
	# go thru 1st time to get # nodes, max_clauses, max_literals, and negative nodes:
	with open(net_file,'r') as file:
		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if loop > 10000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 

			line = line.strip().split('\t')
			clauses = line[1].split(' ')
			max_clauses = max(max_clauses, len(clauses))
			for clause in clauses:
				for ele in clause.split('&'):
					if '-' in ele:
						if ele.replace('-','') not in node_name_to_num.keys():
							node_name_to_num[ele.replace('-','')]=len(neg_nodes)
							node_num_to_name[len(neg_nodes)] = ele.replace('-','')
							neg_nodes += [ele]
				max_literals = max(max_literals, len(clause.split('&')))

			loop += 1
			n += 1

	# go thru 2nd time to fill in rest of node names
	with open(net_file,'r') as file:
		node_num = len(neg_nodes)
		for _ in range(n):
			line = file.readline() 
			line = line.strip().split('\t')
			if line[0] not in node_name_to_num.keys():
						
				node_name_to_num[line[0]] = node_num
				node_num_to_name[node_num] = line[0]
				node_num+=1
	# put the negative nodes at the end
	for i in range(len(neg_nodes)):
		node_name_to_num[neg_nodes[i]] = node_num
		node_num_to_name[node_num] = neg_nodes[i]
		node_num+=1

	assert(len(node_name_to_num)==len(node_num_to_name)==n+len(neg_nodes)) #sanity check


	# building clause_index, i.e. the index for the function of each node
	if n<256: 
		index_dtype = cp.uint8
	elif n<65536:
		index_dtype = cp.uint16
	else:
		index_dtype = cp.uint32

	clause_index = cp.zeros((n,max_clauses, max_literals),dtype=index_dtype)
	
	# element at i,j,k is the node # of the kth literal corresponding to the jth clause of the ith node's function
	# need square matrix for numpy/cupy, so copy the last literal or clause to fill space but maintain same output

	# go thru 3rd time to parse the clauses:
	with open(net_file,'r') as file:

		for _ in range(n): 

			line = file.readline()
			line = line.strip().split('\t')
			node = node_name_to_num[line[0]]

			clauses = line[1].split(' ')

			for i in range(len(clauses)):
				literals = clauses[i].split('&')
				for j in range(len(literals)):
					literal_node = node_name_to_num[literals[j]]
					clause_index[node,i,j] = literal_node
				for j in range(len(literals),max_literals):
					clause_index[node,i,j] = literal_node
					# need square matrix with same truth output, just repeat the last literal
			for i in range(len(clauses),max_clauses):
				clause_index[node,i] = clause_index[node,len(clauses)-1]
				# need square matrix with same truth output, just repeat the last clause


	return clause_index, node_num_to_name, node_name_to_num, n, len(neg_nodes)