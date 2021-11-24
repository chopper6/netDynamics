# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:44:08 2021

@author: jwKim
"""
import os, sys, yaml, util, math

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed

class Params:
    def __init__(self, params_file):
        self.param_file = params_file
        self.params = None
        self._read_file()
    
    def _read_file(self):
        if not os.path.isfile(self.param_file):
            sys.exit("Can't find network file: " + str(self.param_file)) #
            
        with open(self.param_file,'r') as f:
            self.params = yaml.load(f,Loader=yaml.SafeLoader)
        
        self._refine_params()
        self._check_and_read_model_file()
        self._input_nodes_processing()
    
    def _refine_params(self):
        """change or add some values in self.params"""
        # yaml doesn't maintain json's 10e5 syntax, so here is support for scientific notation. Syntax: 10^5
        for k in self.params.keys():
            if isinstance(self.params[k], str) and '^' in self.params[k]:
                parts = self.params[k].split('^')
                self.params[k] = int(parts[0])**int(parts[1])
        
        self.params['parallelism'] = int(max(self.params['parallelism'],1)) #1 is actually sequential, i.e. run 1 at a time

        CUPY, cp = util.import_cp_or_np(try_cupy=1) #test import
        self.params['cupy'] = CUPY
    
    def _check_and_read_model_file(self):
        model_file = self.params['model_file']
        if not os.path.isfile(model_file):
            sys.exit("Can't find model_file: " + model_file + ', check path in parameter file.')
        if os.path.splitext(model_file)[-1].lower() != '.yaml':
            sys.exit("model_file must be yaml format.")
            
        with open(self.params['model_file'],'r') as f:
            model = yaml.load(f,Loader=yaml.SafeLoader)
        
        if self.params['debug']:
            shared_items = {k: self.params[k] for k in self.params if k in model}
            assert(len(shared_items)==0) #should not have overlaping keys between params and model files
        self.params = {**model, **self.params} # in python3.9 can use params_model | params, but may require some people to update python
    
    def _input_nodes_processing(self):
        if 'inputs' in self.params.keys():
            k = len(self.params['inputs'])
            actual_num_samples = math.floor(self.params['parallelism']/(2**k))*2**k
            assert(actual_num_samples>0) # possible to have too low parallelism for the number of inputs used (min 2^k required)
        if actual_num_samples!=self.params['num_samples']:
            print("\nWARNING: only", str(actual_num_samples),"used to maintain even ratio of input samples.\n")
            print("\nWARNIN: temp solution in parse.py sets num_samples == parallelism")
            # TODO: should be a multiple of parallellism, not nec ==
            # see also basin.get_init_sample()
            self.params['num_samples'] = self.params['parallelism'] = actual_num_samples
    
    def get_logic(self):
        # returns F, V
        # F: the logic of the network, where F={}, F[node_name] = [clause1,clause2,...] and each clause=[literal1,literal2,...]
        #	DNF so F[node] = clause1 OR clause2 ..., whereas clause = literal1 AND literal2 ...
        # V['#2name'] = [] that maps node number's to their original names
        # V['name2#'] = {} that maps node's name to their number
        pass
    
    

class Net:
    def __init__(self, net_file, debug=True):
        self.net_file = net_file
        # net file should be in DNF, see README for specifications
        self.debug = debug
        
        self.node_name_to_num = {}
        self.node_num_to_name = []
        self.F= {}
        self.V = {}#{'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
        self._initialize()
        
        self.max_literals = 1
        self.max_clauses = 1
        self.num_clauses = 1
        self.n = 1
        
        self.loop_limit = 1000000
        
        self._read_net_file()
        self.V = {'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
    
    def _initialize(self):
        self.node_name_to_num['0']=0 # always OFF node is first
        self.node_num_to_name += ['0']
        self.F['0'] = [[0]]
    
    def _read_net_file(self):
        net_file = self.net_file
        # net file should be in DNF, see README for specifications
        
        if not os.path.isfile(net_file):
            sys.exit("Can't find network file: " + str(net_file)) 
            
        with open(net_file,'r') as file:
            extension = os.path.splitext(net_file)[-1]
            if extension == '.bnet':
                format_name='bnet'
            else:
                format_name = file.readline().replace('\n','')
            self._get_file_format(format_name)
            
            loop = 0
            while True:
                line = file.readline()
                if not line: #i.e eof
                    break
                if loop > self.loop_limit:
                    sys.exit("Hit an infinite loop, unless net is monstrously huge")
                self._parse_net_file_line(line)
                loop += 1
        
        self._put_negative_nodes_as_2nd_half_of_node_array()
        
        self.V['#2name'] = self.node_num_to_name
        self.V['name2#'] = self.node_name_to_num
                
                
    
    def _get_file_format(self, format_name):
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)
        self.node_fn_split = node_fn_split
        self.clause_split = clause_split
        self.literal_split = literal_split
        self.not_str = not_str
        self.strip_from_clause = strip_from_clause
        self.strip_from_node = strip_from_node
        
    def _parse_net_file_line(self, line):
        line = line.strip().split(self.node_fn_split)
        node_name = line[0].strip()
        clauses = line[1].split(self.clause_split)
        
        self._process_node_name_part(node_name)
        self._process_clauses(node_name, clauses)
        
        
        
    def _process_node_name_part(self, node_name):
        for symbol in self.strip_from_node:
            node_name = node_name.replace(symbol,'')
        if self.debug:
            assert(node_name not in self.node_name_to_num.keys())
        
        self.node_name_to_num[node_name] = self.n
        self.node_num_to_name += [node_name]
        self.F[node_name] =  []
        self.n+=1
    
    def _process_clauses(self,node_name, clauses):
        self.num_clauses += len(clauses)
        self.max_clauses = max(self.max_clauses, len(clauses))
        for clause in clauses:
            this_clause=[]
            self.max_literals = max(self.max_literals, len(clause.split(self.literal_split)))
            for symbol in self.strip_from_clause:
                clause = clause.replace(symbol,'')
            literals = clause.split(self.literal_split)
            for j in range(len(literals)):
                literal_name = literals[j]
                for symbol in self.strip_from_node:
                    literal_name = literal_name.replace(symbol,'')
                this_clause += [literal_name]
            self.F[node_name] += [this_clause]
    
    def _put_negative_nodes_as_2nd_half_of_node_array(self):
        self.node_name_to_num['1'] = self.n #add the always ON node (i.e. not 0)
        self.node_num_to_name += ['1']
        
        node_num=self.n+1
        for i in range(1,self.n):
            self.node_name_to_num[self.not_str + self.node_num_to_name[i]] = node_num
            self.node_num_to_name += [self.not_str + self.node_num_to_name[i]]
            node_num+=1
        
        assert(len(self.node_name_to_num)==len(self.node_num_to_name)==self.n*2) #sanity check
        
    def apply_mutations(self, params):
        if 'mutations' in params:
            for f in self.F:
                if f in self.params['mutations']:
                    self.F[f] = [str(params['mutations'][f])] #should be '0' or '1' which refer to the always OFF and ON nodes
                if params['debug']:
                    assert(self.F[f][0] in ['0','1'])
    
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

def catch_errs(params, clause_mapping, V):
	if params['debug']:
		assert(len(V['name2#'])==len(V['#2name']))
		assert(len(V['name2#'])%2==0) #since 1/2 should be the negative copies

		if 'mutations' in params.keys():
			for k in params['mutations']:
				assert(k in V['name2#'].keys())


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