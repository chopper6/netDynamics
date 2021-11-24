# -*- coding: utf-8 -*-
"""
Created on Wed Nov 24 11:44:08 2021

@author: jwKim
"""
import os, sys, yaml, util, math

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed

def params(param_file):
    param_object = Params(param_file)
    
    return param_object.params

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

def net(params):
    net_file = params["net_file"]
    debug = params['debug']
    clause_bin_size = params["clause_bin_size"]
    parallelism = params['parallelism']
    
    net_object = Net(net_file, debug, clause_bin_size, parallelism)
    net_object.apply_mutations(params)
    net_object.get_clause_mapping()
    
    F = net_object.F
    V = net_object.V
    F_mapd = net_object.clause_mapping
    A = net_object.A
	# F is the logic function, and V are the nodes (specifically how node names map to numbers)
    return F, F_mapd, A, V
    

class Net_reading_part:
    def __init__(self, net_file, debug):
        self.net_file = net_file
        # net file should be in DNF, see README for specifications
        self.debug = debug
        
        self.node_name_to_num = {}
        self.node_num_to_name = []
        
        self.max_literals = None
        self.max_clauses = None
        self.num_clauses = None
        self.n = None
        
        self.loop_limit = 1000000
        
        self.format_name = None
        
    def _initialize(self):
        print("you should override this function")
        
    def _read_net_file(self):
        net_file = self.net_file
        # net file should be in DNF, see README for specifications
        
        if not os.path.isfile(net_file):
            sys.exit("Can't find network file: " + str(net_file)) 
            
        with open(net_file,'r') as file:
            extension = os.path.splitext(net_file)[-1]
            if extension == '.bnet':
                self.format_name='bnet'
            else:
                self.format_name = file.readline().replace('\n','')
            self._get_file_format(self.format_name)
            
            loop = 0
            while True:
                line = file.readline()
                if not line: #i.e eof
                    break
                if loop > self.loop_limit:
                    sys.exit("Hit an infinite loop, unless net is monstrously huge")
                self._parse_net_file_line(line)
                loop += 1
        
               
    def _parse_net_file_line(self, line):
        line = line.strip().split(self.node_fn_split)
        node_name = line[0].strip()
        clauses = line[1].split(self.clause_split)
        
        self._process_node_name_part(node_name)
        self._process_clauses(node_name, clauses)
        
    def _process_node_name_part(self, node_name):
        node_name = self._replace_symbol_in_strip_from_node(node_name, '')
        if self.debug:
            assert(node_name not in self.node_name_to_num.keys())
        
        self.node_name_to_num[node_name] = self.n
        self.node_num_to_name += [node_name]
        self.n+=1
    
    def _process_clauses(self,node_name, clauses):
        print("you should override it")
    
    def _get_file_format(self, format_name):
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)
        self.node_fn_split = node_fn_split
        self.clause_split = clause_split
        self.literal_split = literal_split
        self.not_str = not_str
        self.strip_from_clause = strip_from_clause
        self.strip_from_node = strip_from_node
    
    def _get_index_dtype(self):
        if self.n<256: 
            self.index_dtype = cp.uint8
        elif self.n<65536:
            self.index_dtype = cp.uint16
        else:
            self.index_dtype = cp.uint32
            
    def _replace_symbol_in_strip_from_node(self, variable, replace_value=''):
        for symbol in self.strip_from_node:
            variable = variable.replace(symbol,replace_value)
        return variable


class Net(Net_reading_part):
    def __init__(self, net_file, debug=True, clause_bin_size=99999, parallelism=128):
        super().__init__(net_file, debug)
        self.F= {}
        self.V = {}#{'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
        self._initialize()
        
        
        self._read_net_file()
        self.V = {'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
                  
        self.index_dtype = None
        self._get_index_dtype()
        
        self.clause_bin_size = clause_bin_size
        self.thread_dtype = self._get_thread_dtype(parallelism)
        
        self.nodes_to_clauses = None
        self.nodes_clause = None
        self._curr_clause = 0
        self.A = None # ADJACENCY MATRIX, unsigned
        self.clauses_to_threads = []
        self.threads_to_nodes = []
        self.clause_mapping = {}
    
    def _initialize(self):
        self.node_name_to_num['0']=0 # always OFF node is first
        self.node_num_to_name += ['0']
        self.F['0'] = [[0]]
        self.max_literals = 1
        self.max_clauses = 1
        self.num_clauses = 1
        self.n = 1
    
    def _read_net_file(self):
        super()._read_net_file()
        self._put_negative_nodes_as_2nd_half_of_node_array()
        
        self.V['#2name'] = self.node_num_to_name
        self.V['name2#'] = self.node_name_to_num        
        
        
    def _process_node_name_part(self, node_name):
        super()._process_node_name_part(node_name)
        
        self.F[node_name] =  []
    
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
                literal_name = self._replace_symbol_in_strip_from_node(literal_name, '')
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
                if f in params['mutations']:
                    self.F[f] = [str(params['mutations'][f])] #should be '0' or '1' which refer to the always OFF and ON nodes
                    if params['debug']:
                        assert(self.F[f][0] in ['0','1'])
                    
    def get_clause_mapping(self):
        if self.debug:
            assert(self.n==len(self.V['#2name'])/2)
                                 
        # 0th clause is an always false clause
        self.nodes_to_clauses = cp.zeros((self.num_clauses,self.max_literals),dtype=self.index_dtype) # the literals in each clause
        self.nodes_clause = {i:[] for i in range(self.n)}
        self._curr_clause = 0
        
        # ADJACENCY MATRIX, unsigned
        self.A = cp.zeros((self.n,self.n),dtype=bool)
        
        # go thru F to parse the clauses
        self.nodes_clause[0] += [self._curr_clause]
        self.A[0,0]=1
        self._curr_clause += 1
        for node_num in range(1,self.n): #(1,n) due to the 0 node
            self._process_clause_to_node_num(node_num)
        
        self._get_thread()
        
        self.clauses_to_threads = cp.array(self.clauses_to_threads, dtype=self.thread_dtype)
        self.clause_mapping = {'nodes_to_clauses':self.nodes_to_clauses, 
                               'clauses_to_threads':self.clauses_to_threads, 
                               'threads_to_nodes':self.threads_to_nodes}
        
    
    def _process_clause_to_node_num(self, node_num: "self.n > node_num >=2 only"):
        node_name = self.node_num_to_name[node_num]
        clauses = self.F[node_name]
        for clause in clauses:
            clause_fn = []
            for literal_node in clause:
                literal_node_num = self.node_name_to_num[literal_node]
                clause_fn += [literal_node_num]
                if literal_node_num >= self.n:#i.e. is a negative node
                    self.A[node_num, literal_node_num-self.n] = 1
                else:
                    self.A[node_num, literal_node_num] = 1
            for _ in range(len(clause), self.max_literals): # filling to make sq matrix
                clause_fn += [literal_node_num]
            
            self.nodes_to_clauses[self._curr_clause] = clause_fn
            self.nodes_clause[node_num] += [self._curr_clause]
            self._curr_clause += 1
    
    @staticmethod
    def _get_thread_dtype(parallelism):
        if parallelism <256:
            thread_dtype = cp.uint8
        elif parallelism <65535:
            thread_dtype = cp.uint16
        else:
            thread_dtype = cp.uint32
        return thread_dtype
            
    def _get_thread(self):
        m = min(self.clause_bin_size, self.max_clauses) #later make this a param, will be max clauses compressed per thread
        
        i = 0
        while sum(len(self.nodes_clause[i2]) for i2 in range(self.n)) > 0:
            # ie when have mapped all clauses
            
            # after each clauses_to_threads[i], need to add an index for where the new (compressed) clauses will go
            this_set=[[] for _ in range(self.n)]
            self.threads_to_nodes += [cp.zeros((self.n,self.n),dtype=bool)]
            sorted_keys = sorted(self.nodes_clause, key=lambda k: len(self.nodes_clause[k]), reverse=True)
            self.nodes_clause = {k:self.nodes_clause[k] for k in sorted_keys}
            node_indx = 0
            
            for j in range(self.n):
                take_from_node = sorted_keys[node_indx]
                self.threads_to_nodes[i][j,take_from_node]=1
                if sum([len(self.nodes_clause[i2]) for i2 in range(self.n)]) > 0:
                    if len(self.nodes_clause[take_from_node]) >= m:
                        this_set[j] = self.nodes_clause[take_from_node][:m]
                        if len(self.nodes_clause[take_from_node]) == m:
                            node_indx += 1
                        del self.nodes_clause[take_from_node][:m]
                    else:
                        top = len(self.nodes_clause[take_from_node])
                        this_set[j] = self.nodes_clause[take_from_node][:top]
                        rem = m-(top)
                        this_set[j] += [0 for _ in range(rem)] #use a false clause to make matrix square (assuming DNF)
                        del self.nodes_clause[take_from_node][:top]
                        node_indx += 1
                else:#finished, just need to filll up the array
                    this_set[j] = [0 for _ in range(len(this_set[j-1]))] #alt could copy this_set[j-1]
            self.clauses_to_threads += [this_set]
                    
            
            i += 1
            if i > self.loop_limit:
                sys.exit("ERROR: infinite loop likely in parse.net()")
    

    
    def catch_errs(self, params):
        if params['debug']:
            assert(len(self.V['name2#'])==len(self.V['#2name']))
            assert(len(self.V['name2#'])%2==0) #since 1/2 should be the negative copies
        
        if 'mutations' in params.keys():
            for k in params['mutations']:
                assert(k in self.V['name2#'].keys())

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


class Expanded_net(Net_reading_part):
    def __init__(self, net_file, debug):
        super().__init__(net_file, debug)
        self._initialize()
        self._read_net_file()
        
        self.V = {}#{'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
        
    def _initialize(self):
        self.n = 0
        self.num_clauses = 0
        
    def _process_clauses(self,node_name, clauses):
        for clause in clauses:
            literals = clause.split(self.literal_split)
            if len(literals)>1:
                self.num_clauses += 1
    
    def buile_composite_nodes_and_A(self):
        self.N = self.num_clauses+self.n
        self.A = cp.zeros((self.N,self.N))
        assert(self.n%2==0) #should be even since half are regular and half are complement nodes

        assert(len(self.node_name_to_num)==len(self.node_num_to_name)==self.n) #sanity check
        self._get_index_dtype()
        
        self.composite_counter = 0
        
        self._read_second_time()
        
        
    def _read_second_time(self):
        # go thru 2nd time to build composite nodes and A
        with open(self.net_file,'r') as file:
            if self.format_name!='bnet':
                line = file.readline() #again skip 1st line, the encoding
            
            for _ in range(self.n): 
                line = file.readline()
                line = line.strip().split(self.node_fn_split)
                node = self.node_name_to_num[line[0].strip()]
                
                clauses = [splited.strip() for splited in line[1].split(self.clause_split)]
                
                # fill A with A[2*n+composite_counter,node] = 1 for each composite node
                # and A[i,2*n+composite_counter]=1 for each input i to the composite node
                
                for i in range(len(clauses)):
                    clause = clauses[i]
                    for symbol in self.strip_from_clause:
                        clause = clause.replace(symbol,'')
                        
                    literals = clause.split(self.literal_split)
                    
                    if len(literals)>1:
                        self.A[self.n+self.composite_counter,node]=1
                        for j in range(len(literals)):
                            literal_name = literals[j]
                            literal_name =self._replace_symbol_in_strip_from_node(literal_name, '')
                            literal_node = self.node_name_to_num[literal_name]
                            self.A[literal_node,self.n+self.composite_counter]=1
                        self.composite_counter+=1
                    else:
                        literal_name = literals[0]
                        literal_name =self._replace_symbol_in_strip_from_node(literal_name, '')
                        literal_node = self.node_name_to_num[literal_name]
                        self.A[literal_node,node]=1
        
        self.V = {'#2name':self.node_num_to_name, 'name2#':self.node_name_to_num}
        if self.debug:
            assert(self.n%2==0) #half should be negative nodes
                        
                            
                
        
    
    

def expanded_net(params, net_file):
    Expanded_net_object = Expanded_net(net_file, params["debug"])
    Expanded_net_object.buile_composite_nodes_and_A()
    A = Expanded_net_object.A
    n = Expanded_net_object.n
    N = Expanded_net_object.N
    V = Expanded_net_object.V
    return A,int(n/2),N,V



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