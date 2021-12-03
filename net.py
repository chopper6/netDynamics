import os, sys, yaml, util, math, logic
from copy import deepcopy

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# in particular Fmapd should be cupy (if cupy is being used instead of numpy)

# TODO:
#	Net: add a write_to_file(self), just using F and the format
#			read_from_file(self,file), as one init option
#			build_from_G(self,G), as another init option 
#				why not just deepcopy then, and then expand to expanded?
#	add explicit debug function for net construction
#		check that certain objs same, certain diff
#		ex that G.F[name] = G.nodes[G.nodeNames[name]].F
#	passing net file sep is awk af

# TODO LATER:
#	add time-reverse net (and inheritance with it)
# 	Net._build_net() and build_Fmap() are still messy af
#	function that updates expanded net due to input or mutation holding it static? ldoi cover that case anyway tho



class Net:
	def __init__(self,params,net_key='model_file',G=None, complete=False, negatives=False):
		# complete will also apply any mutations in params and build Fmapd
		# if negatives: assume the net file explicitly lists negatives and their functions
		# net_key is passed if want to use a different file than the default, such as the expanded one

		# will first try and build with an existing Net G
		# then try load from file using net_key
		# otherwise just makes an empty Net

		self.F= {} # logic function for each node
		self.A=None # adjacency matrix, unsigned
		self.Fmapd = None

		self.n = 0  # # regular nodes
		self.n_neg = 0 # curr # regular nodes + # negative nodes

		self.nodes = [] # object to hold name & number, redundant with nodeNames and nodeNums
		self.nodeNames = [] # num -> name
		self.nodeNums = {} # name -> num

		self.regularNodes = []
		self.negativeNodes = []

		self.encoding = None #encoding format used by the input file
		self.not_string = None # called often enough that it is useful to store
		
		self.num_clauses=0 # total number of clauses
		self.max_literals=0 # max literals per clause
		self.max_clauses=0 # max clauses per node

		self._complete = complete #i.e. including building Fmapd and A
		self._negatives = negatives #i.e. negatives included in original net file

		if G is not None: # don't really like this solution...
			self.__dict__ = G.__dict__
		elif net_key is not None:
			self._read_from_file(params, params[net_key])


	def __str__(self):
		# TODO: make this more complete
		return "Net:\nF =" + str(self.F)

	def _read_from_file(self, params, net_file):
		# inits network, except for F_mapd (which maps the logic to an actual matrix execution)
		
		# net file should be in DNF, see README for specifications

		if not os.path.isfile(net_file):
			sys.exit("Can't find network file: " + str(net_file)) 
		
		with open(net_file,'r') as file:
			format_name=self._get_encoding(file,net_file)
			node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(format_name)

			self.encoding = format_name
			self.not_string = not_str 

			loop = 0
			while True:
				line = self._read_line(file, loop)
				if not line: #i.e eof
					break 

				line = line.strip().split(node_fn_split)
				node_name = line[0].strip()
				for symbol in strip_from_node:
					node_name = node_name.replace(symbol,'')
				if params['debug']:
					assert(node_name not in self.nodeNames)
				
				assert(node_name not in ['0','1']) # these are reserved for constant off and on
				self.add_node(node_name)

				clauses = line[1].split(clause_split)
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
					
					self.nodes[-1].F += [this_clause]

				# if function is 0 or 1 set to a self-loop and an init call
				if self.F[node_name] in [[['0']],[['1']]]:
					if 'init' not in params.keys():
						params['init'] = {}
					params['init'][node_name] = int(self.F[node_name][0][0])
					self.reset_F_selfLoop(node_name)
				loop += 1
		
		if not self._negatives:
			self.build_negative_nodes()

		self.count_clauses_and_lits()

		if self._complete:
			self.apply_mutations(params)
			self.build_Fmapd_and_A(params)

	
	def add_node(self,nodeName,isNegative=False):
		newNode = Node(nodeName,self.n_neg,isNegative=isNegative)
		self.nodes += [newNode]
		self.nodeNames += [nodeName]
		self.nodeNums[nodeName] = self.n_neg

		if isNegative:
			self.negativeNodes += [newNode]
			self.n_neg += 1
			# note that F is not added, but would be in the case of the expanded net
		else:
			self.regularNodes += [newNode]
			self.F[nodeName] = newNode.F 
			self.n += 1
			self.n_neg += 1

	def build_negative_nodes(self):
		# put the negative nodes as 2nd half of node array
		for i in range(self.n):
			self.add_node(self.not_string + self.nodeNames[i],isNegative=True)

	def apply_mutations(self, params):
		if 'mutations' in params.keys():
			for node in self.nodeNames:
				if node in params['mutations']:
					lng = len(self.F[node])
					self.reset_F_selfLoop(node)
					params['init'][node] = int(params['mutations'][node])
					if params['debug']:
						assert(params['init'][node] in [0,1]) # for PBN may allow floats

	def reset_F(self,nodeName,new_fn):
		self.F[nodeName].clear() # cannot reassign since that would decouple node.F & self.F[node]
		self.F[nodeName] += new_fn	

	def reset_F_selfLoop(self,nodeName):
		# maintains lng of F
		lng = len(self.F[nodeName])	
		self.reset_F(nodeName,[[nodeName] for _ in range(lng)])

	def count_clauses_and_lits(self):
		for node in self.nodes:
			if not node.isNegative:
				self.num_clauses += len(self.F[node.name]) 
				self.max_clauses = max(self.max_clauses, len(self.F[node.name]))
				for clause in self.F[node.name]:
					self.max_literals = max(self.max_literals, len(clause))

	def build_Fmapd_and_A(self, params): 

		n = self.n 

		# building clauses_to_threads, i.e. the index for the function of each node
		#nodes_to_clauses = cp.zeros((self.num_clauses,self.max_literals),dtype=self._get_index_dtype(self.n)) # the literals in each clause
		nodes_to_clauses = [[] for i in range(self.num_clauses)]
		nodes_clause = {i:[] for i in range(n)}

		# ADJACENCY MATRIX, unsigned
		self.A = cp.zeros((n,n),dtype=bool)
		curr_clause = 0

		for i in range(n):
			clauses = self.nodes[i].F
			for j in range(len(clauses)):
				clause = clauses[j]
				clause_fn = []
				for k in range(len(clause)):
					literal_node = self.nodeNums[clause[k]]
					clause_fn += [literal_node]
					if self.nodes[literal_node].isNegative: 
						self.A[i, literal_node-n] = 1
					else:
						self.A[i, literal_node] = 1
				for k in range(len(clause), self.max_literals): # filling to make sq matrix
					clause_fn += [literal_node]

				nodes_to_clauses[curr_clause] = clause_fn
				nodes_clause[i] += [curr_clause]
				curr_clause += 1

		nodes_to_clauses = cp.array(nodes_to_clauses,dtype=self._get_index_dtype(self.n)) # the literals in each clause
		
		if self.num_clauses>0:
			assert(nodes_to_clauses.shape == (self.num_clauses,self.max_literals))

		# bluid clause index with which to compress the clauses -> nodes
		#	nodes_to_clauses: computes which nodes are inputs (literals) of which clauses
		#	clauses_to_threads: maps which clauses will be processed by which threads (and #threads = #nodes)
		#	threads_to_nodes: maps which threads are functions of which nodes
		clauses_to_threads = []
		threads_to_nodes = [] 
		m=min(params['clause_bin_size'],self.max_clauses) 

		i=0
		while sum([len(nodes_clause[i2]) for i2 in range(n)]) > 0:
			# ie exit when have mapped all clauses

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
						for k in range(rem):
							this_set[j] += [this_set[j][-1]] #use a copy of prev clause to make matrix square (assuming DNF)
						del nodes_clause[take_from_node][:top]
						node_indx += 1
				else: #finished, just need to filll up the array
					this_set[j] = this_set[j-1]

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
		threads_to_nodes = cp.array(threads_to_nodes,dtype=thread_dtype)
		# nodes_to_clauses already converted
		clause_mapping = {'nodes_to_clauses':nodes_to_clauses, 'clauses_to_threads':clauses_to_threads, 'threads_to_nodes':threads_to_nodes}
		
		#print('nodes->clauses\n',nodes_to_clauses,'\n\nclauses->threads\n',clauses_to_threads,'\n\nthreads->nodes\n',threads_to_nodes)
		self.Fmapd = clause_mapping


	def _get_index_dtype(self, max_n):
		if max_n<256: 
			return cp.uint8
		elif max_n<65536:
			return cp.uint16
		else:
			return cp.uint32

	def _read_line(self,file,loop):
		line = file.readline()
		if loop > 1000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 
		return line


	def _get_encoding(self,file,net_file):
		extension = net_file.split('.')
		if extension[-1] == 'bnet':
			return 'bnet'
		else:
			return file.readline().replace('\n','')





class Node:
	def __init__(self,name,num,isNegative=False):
		if not isNegative:
			self.F = [] #its clauses organized as [clause1,clause2,...] where clause=[lit1,lit2,...]
		self.name=name 
		self.num=num 
		self.isNegative = isNegative 


class Expanded_Net(Net):
	def __init__(self,params,G=None,net_key='model_expanded'):
		if G is None:
			build_from_file=True
			complete, negatives = False, False
		else:
			build_from_file = False
			complete, negatives = True, True

		super().__init__(params,net_key=net_key,G=G, complete=complete, negatives=negatives)
		# assumes that negative nodes are already in expanded form (i.e. have their logic)
		# note that composite nodes aren't formally added, since only A_exp is actually used

		if not build_from_file: # instead build from a regular net, so add composite nodes, ect

			self.n_exp = self.n_neg # will add this up during build_Aexp()
			# build F for negative nodes
			logic.DNF_via_QuineMcCluskey_nofile(params, G,expanded=True)
			self.build_Aexp(params)


	def build_Aexp(self,params):
		N = self.n_neg+self._num_and_clauses(self.F)
		self.A_exp = cp.zeros((N,N)) #adj for expanded net 

		for node in self.nodes:
			for clause in node.F:
				if len(clause)>1: # make a composite node
					self.A_exp[self.n_exp,node.num]=1
					for j in range(len(clause)):
						self.A_exp[self.nodeNums[clause[j]],self.n_exp]=1
					self.n_exp+=1
				else:
					self.A_exp[self.nodeNums[clause[0]],node.num]=1
		
		if params['debug']:
			assert(N==self.n_exp)

	def _num_and_clauses(self,F):
		count=0
		for node in F:
			for clause in F[node]:
				if len(clause)>1:
					count+=1
		return count



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

