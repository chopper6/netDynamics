# contructs Net objects that include graph structure and the associated logic functions

import os, sys, yaml, util, math, itertools
import util, logic, deep, PBN
from copy import deepcopy
import espresso

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# in particular Fmapd should be cupy (if cupy is being used instead of numpy)

# TODO: 
#   add explicit debug function for net construction
#       check that certain objs same, certain diff
#       ex G.F and node.F()
#   Gpar should have all parity nodes in G.nodes too..but worried will break other things
#       like with Gpar.n vs .n_neg problem is on the one had want to be able to Gpar in place of G without changing anything
#           on the other hand want Gpar to be a proper network where !x is a sep node altogether...

# TODO LATER:
#   node.F should be composed of other node object, rather than names jp
#   util: custom print that auto states the file/fn calling it
#   add time-reverse net (and inheritance with it)
#   Net._build_net() and build_Fmap() are still messy af
#   check for repeated clauses in build_Fmapd_and_A()


class Net:
    def __init__(self, params, model_file=None, G=None):
        # will construct from graph G if passed
        # otherwise will construct from the default model_file in params (unless otherwise specified)

        if model_file is None:
            model_file=params['model_file']

        self.F= {} # logic function for each node
        self.F_orig = {} # back up to restore after a mutation 
        self.A=None # adjacency matrix, unsigned
        self.Fmapd = None

        self.n = 0  # # regular nodes
        self.n_neg = 0 # num regular nodes + num negative nodes

        self.nodes = [] # object to hold name & number, only regular nodes included
        self.allNodes = [] #includes complement nodes too

        self.nodeNames = [] # num -> name
        self.nodeNums = {} # name -> num

        self.encoding = None # encoding format used by the input file
        self.not_string = None # from encoding, called often enough that it is useful to store
        
        self.num_clauses=0 # total number of clauses
        self.max_literals=0 # max literals per clause
        self.max_clauses=0 # max clauses per node

        self.debug = params['debug']

        if util.istrue(params,['PBN','active']):
            assert(not isinstance(self,ParityNet)) # haven't implemented PBN with ParityNet yet
            self.add_node('OFF')
            self.F['OFF'] = [['OFF']]
            self.PBN=True
        else:
            self.PBN = False

        # CONSTRUCTION FROM PREVIOUS NET
        if G is not None: 
            self.copy_from_net(G)

        # CONSTRUCTION FROM FILE
        elif model_file is not None:
            self.read_from_file(model_file)

        # APPLY MUTATIONS AND BUILD Fmapd
        self.prepare(params)


    def __str__(self):
        # TODO: make this more complete
        return "Net:\nF =" + str(self.F)

    def prepare(self,params): 
        # applies setting mutations and builds Fmapd
        self.params=params # since sometimes alter the params applied to same net here
        self.restore_from_prev_mutations()
        self.add_self_loops()
        self.check_mutations()
        self.apply_mutations()
        self.build_Fmapd_and_A()

    def copy_from_net(self,G):
        self.__dict__ = deepcopy(G.__dict__) # blasphemy to some

    def add_self_loops(self):
        for node in self.nodes:
            # if function is 0 or 1 set to a self-loop and an init call
            if self.F[node.name] in [[['0']],[['1']]]:
                self.params['init'][node.name] = int(self.F[node.name][0][0])
                self.F[node.name] = [[node.name]]

    def read_from_file(self, net_file):
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
                if self.debug:
                    assert(node_name not in self.nodeNames)

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
                    
                    self.F[node_name] += [this_clause]

                loop += 1
        
        self.build_negative_nodes() # this is just a pass function for Parity & DeepNet


    def write_to_file(self, output_file):

        with open(output_file,'w') as ofile:
            if self.encoding != 'bnet':
                ofile.write(self.encoding + '\n')
            else:
                pass # write it in case does not exist
            # note this overwrites
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(self.encoding)
        
        for node in self.nodes:
            if not node.isNegative:
                fn=node.F()
                with open(output_file,'a') as ofile:
                    ofile.write(node.name + node_fn_split + self._fn_to_str(fn) + '\n')


    def _fn_to_str(self,fn):
        # fn = F['nodeName'] = [clause1,clause2,...] where clause = [lit1,lit2,...] 
        # input_nodes are names

        # maybe should store all of these in net obj itself? ex G.format.not_str
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(self.encoding)
        fn_str = ''
        i=0
        for clause in fn:
            assert(len(clause)>0)
            if i!=0:
                fn_str += clause_split
            if len(strip_from_clause) > 0 and len(clause)>1:
                fn_str += strip_from_clause[0]
            j=0
            for lit in clause:
                if j!=0:
                    fn_str += literal_split
                fn_str += lit
                j+=1 
            if len(strip_from_clause) > 0 and len(clause)>1:
                fn_str += strip_from_clause[1]
            i+=1
        return fn_str

    
    def add_node(self,nodeName,isNegative=False):
        # note that F is not added for negatives, since simulation just negates their positive nodes directly

        newNode = Node(self,nodeName,self.n_neg,isNegative=isNegative)
        self.nodeNames += [nodeName]
        self.nodeNums[nodeName] = self.n_neg
        self.allNodes += [newNode]
        self.n_neg += 1
        
        if not isNegative:
            self.F[nodeName] = []
            self.nodes += [newNode]
            self.n+=1

    def nodesByName(self,name):
        return self.nodes[self.nodeNums[name]]

    def build_negative_nodes(self):
        # put the negative nodes as 2nd half of node array
        if not isinstance(self,ParityNet) and not isinstance(self,DeepNet):
            for i in range(self.n):
                self.add_node(self.not_string + self.nodeNames[i],isNegative=True)

    def check_mutations(self): 
        if 'mutations' in self.params.keys():
            for k in self.params['mutations'].keys():
                if k not in self.nodeNames:
                    sys.exit("\nSetting file specifies mutation on " + str(k) +", but this node is not in the network!\n") 

    def restore_from_prev_mutations(self):
        for node in self.nodeNames:
            if node in self.F_orig:
                self.F[node] = deepcopy(self.F_orig[node])

    def apply_mutations(self):
        if 'mutations' in self.params.keys() and len(self.params['mutations']) > 0:
            for node in self.nodeNames:
                if node in self.params['mutations']:
                    self.F_orig[node] = deepcopy(self.F[node])
                    lng = len(self.F[node]) 
                    # will be rendundant, but avoids possible issues with changing # of clauses
                    # otherwise need to rebuild Fmapd each time
                    self.F[node] = [[node] for _ in range(lng)]
                    self.params['init'][node] = int(self.params['mutations'][node])
                    if self.debug:
                        assert(self.params['init'][node] in [0,1]) # for PBN may allow floats


    def count_clauses_and_lits(self):
        assert(0) # can rm, just checking if this fn is still used
        assert(not self.PBN) # otherwise wrong
        for node in self.nodes:
            if not node.isNegative:
                self.num_clauses += len(node.F()) 
                self.max_clauses = max(self.max_clauses, len(node.F()))
                for clause in node.F():
                    self.max_literals = max(self.max_literals, len(clause))


    def build_Fmapd_and_A(self): 
        
        # Fmapd is a clause index with which to compress the clauses -> nodes
        #   nodes_to_clauses: computes which nodes are inputs (literals) of which clauses
        #   clauses_to_threads: maps which clauses will be processed by which threads (and #threads = #nodes)
        #   threads_to_nodes: maps which threads are functions of which nodes
        
        n = self.n

        nodes_to_clauses = []        
        clauses_to_threads = []
        threads_to_nodes = [] 
        nodes_clause = {i:[] for i in range(n)}
        if self.PBN:
            clauses_multiplier = []

        # ADJACENCY MATRIX, unsigned
        self.A = cp.zeros((n,n),dtype=bool)
        self.num_clauses = 0

        # BUILDING NODES->CLAUSES
        for i in range(n):
            numbered_clauses = [self._clause_num_and_add_to_A(i,c) for c in self.nodes[i].F()]
            if self.PBN and util.istrue(self.params,['PBN','float']):
                numbered_clauses, clauses_mult = PBN.get_node_float_clauses(numbered_clauses, n)
                self.max_literals = max([self.max_literals]+ [len(x) for x in numbered_clauses])
                clauses_multiplier += clauses_mult
            self.max_clauses = max(self.max_clauses, len(numbered_clauses))
            nodes_to_clauses += numbered_clauses
            nodes_clause[i] = [z for z in range(self.num_clauses, self.num_clauses+len(numbered_clauses))]
            self.num_clauses += len(numbered_clauses)

        self._make_square_clauses(nodes_to_clauses) 
        nodes_to_clauses = cp.array(nodes_to_clauses,dtype=self._get_index_dtype(self.n)) # the literals in each clause
        if self.PBN:
            clauses_multiplier = cp.array(clauses_multiplier,dtype=cp.int8) # the literals in each clause

        if self.num_clauses>0:
            assert(nodes_to_clauses.shape == (self.num_clauses,self.max_literals))

        # BUILDING CLAUSES->THREADS
        m=min(self.params['clause_bin_size'],self.max_clauses) 

        i=0
        while sum([len(nodes_clause[i2]) for i2 in range(n)]) > 0:
            # ie exit when have mapped all clauses

            # after each clauses_to_threads[i], need to add an index for where the new (compressed) clauses will go
            this_set=[[] for _ in range(n)]
            threads_to_nodes += [cp.zeros((n,n),dtype=bool)]
            sorted_keys = sorted(nodes_clause, key=lambda k: len(nodes_clause[k]), reverse=True)
            nodes_clause = {k:nodes_clause[k] for k in sorted_keys}
            node_indx = 0
            prev_take_from_node = take_from_node = sorted_keys[node_indx]

            for j in range(n):
                if sum([len(nodes_clause[i2]) for i2 in range(n)]) > 0:
                    take_from_node = sorted_keys[node_indx]
                    threads_to_nodes[i][j,take_from_node]=1 
                    if len(nodes_clause[take_from_node]) >= m:
                        this_set[j] = nodes_clause[take_from_node][:m]
                        if len(nodes_clause[take_from_node]) == m:
                            node_indx += 1
                        del nodes_clause[take_from_node][:m]
                    else:
                        top = len(nodes_clause[take_from_node])
                        this_set[j] = nodes_clause[take_from_node][:top] # why is [:top] nec?
                        rem = m-(top)
                        for k in range(rem):
                            if not self.PBN:
                                this_set[j] += [this_set[j][-1]] #use a copy of prev clause to make matrix square (assuming DNF)
                            else:
                                this_set[j] += [0] # add the OFF node instead (since sum in PBN)
                                # note that BOTH the 0th clause and 0th node are corresp to OFF node

                        del nodes_clause[take_from_node][:top]
                        node_indx += 1

                else: #finished, just need to filll up the array
                    if not PBN:
                        threads_to_nodes[i][j,prev_take_from_node]=1 
                        this_set[j] = this_set[j-1]
                    else:
                        # need to add onto the OFF node with its OFF clause
                        threads_to_nodes[i][j,0]=1 
                        this_set[j] = [0 for _ in range(m)]


                prev_take_from_node = take_from_node

            clauses_to_threads += [this_set]    
            i+=1
            if i>1000000:
                sys.exit("ERROR: infinite loop likely in net.build_Fmapd_and_A()")
        
        if self.params['parallelism']<254 and self.num_clauses<254:
            thread_dtype = cp.uint8
        elif self.params['parallelism']<65533 and self.num_clauses<65533:
            thread_dtype = cp.uint16
        else:
            thread_dtype = cp.uint32
        clauses_to_threads = cp.array(clauses_to_threads,dtype=thread_dtype)
        threads_to_nodes = cp.array(threads_to_nodes,dtype=bool)
        # nodes_to_clauses already converted
        clause_mapping = {'nodes_to_clauses':nodes_to_clauses, 'clauses_to_threads':clauses_to_threads, 'threads_to_nodes':threads_to_nodes}
        if self.PBN:
            clause_mapping['clauses_multiplier'] = clauses_multiplier
        #print('nodes->clauses\n',nodes_to_clauses,'\n\nclauses->threads\n',clauses_to_threads,'\n\nthreads->nodes\n',threads_to_nodes)
        self.Fmapd = clause_mapping

    def _clause_num_and_add_to_A(self, source_node_num, clause):            
        clause_fn = []
        for k in range(len(clause)):
            self.max_literals = max(self.max_literals, len(clause))
            literal_node = self.nodeNums[clause[k]]
            clause_fn += [literal_node]
            if self.allNodes[literal_node].isNegative: 
                self.A[source_node_num, literal_node-self.n] = 1
            else:
                self.A[source_node_num, literal_node] = 1
        #for k in range(len(clause), self.max_literals): # filling to make sq matrix
        #    clause_fn += [literal_node]
        return clause_fn

    def _make_square_clauses(self, nodes_to_clauses):
        for clause in nodes_to_clauses:
            for k in range( len(clause), self.max_literals ):
                if not self.PBN:
                    clause += [clause[-1]]
                else:
                    clause += [self.n] # this is the '1' node

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


    def input_indices(self):
        return [self.nodeNums[self.params['inputs'][i]] for i in range(len(self.params['inputs']))]
    
    def output_indices(self):
        return [self.nodeNums[self.params['outputs'][i]] for i in range(len(self.params['outputs']))]
    
    def mutant_indices(self):
        return [self.nodeNums[k] for k in self.params['mutations'].keys()]

    def get_input_sets(self):
        # assumes that 2^#inputs can fit in memory
        #input_indices = self.input_indices(params)
        return list(itertools.product([0,1],repeat=len(self.params['inputs'])))

    def print_matrix_names(self,X):
        # assumes X is a set of network states
        print('\nnetwork state:')
        for x in X: # one network state
            s=''
            for i in range(len(x)):
                s+=self.nodeNames[i] + ":" + str(x[i]) + " "
            print(s)
        print('\n')

    def node_vector_to_names(self,v):
        # where v is a list of node indices
        names = []
        for i in v:
            names+=[self.nodeNames[i]]
        return names

    def certain_nodes_to_names(self,s):
        # where s is a list of node states
        names = []
        for i in range(len(s)):
            if s[i] != 2:
                assert(s[i] in [0,1])
                names += [self.nodeNames[i]+'='+str(int(s[i]))]
        return names


##################################################################################################


class Node:
    def __init__(self,G,name,num,isNegative=False):
        self.name=name 
        self.num=num 
        self.isNegative = isNegative 
        self.G=G

    def F(self):
        #if self.name not in self.G.F:
        #   print("\nERROR: node ",self.name," not assigned a function in Net")
        #   sys.exit()
        return self.G.F[self.name]  # will raise error if node name not assigned in Net's F

    def setF(self,new_fn):
        self.G.F[self.name] = new_fn



##################################################################################################

class ParityNet(Net):
    def __init__(self,params, parity_model_file=None):
        
        if parity_model_file is None:
            parity_model_file=params['parity_model_file'] # i.e. default

        self.parityNodes = [] # these include all regular nodes and their complements
        self.debug = params['debug']
        super().__init__(params, model_file=parity_model_file)
        # assumes that negative nodes are already in parity form (i.e. have their logic)
        # note that composite nodes aren't formally added, since only A_exp is actually used

        self.build_Aexp()


    def add_node(self,nodeName,isNegative=False):
        if self.not_string in nodeName:
            isNegative=True
        newNode = Node(self,nodeName,self.n_neg,isNegative=isNegative)
        self.parityNodes += [newNode]
        self.nodeNames += [nodeName]
        self.nodeNums[nodeName] = self.n_neg
        self.n_neg += 1
        
        self.F[nodeName] = []
        self.allNodes += [newNode]

        if not isNegative:
            self.nodes += [newNode]
            self.n += 1

        if self.debug:
            if self.not_string in nodeName:
                positive_name = nodeName.replace(self.not_string,'')
                assert(self.nodesByName(nodeName).num - self.n == self.nodesByName(positive_name).num)
                # for example, LDOI relies on this precise ordering

    def build_Aexp(self):
        self.n_exp = self.n_neg # build_Aexp will iterate this
        N = self.n_neg+self._num_and_clauses()
        self.A_exp = cp.zeros((N,N)) #adj for expanded net 
        composites = []

        for node in self.parityNodes:
            if self.debug:
                assert(node.F() == self.F[node.name]) 

            for clause in node.F():
                if len(clause)>1: # make a composite node
                    self.A_exp[self.n_exp,node.num]=1
                    for j in range(len(clause)):
                        self.A_exp[self.nodeNums[clause[j]],self.n_exp]=1
                    self.n_exp+=1
                elif clause not in ['0','1']: # ignore tautologies
                    # wouldn't tautologies be [['0'],['1']] instead??
                    self.A_exp[self.nodeNums[clause[0]],node.num]=1
                else:
                    print("net.py build_Aexp(): tautology ignored")
        
        if self.debug:
            assert(N==self.n_exp)
    

    def nodesByName(self,name):
        return self.parityNodes[self.nodeNums[name]]


    def _num_and_clauses(self):
        count=0
        for node in self.parityNodes:
            for clause in node.F():
                if len(clause)>1: 
                    count+=1
        return count

    def count_clauses_and_lits(self):
        for node in self.parityNodes:
            self.num_clauses += len(node.F()) 
            self.max_clauses = max(self.max_clauses, len(node.F()))
            for clause in node.F():
                self.max_literals = max(self.max_literals, len(clause))

    def input_names_from_vector(self,v):
        inpt_ind = self.input_indices()
        #print(inpt_ind,v)
        names = []
        for i in range(len(v)):
            if v[i]==0:
                names+=[self.nodeNames[inpt_ind[i]+self.n]]
            else:
                assert(v[i]==1)
                names+=[self.nodeNames[inpt_ind[i]]]
        return names

    def write_to_file(self, output_file):
        with open(output_file,'w') as ofile:
            if self.encoding != 'bnet':
                ofile.write(self.encoding + '\n')
            else:
                pass # write it in case does not exist
            # note this overwrites
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(self.encoding)
        
        for node in self.parityNodes: # pretty sure this will throw an err on regular net...
            fn=node.F()
            with open(output_file,'a') as ofile:
                ofile.write(node.name + node_fn_split + self._fn_to_str(fn) + '\n')


##################################################################################################

class DeepNet(Net):
    def __init__(self,params, parity_model_file):

        assert(0) # have not updated this in too long, use with caution

        if parity_model_file is None:
            parity_model_file=params['parity_model_file'] # i.e. default

        self.complement = {}
        self.debug=params['debug']

        super().__init__(params, model_file=parity_model_file)
        # assumes that negative nodes are already in parity form (i.e. have their logic)
        # note that composite nodes aren't formally added, since only A_exp is actually used

        #self.build_Aexp(debug=debug)


    def add_node(self,nodeName):
        newNode = Node(self,nodeName,self.n)
        self.nodes += [newNode]
        self.nodeNames += [nodeName]
        self.nodeNums[nodeName] = self.n
        self.n += 1
        self.F[nodeName] = []

        compl = espresso.negate_ele(nodeName, self)
        if 'ErbB2_3+!Akt1&ErbB2_3+!ERa' in nodeName:
            print('net: adding node',nodeName,'and compl',compl)
        self.complement[nodeName] = compl 
        self.complement[compl] = nodeName


    def reorder(self):
        # sorts nodes such that 2nd half are the complements of the 1st half
        visited = [0 for i in range(self.n)]
        new_nodes = []

        for i in range(self.n):
            node = self.nodes[i]
            if self.debug:
                assert(node.F() != [])
            compl = self.get_complement(node)
            if compl.num > i:
                node.num = len(new_nodes)
                new_nodes += [node]
        half_n = len(new_nodes)
        if self.debug:
            assert(half_n == int(self.n/2))
        for i in range(len(new_nodes)):
            compl = self.get_complement(new_nodes[i])
            if self.debug:
                assert(compl.F() != [])
            compl.num = len(new_nodes)
            new_nodes += [compl]

        self.nodes = new_nodes 
        self.nodeNames = [node.name for node in self.nodes] 
        self.nodeNums = {node.name:node.num for node in self.nodes} 

    def build_Aexp(self):
        self.reorder()

        self.n_exp = self.n 
        N = self.n+self._num_and_clauses()

        print("\nnet.build_Aexp, n=",self.n,"n_exp=",N,"\n")

        self.A_exp = cp.zeros((N,N)) #adj for expanded net 

        expanded_node_map = {}
        for node in self.nodes:
            if self.debug:
                assert(node.F() == self.F[node.name]) 
            for clause in node.F():
                if len(clause)>1: 
                    if str(clause) in expanded_node_map: 
                        exp_num = expanded_node_map[str(clause)]
                        self.A_exp[exp_num,node.num]=1
                    else: # make a new expanded node
                        self.A_exp[self.n_exp,node.num]=1
                        expanded_node_map[str(clause)] = self.n_exp
                        for j in range(len(clause)):
                            self.A_exp[self.nodeNums[clause[j]],self.n_exp]=1
                        self.n_exp+=1
                elif clause not in ['0','1',['0'],['1']]: # ignore tautologies
                    self.A_exp[self.nodeNums[clause[0]],node.num]=1
        
        if self.debug:
            assert(N==self.n_exp)
    
    def _num_and_clauses(self):
        count=0
        seen = []
        for node in self.nodes:
            for clause in node.F():
                if len(clause)>1: 
                    if str(clause) not in seen:
                        count+=1
                        seen+=[str(clause)]
        return count


    def get_complement(self, node):
        # returns Node objects, whereas self.complements[name] returns string of the name
        return self.nodesByName(self.complement[node.name])


##################################################################################################

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

