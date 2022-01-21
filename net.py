import os, sys, yaml, util, math
import util, logic, deep
from copy import deepcopy
import espresso

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# in particular Fmapd should be cupy (if cupy is being used instead of numpy)

# TODO: 
# should also be fine to start w a regular net..right?
# debug ParityNet & DeepNet
#   G.nodes() should be the default, not AllNodes!
#       use G.parityNodes() or something to distinguish
#           --> need to check all over the place where allNodes may have been used
#   check that ParityNet name (vs Parity_Net) is fixed elsewhere

# TODO 2:
#   add explicit debug function for net construction
#       check that certain objs same, certain diff
#   debug changes to G.F and node.F()
#   passing net file sep is awk af

# TODO LATER:
#   add time-reverse net (and inheritance with it)
#       then later higher order Gexp's
#   Net._build_net() and build_Fmap() are still messy af
#   function that updates expanded net due to input or mutation holding it static? ldoi cover that case anyway tho



class Net:
    def __init__(self, model_file=None, G=None, debug=False):
        # complete will also apply any mutations in params and build Fmapd [TODO rename complete]
        
        # use a model_file OR an existing net
        assert(model_file is not None or G is not None)
        assert(model_file is None or G is None)

        self.F= {} # logic function for each node
        self.A=None # adjacency matrix, unsigned
        self.Fmapd = None

        self.n = 0  # # regular nodes
        self.n_neg = 0 # curr num regular nodes + num negative nodes

        self.nodes = [] # object to hold name & number, only regular nodes included
        self.allNodes = [] #includes complement nodes too

        self.nodeNames = [] # num -> name
        self.nodeNums = {} # name -> num

        self.encoding = None # encoding format used by the input file
        self.not_string = None # from encoding, called often enough that it is useful to store
        
        self.num_clauses=0 # total number of clauses
        self.max_literals=0 # max literals per clause
        self.max_clauses=0 # max clauses per node

        if G is not None: 
            self.copy_from_net(G)
        elif model_file is not None:
            self.read_from_file(model_file, debug=debug)


    def __str__(self):
        # TODO: make this more complete
        return "Net:\nF =" + str(self.F)

    def prepare_for_sim(self,params): 
        # applies setting mutations and builds Fmapd
        self.add_self_loops(params)
        self.apply_mutations(params,debug=params['debug'])
        self.build_Fmapd_and_A(params)

    def copy_from_net(self,G):
        self.__dict__ = deepcopy(G.__dict__) # blasphemy to some

    def add_self_loops(self,params):
        for node in self.nodes:
            # if function is 0 or 1 set to a self-loop and an init call
            if self.F[node.name] in [[['0']],[['1']]]:
                params['init'][node.name] = int(self.F[node.name][0][0])
                self.F[node.name] = [[node.name]]

    def read_from_file(self, net_file, debug=False):
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
                if debug:
                    assert(node_name not in self.nodeNames)

                self.add_node(node_name,debug=debug)

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
        
        self.build_negative_nodes(debug=debug) # this is just a pass function for Parity & DeepNet

        self.count_clauses_and_lits()


    def write_to_file(self, output_file,parity=False):
        # parity is an extra switch to treat the network like it is a Parity Net
        #   for ex when building a parity net from a regular net

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

    
    def add_node(self,nodeName,isNegative=False,debug=False):
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

    def build_negative_nodes(self,debug=False):
        # put the negative nodes as 2nd half of node array
        for i in range(self.n):
            self.add_node(self.not_string + self.nodeNames[i],isNegative=True,debug=debug)


    def apply_mutations(self,params, debug=False):
        if 'mutations' in params.keys() and len(params['mutations']) > 0:
            for node in self.nodeNames:
                if node in params['mutations']:
                    lng = len(self.F[node]) 
                    # will be rendundant, but avoids possible issues with changing # of clauses
                    # otherwise need to rebuild Fmapd each time
                    self.F[node] = [[node] for _ in range(lng)]
                    params['init'][node] = int(params['mutations'][node])
                    if debug:
                        assert(params['init'][node] in [0,1]) # for PBN may allow floats


    def count_clauses_and_lits(self):
        for node in self.nodes:
            if not node.isNegative:
                self.num_clauses += len(node.F()) 
                self.max_clauses = max(self.max_clauses, len(node.F()))
                for clause in node.F():
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
            clauses = self.nodes[i].F()
            for j in range(len(clauses)):
                clause = clauses[j]
                clause_fn = []
                for k in range(len(clause)):
                    literal_node = self.nodeNums[clause[k]]
                    clause_fn += [literal_node]
                    if self.allNodes[literal_node].isNegative: 
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
        #   nodes_to_clauses: computes which nodes are inputs (literals) of which clauses
        #   clauses_to_threads: maps which clauses will be processed by which threads (and #threads = #nodes)
        #   threads_to_nodes: maps which threads are functions of which nodes
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
                sys.exit("ERROR: infinite loop likely in net.build_Fmapd_and_A()")
        
        if params['parallelism']<256:
            thread_dtype = cp.uint8
        elif params['parallelism']<65535:
            thread_dtype = cp.uint16
        else:
            thread_dtype = cp.uint32
        clauses_to_threads = cp.array(clauses_to_threads,dtype=thread_dtype)
        threads_to_nodes = cp.array(threads_to_nodes,dtype=bool)
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
    def __init__(self,parity_model_file,debug=False):

        self.parityNodes = [] # these include all regular nodes and their complements
        super().__init__(model_file=parity_model_file,debug=debug)
        # assumes that negative nodes are already in parity form (i.e. have their logic)
        # note that composite nodes aren't formally added, since only A_exp is actually used

        self.build_Aexp(debug=debug)


    def add_node(self,nodeName,isNegative=False,debug=False):
        if self.not_string in nodeName:
            isNegative=True
        newNode = Node(self,nodeName,self.n_neg,isNegative=isNegative)
        self.parityNodes += [newNode]
        self.nodeNames += [nodeName]
        self.nodeNums[nodeName] = self.n_neg
        self.n_neg += 1
        
        self.F[nodeName] = []

        if not isNegative:
            self.nodes += [newNode]
            self.n += 1

        if debug:
            if self.not_string in nodeName:
                positive_name = nodeName.replace(self.not_string,'')
                assert(self.nodesByName(nodeName).num - self.n == self.nodesByName(positive_name).num)
                # for example, LDOI relies on this precise ordering

    def build_Aexp(self,debug=False):
        self.n_exp = self.n_neg # build_Aexp will iterate this
        N = self.n_neg+self._num_and_clauses()
        self.A_exp = cp.zeros((N,N)) #adj for expanded net 
        composites = []

        for node in self.parityNodes:
            assert(node.F() == self.F[node.name]) # TODO rm this safety check (or only w debug)
            for clause in node.F():
                if len(clause)>1: # make a composite node
                    self.A_exp[self.n_exp,node.num]=1
                    for j in range(len(clause)):
                        self.A_exp[self.nodeNums[clause[j]],self.n_exp]=1
                    self.n_exp+=1
                elif clause not in ['0','1']: # ignore tautologies
                    self.A_exp[self.nodeNums[clause[0]],node.num]=1
        
        if debug:
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

    def build_negative_nodes(self, debug=False):
        # just to make compatible with default net def
        pass 

    def write_to_file(self, output_file,parity=False):
        # parity is an extra switch to treat the network like it is a Parity Net
        #   for ex when building a parity net from a regular net

        with open(output_file,'w') as ofile:
            if self.encoding != 'bnet':
                ofile.write(self.encoding + '\n')
            else:
                pass # write it in case does not exist
            # note this overwrites
        node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = get_file_format(self.encoding)
        
        for node in self.parityNodes:
            fn=node.F()
            with open(output_file,'a') as ofile:
                ofile.write(node.name + node_fn_split + self._fn_to_str(fn) + '\n')


##################################################################################################

class DeepNet(Net):
    def __init__(self,parity_model_file,debug=False):
        self.complement = {}

        super().__init__(model_file=parity_model_file,debug=debug)
        # assumes that negative nodes are already in parity form (i.e. have their logic)
        # note that composite nodes aren't formally added, since only A_exp is actually used

        #self.build_Aexp(debug=debug)


    def add_node(self,nodeName,debug=False):
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


    def reorder(self,debug=True):
        # sorts nodes such that 2nd half are the complements of the 1st half
        visited = [0 for i in range(self.n)]
        new_nodes = []

        for i in range(self.n):
            node = self.nodes[i]
            if debug:
                assert(node.F() != [])
            compl = self.get_complement(node)
            if compl.num > i:
                node.num = len(new_nodes)
                new_nodes += [node]
        half_n = len(new_nodes)
        if debug:
            assert(half_n == int(self.n/2))
        for i in range(len(new_nodes)):
            compl = self.get_complement(new_nodes[i])
            if debug:
                assert(compl.F() != [])
            compl.num = len(new_nodes)
            new_nodes += [compl]

        self.nodes = new_nodes 
        self.nodeNames = [node.name for node in self.nodes] 
        self.nodeNums = {node.name:node.num for node in self.nodes} 

    def build_Aexp(self,debug=False):
        self.reorder()

        self.n_exp = self.n 
        N = self.n+self._num_and_clauses()

        print("\nnet.build_Aexp, n=",self.n,"n_exp=",N,"\n")

        self.A_exp = cp.zeros((N,N)) #adj for expanded net 

        expanded_node_map = {}
        for node in self.nodes:
            if debug:
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
                #else: JUST DEBUGGING STUFF
                #    num = node.num 
                #    if clause == ['0']:
                #       node.num = (node.num + int(self.n/2)) % self.n
                #    print("net.build_Aexp of deepNet: tautology found for",node.name,": ",clause)
                #    print("\tcorresp node function:",node.F())
        
        if debug:
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

    def build_negative_nodes(self,debug=False):
        # just to make compatible with default net def
        pass 

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

