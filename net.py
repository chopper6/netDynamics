import os, sys, yaml, util, math
import util, logic, deep
from copy import deepcopy

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# in particular Fmapd should be cupy (if cupy is being used instead of numpy)

# TODO:
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
    def __init__(self, model_file=None, G=None, parity=False, debug=False):
        # complete will also apply any mutations in params and build Fmapd [TODO rename complete]
        # if parity: assume the net file explicitly lists negatives and their functions
        
        # use a model_file OR an existing net
        assert(model_file is not None or G is not None)
        assert(model_file is None or G is None)

        self.F= {} # logic function for each node
        self.A=None # adjacency matrix, unsigned
        self.Fmapd = None

        self.n = 0  # # regular nodes
        self.n_neg = 0 # curr # regular nodes + # negative nodes

        self.nodes = [] # object to hold name & number, only regular nodes included
        self.allNodes = [] # includes both regular and parity nodes

        self.nodeNames = [] # num -> name
        self.nodeNums = {} # name -> num

        self.encoding = None #encoding format used by the input file
        self.not_string = None # called often enough that it is useful to store
        
        self.num_clauses=0 # total number of clauses
        self.max_literals=0 # max literals per clause
        self.max_clauses=0 # max clauses per node

        if G is not None: 
            self.copy_from_net(G)
        elif model_file is not None:
            self.read_from_file(model_file, debug=debug,parity=parity)


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

    def read_from_file(self, net_file, debug=False, parity=False):
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

                self.add_node(node_name,debug=debug,parity=parity)

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
        
        if not parity:
            self.build_negative_nodes(debug=debug)

        self.count_clauses_and_lits(parity=parity)


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
        
        for node in self.allNodes:
            if not node.isNegative or isinstance(self,Parity_Net) or parity:
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

    
    def add_node(self,nodeName,isNegative=False,debug=False,parity=False,deep=False):
        # note that F is not added for negative, unless a parity net
        # TODO: have sep flags for isNegative and parity is confusing
        #                and now deep too..this is becoming a shitshow
        if parity and self.not_string in nodeName:
            isNegative=True

        newNode = Node(self,nodeName,self.n_neg,isNegative=isNegative)
        self.allNodes += [newNode]
        self.nodeNames += [nodeName]
        if not deep:
            self.nodeNums[nodeName] = self.n_neg
            self.n_neg += 1
        else:
            self.nodeNums[nodeName] = self.n_neg + self.n_deep
            self.n_deep += 1
        
        if not isNegative or parity or deep:
            self.F[nodeName] = []
        if not isNegative:
            self.nodes += [newNode]
            if not deep:
                self.n += 1

        if parity and debug and not deep:
            if self.not_string in nodeName:
                positive_name = nodeName.replace(self.not_string,'')
                assert(self.nodesByName(nodeName).num - self.n == self.nodesByName(positive_name).num)
                # for example, LDOI relies on this precise ordering

    def rm_node(self,node,resort=True):
        assert(0) # for now doesn't work since self.A cannot be resized if using cupy
        # TODO: this is messy...and likely leaves some gaps
        # note that this does NOT update Fmapd (since may require rebuilding it entirely)
        self.nodeNames.remove(node.name)
        del self.nodeNums[node.name] 
        self.A = cp.delete(self.A, (node.num), axis=0)
        self.A = cp.delete(self.A, (node.num), axis=1)
        if node.name in self.F:
            del self.F[node.name]

        self.nodes.remove(node)
        self.allNodes.remove(node)
        self.n -=1
        if self.n_neg > self.n: # ehh kinda sloppy
            self.n_neg -=2

        if resort: # complexity of this is not great...
            for node2 in G.nodes(): 
                if node2.num > node.num:
                    node2.num -= 1
    
        if isinstance(self,Parity_Net):
            self.n_exp -=2
            self.A_exp = cp.delete(self.A_exp, (node.num),axis=0)
            self.A_exp = cp.delete(self.A_exp, (node.num+self.n),axis=0) # i.e. also rm the complement
            self.A_exp = cp.delete(self.A_exp, (node.num),axis=1)
            self.A_exp = cp.delete(self.A_exp, (node.num+self.n),axis=1) 


    def nodesByName(self,name):
        return self.allNodes[self.nodeNums[name]]

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


    def count_clauses_and_lits(self, parity=False):
        for node in self.allNodes:
            if not node.isNegative or parity or isinstance(self,Parity_Net):
                self.num_clauses += len(node.F()) 
                self.max_clauses = max(self.max_clauses, len(node.F()))
                for clause in node.F():
                    self.max_literals = max(self.max_literals, len(clause))

    def count_HO_clauses(self):
        # temp function, can remove
        HO = {}
        max_indeg=0
        max_clauses=0
        used_pairs = []
        for node in self.nodes:
            i=node.num
            max_indeg = max(max_indeg,cp.sum(self.A[i]))
            for j in range(len(self.A[i])):
                if self.A[i,j]==1:
                    for k in range(len(self.A[i])):
                        if self.A[i,k]==1 and [j,k] not in used_pairs:
                            max_clauses += 2 # mult or add
                            used_pairs += [[j,k]]
            for clause in node.F():
                cl = clause.copy()
                cl.sort()
                if len(clause) not in HO.keys():
                    HO[len(clause)]=[cl] 
                elif clause.sort() not in HO[len(clause)]:
                    HO[len(clause)]+=[cl]
                else:
                    print(cl,' IS IN: ',HO[len(clause)])

        print("total # nodes = ",self.n)
        print("max indeg = ",max_indeg, '\tmax clauses =',max_clauses)
        for k in HO:
            print("# order",k,"virtual nodes=",2*len(HO[k])) # incld complement



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
            clauses = self.allNodes[i].F()
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


class Parity_Net(Net):
    def __init__(self,parity_model_file,debug=False,deep=False):

        super().__init__(model_file=parity_model_file,debug=debug,parity=True)
        # assumes that negative nodes are already in parity form (i.e. have their logic)
        # note that composite nodes aren't formally added, since only A_exp is actually used

        if deep:
            self.n_deep = 0 
        else:
            self.build_Aexp(debug=debug)



    def build_Aexp(self,debug=False,deep=False):
        self.n_exp = self.n_neg # build_Aexp will iterate this
        N = self.n_neg+self._num_and_clauses(self.F,deep=deep)
        if deep:
            self.n_exp += self.n_deep
            N += self.n_deep
        self.A_exp = cp.zeros((N,N)) #adj for expanded net 
        composites = []

        for node in self.allNodes:
            for clause in node.F():
                if len(clause)>1: # make a composite node
                    make_composite=True
                    if deep:
                        cName, negName = deep.get_composite_name(clause)
                        if cName in self.nodeNames:
                            assert(0) # don't think this should happen...
                            self.A_exp[self.nodeNums[cName],node.num]=1
                            make_composite=False

                        # TODO: this shouldn't just be for deep!
                        if cName in composites:
                            assert(0) # TODO: if this happens, need to save the composite # and use as index now
                            make_composite=False
                        else:
                            composites += [cName]
                    if make_composite:
                        self.A_exp[self.n_exp,node.num]=1
                        for j in range(len(clause)):
                            self.A_exp[self.nodeNums[clause[j]],self.n_exp]=1
                        self.n_exp+=1
                elif clause not in ['0','1']: # ignore tautologies
                    self.A_exp[self.nodeNums[clause[0]],node.num]=1
        
        if debug:
            print(N, self.n_exp,self.n_deep,self.n_neg,'\n\n\n')
            # N + n_neg = n_exp --> actual number composites added is n_neg less than expected... wtf
            assert(N==self.n_exp)

    def _num_and_clauses(self,F,deep=False):
        count=0
        for node in self.allNodes:
            for clause in F[node]:
                if len(clause)>1:
                    if not deep: 
                        count+=1
                    else:
                        cName, negName = deep.get_composite_name(clause)
                        if cName not in self.nodeNames:
                            count+=1
                        else:
                            assert(False) # again i don't think this should occur!
        return count



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

