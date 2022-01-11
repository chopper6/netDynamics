import sys, pickle
import ldoi, logic, param, net
import espresso # this file uses pyeda, and so will require linux

# BUG: calling LDOI at end 
#   not same as directly building net and running LDOI, so something about parity net construction
#       which should be a deep net anyhow
#   not symmetric when it should be...

# TODO: add DeepNet to net.py and clean that file in general
#       calc OR nodes too

def build_deep(G,kmax,output_file,minimizer='espresso',debug=True):
    # G should be a parity net (see Net.py)
    # kmax is the highest order term to expand to

    #if minimizer == 'espresso':
    #   import espresso # this is imported here since espresso does not compile on windows
    #   # i.e. to allow windows to use other parts of this program

    k=2
    while k<=kmax:
        added = True
        while added:
            added=False
            nodes_to_add = {} # will add these nodes (which are products) and their complements (which are sums)
            for V in G.nodes:
                for i in range(len(G.F[V.name])):
                    clause = G.F[V.name][i] 
                    if len(clause) > 1 and len(clause) <= k:
                        if True: #parent_clauses_overlap(G,clause):
                            cName, negName = get_composite_name(clause,G.not_string)
                            if cName not in G.nodeNames and cName not in nodes_to_add:
                                print('\nreducing',cName,'vs',negName,'using',clause)
                                ON_fn, OFF_fn = calc_deep_fn(G,clause,minimizer=minimizer)
                                nodes_to_add[cName] = ON_fn
                                nodes_to_add[negName] = OFF_fn
                                added=True
                            
                            complement = get_complement_name(V.name, G.not_string)#can rm
                            print('in addition to replacing term in',V.name,'factoring its complement',get_complement_name(cName, G.not_string))
                            print('prev function=',G.F[complement])
                            G.F[complement] = complement_factor(G.F[V.name],clause,cName,G.not_string)
                            print('factored function=',G.F[complement])
                            # just assuming espresso now
                            G.F[V.name][i] = [cName] # TODO check if leaving as mult is better for LDOI

            for name in nodes_to_add:
                G.add_node(name,debug=False,deep=True)  
                G.F[name] = nodes_to_add[name]
                # poss indicate as sep type of node? 
                # jp don't care about isNegative for LDOI, but check
            print("completed a pass with k=",k,", and",len(nodes_to_add)," new virtual nodes.")
        k += 1

    G.build_Aexp(debug=debug)
    if output_file is not None:
        with open(output_file,'wb') as f:
            pickle.dump([G, params],f)
        #G.write_to_file(output_file,parity=True)

    return G


def complement_factor(fn,clause, composite_name, not_str):
    partial_fn = []
    for clause2 in fn:
        if clause2 != clause:
            partial_fn += [clause2]
    complement = get_complement_name(composite_name, not_str)
    if partial_fn!=[]:
        fn = espresso.reduce_complement(partial_fn, not_str,complement)
    return fn

def parent_clauses_overlap(G, clause):
    # TODO might've fucked this up
    assert(0)
    for i in range(len(clause)):
        ele1 = clause[i]
        for clause1 in G.F[ele1]:
            for a in clause1:
                for j in range(i,len(clause)):
                    ele2 = clause[j]
                    if ele2 == a:
                        return True
                    for clause2 in G.F[ele2]: 
                        for b in clause2:
                            if a==b or ele1==b:
                                return True
    return False


def get_complement_name(name, not_str):
    assert('&' not in name or '+' not in name)
    print('in get_complement_name(): starting with',name)
    if '&' in name:
        name = name.split('&')
        for i in range(len(name)):
            part=name[i]
            if not_str in part:
                name[i] = part.replace(not_str,'')
            else:
                name[i] = not_str + part 
        name ='+'.join(name)
    elif '+' in name:
        name = name.split('+')
        for i in range(len(name)):
            part=name[i]
            if not_str in part:
                name[i] = part.replace(not_str,'')
            else:
                name[i] = not_str + part 
        name = '&'.join(name)
    else:
        if not_str in name:
            name = name.replace(not_str, '')
        else:
            name = not_str + name
    print('\tcomplement=',name,'\n')
    return name

def get_composite_name(clause, not_str):
    s,snot = '',''
    i=0
    for ele in clause:
        if i!=0:
            s+='&'
            snot+='+'
        s += str(ele)
        snot += not_str + str(ele)
        i+=1
    return s,snot


def calc_deep_fn(G,clause,minimizer='espresso',complement=True):
    # returns builds and reduces higher order function for the clause
    # uses either Quine-McCluskey or Espresso
    # Off fn may be v. slow since is the sum term
    if minimizer == 'espresso':
        import espresso # imported here since espresso does not compile on windows (so someone using windows uses the 'qm' minimizer instead)
        
        fns, varbs = [],[]
        for ele in clause:
            varbs += [ele]
            fns += [G.F[ele]]
        if complement:
            ON_fn, OFF_fn = espresso.reduce_async_AND_espresso(fns, varbs, G.not_string, complement=complement)
        else:
            ON_fn = espresso.reduce_async_AND_espresso(fns, varbs, G.not_string, complement=complement)
    elif minimizer in ['qm','QM']:
        # note that this can also calc complement
        #print("\tCalculating deep function of ",clause)
        inputs_num2name, inputs_name2num = organize_inputs(G, clause)
        ON_clauses, OFF_clauses = on_off_terms(G,clause,inputs_name2num)
        #print('\tdetails:',inputs_num2name,ON_clauses, OFF_clauses)
        ON_fn = logic.run_qm(ON_clauses, len(inputs_name2num), G.encoding, inputs_num2name)
        if complement:
            OFF_fn = logic.run_qm(OFF_clauses, len(inputs_name2num), G.encoding, inputs_num2name)
    else:
        assert(0) # unrecognized argument for 'minimizer'
    if complement:
        return ON_fn, OFF_fn 
    else:
        return ON_fn


def organize_inputs(G, clause):
    inputs_name2num = {}    # name -> num
    inputs_num2name = {}  # num -> name

    for ele in clause:
        eleName = ele.replace(G.not_string,'')
        if eleName not in inputs_name2num:
            inputs_num2name[len(inputs_name2num)] = eleName
            inputs_name2num[eleName] = len(inputs_name2num)

    for ele in clause:
        logic.add_to_input_nodes(G.F[ele], G.not_string, inputs_name2num, inputs_num2name)

    return inputs_num2name, inputs_name2num

def on_off_terms(G,clause,inputs_name2num):
    # jp input nodes need to be inputs of any of the ele's
    # then each cl of ele * all eles except that one, need to make sure don't cancel
    # poss that some inputs will then be irrelv

    num_inputs = len(inputs_name2num)
    unreduced_clauses = []

    all_str = ['x' for i in range(num_inputs)]
    for ele in clause:
        eleName = ele.replace(G.not_string,'')
        if G.not_string in ele:
            all_str[inputs_name2num[eleName]] = '0'
        else:
            all_str[inputs_name2num[eleName]] = '1'

    for ele in clause:
        for cl in G.F[ele]:     
            expanded_clause = all_str.copy()
            skip=False
            for lit in cl: 
                literal_name = str(lit)
                if G.not_string in literal_name:
                    sign = '0'
                    literal_name = literal_name.replace(G.not_string,'')
                else:
                    sign = '1'
                if expanded_clause[inputs_name2num[literal_name]] != 'x' and expanded_clause[inputs_name2num[literal_name]] != sign:
                    skip=True
                    break #contradicts all_str, so ignore this clause
                expanded_clause[inputs_name2num[literal_name]]=sign
            
            if not skip:
                expanded_clauses = [''.join(expanded_clause)] #if a literal is not in a string, need to make two string with each poss value of that literal    
                final_expanded_clauses = []
                while expanded_clauses != []:
                    for j in range(num_inputs):
                        for ec in expanded_clauses:
                            if 'x' not in ec:
                                final_expanded_clauses += [ec]
                                expanded_clauses.remove(ec)
                            elif ec[j]=='x':
                                expanded_clauses+= [ec[:j] + "1" + ec[j+1:]]
                                expanded_clauses+= [ec[:j] + "0" + ec[j+1:]]
                                expanded_clauses.remove(ec)

                for ec in final_expanded_clauses:
                    unreduced_clauses += [int('0b'+ec,2)]

    ON_clauses = list(set(unreduced_clauses)) #remove duplicates
    OFF_clauses = [i for i in range(2**num_inputs)]
    for term in ON_clauses:
        OFF_clauses.remove(term)

    return ON_clauses, OFF_clauses



if __name__ == "__main__":
    if len(sys.argv) not in [2,3]:
        sys.exit("Usage: python3 deep.py PARAMS.yaml [output_file]")

    if len(sys.argv) == 2:
        output_file=None 
    else:
        output_file=sys.argv[2]
    
    params = param.load(sys.argv[1])
    Gpar = net.Parity_Net(params['parity_model_file'],debug=params['debug'],deep=True)
    Gdeep = build_deep(Gpar,5,output_file,minimizer='espresso',debug=True)
    init = ldoi.get_const_node_inits(Gdeep,params)
    ldoi.test(Gdeep,init=init)
    ldoi.ldoi_sizes_over_all_inputs(params,Gdeep) # todo: change this fn so that it returns actual solns jp
