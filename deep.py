import sys, pickle
import ldoi, logic, param, net
import espresso # this file uses pyeda, and can be a bit tricky to install on windows

# TODO HERE AND  ELSEWHERE:
#   double check whenever call 'not_str' (since generally shouldn't)

# LATER: add term condition to prev building unnec new nodes

# move QM section to logic.py? [and update it once espresso version done jp]

def build_deep(G,kmax,output_file,minimizer='espresso',debug=True):
    # G should be a DeepNet (see Net.py)
    # kmax is the highest order term to expand to

    assert(minimizer=='espresso') # put back QM later (maybe)

    k=2
    while k<=kmax:
        added = True
        while added:
            added=0
            for i in range(len(G.nodes)): 
                V = G.nodes[i]
                cmps,cmpl = [],[] #composites and complements, resp
                for j in range(len(G.F[V.name])):
                    clause = G.F[V.name][j]
                    if len(clause) > 1 and len(clause) <= k:
                        cmpsName = G.composite_name([clause])
                        if cmpsName not in G.nodeNames:
                            cmplName = complement_clause(clause,G)
                            assert(cmplName not in G.nodeNames)
                            if 'ErbB2_3+!Akt1&ErbB2_3+!ERa' == cmpsName:
                                print('\ndeep: reducing',cmpsName,'vs compl',cmplName ,'using',clause)
                            ON_fn, OFF_fn = calc_deep_fn(G,clause,minimizer=minimizer)

                            G.add_node(cmpsName,debug=debug) 
                            G.F[cmpsName] = ON_fn
                            G.add_node(cmplName,debug=debug) 
                            G.F[cmplName] = OFF_fn
                            #print("F[",cmpsName,"]=",ON_fn, 'from clause=',clause)
                            #print("\tOFF=",OFF_fn)
                            added += 1
                            cmps += [cmpsName]
                            cmpl += [cmplName]


                        G.F[V.name][j] = [cmpsName] 


                if len(cmps)>0:
                    complement = G.complement[V.name]
                    #print('prev function of',complement,'=',G.F[complement])
                    G.F[complement] = complement_factor(G.F[V.name],cmps,cmpl,G)
                    #print('\tfactored function=',G.F[complement])

            print("completed a pass with k=",k,", and",added," new virtual nodes.")
            
        k += 1

    G.build_Aexp(debug=debug)
    if output_file is not None:
        with open(output_file,'wb') as f:
            pickle.dump([G, params],f)
        #G.write_to_file(output_file,parity=True)

    return G


def debug_check_fn(fn): # can rm this fn
    for clause in fn:
        for ele in clause:
            assert('placeholder' not in ele)


def calc_deep_fn(G,clause,minimizer='espresso',complement=True):
    # returns builds and reduces higher order function for the clause
    # uses either Quine-McCluskey or Espresso
    # Off fn may be v. slow since is the sum term
    if minimizer == 'espresso':
        #import espresso # imported here since espresso does not compile on windows (so someone using windows uses the 'qm' minimizer instead)
        
        fns, varbs = [],[]
        for ele in clause:
            varbs += [ele]
            fns += [G.F[ele]]
        if complement:
            ON_fn, OFF_fn = espresso.reduce_AND_async(fns, varbs, G, complement=complement)
        else:
            ON_fn = espresso.reduce_AND_async(fns, varbs, G, complement=complement)
    elif minimizer in ['qm','QM']:
        assert(0) #haven't updated this in awhile
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


def complement_factor(fn, cmpsNames, cmplNames, G):
    #print('deep.compl_factor(): original +ve fn=',fn)
    # compositeNames & complementNames should match
    partial_fn = []
    for clause2 in fn:
        if len(clause2)>1 or clause2[0] not in cmpsNames:
            partial_fn += [clause2]

    fn = espresso.reduce_complement(partial_fn, cmplNames, G) 
    return fn


def complement_clause(fn,G):
    fn = [fn]
    fn_new = espresso.not_to_dnf(fn,G)
    fn_name_new = espresso.F_to_str(fn_new)
    #print('deep.complement_name: original name=',fn,'\tcomplement name=',fn_name_new)
    return fn_name_new



########### Quine McC Only ##############

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

#############################################

if __name__ == "__main__":
    if len(sys.argv) not in [2,3]:
        sys.exit("Usage: python3 deep.py PARAMS.yaml [output_file]")

    if len(sys.argv) == 2:
        output_file=None 
    else:
        output_file=sys.argv[2]
    
    params = param.load(sys.argv[1])
    G = net.DeepNet(params['parity_model_file'],debug=True)
    Gdeep = build_deep(G,2,output_file,minimizer='espresso',debug=True)
    print("\nDone (not yet using LDOI)\n")
    #init = ldoi.get_const_node_inits(Gdeep,params)
    #ldoi.test(Gdeep,init=init)
    #ldoi.ldoi_sizes_over_all_inputs(params,Gdeep) # todo: change this fn so that it returns actual solns jp
