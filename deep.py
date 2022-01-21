import sys, pickle, itertools
import ldoi, logic, param, net
import espresso # this file uses pyeda, and can be a bit tricky to install on windows
from copy import deepcopy 

# move QM section to logic.py? [and update it once espresso version done jp]

def build_deep(G,kmax,output_file,minimizer='espresso',debug=True):
    # G should be a DeepNet (see Net.py)
    # kmax is the highest order term to expand to

    assert(minimizer=='espresso') # put back QM later (maybe)
    checked = []
    terminates = {}
    k=2
    while k<=kmax:
        added = True
        while added:
            added=0
            for i in range(len(G.nodes)): 
                V = G.nodes[i]
                cmps,cmpl = [],[] #composites and complements, resp
                for j in range(len(G.F[V.name])):
                    clause = deepcopy(G.F[V.name][j])
                    if len(clause) > 1 and clause_order(clause) <= k:
                        cmpsName = composite_name([clause],G)
                        cmplName = complement_name(clause,G)
                        just_added=False
                        # need to check "checked" here sT other nodes with the same clause still add it
                        if cmpsName not in G.nodeNames and clause not in checked:
                            checked += [clause]
                            assert(cmplName not in G.nodeNames)
                            ON_fn, OFF_fn = calc_deep_fn(G,clause,minimizer=minimizer)

                            on_terminates, off_terminates = check_termination(ON_fn, OFF_fn, clause,G)
                            terminates[cmpsName] = on_terminates
                            terminates[cmplName] = off_terminates
                            if not on_terminates or not off_terminates:
                                G.add_node(cmpsName,debug=debug) 
                                G.F[cmpsName] = ON_fn
                                G.add_node(cmplName,debug=debug) 
                                G.F[cmplName] = OFF_fn
                                added += 1
                                just_added=True
                            if on_terminates:
                                checked += [cmpsName]
                            if off_terminates:
                                checked += [cmplName]

                        # node that if cmps was checked and considered to terminate, then it won't be a node
                        if (cmpsName in G.nodeNames and [cmpsName] not in G.F[V.name]) or just_added:
                            #if not terminates[cmpsName]: #temp not adding them ands
                            #   #G.F[V.name][j] = [cmpsName] #rm lower order terms
                            #    G.F[V.name] += [[cmpsName]]

                            if not terminates[cmplName]:
                                cmpl += [cmplName]
                                #print('cmpl of',cmpsName,'is',cmplName)
                                cmpl = rm_list_in_list_dups(cmpl)

                    
                if len(cmpl)>0:
                    complement = G.complement[V.name]
                    G.F[complement] += complement_factor(G.F[complement],cmpl) # again if rm lower order terms, change to G.F[complement] += [complement_factor]
                        # use += instead if want to keep lower level terms (and rm red as w below)
                        # whereas use just = to rm lower orders
                    G.F[complement] = rm_list_in_list_dups(G.F[complement])

                    #rm_rendunant_nodes(G.F[complement],cmpl,k)
                    ## note that if lower orders are NOT kept, then have to track which clauses are replaced
                    #G.F[V.name] += [cmps]
                    #G.F[V.name] = rm_list_in_list_dups(G.F[V.name])

            print("completed a pass with k=",k,", and",added," new virtual nodes.")

        k += 1

    G.build_Aexp(debug=debug)
    if output_file is not None:
        with open(output_file,'wb') as f:
            pickle.dump([G, params],f)
        #G.write_to_file(output_file,parity=True)

    return G


def rm_rendunant_nodes(fn,cmpl,k):
    # curr unused
    assert(k<3)
    rm = []
    for cml in cmpl:
        pieces = cml.split('+')
        for clause in fn: 
            for piece in pieces:
                if clause in pieces:
                    rm+=[clause]
    for clause in rm:
        if clause in fn:
            fn.remove(clause)

def rm_list_in_list_dups(k):
    k = sorted(k)
    k =[k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
    return k

def clause_order(clause):
    # order = # base nodes included
    # for example 'A&B+C+D' is considered 4th order
    order = 0
    for ele in clause:
        parts = ele.split('&')
        for part in parts:
            frags = part.split('+')
            order += len(frags)
    return order


def check_termination(ON_fn, OFF_fn, clause,G):
    fns, fns_compl = [],[]
    for ele in clause:
        fns += [sorted(G.F[ele])]
        fns_compl += [sorted(G.F[G.complement[ele]])]
    espresso.sort_fn_of_fns(fns)
    espresso.sort_fn_of_fns(fns_compl)
    ON_fn.sort()
    OFF_fn.sort()
    on_terminates = espresso.check_equiv_AND(fns, ON_fn, G)
    off_terminates = espresso.check_equiv_OR(fns_compl, OFF_fn, G)
    return on_terminates, off_terminates


def calc_deep_fn(G,clause,minimizer='espresso',complement=True):
    # returns builds and reduces higher order function for the clause
    # uses either Quine-McCluskey or Espresso

    if minimizer == 'espresso':
        
        fns, varbs = [],[]
        fns_compl, varbs_compl = [],[]
        for ele in clause:
            varbs += [ele]
            varbs_compl += [G.complement[ele]]
            fns += [G.F[ele]]
            fns_compl += [G.F[G.complement[ele]]]

        ON_fn = espresso.reduce_AND_async(fns, varbs, G)
        if complement:
            OFF_fn = espresso.reduce_OR_async(fns_compl, varbs_compl, G)

    elif minimizer in ['qm','QM']:
        assert(0) #haven't updated this in awhile
        inputs_num2name, inputs_name2num = organize_inputs(G, clause)
        ON_clauses, OFF_clauses = on_off_terms(G,clause,inputs_name2num)
        ON_fn = logic.run_qm(ON_clauses, len(inputs_name2num), G.encoding, inputs_num2name)
        if complement:
            OFF_fn = logic.run_qm(OFF_clauses, len(inputs_name2num), G.encoding, inputs_num2name)
    else:
        assert(0) # unrecognized argument for 'minimizer'
    
    if complement:
        return ON_fn, OFF_fn 
    else:
        return ON_fn


def complement_factor(fn, factor_strs):
    # this is effectively dividing and finding quotient + remainder, in boolean alg
    # fn and factor are both functions of the same form, although factor is typically smaller
    
    factors = [espresso.str_to_F(factor) for factor in factor_strs]
    quotient = [[[] for _ in range(len(factors[i]))] for i in range(len(factors))]
    quotient_used_clauses = [[[] for _ in range(len(factors[i]))] for i in range(len(factors))] # the clause that were used to generate the quoient
    
    for clause in fn:
        for i in range(len(factors)):
            for j in range(len(factors[i])):
                contained=True 
                for factor_ele in factors[i][j]:
                    if factor_ele not in clause:
                        contained=False
                        break 
                if contained:
                    q = []
                    for fn_ele in clause:
                        if fn_ele not in factors[i][j]:
                            q += [fn_ele]
                    quotient[i][j] += [q]
                    quotient_used_clauses[i][j]+=[clause]
        
    new_fn=[]
    remainder = deepcopy(fn)
    for i in range(len(factors)):
        new_fn_part = []
        for j in range(len(quotient[i][0])):
            term = quotient[i][0][j]
            in_all = True
            for_this_term_used = []
            for k in range(len(quotient[i])): # 0 incld because did that else statement (add its used quotients)
                if term not in quotient[i][k]:
                    in_all=False 
                    break 
                else:
                    indx = quotient[i][k].index(term) 
                    # worry if two quotients in i,k are the same
                    # poss make another dim for clausei n fn
                    for_this_term_used += [quotient_used_clauses[i][k][indx]]
            if in_all:
                new_fn_part += [term + [factor_strs[i]]]
                for used_term in for_this_term_used: # if used by ANY of the factors, is not a remainder
                    if used_term in remainder:
                        remainder.remove(used_term)

        # check for additional factors:
        other_factors = deepcopy(factors)
        other_factors.remove(factors[i])
        other_factors = [espresso.F_to_str(factor) for factor in other_factors]

        # i assume that this factor will not appear again after checking other factors
        if new_fn_part != [] and other_factors != []:
            new_fn_part = complement_factor(new_fn_part, other_factors)

        for clause in new_fn_part:
            clause.sort()
            if clause not in new_fn:
                new_fn += [clause]

    # add back remainder
    # note that fn - new_fn*factor might also work
    for clause in remainder:
        new_fn += [clause]
    return new_fn


def complement_name(fn,G):
    fn = [fn]
    fn_new = espresso.not_to_dnf(fn,G) # this fn can be prohibatively slow for large fns
    fn_name_new = espresso.F_to_str(fn_new)
    return fn_name_new

def composite_name(fn, G):
    fn_new = espresso.reduce_deep(fn, G)
    cName = espresso.F_to_str(fn_new)
    return cName


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

    if False:
        #fn = [['A','B','D'],['A','C','D'],['E','D'],['C','E'],['A','B','F']] #,['A','E']
        #factors = [[['A','B'],['A','C'],['E']]]
        fn = [['A','C'],['A','D'],['B','C'],['B','D'],['A','B','E'],['Z','X']] 
        factors = [[['A'],['B']],[['C'],['D']],[['B','E']]]
        newFn = complement_factor(fn,factors)

    else:
        # TODO: clean up this general flow as well (ex ldoi.test())
        params = param.load(sys.argv[1])
        if 0:
            print("\n~~~Original LDOI~~~\n")
            G = net.ParityNet(params['parity_model_file'],debug=True)
            init = ldoi.get_const_node_inits(G,params)
            print('initialization:',[G.nodeNames[num] for num in init])
            regular_ldoi = ldoi.test(G,params, init=init)

        #sys.setrecursionlimit(4000) #this is occuring in pyeda when reducing an exapnded str -> f
        G = net.DeepNet(params['parity_model_file'],debug=True)
        Gdeep = build_deep(G,2,output_file,minimizer='espresso',debug=True)
        
        if 1:
            print("\nDEEP G")
            for node in Gdeep.nodes:
                print(node.name,node.num,' F=',node.F())
            print('\n')
        if 0:
            print("\n~~~Deep LDOI~~~\n")
            init = ldoi.get_const_node_inits(Gdeep,params)
            print('initialization:',[G.nodeNames[num] for num in init])

            deep_ldoi = ldoi.test(Gdeep,params, init=init)
            #ldoi.ldoi_sizes_over_all_inputs(params,Gdeep) # todo: change this fn so that it returns actual solns jp
            
            if 1:
                print('\n\n~~~Comparison~~~\n')
                for k in regular_ldoi.keys():
                    if k not in deep_ldoi.keys():
                        print("only REGULAR ldoi has solution for",k,'=',regular_ldoi[k]) 
                for k in deep_ldoi.keys():
                    if k not in regular_ldoi.keys():
                        print("only DEEP ldoi has solution for",k,'=',deep_ldoi[k]) 
                    elif deep_ldoi[k] != regular_ldoi[k]:
                        print("Deep vs Regular possibly diff for:",k)
                        for soln in deep_ldoi[k]:
                            if soln not in regular_ldoi[k]:
                                print('\tonly DEEP:',soln)
                        for soln in regular_ldoi[k]:
                            if soln not in deep_ldoi[k]:
                                print('\tonly REGULAR:',soln)
        else:
            print("Done. no ldoi atm")