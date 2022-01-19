import sys, pickle, itertools
import ldoi, logic, param, net
import espresso # this file uses pyeda, and can be a bit tricky to install on windows
from copy import deepcopy 

# TODO HERE AND  ELSEWHERE:
#   double check whenever call 'not_str' (since generally shouldn't)

# LATER: add term condition to prev building unnec new nodes
#   rm redundancy in complement_factor(): if decide to keep lower order terms, composite names shouldn't be repeatedly called jp

# move QM section to logic.py? [and update it once espresso version done jp]

def build_deep(G,kmax,output_file,minimizer='espresso',debug=True):
    # G should be a DeepNet (see Net.py)
    # kmax is the highest order term to expand to

    assert(minimizer=='espresso') # put back QM later (maybe)
    checked = []
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
                        cmpsName = G.composite_name([clause])
                        cmplName = complement_clause(clause,G)

                        # need to check "checked" here sT other nodes with the same clause still add it
                        if cmpsName not in G.nodeNames and clause not in checked:
                            assert(cmplName not in G.nodeNames)
                            ON_fn, OFF_fn = calc_deep_fn(G,clause,minimizer=minimizer)

                            G.add_node(cmpsName,debug=debug) 
                            G.F[cmpsName] = ON_fn
                            G.add_node(cmplName,debug=debug) 
                            G.F[cmplName] = OFF_fn
                            if cmplName == '!a+c':
                                print("OFF_fn for !a+c: ",OFF_fn)
                            #print("F[",cmpsName,"]=",ON_fn, 'from clause=',clause)
                            #print("\tOFF=",OFF_fn)
                            added += 1

                            checked += [clause]

                        cmps += [cmpsName] 
                        cmpl += [cmplName]

                        # rm'g duplicates, TODO: see if this is nec (same for further below)
                        cmps = rm_list_in_list_dups(cmps)
                        cmpl = rm_list_in_list_dups(cmpl)

                        # TODO: still end up with duplicates here, due to diff orderings
                        if [cmpsName] not in G.F[V.name]: # this is also ineffic
                            G.F[V.name] += [[cmpsName]]  # poss later change to G.F[V.name][j] = [cmpsName], if use term condition
                        
                            G.F[V.name] = rm_list_in_list_dups(G.F[V.name])
                    
                if len(cmps)>0:
                    complement = G.complement[V.name]
                    G.F[complement] += complement_factor(G.F[V.name],cmpl) # again if rm lower order terms, change to G.F[complement] += [complement_factor]
                    G.F[complement] = rm_list_in_list_dups(G.F[complement])

            print("completed a pass with k=",k,", and",added," new virtual nodes.")
            
        k += 1

    G.build_Aexp(debug=debug)
    if output_file is not None:
        with open(output_file,'wb') as f:
            pickle.dump([G, params],f)
        #G.write_to_file(output_file,parity=True)

    return G

def rm_list_in_list_dups(k):
    k = sorted(k)
    k =[k[i] for i in range(len(k)) if i == 0 or k[i] != k[i-1]]
    return k

def clause_order(clause):
    order = 0
    for ele in clause:
        parts = ele.split('&')
        for part in parts:
            frags = part.split('+')
            order += len(frags)
    return order

def debug_check_fn(fn): # can rm this fn
    for clause in fn:
        for ele in clause:
            assert('placeholder' not in ele)


def calc_deep_fn(G,clause,minimizer='espresso',complement=True):
    # returns builds and reduces higher order function for the clause
    # uses either Quine-McCluskey or Espresso
    # Off fn may be v. slow since is the sum term
    if minimizer == 'espresso':
        
        fns, varbs = [],[]
        fns_compl, varbs_compl = [],[]
        for ele in clause:
            varbs += [ele]
            varbs_compl += [G.complement[ele]]
            fns += [G.F[ele]]
            fns_compl += [G.F[G.complement[ele]]]
        if complement:
            ON_fn, OFF_fn = espresso.reduce_AND_async(fns, varbs,fns_compl, varbs_compl, G, complement=complement)
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


def complement_factor(fn, factor_strs):
    # this is effectively dividing and finding quotient + remainder, in boolean alg
    # fn and factor are both functions of the same form, although factor is typically smaller
    
    #factor_strs = [espresso.F_to_str(factor) for factor in factors]
    #print('in compl_factor, factors=',[factor for factor in factor_strs])
    factors = [espresso.str_to_F(factor) for factor in factor_strs]
    #print('init fn=',fn)
    #print('factors=',factors,'factor_str=',factor_strs)
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
                    #print(clause,'/',factors[i][j],'=',q,'with used clause:',clause)
        
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
                    #print('a used term:',used_term)
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
    # note that fn - new_fn*factor should also work
    for clause in remainder:
        new_fn += [clause]
    #print("final function =",new_fn)
    return new_fn




def complement_factor_old(fn, cmpsNames, cmplNames, G): # del soon!
    #print('deep.compl_factor(): original +ve fn=',fn)
    # compositeNames & complementNames should match
    partial_fn = []
    for clause in fn:
        if (len(clause)>1 or clause[0] not in cmpsNames) and (G.composite_name([clause]) not in cmpsNames): 
            # 2nd condition (and ..) due to fact that lower order terms are included atm
            partial_fn += [clause]

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

    if False:
        #fn = [['A','B','D'],['A','C','D'],['E','D'],['C','E'],['A','B','F']] #,['A','E']
        #factors = [[['A','B'],['A','C'],['E']]]
        fn = [['A','C'],['A','D'],['B','C'],['B','D'],['A','B','E'],['Z','X']] 
        factors = [[['A'],['B']],[['C'],['D']],[['B','E']]]
        newFn = complement_factor(fn,factors)

    else:
        # TODO: clean up this general flow as well (ex ldoi.test())

        #sys.setrecursionlimit(4000) #this is occuring in pyeda when reducing an exapnded str -> fn
        params = param.load(sys.argv[1])
        G = net.DeepNet(params['parity_model_file'],debug=True)
        Gdeep = build_deep(G,2,output_file,minimizer='espresso',debug=True)
        
        print("\nDEEP G")
        for node in Gdeep.nodes:
            print(node.name,node.num,' F=',node.F())
        if 0:
            init = ldoi.get_const_node_inits(Gdeep,params)
            print('initialization:',[G.nodeNames[num] for num in init])

            ldoi.test(Gdeep,init=init)
            #ldoi.ldoi_sizes_over_all_inputs(params,Gdeep) # todo: change this fn so that it returns actual solns jp
        else:
            print("Done. no ldoi atm")