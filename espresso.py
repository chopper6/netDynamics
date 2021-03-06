from pyeda import inter as eda
from copy import deepcopy

# Anything in espresso.py should use F form and internally convert to pyEda and back as needed
# F form: fn = [clause1, clause2, ...], where clause = [ele1, ele2, ...]

def reduce_deep(fn,G):
    # reduce, while seperating any composites in fn
    # assumes each ele is in dnf
    fn_eda = to_pyEda_deep(fn,G)
    fn_eda = fn_eda.to_dnf()
    if not fn_eda:
        return [['0']] # i.e. function evals to false always
    fn_reduced, = eda.espresso_exprs(fn_eda)
    fn_reduced = from_pyEda(fn_reduced,G)
    return fn_reduced

def to_pyEda_deep(fn,G):
    # unlike regular to_pyEda, this reduces all composite nodes as well
    # assumes each element of fn is in dnf
    fn = deepcopy(fn)
    fnStr = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            fnStr += ' | '
        for ele in clause:
            if j!=0:
                fnStr += ' & '
            sum_terms = [ele]
            if '+' in ele: 
                sum_terms = ele.split('+')
            fnStr += '('
            k=0
            for sterm in sum_terms:
                if k!=0:
                    fnStr += ' | '
                prod_terms = [sterm]
                if '&' in sterm:
                    prod_terms = sterm.split('&')

                fnStr += '(' 
                l=0   
                for pterm in prod_terms:
                    if l!=0:
                        fnStr+=' & '
                    if G.not_string in pterm:
                        fnStr += '~'
                        pterm = pterm.replace(G.not_string,'')
                    fnStr += pterm
                    l+=1
                k+=1
                fnStr += ')'    
            fnStr += ')'
            j+=1
        i+=1
    return eda.expr(fnStr)


def reduce(fn, G,pyeda_form=False):
    if not pyeda_form:
        fn, placeholders = to_pyEda(fn,G)
    fn_eda = fn.to_dnf()
    if not fn_eda:
        return [['0']] # i.e. function evals to false always
    elif fn_eda == True or str(fn_eda)=='1': # eda's 1 is not == 1 or '1' lol
        return [['1']]
    fn_reduced, = eda.espresso_exprs(fn_eda)

    if not pyeda_form:
        fn_reduced = from_pyEda(fn_reduced,G,placeholders)
    return fn_reduced


def reduce_AND_async(fns, varbs, G):
    # where fns = [fn1, fn2, ...]
    # and varbs = [var1, var2, ...] correspd to those functions
    ph={}
    part1 = 1
    for fn in fns:
        fn, ph = to_pyEda(fn,G,placeholders=ph)
        part1 = eda.And(part1,fn) 
    part2 = 0
    for v in varbs:
        term = []
        for u in varbs:
            if v!=u:
                term += [u]
                #term += [negate_ele(u, G)]
        fn, ph = to_pyEda([term],G,placeholders=ph)
        part2 = eda.Or(part2,fn)
    fn = eda.And(part1,part2).to_dnf()

    return reduce_and_return(fn,G,placeholders=ph)

def reduce_OR_async(fns, varbs, G):
    # TODO: unsure of proper equations for part 2!

    part1 = 0
    ph={}
    for i in range(len(fns)):
        fns[i], ph = to_pyEda(fns[i],G,placeholders=ph)
        varbs[i], ph = to_pyEda([[varbs[i]]],G,placeholders=ph)
        part1 = eda.Or(part1,eda.And(fns[i],varbs[i]) )
    part2 = 0
    for i in range(len(varbs)):
        for j in range(i+1,len(varbs)):
            part2 = eda.Or(part2,eda.And(varbs[i],varbs[j]))
    part3 = 1
    for i in range(len(fns)): 
        part3 = eda.And(part3,fns[i])
    fn = eda.Or(part1,part2,part3).to_dnf()
    return reduce_and_return(fn,G,placeholders=ph)


def reduce_and_return(fn,G,placeholders={}):
    if not fn:
        fn_reduced = 0 # i.e. function evals to false always
    else:
        fn_reduced, = eda.espresso_exprs(fn)
    if not fn_reduced:
        fn_reduced = 0
    return from_pyEda(fn_reduced, G,placeholders=placeholders)

def check_equiv_AND(fns, ON_fn, G):
    naive = 1
    ph = {}
    for fn in fns:
        fn, ph = to_pyEda(fn,G,placeholders=ph)
        naive = eda.And(fn,naive)
    if naive not in [0,1]:
        naive = reduce(naive, G,pyeda_form=True)
    naive = from_pyEda(naive,G,ph)
    sort_fn(naive)
    return ON_fn==naive

def check_equiv_OR(fns, OFF_fn, G):
    naive = 0
    ph = {}
    for fn in fns:
        fn, ph = to_pyEda(fn,G,placeholders=ph)
        naive = eda.Or(fn,naive)
    if naive not in [0,1]:
        naive = reduce(naive, G,pyeda_form=True)
    naive = from_pyEda(naive,G,ph)
    sort_fn(naive)
    return OFF_fn==naive

def sort_fn(fn):
    for clause in fn:
        clause.sort()
    fn.sort()

def sort_fn_of_fns(F):
    for fn in F:
        sort_fn(fn)
    F.sort()

######### CONVERSION ###########

def not_to_dnf(fn,G,cap=100,pyeda_form=False):
    # when |new terms| > cap, will reduce using espresso

    # TODO: poss problematic w/ phs
    # if poss get rid of negate_ele ect recursion, but the phs may not be in G.complements
    # esp 'if pyeda' clause at the end...

    # CURR: recursion via not_to_dnf() -> negate_ele() -> not_to_dnf(); works but poss confusing
    # ALT: could make 2 dicts of higher order terms and their complements, sT don't have to reconstruct each time
    #           (although would still have to construct the 1st time, ect)
    if pyeda_form:
        fn = from_pyEda(fn,G) 
    if fn in [[['1']],['1'],[1],[[1]],1,'1']:
        return 0
    elif fn in [[['0']],['0'],[0],[[0]],0,'0']:
        return 1

    terms = []
    clause1 = fn[0]

    for ele1 in clause1:
        terms += [[negate_ele(ele1,G)]]  
        for j in range(1,len(fn)):
            clause2=fn[j]
            new_terms = []
            for ele2 in clause2:
                for term in terms:
                    new = term + [negate_ele(ele2,G)]
                    new_terms += [list(set(new))]      # poss skip list(set())? just rms duplicates      
            terms = new_terms

            if len(terms) > cap:
                #print("reducing terms=",terms)
                terms = reduce(terms, G)
    terms = reduce(terms, G)

    if pyeda_form:
        terms, phs = to_pyEda(terms, G) # should these phs ever be returned?
        #print('terms,phs=',terms,phs)
    return terms


def negate_ele(ele, G):
    # poss change this to "complement_name"
    # ele is str
    
    # base case
    if '&' not in ele and '+' not in ele:
        not_str = G.not_string
        if not_str in ele:
            return ele.replace(not_str,'')
        else:
            return not_str + ele

    # else higher order node
    ele_fn = str_to_F(ele)
    ele_not = not_to_dnf(ele_fn,G)
    if not ele_not:
       ele_not = '0'
    return F_to_str(ele_not)

######### UTILITY FNS ############

def to_pyEda(fn, G,placeholders={}):
    # takes my function format and converts to pyEda 
    # pass placeholders if calling multiple times before from_pyEda
    #print("in to_pyEda converting fn:",fn)
    fn=deepcopy(fn)
    placeholders = insert_placeholders(fn,G,placeholders=placeholders)
    fnStr = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            fnStr += ' | '
        for ele in clause:
            if j!=0:
                fnStr += ' & '
            if G.not_string in ele:
                fnStr += '~'
                ele = ele.replace(G.not_string,'')
            fnStr += ele
            j+=1
        i+=1
    return eda.expr(fnStr), placeholders

def from_pyEda(fn, G,placeholders=None):
    newF = []
    fn = str(fn).replace('Or(','')
    assert('Or(' not in fn) #since this is DNF, should only be one Or which is rm'd in prev line
    if 'And' in fn:
        clauses = fn.split('And(')
    else:
        clauses = fn.split(',')

    for clause in clauses:
        eles = clause.split(', ')
        newClause = []
        for ele in eles:
            newEle = ''
            if '~' in ele:
                newEle += G.not_string
                ele = ele.replace('~','')
            while ' ' in ele or ')' in ele:
                ele = ele.replace(' ','')
                ele = ele.replace(')','')
            newEle += ele
            if ele != '':
                newClause += [newEle]
        if newClause != []:
            newF += [newClause]

    if placeholders is not None and len(placeholders):
        replace_placeholders(newF, G, placeholders)
    return newF


def insert_placeholders(fn,G,placeholders={}):
    # this is to use names that have illegal symbols as variables, such as "A+B"
    # implicitly modifies fn
    # if for some crazy reason this is the name of a node ("placeholder1"), then may cause an issue
    originals = []
    for clause in fn:
        for i in range(len(clause)):
            name = clause[i]
            if '&' in name or '+' in name:
                #name = ele.replace(not_str,'')
                
                # this breaks cyclic: new node -> find it's complement -> not_to_dnf -> reduce -> insert_phs -> but compl doesn't exist yet (bc finding)
                # BUT should be careful..
                if name in G.complement:
                    compl = G.complement[name]
                else:
                    compl = None

                if name not in originals and (not compl or compl not in originals):
                    originals += [name]
                    placeholders['placeholder'+str(originals.index(name))] = clause[i]
                    clause[i] = 'placeholder'+str(originals.index(name))
                elif not compl or compl in originals:
                    clause[i] = '!placeholder'+str(originals.index(compl))
                else:
                    clause[i] = 'placeholder'+str(originals.index(name))
    return placeholders #note that fn has been implicitly modified


def replace_placeholders(fn, G, placeholders):
    # this is to use names that have illegal symbols as variables, such as "A+B"
    # implicitly modifies fn
    for clause in fn:
        for i in range(len(clause)):
            if clause[i] in placeholders:
                clause[i] = placeholders[clause[i]]
            elif clause[i].replace('!','') in placeholders:
                clause[i] = G.complement[placeholders[clause[i].replace('!','')]]



def F_to_str(fn):
    s = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            s+='+'
        for ele in clause:
            if j!=0:
                s+='&'
            s+=ele 
            j+=1
        i+=1
    return s 

def str_to_F(s):
    # assumes dnf
    fn = []
    clauses = s.split('+') 
    for clause in clauses:
        fn_clause = []
        eles = clause.split('&')
        for ele in eles:
            fn_clause += [ele]
        fn += [fn_clause]
    return fn 


if __name__ == '__main__':
    print("\nwell watcha wanna do? ain't got nothing planned for espresso.py's main")

