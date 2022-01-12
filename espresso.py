from pyeda import inter as eda
import deep # TODO: should move fns from deep to here instead, or some utility file

##########################################
fa = [['!a','c']]
fb = [['a','!b']]
vars1 = ['a','b']
vars2 = ['b']
f1 = [['a','b'],['a','c'],['d']]
f2 = [['a','!b'],['d']]
not_str = '!'
##########################################


def not_to_dnf(fn,not_str,cap=100):
    # assumes fn is already in dnf and in a pyeda form
    # returns in pyeda form
    # when |new terms| > cap, will reduce using espresso
    fn = from_pyEda_fn(fn,not_str) 
    if fn == ['1']:
        return 0
    elif fn == ['0']:
        return 1

    terms = []
    clause1 = fn[0]
    for ele1 in clause1:
        terms += [[deep.composite_name([[ele1]], not_str, negate=True)]]  # get_complement_name(ele1, not_str)]]
        for j in range(1,len(fn)):
            clause2=fn[j]
            new_terms = []
            for ele2 in clause2:
                for term in terms:
                    new_terms += [term + [deep.composite_name([[ele2]], not_str, negate=True)]]              #deep.get_complement_name(ele2, not_str)]]
            terms = new_terms

            if len(terms) > cap:
                terms = reduce_espresso(terms, not_str)
    terms = reduce_espresso(terms, not_str)
    return to_pyEda_fn(terms, not_str)   


def F_to_str(fn):
    # TODO: move to util or net or something
    # assumes F is of the form [clause1,clause2,...] where each clause is [ele1,ele2,...]
    s = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            s+='|'
        for ele in clause:
            if j!=0:
                s+='&'
            s+=ele 
            j+=1
        i+=1
    return s 

def str_to_F(s):
    fn = []
    clauses = s.split('&') 
    for clause in clauses:
        fn_clause = []
        eles = clause.split('|')
        for ele in eles:
            fn_clause += [ele]
        fn += [fn_clause]
    return fn 

def reduce_espresso(fn, not_str):
    fn_eda = to_pyEda_fn(fn,not_str).to_dnf()
    if not fn_eda:
        return False # i.e. function evals to false always
    fn_reduced, = eda.espresso_exprs(fn_eda)
    return from_pyEda_fn(fn_reduced,not_str)


def reduce_async_AND_espresso(fns, varbs, not_str, complement=True):
    # where fns = [fn1, fn2, ...]
    # and varbs = [var1, var2, ...] correspd to those functions
    part1 = 1
    for fn in fns:
        part1 = eda.And(part1,to_pyEda_fn(fn,not_str))
    part2 = 0
    for v in varbs:
        term = 1
        for u in varbs:
            if v!=u:
                if not_str in u:
                    u=u.replace(not_str,'')
                    term = eda.And(term,eda.Not(u))
                else:
                    term = eda.And(term,u)
        part2 = eda.Or(part2,term)

    fn = eda.And(part1,part2).to_dnf()
    #print('espresso attempting to reduce:',eda.And(part1,part2).to_dnf())
    if complement:
        fn_not = not_to_dnf(fn,not_str)
        if not fn:
            return ['0'],['1']
        elif not fn_not:
            return ['1'],['0']
        ON_fn, OFF_fn = eda.espresso_exprs(fn, fn_not)
        if not ON_fn:
            return ['0'],['1']
        elif not OFF_fn:
            return ['1'],['0']
        return from_pyEda_fn(ON_fn, not_str), from_pyEda_fn(OFF_fn, not_str)
    else:  
        if not fn:
            return ['0'] # i.e. function evals to false always
        fn_reduced, = eda.espresso_exprs(fn)
        if not fn_reduced:
        	return ['0']
        return from_pyEda_fn(fn_reduced, not_str)

def reduce_async_AND_espresso_old(fn1, vars1, fn2, vars2, not_str):
    assert(0) #old version
    # where vars1 = [varA,...,varX] to represent poss product term
    fn1 = to_pyEda_fn(fn1,not_str)
    fn2 = to_pyEda_fn(fn2,not_str)
    for v in vars1:
        fn2 = eda.And(fn2,v,simplify=False)
    for v in vars2:
        fn1 = eda.And(fn1,v,simplify=False)

    fnCombined = eda.Or(fn1,fn2,simplify=False).to_dnf()
    if not fnCombined:
        return False
    fn_reduced, = eda.espresso_exprs(fnCombined)
    return from_pyEda_fn(fn_reduced,not_str)


def reduce_AND_espresso(fn1, fn2, not_str):
    fn1 = to_pyEda_fn(fn1,not_str)
    fn2 = to_pyEda_fn(fn2,not_str)
    fnAnd = eda.And(fn1,fn2,simplify=False).to_dnf()
    if not fnAnd:
        return False
    fnAnd_reduced, = eda.espresso_exprs(fnAnd)
    return from_pyEda_fn(fnAnd_reduced,not_str)

def to_pyEda_fn(fn, not_str):
    # takes my function format and converts to pyEda
    fnStr = ''
    i=0
    for clause in fn:
        j=0
        if i!=0:
            fnStr += ' | '
        for ele in clause:
            if j!=0:
                fnStr += ' & '
            if not_str in ele:
                fnStr += '~'
                ele = ele.replace(not_str,'')
            fnStr += ele
            j+=1
        i+=1
    return eda.expr(fnStr)

def reduce_complement(fn, not_str, complements):
    # assumes fn is non-zero
    if fn==[]:
        full_unred = 1
    else:
        f = to_pyEda_fn(fn,not_str)
        print("esp.reduce_compl, partial fn=",f)
        fNot = not_to_dnf(f,not_str)
        if not fNot:
            assert(0) # have to think about what to do
        partial, = eda.espresso_exprs(fNot) # TODO: reduce at end of not_to_dnf, sT this is unnec
        if not partial:
            assert(0) # have to think about what to do
        full_unred = partial
    compls = map(eda.exprvar,complements) # TODO need mult
    print("esp.reduce_compl, compls=",compls)
    print("esp.reduce_compl, partialNot=",full_unred)
    for compl in compls:
        full_unred = eda.And(full_unred, compl)
    full, = eda.espresso_exprs(full_unred.to_dnf())
    full = from_pyEda_fn(full, not_str)
    # could check if full is smaller than original fn
    print("esp.reduce_compl yields",full)
    return full

def reduce(fn, not_str):
    fn = to_pyEda_fn(fn,not_str).to_dnf()
    fn_rd, = eda.espresso_exprs(fn)
    return from_pyEda_fn(fn_rd, not_str)

def from_pyEda_fn(fn, not_str):
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
                newEle += not_str
                ele = ele.replace('~','')
            while ' ' in ele or ')' in ele:
                ele = ele.replace(' ','')
                ele = ele.replace(')','')
            newEle += ele
            if ele != '':
                newClause += [newEle]
        if newClause != []:
            newF += [newClause]
    return newF

if __name__ == '__main__':
    f=to_pyEda_fn(fa,'!')
    fnot = not_to_dnf(f,'!')
    print('f=>!f : ',f,'=>',fnot)
    if False:
        fn = reduce_async_AND_espresso([fa,fb], vars1, not_str)
        print('Final form = ',fn)
    if False:
        f1_rd = reduce_espresso(f1, not_str)
        print("Reduced f1 = ",f1_rd)
        fAnd_rd = reduce_AND_espresso(f1, f2, not_str)
        print("Reduced f1&f2 = ",fAnd_rd)

