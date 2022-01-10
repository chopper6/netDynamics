from pyeda import inter as eda

##########################################
fa = [['a','b'],['c']]
fb = [['a','!b']]
vars1 = ['a','b']
vars2 = ['b']
f1 = [['a','b'],['a','c'],['d']]
f2 = [['a','!b'],['d']]
not_str = '!'
##########################################

def reduce_espresso(fn, not_str):
    fn_eda = to_pyEda_fn(fn,not_str).to_dnf()
    if not fn_eda:
        return False # i.e. function evals to false always
    fn_reduced, = eda.espresso_exprs(fn_eda)
    return from_pyEda_fn(fn_reduced,not_str)


def reduce_async_AND_espresso(fns, varbs, not_str):
    # where fns = [fn1, fn2, ...]
    # and varbs = [var1, var2, ...] correspd to those functions
    part1 = 1
    for fn in fns:
        part1 = eda.And(part1,fn)
    part2 = 0
    for v in varbs:
        term = 1
        for u in varbs:
            if v!=u:
                term = eda.And(term,u)
        part2 = eda.Or(part2,term)
    fn = eda.And(part1,part2).to_dnf()
        print(part1,'\tthen',part2)
    if not fn:
        return False # i.e. function evals to false always
    fn_reduced, = eda.espresso_exprs(fn)
    return fn_reduced

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

def from_pyEda_fn(fn, not_str):
    newF = []
    fn = str(fn).replace('Or(','')
    assert('Or(' not in fn) #since this is DNF, should only be one Or which is rm'd in prev line
    clauses = fn.split('And(')
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
    fn = reduce_async_AND_espresso([fa,fb], vars1, not_str)
    print('Final form = ',fn)
    if False:
        f1_rd = reduce_espresso(f1, not_str)
        print("Reduced f1 = ",f1_rd)
        fAnd_rd = reduce_AND_espresso(f1, f2, not_str)
        print("Reduced f1&f2 = ",fAnd_rd)

