from pyeda import inter as eda
from copy import deepcopy
##########################################
fa = [['!a','c']]
fb = [['a','!b']]
vars1 = ['a','b']
vars2 = ['b']
f1 = [['a','b'],['a','c'],['d']]
f2 = [['a','!b'],['d']]
not_str = '!'
##########################################

# Anything in espresso.py should use F form and internally convert to pyEda and back as needed
# F form: fn = [clause1, clause2, ...], where clause = [ele1, ele2, ...]

# TODO: not_dnf() and negate_ele() still need cleaning, esp w.r.t phs
# pass G to these fns from wherever
# when debugging check edge cases such as: return ['0'],['1']

def reduce(fn, not_str, G):
    fn_eda, placeholders = to_pyEda_fn(fn,not_str,G)
    fn_eda = fn_eda.to_dnf()
    if not fn_eda:
        return False # i.e. function evals to false always
    fn_reduced, = eda.espresso_exprs(fn_eda)
    fn_reduced = from_pyEda_fn(fn_reduced,not_str,placeholders,G)
    #print("\n\treduced fn:",fn_reduced)
    return fn_reduced


def reduce_complement(fn, not_str, complements, G):
    if fn==[]:
        full_unred = 1
        ph = {}
    else:
        f, ph = to_pyEda_fn(fn,not_str)
        print("esp.reduce_compl, partial fn=",f)
        partial = not_to_dnf(f,not_str,pyeda_form=True) #again ok to skip ph here only bc fwd fn's ph already recorded
        if not partial:
            assert(0) # have to think about what to do
        full_unred = partial

    compls, ph = to_pyEda([[compl for compl in complements]], not_str,G,placeholders=ph) 
    full_unred = eda.And(full_unred, compls)
    full, = eda.espresso_exprs(full_unred.to_dnf())
    full = from_pyEda_fn(full, not_str,ph,G)
    return full


def reduce_AND_async(fns, varbs, not_str,G, complement=True):
    # where fns = [fn1, fn2, ...]
    # and varbs = [var1, var2, ...] correspd to those functions
    used_compls = []
    part1 = 1
    ph={}
    for fn in fns:
        fn, ph = to_pyEda_fn(fn,not_str,G,placeholders=ph)
        part1 = eda.And(part1,fn) 
    part2 = 0
    for v in varbs:
        term = []
        for u in varbs:
            if v!=u:
                term += [negate_ele(u, not_str)]
        fn, ph = to_pyEda_fn(term,not_str,G,placeholders=ph)
        part2 = eda.Or(part2,term)

    fn = eda.And(part1,part2).to_dnf()
    #print('espresso attempting to reduce:',eda.And(part1,part2).to_dnf())
    if complement:
        OFF_fn = not_to_dnf(fn,not_str,pyeda_form=True)
        # note that any placeholders during not are lost, this is only ok since the complement of any such ph should be in the On fn
        # but in general this is smthg not great about not_to_dnf(pyeda_form=True)
        if not fn:
            return ['0'],['1']
        elif not OFF_fn:
            return ['1'],['0']
        ON_fn, = eda.espresso_exprs(fn)
        if not ON_fn:
            return ['0'],['1']
        elif not OFF_fn:
            return ['1'],['0']
        return from_pyEda_fn(ON_fn, not_str,placeholders=ph,G=G), from_pyEda_fn(OFF_fn, not_str,placeholders=ph,G=G)
    else:  
        if not fn:
            return ['0'] # i.e. function evals to false always
        fn_reduced, = eda.espresso_exprs(fn)
        if not fn_reduced:
            return ['0']
        return from_pyEda_fn(fn_reduced, not_str,placeholders=ph,G=G)


######### CONVERSION ###########

def not_to_dnf(fn,not_str,G,cap=100,pyeda_form=False):
    # when |new terms| > cap, will reduce using espresso

    # TODO: poss problematic w/ phs
    # if poss get rid of negate_ele ect recursion, but the phs may not be in G.complements

    # CURR: recursion via not_to_dnf() -> negate_ele() -> not_to_dnf(); works but poss confusing
    # ALT: could make 2 dicts of higher order terms and their complements, sT don't have to reconstruct each time
    #           (although would still have to construct the 1st time, ect)
    if pyeda_form:
        fn = from_pyEda_fn(fn,not_str) 
    if fn == ['1']:
        return 0
    elif fn == ['0']:
        return 1

    terms = []
    clause1 = fn[0]

    for ele1 in clause1:
        terms += [[negate_ele(ele1,not_str)]]  
        for j in range(1,len(fn)):
            clause2=fn[j]
            new_terms = []
            for ele2 in clause2:
                for term in terms:
                    new_terms += [term + [negate_ele(ele2,not_str)]]           
            terms = new_terms

            if len(terms) > cap:
                terms = reduce_espresso(terms, not_str)
    terms = reduce_espresso(terms, not_str)

    if pyeda_form:
        return to_pyEda_fn(terms, not_str,G)   
    else:
        return terms


def negate_ele(ele, not_str):
    # ele is str
    
    # base case
    if '&' not in ele and '+' not in ele:
        if not_str in ele:
            return ele.replace(not_str,'')
        else:
            return not_str + ele

    # else higher order node
    ele_fn = str_to_F(ele)
    ele_not_eda = espresso.not_to_dnf(ele_fn,not_str)
    return F_to_str(ele_not)

######### UTILITY FNS ############

def to_pyEda(fn, not_str,G,placeholders={}):
    # takes my function format and converts to pyEda 
    # pass placeholders if calling multiple times before from_pyEda
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
            if not_str in ele:
                fnStr += '~'
                ele = ele.replace(not_str,'')
            fnStr += ele
            j+=1
        i+=1
    return eda.expr(fnStr), placeholders

def from_pyEda(fn, not_str,placeholders=None,G=None):
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

    if placeholders is not None and len(placeholders):
        replace_placeholders(newF, placeholders,G)
    return newF


def insert_placeholders(fn,G,placeholders={}):
    # this is to use names that have illegal symbols as variables, such as "A+B"
    # implicitly modifies fn
    # if for some crazy reason this is the name of a node ("placeholder1"), then may cause an issue
    #fn = deepcopy(fn) # jp DON"T want this, but lets see
    originals = []
    for clause in fn:
        for i in range(len(clause)):
            ele = clause[i]
            if '&' in ele or '+' in ele:
                name = ele.replace(not_str,'')
                compl = G.complement[name]
                if name not in originals and compl not in originals:
                    originals += [name]
                    placeholders['placeholder'+str(originals.index(name))] = clause[i]
                    clause[i] = 'placeholder'+str(originals.index(name))
                elif compl in originals:
                    clause[i] = '!placeholder'+str(originals.index(name))
                else:
                    clause[i] = 'placeholder'+str(originals.index(name))
    return placeholders #note that fn has been implicitly modified


def replace_placeholders(fn, placeholders, G):
    # this is to use names that have illegal symbols as variables, such as "A+B"
    # implicitly modifies fn
    for clause in fn:
        for i in range(len(clause)):
            if clause[i] in placeholders:
                clause[i] = placeholders[clause[i]]
            elif clause[i].replace('!','') in placeholders:
                clause[i] = G.complement[placeholders[clause[i]]]




if __name__ == '__main__':
    print("\nwell watcha wanna do? ain't got nothing planned for espresso.py's main")

