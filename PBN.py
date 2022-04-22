from itertools import chain, combinations
import numpy as np # may have to replace w cp

# TODO: 
#		in net: self.max_literals is different..
#		integrate into net.build_Fmapd_and_A()
#			if poss, clean that funtion while you're there
#			need to replace max_literals with the 1 node
#			integrate clauses_mult
#		and fix lap.step(), incld 1-x for negatives, and multiplier for clauses
#		anYwhere that F is modified (self loops, mutations...) will need to use this, not just during init
#		then a whole lotto test based debug

def get_node_float_clauses(fn, n):
	# for PBN
	# n should be the number of regular nodes (not complements)
	clauses = []
	clauses_mult = [] # what the clause is multiplied by
	pset = powerset(fn)
	for ps in pset:
		lng = len(ps)
		flat_ps = list(chain.from_iterable(ps))
		flat_ps.sort() # sort sT can tell if already used this clause
		clause = np.unique(np.array(flat_ps)) # remove repeats
		print('clause=',clause, 'clauses=',clauses)
		if not np.any(np.isin(clause+n,clause)): # check there does not exist a node and its complement in the clause
			clause = list(clause) # don't want numpy now (since clauses won't be square)
			if clause not in clauses:
				clauses += [clause]
				clauses_mult += [(-1)**(lng+1)]
			else:
				indx = clauses.index(clause)
				clauses_mult[indx] += (-1)**(lng+1)

	return clauses, clauses_mult


def powerset(iterable):
    "powerset([1,2,3]) --> (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))


if __name__ == "__main__":
	fn = [[1,0],[3]]
	n=2
	get_node_clauses(fn, n)