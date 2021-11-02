import parse, util
import itertools, sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

def test(param_file):
	A,n,N,node_mapping = parse.expanded_net(sys.argv[1])
	ldoi_solns, negated = ldoi_bfs(A,n,N,pinning=0)
	#print('\nFinished\nnode mapping:',node_mapping['name_to_num'],'\nnegated:',negated)#,'\nLDOI:',ldoi_solns)
	for i in range(len(ldoi_solns)):
		soln_names = ''
		for j in range(len(ldoi_solns[i])):
			if ldoi_solns[i,j]:
				soln_names += node_mapping['num_to_name'][j] + ', '
		print("LDOI(",node_mapping['num_to_name'][i],') =',soln_names)

def ldoi_bfs(A,n,N,pinning=True):
	# A should be adjacency matrix of Gexp
	# and the corresponding Vexp to A should be ordered such that:
	#	 Vexp[0:n] are normal nodes, [n:2n] are negative nodes, and [2n:N] are composite nodes

	# pinning means that sources cannot drive their complement (they are fixed to be ON)
	# without pinning the algorithm will run faster, and it is possible that ~A in LDOI(A)

	X = cp.zeros((N,N),dtype=bool) # the current sent of nodes being considered
	visited = cp.zeros((N,N),dtype=bool) # if visited[i,j]=1, then j is in the LDOI of i
	negated = cp.zeros(N,dtype=bool) # if negated[i] then the complement of i is in the LDOI of i
	D = cp.diag(cp.ones(N,dtype=bool))
	D[cp.arange(2*n,N)]=0 #don't care about source nodes for complement nodes
	D_compl = D.copy()
	D_compl[:2*n] = cp.roll(D_compl[:2*n],n,axis=0)
	A = cp.array(A, dtype=bool)

	max_count = max(cp.sum(A,axis=0))
	if max_count<128: 
		index_dtype = cp.int8
	else: #assuming indeg < 65536/2:
		index_dtype = cp.int16

	num_to_activate = cp.sum(A,axis=0,dtype=index_dtype)
	num_to_activate[:2*n] = 1 # non composite nodes are OR gates
	counts = cp.tile(num_to_activate,N).reshape(N,N) # num input nodes needed to activate
	zero = cp.zeros((N,N))

	X[D] = 1
	X = counts-cp.matmul(X.astype(index_dtype),A.astype(index_dtype))<=0 
	#i.e. the descendants of X, including composite nodes iff X covers all their inputs
	
	if pinning:
		negated = negated | cp.any(X & D_compl,axis=1) #TODO: check it
		X = cp.logical_not(D_compl) & X  

	loop=0
	while cp.any(X): 
		
		visited = visited | X #note that D is included, i.e. source nodes

		if pinning:
			# first mask visited|D to only include those that do not have it's complement in their solution
			# for visited[i] this means visited[i] & not visited.T[2*i]?
			visited_rolled = visited.copy()
			visited_rolled[:2*n] = cp.roll(visited_rolled[:2*n],n,axis=0)
			#memoX = cp.matmul(visited|D & cp.logical_not(cp.any(D_compl & visited,axis=1)),visited)
			memoX = cp.matmul(visited|D & cp.logical_not(visited_rolled.T), visited) #this still doesn't cancel out the whole row...
			#print(visited|D.astype(int),'\n\n',visited.astype(int) & cp.logical_not(visited_rolled.T).astype(int),'\n\n',memoX.astype(int))
			
			# TODO: check logical_not(...) should be size N, and should mask rows of visited
			# if A has visited B, then add B's visited to A if ~A not in B's visited
			# otherwise must cut all of B's contribution!
			# can't just 0 entire row of visited & cp.logical_not(...), since it will be used depending on visited|D
			X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) | memoX
			negated = negated | cp.any(X & D_compl,axis=1) #TODO: check it
			X = cp.logical_not(visited) & cp.logical_not(D_compl) & X 
			
		else:
			memoX = cp.matmul(visited|D, visited)
			X = (counts-cp.matmul(visited|D.astype(index_dtype),A.astype(index_dtype))<=0) | memoX
			X = cp.logical_not(visited) & X

		# debugging:
		loop+=1
		if loop>N:
			print("WARNING: N steps exceeded")
		if loop > N**4:
			sys.exit("ERROR: infinite loop in ldoi_bfs!")

	if not pinning:
		negated = cp.any(visited & D_compl,axis=1) #TODO: check it
	return visited[:2*n,:2*n], negated[:2*n] 


def trash(A,n,N,pinning=True):
	# A should be adjacency matrix of Gexp
	# and the corresponding Vexp to A should be ordered such that:
	#	 Vexp[0:n] are normal nodes, [n:2n] are negative nodes, and [2n:N] are composite nodes

	# pinning means that sources cannot drive their complement (they are fixed to be ON)
	# without pinning the algorithm will run faster, and it is possible that ~A in LDOI(A)

	X = cp.zeros((2*n,N),dtype=bool) # the current sent of nodes being considered
	visited = cp.zeros((N,N),dtype=bool) # if visited[i,j]=1, then j is in the LDOI of i
	negated = cp.zeros(2*n,dtype=bool) # if negated[i] then the complement of i is in the LDOI of i
	D = cp.vstack((cp.diag(cp.ones(2*n,dtype=bool)),cp.zeros((N-2*n,2*n),dtype=bool))).T # indices of source nodes in X
	D_compl = cp.roll(D.copy(),n) # indices of complement of source nodes in X
	A = cp.array(A, dtype=bool)

	max_count = max(cp.sum(A,axis=0))
	if max_count<256: 
		index_dtype = cp.uint8
	else: #assuming indeg < 65536:
		index_dtype = cp.uint16

	counts = cp.tile(cp.sum(A,axis=0,dtype=index_dtype),2*n).reshape(2*n,N) # num input nodes needed to activate
	zero = cp.zeros((2*n,N))

	X[D] = 1
	X = cp.matmul(X,A.T)
	#print(cp.array(X,dtype=int)[5])

	loop=0
	while cp.any(X):
		negated = negated | cp.any(X & D_compl,axis=1) #TODO
		X = cp.logical_not(visited) & cp.logical_not(D_compl) & X
		counts = cp.maximum(counts-X,zero)
		X = X & (counts==0)

		visited = visited | X #note that D is included, i.e. source nodes
		if pinning:
			memoX = cp.matmul(visited, (visited & cp.logical_not(D_compl & visited)).T)
			# TODO: pretty sure this is wrong
			# if A has visited B, then add B's visited to A if ~A not in B's visited
			# otherwise must cut all of B's contribution!
			X = cp.matmul(X & cp.logical_not(D),A) | memoX
		else:
			print(X.shape)
			X = cp.matmul(visited, visited.T)
			print(X.shape)
			assert(0)

		# debugging:
		loop+=1
		if loop>2*n:
			print("WARNING: 2n steps exceeded")
		elif loop>N:
			print("WARNING: N steps exceeded")
		elif loop > N**4:
			sys.exit("ERROR: infinite loop in ldoi_bfs!")

	return visited, negated

if __name__ == "__main__":
	test(sys.argv[1])
