# finds control set to drive system into a single attractor
# haven't looked at this in a long time

import sys
import ldoi, param, net, util
import numpy as np
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


CONTROL_PARAMS = {
	'method':'det' # det or stoch
}


def deterministic(G, params):
	assert(isinstance(G,net.Parity_Net))

	fixed = np.array([0 for _ in range(G.n_neg)],dtype=bool)
	#fixed = trim(G, params,fixed)
	solution = np.array([0 for _ in range(G.n_neg)], dtype=bool)

	loop=0
	while np.sum(solution+fixed) != G.n_neg:
		if loop%1==0:
			print("Searching for candidate #",loop)
		loop+=1

		# run ldoi with existing solution
		pinned = np.where(solution==1)[0]
		visited, negated = ldoi.ldoi_bfs(G,pinning=True,init=pinned) 
		if CUPY:
			visited = visited.get() #cast cupy->numpy

		# take node with largest ldoi set, excluding existing solution and already fixed nodes
		#s_indx, f_indx = np.where(solution==1)[0],np.where(fixed==1)[0]
		visited[:,solution]=0
		visited[:,fixed]=0
		visited[solution,:]=0 # these neither count towards a node's score, nor are new candidates to add to the solution
		visited[fixed,:]=0
		best = np.argmax(np.sum(visited,axis=1)) # todo: check axis is correct

		# add that node to the solution
		if np.sum(visited[best]) == 0:
			assert(0) # may need to debug this section
			indx = np.random(np.where(fixed==0 and solution==0)[0]) # pin one of the remaining independent nodes
			compl = (indx + G.n) % G.n_neg
			solution[indx]=1
			fixed[compl]=1
		else:
			assert(solution[best]==0)
			newly_fixed = np.where(visited[best]==1)[0] 
			assert(np.all(np.logical_not(fixed[newly_fixed]))) # new candidate should not have counted any prev fixed nodes
			assert(np.all(np.logical_not(solution[newly_fixed]))) # new candidate should not have counted any prev fixed nodes
			newly_fixed = np.append(newly_fixed, (best + G.n) % G.n_neg) # also consider it's complement fixed
			solution[best]=1
			fixed[newly_fixed]=1
			fixed[best]=0 # since this will occur if best is part of a stable motif
			assert(np.all(np.logical_not(np.logical_and(fixed, solution))))
			#solns = np.where(solution==1)[0]
			#solution_names = [G.nodeNames[i] for i in solns]
			#print(solution_names)
	solns = np.where(solution==1)[0]
	solution_names = [G.nodeNames[i] for i in solns]
	return solution_names

##############################################

def trim(G, params,fixed):
	# build expanded net, then trim nodes with only 1 in or out-deg
	# put these in 'fixed' sT they are ignored later

	to_rm = []
	A = G.A_exp[:G.n_neg, :G.n_neg] # clip out expanded nodes
	rmd = True
	while rmd: # iteratively take out nodes w/ deg = 1: either must pin (if in deg) or just wait until pred pinned
		rmd=False
		for node in G.nodes:
			if sum(A[node.num]) + sum(A[:,node.num]) == 1: 
				to_rm += [node]
				rmd=True
		for node in list(set(to_rm)): # poss that expanded nodes will linger but oh well
			G.A_exp[node.num,:] = 0 
			G.A_exp[:,node.num] = 0 
			compl = (node.num + G.n) % G.n_neg
			G.A_exp[compl,:] = 0 
			G.A_exp[:,compl] = 0 
			fixed[node.num] = 1
			fixed[compl] = 1
	return fixed



if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 control.py PARAMS.yaml")

	print('\nGlobally controlling, using these control params:',CONTROL_PARAMS,'\n\n')	
	params = param.load(sys.argv[1])
	G = net.Parity_Net(params['parity_model_file'],debug=params['debug'])
	if CONTROL_PARAMS['method'] == 'det':
		solution = deterministic(G,params)
	elif CONTROL_PARAMS['method'] == 'stoch':
		assert(0) # not implem yet
	else:
		print("ERROR: unrecognized control method in CONTROL_PARAMS.")
		sys.exit()
	print("\nControl Solution:",solution)