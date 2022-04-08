import lap, basin, util, param, net, ldoi, control
import sys
from copy import deepcopy
import numpy as np

CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# jp will del this file and merge into ldoi.py

# TODO
# finish unfair compare 
# ofc gotta clean a lot
# exh with mult copies for async (poss after clean)

# LATER
# merge mult As (intersection)
# then do comparison over inputs
# compare #steps reqd to async

def apply_pins(X,n):
	# where n is number of nodes (i.e. 1/2 height of X)
	assert(len(X)==n*2)
	cp.fill_diagonal(X,1)
	X_rolled = cp.roll(X,n,axis=0)
	cp.fill_diagonal(X_rolled,0)
	X = cp.roll(X_rolled,n,axis=0) 
	return X

def build_A0(G,A0_avg):
	assert(isinstance(G,net.ParityNet))
	A0 = cp.ones(A0_avg.shape,dtype=cp.int8)*2 # only need 0,1,2, but int8 is smallest avail
	A0[cp.isclose(A0_avg,0)] = 0 
	A0[cp.isclose(A0_avg,1)] = 1

	A0not = cp.ones(A0_avg.shape,dtype=cp.int8)*2 
	A0not[A==1]=0 
	A0not[A==0]=1

	A0exp = cp.zeros(G.n_exp-2*G.n)
	assert(G.n_exp-2*G.n>=0)

	A0 = cp.hstack([A0,A0not,A0exp])
	return A0 

def contextual(params, G, A0_avg):
	# TODO: add pinned nodes, AFTER checking that this just returns A0 again

	# A0_avg should be a numpy or cupy object with the average activity level of each node in A0
	x = cp.ones(A0_avg.shape,dtype=cp.int8)*2 # only need 0,1,2, but int8 is smallest avail
	x[cp.isclose(A0_avg,0)] = 0 
	x[cp.isclose(A0_avg,1)] = 1

	X = cp.vstack([x for _ in range(2*G.n)])
	changed = cp.zeros(X.shape,dtype=bool)
	X = apply_pins(X,G.n)

	cont = True
	loop = 0 
	print('starting conextualdns')
	while cont:
		print('\nX  = ',X[5])
		xON = X.copy()
		xON[xON==2]=0

		xOFF = X.copy()
		xOFF[xOFF==2]=1

		# if only looking at a single attractor:
		#xON = lap.step(params, xON[cp.newaxis,:].astype(bool), G)[0]  # check parallelism arg, newaxis just to increase dim, [0] to return down
		#xOFF = lap.step(params, xOFF[cp.newaxis,:].astype(bool), G)[0]
		
		xON = lap.step(params, xON.astype(bool), G)
		xOFF = lap.step(params, xOFF.astype(bool), G)

		x_next = X.copy() #cp.ones(A0_avg.shape,dtype=cp.int8)*2
		x_next[xON==1]=1
		x_next[xOFF==0]=0  	# can change 2->1 if 1st change...right?
		x_next[(x_next!=X) & changed]=2 
		changed[x_next!=X]=True

		x_next = apply_pins(x_next,G.n)
		cont = 1-cp.all(x_next==X)
		X=x_next
		#print("xON=",xON.astype(int),'\nxOFF=',xOFF.astype(int),'\nx_next=',x_next.astype(int))

		loop+=1
		if loop>100:
			assert(0) # infinite loop?

	#print('num steps required =',loop) # maybe track this (compared with async # steps reqd)
	return X


def exh_compare():
	# TODO: other than cleaning the shitstorm that is this file
	# run exhaustive control...how to compare?
	# 	a) exh almost never 100% fixes
	#	b) partial coverage was a control goal...
	exh_perturbs = control.exhaustive(params, G, CONTROL_PARAMS['mut_thresh'], CONTROL_PARAMS['cnt_thresh'], max_mutator_size=CONTROL_PARAMS['max_mutator_size'], max_control_size = CONTROL_PARAMS['max_control_size'],norm=CONTROL_PARAMS['norm'])
		

def unfair_compare(params, include_exhaustive=True):
	# TODO: the whole process of running a single attractor in basin is WAY more convoluted than it should be 

	G = net.Net(model_file=params['model_file'],debug=params['debug'])
	Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])		

	ldoi_soln = ldoi.ldoi_bfs(Gpar)[0]
	

	cp.fill_diagonal(ldoi_soln, 1) # ldoi usually doesn't consider the drive to be in its soln unless it forms a stable motif
	ldoi_percent = cp.sum(ldoi_soln)/(2*G.n**2) 

	G.prepare_for_sim(params)
	SS = basin.measure(params, G)
	avg_canal_percent = 0
	avg_combo_percent = 0
	total_num_missed  = 0
	for k in SS.attractors:
		A0 = SS.attractors[k]
		z=[(G.nodeNames[i],float(A0.avg[i])) for i in range(len(A0.avg))]
		print("\nA0 starting...",z)
		canal_soln = ldoi.ldoi_bfs(Gpar,A0=A0.avg)[0]
		total_pinned = 0

		for node in G.nodes:
			print('\tpinning',node.name)
			for b in [0,1]:
				#print('\n\ncanal',node.name,b)
				SS0 = deepcopy(SS) # only want to start from single A0
				paramscopy, Gcopy = deepcopy(params), deepcopy(G)
				paramscopy['mutations'][node.name]=b 
				paramscopy['inputs']=[]
				Gcopy.prepare_for_sim(paramscopy)
				SS0.attractors = {k:SS0.attractors[k]}
				SS0.attractors[k].id = util.char_in_str(SS0.attractors[k].id, node.num, b)
				SS0.attractors[k].size=1
				SS0.build_A0()
				SS_result = basin.calc_size(paramscopy, Gcopy, SS0=SS0)
				exh_A = np.array([SS_result.attractors[k].avg for k in SS_result.attractors][0]) 
				# note that this assume only 1 resulting A, better for sync

				if not (exh_A[node.num]==b): # pinned node should stay pinned
					print("node",node.name,node.num,'set to',b,'doesnt stay pinned!',exh_A[node.num])
					assert(0)

				total_pinned += (exh_A==1).sum()+(exh_A==0).sum()


				# check vs canal soln
				canal_indx = node.num + (1-b)*G.n
				for j in range(G.n*2):
					if canal_soln[canal_indx,j]==1:
						if j<G.n:
							if not (exh_A[j]==1): # for exh soln look if regular node is same
								print('set',node.name,'#',node.num,'to',b,'target',G.nodeNames[j],'#',j,'-> canal:',canal_soln[canal_indx,j],'vs exh',exh_A[j])
								z={G.nodeNames[i]:exh_A[i] for i in range(len(exh_A))}
								print('\t\tcanal\t\texh')
								for ky in z:
									indx = G.nodeNums[ky]
									if canal_soln[canal_indx,indx] != exh_A[indx]:
										print(ky,canal_soln[canal_indx,indx],exh_A[indx])

								assert(0)
						else:
							assert(exh_A[j-G.n]==0)  # for exh soln look if compl node is opposite
					elif canal_soln[canal_indx,j]==0:
						if j<G.n:
							if not (exh_A[j]==0):
								print('set',node.name,'#',node.num,'to',b,'target',G.nodeNames[j],'#',j,'-> canal:',canal_soln[canal_indx,j],'vs exh',exh_A[j])
								assert(0)
						else:
							assert(exh_A[j-G.n]==1)

		percent_pinned_exh = total_pinned/(2*G.n**2) # each poss pin per each poss mutn 

		if not (cp.all(cp.isin(canal_soln[ldoi_soln], cp.array([1,2])))):
			for i in range(len(canal_soln)):
				for j in range(len(ldoi_soln)):
					if ldoi_soln[i,j]==1 and canal_soln[i,j]==0:
						print("\nerror on",Gpar.nodeNames[j],"when pinning",Gpar.nodeNames[i],'indices=',i,j,'\n')
						assert(0)
		num_missed = int(cp.sum(ldoi_soln[canal_soln!=1]))

		total_num_missed += num_missed/(2*G.n**2)
		avg_canal_percent += cp.sum(canal_soln[canal_soln!=2])/(2*G.n**2)
		avg_combo_percent += (cp.sum(canal_soln[canal_soln!=2]) + num_missed)/(2*G.n**2)

	avg_canal_percent/=len(SS.attractors)
	avg_combo_percent/=len(SS.attractors)
	total_num_missed/=len(SS.attractors)

	ldoi_percent /= percent_pinned_exh
	avg_canal_percent /= percent_pinned_exh
	avg_combo_percent /= percent_pinned_exh

	print("Survived. Ldoi score=",ldoi_percent,"vs canal=",avg_canal_percent)
	print("Combined ldoi and canal score =",avg_combo_percent," canal missed avg",total_num_missed,"of LDOI")
	# note that some nodes are oscillating, so need exhaustive sim to check

if __name__ == "__main__":
	import random as rd
	
	params = param.load(sys.argv[1])
	G = net.Net(model_file=params['model_file'],debug=params['debug'])
	
	if 1:
		unfair_compare(params)

	elif 0:
		G.prepare_for_sim(params) # there should be a 1-liner for making a net and preparing it...
		SS = basin.measure(params, G)
		A0 = rd.choice([A for A in SS.attractors.values()])
		print("A0=",A0.avg.astype(int))

		mutant = rd.choice(G.nodeNames).replace(G.not_string,'')
		val = (int(A0.id[G.nodeNums[mutant]%len(A0.id)])+1)%2

		print("mutating",mutant,'(#',G.nodeNums[mutant]%len(A0.id),') to',val)
		params['mutations'][mutant] = val 
		G.prepare_for_sim(params)
		A0.avg[G.nodeNums[mutant]] = val

		x = contextual(params, G, A0.avg)
		print("returns:",x)

	elif 1:

		Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])	
		G.prepare_for_sim(params) # there should be a 1-liner for making a net and preparing it...
		SS = basin.measure(params, G)
		A0 = rd.choice([A for A in SS.attractors.values()])
		print("A0=",A0.avg)
		x = ldoi.ldoi_bfs(Gpar, A0=A0.avg)[0]
		print("returns:",x.astype(int))
		print("num 2's=",cp.sum(x[x==2])/2,"out of:",len(x)*len(x[0]))

	else: # spc run

		mutant, val = 'x2',1
		A0 = cp.array([0,1,0,0])	

		print("A0=",A0)
		params['mutations'][mutant] = val 
		G.prepare_for_sim(params)
		A0[G.nodeNums[mutant]] = val

		x = contextual(params, G, A0)
		print("returns:",x)