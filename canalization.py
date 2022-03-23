import lap, basin, util, param, net, ldoi
import sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed

# jp will del this file and merge into ldoi.py

# TODO
# finish unfair compare 
#	fix the dumb indexing
# 	add exh!
# 	for spc As so ofc LDOI should be worse
#	but point is anything in LDOI should NEC be included
#	and no contradictions with exh

# LATER
# compare cp vs np
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

def unfair_compare(params):
	G = net.Net(model_file=params['model_file'],debug=params['debug'])
	Gpar = net.ParityNet(params['parity_model_file'],debug=params['debug'])		
	ldoi_soln = ldoi.ldoi_bfs(Gpar)[0]
	#cp.fill_diagonal(ldoi_soln, 1) # ldoi usually doesn't consider the drive to be in its soln unless it forms a stable motif
	# TODO: is diag treatment diff for A0? 

	ldoi_percent = cp.sum(ldoi_soln)/((2*G.n)**2) 

	G.prepare_for_sim(params)
	SS = basin.measure(params, G)
	avg_canal_percent = 0
	avg_combo_percent = 0

	for A0 in SS.attractors.values():
		#print('\nA0=',A0.avg)
		num_missed = 0
		canal_soln = ldoi.ldoi_bfs(Gpar,A0=A0.avg)[0]
		if not (cp.all(cp.isin(canal_soln[ldoi_soln], cp.array([1,2])))):
			for i in range(len(canal_soln)):
				for j in range(len(ldoi_soln)):
					if ldoi_soln[i,j]==1 and canal_soln[i,j]==0:
						print("\nerror on",Gpar.nodeNames[j],"when pinning",Gpar.nodeNames[i],'indices=',i,j,'\n')
						assert(0)
		num_missed += cp.sum(ldoi_soln[canal_soln!=1])/2 # since all = 2 this is just counting 

		avg_canal_percent += cp.sum(canal_soln[canal_soln!=2])/((2*G.n)**2)
		avg_combo_percent += (cp.sum(canal_soln[canal_soln!=2]) + num_missed)/((2*G.n)**2)

	avg_canal_percent/=len(SS.attractors)
	avg_combo_percent/=len(SS.attractors)

	print("Survived. Ldoi score=",ldoi_percent,"vs canal=",avg_canal_percent)
	print("Combined ldoi and canal score =",avg_combo_percent)
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