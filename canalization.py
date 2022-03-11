import lap, basin, util, param, net
import sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


# LATER
# parallelize pinned (controllers)
# jp mutated would just be on the original net, right?
# compare cp vs np
# add fn to compare w LDOI

def contextual(params, G, A0_avg):
	# TODO: add pinned nodes, AFTER checking that this just returns A0 again

	# A0_avg should be a numpy or cupy object with the average activity level of each node in A0
	x = cp.ones(A0_avg.shape,dtype=cp.int8)*2 # only need 0,1,2, but int8 is smallest avail
	x[cp.isclose(A0_avg,0)] = 0 
	x[cp.isclose(A0_avg,1)] = 1
	changed = cp.zeros(A0_avg.shape,dtype=bool)

	cont = True
	loop = 0 

	while cont:
		xON = x.copy()
		xON[xON==2]=0

		xOFF = x.copy()
		xOFF[xOFF==2]=1

		xON = lap.step(params, xON[:,cp.newaxis].astype(bool), G)[0]  # check parallelism arg, newaxis just to increase dim, [0] to return down
		xOFF = lap.step(params, xOFF[:,cp.newaxis].astype(bool), G)[0]

		x_next = x.copy()
		x_next[xON==1]=1
		x_next[xOFF==0]=0  	# can change 2->1 if 1st change...right?
		x_next[x_next!=x & changed]=2  
		changed[x_next!=x]=True

		cont = cp.all(x_next==x)
		loop+=1
		if loop>100:
			assert(0) # infinite loop?

	return x

if __name__ == "__main__":
	import random as rd
	params = param.load(sys.argv[1])
	G = net.Net(model_file=params['model_file'],debug=params['debug'])
	G.prepare_for_sim(params) # there should be a 1-liner for making a net and preparing it...
	SS = basin.measure(params, G)
	A0 = rd.choice([A for A in SS.attractors.values()])
	print("A0=",A0.avg.astype(int))

	# 1 not clear if working, 2 get out of range on nums which means nodeNames and nums ain't flush \:
	mutant = rd.choice(G.nodeNames)
	val = (int(A0.id[G.nodeNums[mutant]])+1)%2
	print("mutating",mutant,'to',val)
	params['mutations'][mutant] = val 
	G.prepare_for_sim(params)
	A0.avg[G.nodeNums[mutant]] = val

	x = contextual(params, G, A0.avg)
	print("returns:",x)