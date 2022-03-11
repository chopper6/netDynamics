import lap, basin
import sys
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


# LATER
# compare cp vs np
# add fn to compare w LDOI

def contextual(params, G, A0_avg):
	# TODO: add pinned nodes, AFTER checking that this just returns A0 again

	# A0_avg should be a numpy or cupy object with the average activity level of each node in A0
	x = cp.ones(A0_avg.shape,dtype=int8)*2 # only need 0,1,2, but int8 is smallest avail
	x[cp.isclose(A0_avg,0)] = 0 
	x[cp.isclose(A0_avg,0)] = 1

	changed = cp.zeros(A0_avg.shape,dtype=bool)

	cont = True
	loop = 0 

	while cont:
		xON = x.copy()
		xON[xON==2]=0

		xOFF = x.copy()
		xOFF[xOFF==2]=1

		xON = lap.step(params, xON.astype(bool), G)
		xOFF = lap.step(params, xOFF.astype(bool), G)

		x0_next = x.copy()
		x0_next[xON==1]=1
		x0_next[xOFF==0]=0  	# can change 2->1 if 1st change...right?
		x0_next[x_next!=x & changed]=2  
		changed[x_next!=x]=True

		cont = cp.all(x_next==x)
		loop+=1
		if loop>100:
			assert(0) # infinite loop?

	return x0 

if __name__ == "__main__":
	import random as rd
	params = param.load(sys.argv[1])
	G = Net(model_file=params['model_file'],debug=params['debug'])
	SS = basin.measure(params, G)
	A0 = rd.choice([A for A in SS.attractors])
	contextual(params, G, A0.avg)