import param, net, basin, canalization
import sys
import numpy as np
import cupy as cp

# TODO: clean tf out of the function
#	merge with canalization
#	only keep parts in ldoi if really nec
#	

def causal_paths(param_file):
	params = param.load(param_file)
	G = net.Net(params)	
	Gpar = net.ParityNet(params)

	SS = basin.measure(params, G)
	dom_pheno = canalization.build_dominant_pheno(G, SS)
	#A_intersect = build_dom_intersection_attractors(params, G, SS, dom_pheno)
	A_intersect = canalization.build_intersection_attractors(params, Gpar, SS, transients=True, composites=False)
	#print([(G.nodeNames[i], A_intersect['[0, 1, 0, 0]'][i][0]) for i in range(G.n)]) 
	#assert(0)
	# transients ensures that nodes labelled "2" are oscil in all attractors
	# no composites to reduce memory required 

	SMs = SM_in_attr(params, Gpar, A_intersect)  # note that expanded network is needed for SM

	causal_A_On_nums, causal_A_Tr_nums = causal_trim(Gpar, A_intersect, dom_pheno, SMs)
	causal_On, causal_Tr = convert2names(G, causal_A_On_nums), convert2names(G, causal_A_Tr_nums, transient=True)
	
	for k in causal_On:
		print('\n\n\n############### for k =',k,'#############')
		causal_inputs = 0 
		for inpt in params['inputs']:
			if inpt in causal_On[k] or inpt in causal_Tr[k] or G.not_string + inpt in causal_On[k] or G.not_string + inpt in causal_Tr[k]:
				causal_inputs+=1
		causal_inputs /= len(params['inputs'])
		print("\nFraction of causal inputs=",causal_inputs)

		if 1:
			print("\n~~~Causal On~~~")
			print(causal_On[k],'\n\n')
			print("~~~Causal Tr~~~")
			print(causal_Tr[k],'\n\n')

def convert2names(G, causal_A, transient=False):
	# if only looking at transient don't distinguish x from !x (since if x is transient so is !x)
	named = {k:[] for k in causal_A}
	if transient:
		n=G.n 
	else:
		n=G.n_neg
	for k in causal_A:
		for i in range(n):
			if causal_A[k][i]:
				named[k] += [G.nodeNames[i]]
		#causal_A[k] = G.certain_nodes_to_names(causal_A[k])
	return named

def causal_trim(Gpar, A_intersect, dom_pheno, SMs):
	# takes a subgraph of fixed points and a target output
	# removes nodes whose states do not drive the output

	causal_A_On = {k:np.zeros(Gpar.n_neg, dtype=bool) for k in A_intersect}  # why are these np and not dict anyway? slightly more mem effic
	causal_A_Tr = {k:np.zeros(Gpar.n_neg, dtype=bool) for k in A_intersect} 
	for k in A_intersect.keys():
		assert(k in dom_pheno.keys())
		outputs = Gpar.output_indices()
		children, checked, checked_transient = [],[], []
		# children is a tuple of (node_number, transient_counts)
		# transient counts if the output node is transient, or if there is a downstream stable motif that is causally related to the output

		for o in outputs:
			for suffix in [0,Gpar.n]:
				onum=o+suffix
				if A_intersect[k][onum] == 1:
					if SMs[k][onum]:
						children += [(onum,True)]
					else:
						children += [(onum,False)]
					causal_A_On[k][onum] = 1
					checked += [onum]
				elif A_intersect[k][onum] == 2: #i.e. transient
					# ARBITRARILY ONLY ADD THE TRUE SIDE
					if suffix==0:
						children += [(onum,True)]
						causal_A_Tr[k][onum] = A_intersect[k][onum]
						checked_transient += [onum]
				else:
					assert(A_intersect[k][onum]==0) # if is 3, then not the dominant pheno!

		while len(children) > 0:
			new_children = []
			for tup in children:
				c, transient = tup[0],tup[1]
				name = Gpar.nodeNames[c]
				for clause in Gpar.F[name]:
					# need indices + nec values instead
					#ind, vals = clause_ind_and_vals(G,clause)
					ind = np.array([Gpar.nodeNums[x] for x in clause])
					#print('here',clause, 'AU',A_intersect[k][ind])
					#if k == '[1, 0, 0, 0]' and name=='Proliferation':
					#	print("prolifs parents:",A_intersect[k][ind])
					if transient:
						causal = np.all((A_intersect[k][ind] == 1) | (A_intersect[k][ind] == 2))
					else:
						causal = np.all(A_intersect[k][ind] == 1)
					if causal:
						parent_names = [Gpar.nodeNames[i] for i in ind]
						#if k == '[1, 0, 0, 0]': # and name=='GRB2':
						#	if 'FRS2' in parent_names or '!FRS2' in parent_names:
						#		print("parents",parent_names,'of child', name)
							
						#causal_A[k][ind] = A_intersect[k][ind]
						for num in ind:
							if A_intersect[k][num] == 1:
								if num not in checked:
									causal_A_On[k][num] = A_intersect[k][num]
									checked += [num]
									child_transient = transient | SMs[k][num]
									new_children += [(num,child_transient)]
							else:
								assert(A_intersect[k][num] == 2)
								if num not in checked_transient and (num + Gpar.n)%Gpar.n_neg not in checked_transient:
									causal_A_Tr[k][num] = A_intersect[k][num]
									checked_transient += [num]
									child_transient = transient | SMs[k][num]
									new_children += [(num,child_transient)]
							
			children = new_children
	return causal_A_On, causal_A_Tr

def clause_ind_and_vals(G,clause):
	ind, vals = [],[]
	for name in clause: 
		if G.not_string in name:
			name=name.replace(G.not_string, '')
			vals += [0]
		else:
			vals += [1]
		ind += [G.nodeNums[name]]

	return ind, vals


def build_dom_intersection_attractors(params, G, SS, dom_pheno):
	# similar to ldoi.build_intersection_attractors() except with regular net and using attractors in dominant pheno only
	assert(not isinstance(G,net.ParityNet)) # can change this, but careful 
	input_ind, output_ind = G.input_indices(), G.output_indices()
	input_orgnzd = {} #{str(k):[] for k in G.get_input_sets(params)}
	for A in SS.attractors.values():
		input_key = str([int(x) for x in A.avg[input_ind]])

		# if A pheno matches domin only, then:
		if np.all(A.pheno_arr == dom_pheno[input_key]):
			if input_key not in input_orgnzd.keys():
				input_orgnzd[input_key] = []
			A_certain = build_A0(G,A.avg)
			input_orgnzd[input_key] += [A_certain]

	A_intersect = {}
	for k in input_orgnzd:
		stacked = np.array([input_orgnzd[k][i] for i in range(len(input_orgnzd[k]))])
		# for each node (col), if all attrs (rows) are not the same, then all equal 2
		isSame = np.array([np.all(np.equal(stacked[:,i],stacked[0,i])) for i in range(len(stacked[0]))]) # should be a better numpy way...
		A_intersect[k] = input_orgnzd[k][0]
		A_intersect[k][isSame==False] = 2 
	
	return A_intersect


def build_A0(G,A0_avg):
	# this is same as in ldoi, except for regular net
	#assert(not isinstance(G,net.ParityNet))
	A0 = np.ones(A0_avg.shape,dtype=np.int8)*2 # only need 0,1,2, but int8 is smallest avail
	A0[np.isclose(A0_avg,0)] = 0 
	A0[np.isclose(A0_avg,1)] = 1

	return A0 

def robust_senstive_structs():
	assert(0) # just saving the code but jp del
	R_in={}
	#print(dom_pheno,'\n\n\n',A_intersect)

	for k1 in dom_pheno.keys():
		assert(k1 in A_intersect.keys())
		r = A_intersect[k1]
		#S = A_intersect[k1]
		for k2 in dom_pheno.keys():
			if k1!=k2: # jp this is a mistake, since k1!=k2 always unless they are the same...
				if np.all(dom_pheno[k1] == dom_pheno[k2]):
					r[r!=A_intersect[k2]]=2
			else:
				pass # TODO build sensitive struct
		R_in[k1]=Gpar.certain_nodes_to_names(r[:G.n]) # G.n to trim out expanded net nodes

	R_out = {}
	S_out = {}
	for k in R_in.keys():
		print("\ninput set",k,'with outputs',dom_pheno[k],':\n',R_in[k])
		pheno_key = str(dom_pheno[k])
		A=A_intersect[k]
		s= np.zeros(A.shape)
		for k2 in dom_pheno.keys():

			# TODO: shouldn't need to cast to np here, likely a mistake in basin.py
			if isinstance(A,cp.ndarray):
				A=A.get()	
			if isinstance(A_intersect[k2],cp.ndarray):
				A_intersect[k2]=A_intersect[k2].get()
			#if np.sum(A!=A_intersect[k2])>0:
			# not sure why worked before, but now need to match assignment size and indices explicitly
			s[A!=A_intersect[k2]]=A[A!=A_intersect[k2]]
		if pheno_key not in R_out.keys():
			R_out[pheno_key] = A_intersect[k]
			S_out[pheno_key] = s
		else:
			r = A_intersect[k]
			r[R_out[pheno_key]!=r]=2
			R_out[pheno_key] = r
			#S_out[pheno_key] = idunnowtf

	for k in R_out.keys():
		R_out[k] = Gpar.certain_nodes_to_names(R_out[k][:G.n])


	for k in R_out.keys():
		print("\noutput set",k,':\n',R_out[k])


def SM_in_attr(params, Gpar, A_intersect):
	SM = {input_state:np.zeros(Gpar.n_neg, dtype=bool) for input_state in A_intersect.keys()}
	for input_state in A_intersect.keys(): 
		for name in Gpar.nodeNames:
			num = Gpar.nodeNums[name]
			if A_intersect[input_state][num]==1:
				SM[input_state][num] = logical_BFS_for_SM(Gpar, name, A_intersect[input_state])

	return SM

def logical_BFS_for_SM(Gpar, origin, A_intersect):
	# note that here A_intersect is just for one input state
	visited = {node:False for node in Gpar.nodeNames}
	tocheck = [origin]
	while len(tocheck) > 0:
		v=tocheck[0]
		for clause in Gpar.F[v]:
			clauseon=True
			for parent in clause:
				parent_num = Gpar.nodeNums[parent]
				if A_intersect[parent_num] != 1:
					clauseon=False
			if clauseon:
				for parent in clause:
					if not visited[parent]:
						visited[parent]=True
						tocheck+=[parent]
					if parent == origin:
						return True
		tocheck.remove(v)
	return False


if __name__ == "__main__":
	if len(sys.argv) not in [2]:
		sys.exit("Usage: $#%^&*(*)*())&")
	
	causal_paths(sys.argv[1])