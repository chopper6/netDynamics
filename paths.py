import param, net, basin, canalization
import sys
import numpy as np
import cupy as cp

# TODO: clean tf out of the function

def causal_paths(param_file):
	params = param.load(param_file)
	G = net.Net(params)	

	SS = basin.measure(params, G)
	dom_pheno = canalization.build_dominant_pheno(G, SS)
	A_intersect = build_dom_intersection_attractors(params, G, SS, dom_pheno)
	causal_A_nums = causal_trim(G, A_intersect, dom_pheno)
	causal_A = convert2names(G, causal_A_nums)
	for k in causal_A:
		print(k,':::',causal_A[k],'\n\n')

def convert2names(G, causal_A):
	for k in causal_A:
		causal_A[k] = G.certain_nodes_to_names(causal_A[k])
	return causal_A

def causal_trim(G, A_intersect, dom_pheno):
	# takes a subgraph of fixed points and a target output
	# removes nodes whose states do not drive the output
	causal_A = {k:np.ones(G.n)*2 for k in A_intersect} # i.e. start all in nodes 'uncertain' state 2 
	for k in A_intersect.keys():
		assert(k in dom_pheno.keys())
		children = G.output_indices()
		causal_A[k][children] = A_intersect[k][children]
		checked = []
		while len(children) > 0:
			new_children = []
			checked += children
			for c in children:
				name = G.nodeNames[c]
				for clause in G.F[name]:
					# need indices + nec values instead
					ind, vals = clause_ind_and_vals(G,clause)
					if np.all(A_intersect[k][ind] == vals):
						causal_A[k][ind] = A_intersect[k][ind]
						for num in ind:
							if num not in checked:
								new_children += [num]
			children = new_children
	return causal_A

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
	assert(not isinstance(G,net.ParityNet))
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


if __name__ == "__main__":
	if len(sys.argv) not in [2]:
		sys.exit("Usage: $#%^&*(*)*())&")
	
	causal_paths(sys.argv[1])