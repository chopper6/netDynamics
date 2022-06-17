import param, net, basin, canalization
import sys
import numpy as np

def sandbox(param_file):
	params = param.load(param_file)
	G = net.Net(params)
	Gpar = net.ParityNet(params)	

	SS = basin.measure(params, G)
	dom_pheno = canalization.build_dominant_pheno(G, SS)
	A_intersect = canalization.build_intersection_attractors(params, Gpar, SS)
	# TODO: A_intersect does not req expanded network form here

	R_in={}
	#print(dom_pheno,'\n\n\n',A_intersect)

	for k1 in dom_pheno.keys():
		assert(k1 in A_intersect.keys())
		r = A_intersect[k1]
		#S = A_intersect[k1]
		for k2 in dom_pheno.keys():
			if k1!=k2:
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
			s[A!=A_intersect[k2]]=A
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
	
	sandbox(sys.argv[1])