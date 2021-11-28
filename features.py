import math
from copy import deepcopy
import control, parse

def entropy(P):
	if isinstance(P, dict):
		P=P.values()
	H=0
	for p in P:
		#p=float(p)
		if p!=0:
			H-= p*math.log2(p)
	return H

def add_to_dict(D,k,v):
	if k not in D.keys():
		D[k]=v 
	else:
		D[k]+=v

def check_corr(params, G, phenos):
	# major issue is how to count entropy of oscils!
	# tons of off by 1 due to that damn 0 nodes (C-1 ect)

	H = {k:0 for k in G.nodes}
	num_inputs = len(params['inputs'])
	for C in G.nodes:
		indx = G.nodeNum[C]
		if C.isNegative:
			indx-=G.n
		xy_prs, x_prs = {},{}
		for l in phenos:
			Pout = phenos[l].outputs
			for k in phenos[l].attractors:
				A = phenos[l].attractors[k]
				if not C.isNegative and float(A.avg[G.n]) > .9:
					add_to_dict(xy_prs,Pout+'1',A.size)
					add_to_dict(x_prs,'1',A.size)
				elif C.isNegative and float(A.avg[G.n]) < .1:
					add_to_dict(xy_prs,Pout+'0',A.size)
					add_to_dict(x_prs,'0',A.size)
				# else presumed to oscil, unsure how to handle
			assert(sum(xy_prs.values()) < 1.001)
		
		H[V['#2name'][C]] = entropy(xy_prs) - entropy(x_prs) #/math.log2(3)

	return H


################ BELOW HERE NEEDS TO BE UPDATED #############################

def calc_entropy(params,ioPairs,num_inputs):
	# note that there is a lot of redundancy with the features and P's calculated here
	# 'state' of an iopair is the attractor, i.e. xf 

	features = {}
	P={'attr,pheno':{},'pheno':{},'attr':{}, 'input,pheno':{}, 'input,attr':{},'xf,attr':[]} 

	H_inputs = num_inputs #since each input has 2 states and are equally likely

	for k in ioPairs.keys():
		iopair = ioPairs[k]
		for i in range(int(iopair['period'])): 
			if params['update_rule'] != 'sync':
				assert(False) #period will not work here
				# want a way to def subattractors, but time spent in each state may be too depd on model rates ect
				# also need to check if i def attractors by their state in the async case
			P['xf,attr']+=[(1/iopair['period'])*iopair['size'] ]

		if iopair['pheno'] not in P['pheno'].keys():
			P['pheno'][iopair['pheno']] = iopair['size']
		else:
			P['pheno'][iopair['pheno']] += iopair['size']

		if iopair['state'] not in P['attr'].keys():
			P['attr'][iopair['state']] = iopair['size']
		else:
			P['attr'][iopair['state']] += iopair['size']


		for pair in [['attr','pheno'],['input','pheno'],['input','attr']]:
			search1, search2 = pair[0], pair[1]
			if search1 == 'attr':
				search1 = 'state'
			if search2 == 'attr':
				search2 = 'state'
			attr_key = 	iopair[search1] + ',' + iopair[search2]	
			P_key =	pair[0] + ',' + pair[1]
			if attr_key not in P[P_key].keys():
				P[P_key][attr_key] = iopair['size']
			else:
				P[P_key][attr_key] += iopair['size']



	features['H(attr)'] = entropy(P['attr'])
	features['H(pheno)'] = entropy(P['pheno'])
	features['H(xf|attr)'] = entropy(P['xf,attr']) - features['H(attr)'] 
	features['H(attr|pheno)'] = entropy(P['attr,pheno']) - features['H(pheno)']
	features['H(attr,pheno)'] = entropy(P['attr,pheno'])

	features['H(input|pheno)'] = entropy(P['input,pheno']) - features['H(pheno)']
	features['H(pheno|input)'] = entropy(P['input,pheno']) - H_inputs
	features['H(input|attr)'] = entropy(P['input,attr']) - features['H(attr)']
	features['H(attr|input)'] = entropy(P['input,attr']) - H_inputs

	if params['debug']:
		for k in P.keys():
			summ=0
			if k!='xf,attr':
				for pr in P[k].keys():
					#assert(P[k][pr] <= 1 and P[k][pr]>=0)
					if not (P[k][pr] <= 1 and P[k][pr]>=0):
						print('\nWARNING probability in features.py may be wrong: p(',k,',',pr,')=',P[k][pr])
					summ+=P[k][pr]
				#print('summ of',k,'=',summ)
			else:
				summ = sum(P[k])
			math.isclose(summ,1)

	return features

