import math

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


def dict_vals(D):
	return [D[k] for k in D.keys()]


def entropy(P):
	if isinstance(P, dict):
		P=dict_vals(P)
	H=0
	for p in P:
		#p=float(p)
		if p!=0:
			H-= p*math.log2(p)
	return H
