import math
from copy import deepcopy
import control, parse


def calc_canalizing(F, chaining=False):
	# FUCK jp this is just LDOI, so del it... \:
	# chaining means that if a node canalizes a child, add that child's canalizing score to the node

	# TODO:
	# chainging is currently wrong, jp should just use LDOI
	# (sep fn) entropy calc, reqs QM

	node_fn_split, clause_split, literal_split, not_str, strip_from_clause,strip_from_node = parse.get_file_format('bnet')
	Vexp = [k for k in F.keys()] + [not_str + k for k in F.keys()]
	Vexp.remove('0') # rm that 0 node
	Vexp.remove('!0')
	canal_sets = {k:set() for k in Vexp}

	for node in F.keys():
		inputs_rev, inputs_fwd =  control.get_inputs_to_node(F[node], not_str)
		# fwd is name2#, rev is #toname
		for inpt in inputs_rev: 
			on, not_on, off, not_off = False, False, True, True
			# off means off when inputs is off
			# not_on means on when input is off
			# not_off means off when not input is off 
			# ...so this is v redunant
			for clause in F[node]:
				if inpt not in clause:
					not_off=False
				if clause == [inpt]:
					on=True
				if not_str + inpt not in clause:
					off=False
				if clause == [not_str + inpt]:
					not_on = True

			if on:
				canal_sets[inpt].update([node])
			if not_on:
				canal_sets[not_str+inpt].update([node])
			if off:
				canal_sets[inpt].update([not_str+node])
			if not_off:
				canal_sets[not_str+inpt].update([not_str+node])

	if chaining:
		while True:
			canal_sets_next = deepcopy(canal_sets)
			for node in canal_sets.keys():
				for child in canal_sets[node]:
					if node not in canal_sets[child]:
						canal_sets_next[node].update(canal_sets[child])

			if canal_sets_next == canal_sets:
				break
			canal_sets = canal_sets_next

	canal_score = {k:len(canal_sets[k])/(len(Vexp)/2) for k in Vexp} 
	total_score = sum(canal_score.values())/len(Vexp)
	print(canal_sets)
	return canal_score, total_score


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

def check_corr(params, V, phenos):
	# major issue is how to count entropy of oscils!
	# tons of off by 1 due to that damn 0 nodes (C-1 ect)

	n = int(len(V['name2#'])/2)
	nodes = [i for i in range(len(V['name2#']))]
	nodes.remove(0)
	nodes.remove(n)
	H = {V['#2name'][k]:0 for k in nodes}
	num_inputs = len(params['inputs'])
	for C in nodes:
		C_offset = C-1 #the fucking 1 and 0 nodes
		if C >n:
			C_offset-=n
		xy_prs, x_prs = {},{}
		for l in phenos:
			Pout = phenos[l].outputs
			for k in phenos[l].attractors:
				A = phenos[l].attractors[k]
				if C<n and float(A.avg[C_offset]) > .9:
					add_to_dict(xy_prs,Pout+'1',A.size)
					add_to_dict(x_prs,'1',A.size)
				elif C>n and float(A.avg[C_offset]) < .1:
					add_to_dict(xy_prs,Pout+'0',A.size)
					add_to_dict(x_prs,'0',A.size)
				# else presumed to oscil, unsure how to handle
			assert(sum(xy_prs.values()) < 1.001)
		
		H[V['#2name'][C]] = entropy(xy_prs) - entropy(x_prs) #/math.log2(3)

	return H


def check_corr_old(params, V, phenos):
	# major issue is how to count entropy of oscils!
	# tons of off by 1 due to that damn 0 nodes (C-1 ect)

	n = int(len(V['name2#'])/2)
	nodes = [i for i in range(len(V['name2#']))]#incld 0 and 1 nodes cause fuck it
	H_joint = {V['#2name'][k]:0 for k in nodes} # joint entropy of output and node
	H = {V['#2name'][k]:0 for k in nodes}
	H_cond = {V['#2name'][k]:0 for k in nodes}
	for C in nodes:
		P1, P0, Poscil = 0,0,0
		for l in phenos:
			P = phenos[l]
			p1,p0,poscil = 0,0,0
			for k in P.attractors:
				A = P.attractors[k]
				if float(A.avg[C]) > .9:
					p1+= A.size
					P1+= A.size
				elif float(A.avg[C]) < .1:
					p0+= A.size
					P0+= A.size
				else:
					poscil+= A.size
					Poscil+= A.size
			summ = p1+p0#+poscil
			for event in [p1,p0]:#,poscil]:
				if summ!=0:	
					p = (event/summ)*P.size
					assert(p<1.0001 and p>=-.0001)
					if p!=0:
						H_joint[V['#2name'][C]] -= p*math.log2(p)
		
		summ = P1+P0+Poscil
		#if Poscil > .3:
		#	H_cond[V['#2name'][C]] = None #since unsure what to do
		#else:
		if C>n:
			p=P0 / summ
		for event in [P1,P0]:#,Poscil]:
			if summ!=0:
				p = (event/summ)
				assert(p<1.0001 and p>=-.0001)
				if p!=0:
					H[V['#2name'][C]] -= p*math.log2(p)

		H_cond[V['#2name'][C]] = H_joint[V['#2name'][C]] - H[V['#2name'][C]]

		#if not (H_cond[V['#2name'][C]] > -.01 and H_cond[V['#2name'][C]] < 1.01): #some room for rounding errors
		# since have more than 1 pheno there are more than 2 events, so entropy may be > 1
	return H_cond


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

