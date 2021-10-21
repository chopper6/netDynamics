import main, parse,plot, features, stats
import sys, os, itertools, datetime, math, pickle
import random as rd
import numpy as np


def input_set(**kwargs):
	param_file = kwargs.get('param_file', None)
	plotpie = kwargs.get('plotpie', False)
	params = kwargs.get('params', None)

	if param_file==None and params==None:
		sys.err("ERROR: must pass a param file or params!")
	if params is None:
		params = parse.params(param_file)
	num_inputs = len(params['phenos']['inputs'])
	all_attractors = {}


	if plotpie==True:
		params['output_dir'] = os.path.join(os.getcwd()+'/'+params['output_dir'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
		os.makedirs(params['output_dir'])

		params['savefig'] = True # i assume you don't want an image popping up for every combination of inputs


	input_sets = itertools.product([0,1], repeat=num_inputs) 
	if params['verbose']:
		print("\nRunning",2**num_inputs,'input combinations.')
	j=1
	for input_set in input_sets:
		if params['verbose']:
			print("\n~~~ Starting input set #",j,'~~~\n')
		label=''
		for i in range(num_inputs):
			input_node = params['phenos']['inputs'][i]
			params['phenos']['init'][input_node] = input_set[i]
			if i!=0:
				label+='_'
			label+=input_node +str(input_set[i])
		attractors, node_mapping = main.find_attractors(params)
		add_attractors(all_attractors,attractors,input_set) # WARNING: not really attractors, but attractors x inputs now

		if plotpie==True:
			plot.pie(params,attractors, node_mapping,external_label=label)
		j+=1

	normz_attractors(all_attractors, num_inputs)
	return all_attractors


def many_sequences(param_file):
	cancer_seqs = parse.sequences('input/efSeq_pos.txt', 'input/efSeq_neg.txt')
	params = parse.params(param_file)

	run_cancer, run_scramble, run_driverScramble, run_sideScramble, run_random = [0,1,1,1,0]

	all_feats = {'cancerous':[],'scrambled':[],'random':[],'scrambled_drivers':[],'scrambled_passengers':[]}
	params['savefig'] = True # i assume
	cap = None
	k=4
	if cap is None: 
		cap = len(cancer_seqs)

	if run_cancer:
		print("\n~~~ Cancerous Sequences~~~\n")
		i=0
		for seq in cancer_seqs[:cap]:
			if params['verbose']:
				print("\tStarting sequence #",i+1)
			all_feats['cancerous'] += [sequence(params,seq)]
			i+=1

	if run_scramble:
		print("\n~~~ Scrambled Sequences~~~\n")	
		i=0	
		for seq in cancer_seqs[:cap]:
			if params['verbose']:
				print("\tStarting sequence #",i+1)
			while seq.index(('APC',0)) < seq.index(('Ras',1)) < seq.index(('PTEN',0)) < seq.index(('Smad',0)) < seq.index(('p53',0)):
				# ensures at least one of the main drivers are out of order
				rd.shuffle(seq)	# shuffles order in place
			all_feats['scrambled'] += [sequence(params,seq)]
			i+=1

	if run_driverScramble:
		print("\n~~~ Scrambled Main Driver Sequences ~~~\n")
		i=0	
		for seq in cancer_seqs[:cap]:
			if params['verbose']:
				print("\tStarting sequence #",i+1)

			drivers = [('APC',0),('Ras',1),('PTEN',0),('Smad',0),('p53',0)]
			driver_seq = [seq.index(drivers[i]) for i in range(len(drivers))]
			rd.shuffle(driver_seq)
			for j in range(len(driver_seq)):
				seq[driver_seq[j]] = drivers[j]

			all_feats['scrambled_drivers'] += [sequence(params,seq)]
			i+=1

	if run_sideScramble:
		print("\n~~~ Scrambled Passenger Sequences ~~~\n")
		i=0	
		for seq in cancer_seqs[:cap]:
			if params['verbose']:
				print("\tStarting sequence #",i+1)

			drivers = [('APC',0),('Ras',1),('PTEN',0),('Smad',0),('p53',0)]
			driver_seq = [seq.index(drivers[i]) for i in range(len(drivers))]
			passenger_seq = [j for j in range(len(seq))]
			for j in driver_seq:
				passenger_seq.remove(j)
			passengers = [seq[j] for j in passenger_seq]
			rd.shuffle(passenger_seq)
			for j in range(len(passenger_seq)):
				seq[passenger_seq[j]] = passengers[j]

			all_feats['scrambled_passengers'] += [sequence(params,seq)]
			i+=1


	if run_random:
		clause_mapping, node_mapping = parse.net(params)
		nodes = node_mapping['num_to_name']
		print("\n~~~ Random Sequences~~~\n")	
		for i in range(cap):
			if params['verbose']:
				print("\n~~~ Starting sequence #",i+1,' ~~~\n')
			# build a seq by picking rd nodes and rd flips
			seq=[]
			for m in range(len(cancer_seqs[0])):
				seq+=[(nodes[rd.randint(1,math.floor(len(nodes)/2))],rd.choice([0,1]))] #note that nodes includes negative nodes
			#print('build random seq:',seq)
			all_feats['random'] += [sequence(params,seq)]

	params['output_dir'] += 'seqs_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs(params['output_dir'])

	pickle_file = params['output_dir'] + '/multi' + '_'+ datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S') + '.pickle'
	print("Pickling a file to: ", pickle_file)
	pickle.dump( {'data':all_feats, 'params':params}, open( pickle_file, "wb" ) )

	cluster_time_series(all_feats, params,k=k)


def sequence(params,seq,plotit=False):
	#run input set over a sequence of mutations

	# a random seq:
	#seq = [('LDHA',0),('RAGS',1),('Snail',1),('TAK1',1),('BAX',1),('cdh1_UbcH10',0),('Bcl_2',1),('AMPK',1),('HIF1',1),('APC',1),('p27',1),('E2F',1),('TGFbeta',0),('p14',1),('PTEN',1),('Gli',1),('p21',0),('Caspase9',1),('eEF2K',0),('UbcH10',1)]

	# an effector seq: 		
	#seq=[('APC',0),('Mdm2',1),('AMPK',1),('Ras',1),('FADD',0),('Dsh',1),('COX412',1),('AKT',1),('BAX',0),('p27',0),('p15',0),('PTEN',0),('Smad',0),('GSH',0),('p14',0),('PI3K',1),('p53',0),('E_cadh',1),('BAD',0),('VEGF',1)]

	feats = []
	num_inputs = len(params['phenos']['inputs'])
	verbose=params['verbose']

	if verbose:
		print("\t\tStarting with no mutations")
	params['phenos']['mutations'] = {}
	params['verbose']=False
	attractors = input_set(plotpie=False,params=params)
	feats += [features.calc_entropy(params,attractors,num_inputs)]

	for i in range(len(seq)):
		if verbose:
			print("\t\tStarting mutation #",i+1)
		params['phenos']['mutations'][seq[i][0]] = seq[i][1]
		params['verbose']=False
		attractors = input_set(plotpie=False,params=params)
		feats += [features.calc_entropy(params,attractors,num_inputs)]

	orig_dir = params['output_dir']
	if plotit==True and False: #turned this off to set output dir elsewhere
		params['output_dir'] = os.path.join(os.getcwd()+'/'+params['output_dir'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
		os.makedirs(params['output_dir'])
		params['savefig'] = True # i assume

	params['verbose']=verbose
	if plotit:
		plot.features(params,seq,feats)
	params['output_dir'] = orig_dir

	return feats


def add_attractors(all_attractors,attractors,input_set):
	for k in attractors.keys():
		key = k + '|' + str(input_set) 
		if key not in all_attractors.keys():
			all_attractors[key] = attractors[k]
			all_attractors[key]['input'] = str(input_set) 
		else:
			all_attractors[key]['size'] += attractors[k]['size']
			if params['debug']:
				for key2 in attractors[k].keys():
					if key2 != 'size':
						assert(all_attractors[key][key2] == attractors[k][key2])


def normz_attractors(attractors, num_inputs):
	for k in attractors.keys():
		attractors[k]['size'] /= 2**num_inputs






def plot_from_pickle(pickle_file):
	k=4
	a_pickle = pickle.load( open( pickle_file, "rb" ) )
	data, params = a_pickle['data'], a_pickle['params']
	cluster_time_series(data, params,k=k)

def cluster_time_series(data, params, k):
	for key in data.keys():
		if data[key] != [] and data[key] != {}: #ie non empty
			for feat in ['H(pheno)','H(attr|pheno)','H(xf|attr)','H(attr)','H(input|pheno)','H(pheno|input)','H(input|attr)','H(attr|input)']:
				oneSet = []
				for i in range(len(data[key])):
					oneSeries = []
					for j in range(len(data[key][i])):
						oneSeries += [data[key][i][j][feat]]
					oneSet += [oneSeries]
				label = key+'_'+feat
				kmeans_cluster(params, oneSet,label,k=k)


def kmeans_cluster(params, series,label,k=8):
	# series = [[time_series1],...[time_seriesN]]
	from sklearn.cluster import KMeans
	# importing here cause don't want to import sklearn unless abs nec
	#alt: from tslearn.clustering.TimeSeriesKMeans


	clusters = KMeans(n_clusters=k, random_state=0).fit(series, y=None, sample_weight=None).labels_

	avgs, CIS = [],[]
	for i in range(k):
		aCluster = []
		for j in range(len(series)):
			if clusters[j]==i:
				aCluster+=[series[j]]
		if len(aCluster)>0:
			aCluster=np.array(aCluster)
			cluster_avg = aCluster.mean(axis=0)
			if len(aCluster)==1:
				cluster_CIs=None 
			else:
				cluster_CIs = np.array([stats.conf_interval(aCluster[:,t], confidence=0.95,skewed=False) for t in range(len(aCluster[0]))])
				cluster_CIs = np.swapaxes(cluster_CIs,0,1)
				#print(cluster_avg[:3], cluster_CIs[:,:3])
				#assert(0)
			#cluster_label = label + '_cluster' + str(i)
			CIS += [cluster_CIs]
		elif aCluster != []: #i.e. there is at least 1 series in this cluster
			CIS += [None]
		if aCluster != []:
			avgs += [cluster_avg]
	
	plot.clustered_time_series(params,avgs ,label,CIs=CIS)




if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 mult.py PARAMS.yaml run_type")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml' and sys.argv[2]!='plot':
		sys.exit("Parameter file must be yaml format")
	
	if sys.argv[2]=='inputs':
		input_set(param_file=sys.argv[1])
	elif sys.argv[2]=='seq':
		many_sequences(sys.argv[1])
	elif sys.argv[2]=='plot':
		plot_from_pickle(sys.argv[1])
	else:
		sys.exit("Unrecognized run_type (arguments 3).")