import main, parse,plot, features, stats
import sys, os, itertools, datetime, math, pickle
import random as rd
import numpy as np

assert(0) # TODO: this is way outdated by now!


def many_sequences(param_file):
	###### PARAMS FOR THIS: ######
	runSeqType = {'cancerous':1,'scrambled':0,'random':1,'scrambled_drivers':0,'scrambled_passengers':0}
	seqNumCap = 5
	seqLngCap = 4 #mainly for debugging
	k=2
	##############################

	cancer_seqs = parse_sequences('input/efSeq_pos.txt', 'input/efSeq_neg.txt')
	params = parse.params(param_file)

	seqTypes = ['cancerous','scrambled','random','scrambled_drivers','scrambled_passengers']

	all_feats = {}
	params['savefig'] = True # i assume
	pickled_seqs, all_feats = {}, {}

	for seqType in seqTypes:
		one_type_of_seq(params, cancer_seqs, seqType, runSeqType, pickled_seqs, all_feats,  seqNumCap=seqNumCap, seqLngCap=seqLngCap)


	params['output_dir'] += 'seqs_'+datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
	os.makedirs(params['output_dir'])

	pickle_file = params['output_dir'] + '/features.pickle'
	print("Pickling a file to: ", pickle_file)
	pickle.dump( {'data':all_feats, 'params':params, 'seqs':pickled_seqs}, open( pickle_file, "wb" ) )

	clusterplot_time_series(all_feats, params,pickled_seqs,k=k)


def one_type_of_seq(params, cancer_seqs, seqType, runSeqType, pickled_seqs, all_feats, seqNumCap=None, seqLngCap=None):
	if seqNumCap is None: 
		seqNumCap = len(cancer_seqs)
	if seqLngCap is None:
		seqLngCap = len(cancer_seqs[0])

	if runSeqType[seqType]:
		print("\n~~~ " + seqType + " sequences ~~~\n")
		pickled_seqs[seqType] = []
		all_feats[seqType] = []
		i=0
		for seq in cancer_seqs[:seqNumCap]:
			if params['verbose']:
				print("\tStarting sequence #",i+1)

			# if seqType = 'cancerous' just use seq as is
			if seqType == 'scrambled':
				while seq.index(('APC',0)) < seq.index(('Ras',1)) < seq.index(('PTEN',0)) < seq.index(('Smad',0)) < seq.index(('p53',0)):
					# ensures at least one of the main drivers are out of order
					rd.shuffle(seq)	# shuffles order in place
			elif seqType == 'scrambled_drivers':
				drivers = [('APC',0),('Ras',1),('PTEN',0),('Smad',0),('p53',0)]
				driver_seq = [seq.index(drivers[i]) for i in range(len(drivers))]
				rd.shuffle(driver_seq)
				for j in range(len(driver_seq)):
					seq[driver_seq[j]] = drivers[j]
			elif seqType == 'scrambled_passengers':
				drivers = [('APC',0),('Ras',1),('PTEN',0),('Smad',0),('p53',0)]
				driver_seq = [seq.index(drivers[i]) for i in range(len(drivers))]
				passenger_seq = [j for j in range(len(seq))]
				for j in driver_seq:
					passenger_seq.remove(j)
				passengers = [seq[j] for j in passenger_seq]
				rd.shuffle(passenger_seq)
				for j in range(len(passenger_seq)):
					seq[passenger_seq[j]] = passengers[j]
			elif seqType == 'random': 
				F, V = parse.get_logic(params)
				nodes = V['#2name']
				seq=[]
				for m in range(len(cancer_seqs[0])):
					seq+=[(nodes[rd.randint(1,math.floor(len(nodes)/2))],rd.choice([0,1]))] #note that nodes includes negative nodes
			
			pickled_seqs[seqType] += [seq[:seqLngCap]]
			all_feats[seqType] += [sequence(params,seq[:seqLngCap])]
			i+=1


def sequence(params,seq,plotit=False):
	#run input set over a sequence of mutations

	# a random seq:
	#seq = [('LDHA',0),('RAGS',1),('Snail',1),('TAK1',1),('BAX',1),('cdh1_UbcH10',0),('Bcl_2',1),('AMPK',1),('HIF1',1),('APC',1),('p27',1),('E2F',1),('TGFbeta',0),('p14',1),('PTEN',1),('Gli',1),('p21',0),('Caspase9',1),('eEF2K',0),('UbcH10',1)]

	# an effector seq: 		
	#seq=[('APC',0),('Mdm2',1),('AMPK',1),('Ras',1),('FADD',0),('Dsh',1),('COX412',1),('AKT',1),('BAX',0),('p27',0),('p15',0),('PTEN',0),('Smad',0),('GSH',0),('p14',0),('PI3K',1),('p53',0),('E_cadh',1),('BAD',0),('VEGF',1)]

	feats = []
	num_inputs = len(params['inputs'])
	verbose=params['verbose']

	if verbose:
		print("\t\tStarting with no mutations")
	params['mutations'] = {}
	params['verbose']=False
	attractors = input_set(plotpie=False,params=params)
	feats += [features.calc_entropy(params,attractors,num_inputs)]

	for i in range(len(seq)):
		if verbose:
			print("\t\tStarting mutation #",i+1)
		params['mutations'][seq[i][0]] = seq[i][1]
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


############################################################################################################	



def clusterplot_time_series(data, params, pickled_seqs, k):

	# first get ylims, lazy style
	ylims = {}
	ylim_buffer = 2 #note that CI's may get cut off
	for key in data.keys(): #key= scrambled, cancerous, ect
		if data[key] != [] and data[key] != {}: #ie non empty
			ylims[key] = {}
			for feat in data[key][0][0].keys(): #ex 'H(pheno)','H(attr|pheno)'
				ylims[key][feat] = {}
				for i in range(len(data[key])):
					for j in range(len(data[key][i])):
						if i==j==0:
							ylims[key][feat]['min'] = ylims[key][feat]['max'] = data[key][i][j][feat]
						ylims[key][feat]['min'] = min (ylims[key][feat]['min'],data[key][i][j][feat])
						ylims[key][feat]['max'] = max (ylims[key][feat]['max'],data[key][i][j][feat])

	pickled_clusters = {}
	for key in data.keys(): #key= scrambled, cancerous, ect
		if data[key] != [] and data[key] != {}: #ie non empty
			pickled_clusters[key] = {}
			for feat in data[key][0][0].keys(): #ex 'H(pheno)','H(attr|pheno)'
				oneSet = []
				for i in range(len(data[key])):
					oneSeries = []
					for j in range(len(data[key][i])):
						oneSeries += [data[key][i][j][feat]]
					oneSet += [oneSeries]
				label = key+'_'+feat
				clusters = kmeans_cluster(params, oneSet,k=k)

				if ylims[key][feat]['min'] == 0:
					ylims[key][feat]['min'] = -0.1 #otherwise overlaps x-axis
				ybounds = [ylims[key][feat]['min']/ylim_buffer,ylims[key][feat]['max']*ylim_buffer]
				plot.clustered_time_series(params,clusters['avgs'],label,CIs=clusters['CIs'],ybounds=ybounds)
				pickled_clusters[key][feat] = {}
				for i in range(k):
					pickled_clusters[key][feat][i] = []
					for j in range(len(pickled_seqs[key])):
						if clusters['cluster_labels'][j]==i:
							pickled_clusters[key][feat][i] += [pickled_seqs[key][j]]

	pickle_file = params['output_dir'] + '/clusters.pickle'
	pickle.dump( pickled_clusters, open( pickle_file, "wb" ) )


def kmeans_cluster(params, series,k=8):
	# series = [[time_series1],...[time_seriesN]]
	from sklearn.cluster import KMeans
	# importing here cause don't want to import sklearn unless abs nec
	#alt: from tslearn.clustering.TimeSeriesKMeans

	clusters = KMeans(n_clusters=k, random_state=None,max_iter=1000).fit(series, sample_weight=None).labels_

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
			CIS += [cluster_CIs]
		elif aCluster != []: #i.e. there is at least 1 series in this cluster
			CIS += [None]
		if aCluster != []:
			avgs += [cluster_avg]

	return {'avgs':avgs, 'CIs':CIS,'cluster_labels':clusters}
	

def parse_sequences(seq_file_pos, seq_file_neg):
	seqs = []
	with open(seq_file_pos,'r') as file:
		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if loop > 1000000:
				sys.exit("Hit an infinite loop, unless file is monstrously huge") 

			seq = []
			was_tab=True
			word=''
			for c in line:
				if c=='\n':
					if word == '':
						seq+=['']
					else:
						seq+=[(word,1)]
					seqs+=[seq]
					break
				elif c=='\t':
					if was_tab:
						seq+=['']
					else:
						if word == '':
							seq+=['']
						else:
							seq+=[(word,1)]
					word=''
				else:
					was_tab=False
					if c in ['\\','/','-']:
						word+='_'
					else:
						word+=c
			loop += 1
	with open(seq_file_neg,'r') as file:
		i = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			if i > 1000000:
				sys.exit("Hit an infinite loop, unless file is monstrously huge") 
			was_tab=True
			word=''
			j=0
			for c in line:
				if c=='\n':
					if word != '':
						assert(seqs[i][j]=='')
						seqs[i][j]=(word,0)
					break
				elif c=='\t':
					if not was_tab and word != '':
						assert(seqs[i][j]=='')
						seqs[i][j]=(word,0)
					word=''
					j+=1
				else:
					was_tab=False
					if c in ['\\','/','-']:
						word+='_'
					else:
						word+=c
			i += 1
	return seqs



def plot_from_pickle(pickle_file,k):
	a_pickle = pickle.load( open( pickle_file, "rb" ) )
	data, params, seqs = a_pickle['data'], a_pickle['params'], a_pickle['seqs']
	clusterplot_time_series(data, params,seqs, k=int(k))


if __name__ == "__main__":
	if len(sys.argv) not in [3,4]:
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
		if len(sys.argv) not in [4]:
			sys.exit("Usage: python3 mult.py pickle_file run_type K")
		plot_from_pickle(sys.argv[1],sys.argv[3])
	else:
		sys.exit("Unrecognized run_type (arguments 3).")