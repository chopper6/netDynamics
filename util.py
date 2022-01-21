import time, os, math, pickle, warnings
from datetime import datetime, date

TURN_CUPY_OFF_OVERRIDE = 0 #turns off CUPY for everything
if (TURN_CUPY_OFF_OVERRIDE):
	warnings.warn("Cupy is off, see util.py")



def istrue(d,keys):
	# searches if dictionary d[key0][key1]...[keyn] is True
	if isinstance(keys, list):
		curr_dict = d 
		for k in keys:
			if not k in curr_dict.keys():
				return False
			else:
				curr_dict = curr_dict[k]
		return curr_dict
	else: # assume there is only 1 key then
		return keys in d and d[keys]

def get_node_dtype(params):
	if istrue(params,['PBN','active']) and params['PBN']['init'] == 'half':
		return float 
	else:
		return 'bool' 

def import_cp_or_np(try_cupy = True, np_warnings = False):
	if try_cupy and not TURN_CUPY_OFF_OVERRIDE:
		try:
			import cupy as cp
			CUPY = True
		except ImportError:
			import numpy as cp
			CUPY = False
			if not np_warnings:
				import warnings
				warnings.filterwarnings("ignore")
	else:
		import numpy as cp
		CUPY = False
		if not np_warnings:
			import warnings
			warnings.filterwarnings("ignore")

	return CUPY, cp


CUPY, cp = import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed
# need to import here for copy_to_larger_dim

def copy_to_larger_dim(x, num_copies):
	# array will be 1 dim larger, with num_copies of the original array
	# for example, copying array of shape (4,5) with 3 copies would result in an array of shape (3,4,5)
	# could also try cp.newaxis?
	return cp.tile(x,num_copies).reshape(tuple([num_copies])+x.shape)


###### FROM LIGHT SIMULATION #########



def true_in_dict(d,key):
	if key in d and d[key]:
		return True
	else:
		return False


def none(x):
	if x in [None,'none','None',0,'0','False',False,'',' ']:
		return 1
	else:
		return 0

def bool(x):
	if x in [0,'0','False',False,'false','unuh','noway','gtfofh']:
		return False
	elif x in [1,'1','True',True,'true','yeaya','fosho','nodoubt']:
		return True

def timestamp():
	now = datetime.now()
	curr_date = str(date.today()).strip('2020-')
	curr_time = str(datetime.now().strftime("%H-%M-%S"))
	tstamp = curr_date+'_'+curr_time
	return tstamp

def pickle_it(data, file):
	with open(file,'wb') as f:
		pickle.dump(data,f)

def load_pickle(file):
	with open(file, 'rb') as f:
		data = pickle.load(f)
	return data

def rng(x):
	return range(len(x))

def avg(x):
	return sum(x)/len(x)

def avg_by_key(X,key):
	summ = 0
	for x in X:
		summ += x[key]
	return summ/len(x)

def var_by_key(X,key):
	varr = 0
	the_avg = avg_by_key(X,key)
	for x in X:
		varr += math.pow(the_avg-x[key],2)
	return varr/(len(x)-1) 

def var(x,mean=None):
	# calculates sample variance
	if mean is None: mean = avg(x)
	var = sum([math.pow(mean-x[i],2) for i in rng(x)])/ (len(x)-1)
	return var


def check_build_dir(dirr):
	if not os.path.exists(dirr):
		print("\nCreating new directory for output at: " + str(dirr) + '\n')
		os.makedirs(dirr)

def safe_div_array(A,B):
	# a is numerator, b is divisor
	assert(len(A) == len(B))
	z=[]
	for i in rng(A):
		if B[i] == 0: z+=[0]
		else: z+=[A[i]/B[i]]
	return z


def get_timestamp():
	now = datetime.now()
	curr_date = str(date.today()).strip('2020-')
	curr_time = str(datetime.now().strftime("%H-%M-%S"))
	tstamp = curr_date+'_'+curr_time
	return tstamp