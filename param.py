import os, sys, yaml, util, math

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed

def load(param_file):
	check_file(param_file,'parameter')
	with open(param_file,'r') as f:
		params = yaml.load(f,Loader=yaml.SafeLoader)

	clean(params)
	if 'setting_file' in params.keys():
		params= load_model_file(params) # apparently reassigning params within file does not update unless explicitly returned

	assert(params['fraction_per_lap']==1) # poss odd behavior otherwise...spc in async style sims...debug before using

	return params

def check_file(file_path,name):
	if not os.path.isfile(file_path):
		sys.exit("Can't find " + name + " file: " + file_path)
	if os.path.splitext(file_path)[-1].lower() != '.yaml':
		sys.exit(name + " file must be yaml format")

def clean(params):
	for k in params.keys():
		param_pow(params, k)

	params['parallelism'] = int(max(params['parallelism'],1)) #1 is actually sequential, i.e. run 1 at a time

	CUPY, cp = util.import_cp_or_np(try_cupy=1) #test import
	params['cupy'] = CUPY

	if util.istrue(params,['PBN','float_update']):
		assert(params['PBN']['active']) # float_update requires PBN
		assert(params['update_rule']=='Gasync') # float_update only makes sense for Gasync


def param_pow(params,k):
	# yaml doesn't maintain json's 10e5 syntax, so here is support for scientific notation. Syntax: 10^5
	if isinstance(params[k], str) and '^' in params[k]:
		parts = params[k].split('^')
		params[k] = int(parts[0])**int(parts[1])

def load_model_file(params):
	if not os.path.isfile(params['setting_file']):
		sys.exit("Can't find model_file: " + params['setting_file'] + ', check path in parameter file.')
	if os.path.splitext(params['setting_file'])[-1].lower() != '.yaml':
		sys.exit("'setting_file' must be yaml format.")
	
	with open(params['setting_file'],'r') as f:
		model = yaml.load(f,Loader=yaml.SafeLoader)

	if params['debug']:
		shared_items = {k: params[k] for k in params if k in model}
		assert(len(shared_items)==0) #should not have overlaping keys between params and model files
	params = {**model, **params} # in python3.9 can use params_model | params, but may require some people to update python

	adjust_for_inputs(params)
	return params

def adjust_for_inputs(params):
	if 'inputs' in params.keys():
		k = len(params['inputs'])
		actual_num_parallel = round(params['parallelism']/(2**k))*2**(k)

		if params['parallelism'] < 2**k:
			sys.exit("\nERROR: parallelism parameter must be >= # input combinations, since inputs are run in parallel!\n")

		if actual_num_parallel!=params['parallelism']:
			print("\nWARNING: parallelism set to",actual_num_parallel,' for balanced input samples on each parallel iteration.')
			params['parallelism']=actual_num_parallel

		actual_num_samples = round(params['num_samples']/params['parallelism'])*params['parallelism']

		if params['num_samples'] != actual_num_samples:
			print("WARNING: num_samples set to", actual_num_samples,"for full parallelism each iteration.\n")
			params['num_samples'] = actual_num_samples



if __name__ == "__main__": # just for debugging purposes
	print("Debugging parse.py")

	sequences('input/efSeq_pos.txt', 'input/efSeq_neg.txt')