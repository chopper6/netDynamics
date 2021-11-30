import os, sys, yaml, util, math

CUPY, cp = util.import_cp_or_np(try_cupy=0) #should import numpy as cp if cupy not installed


def params(param_file):
	check_file(param_file,'parameter')
	with open(param_file,'r') as f:
		params = yaml.load(f,Loader=yaml.SafeLoader)

	params_clean_start(params)
	if 'setting_file' in params.keys():
		params= load_model_file(params) # apparently reassigning params within file does not update unless explicitly returned
	params_clean_end(params)

	return params

def check_file(file_path,name):
	if not os.path.isfile(file_path):
		sys.exit("Can't find " + name + " file: " + file_path)
	if os.path.splitext(file_path)[-1].lower() != '.yaml':
		sys.exit(name + " file must be yaml format")

def params_clean_start(params):
	for k in params.keys():
		param_pow(params, k)

	params['parallelism'] = int(max(params['parallelism'],1)) #1 is actually sequential, i.e. run 1 at a time

	CUPY, cp = util.import_cp_or_np(try_cupy=1) #test import
	params['cupy'] = CUPY

def params_clean_end(params):
	actual_num_samples = math.ceil(params['num_samples']/params['parallelism'])*params['parallelism']
	if actual_num_samples != params['num_samples']:
		print('WARNING: Due to parallelism, actual number of samples = ',actual_num_samples)
		params['num_samples'] = actual_num_samples


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

	params_adjust_for_inputs(params)
	return params

def params_adjust_for_inputs(params):
	if 'inputs' in params.keys():
		k = len(params['inputs'])
		actual_num_parallel = math.floor(params['parallelism']/(2**k))*2**k

		if actual_num_parallel < 1:
			sys.exit("\nERROR: inputs are run in parallel, so parallelism parameter must be >= # input combinations!\n")

		if actual_num_parallel!=params['parallelism']:
			params['num_samples'] = params['num_samples']*actual_num_parallel/params['parallelism']
			params['parallelism']=actual_num_samples
			print("\nWARNING: only", str(actual_num_samples),"used to maintain even ratio of input samples on each parallel iteration.\n")

			params['num_samples'] = params['parallelism'] = actual_num_samples



if __name__ == "__main__": # just for debugging purposes
	print("Debugging parse.py")

	sequences('input/efSeq_pos.txt', 'input/efSeq_neg.txt')