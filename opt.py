# for now just manual optimization: time code with few different parameters
# later hopefully automatic
# also has not been updated in some time

import basin, control
import sys, os
from timeit import default_timer as timer

# TODO: add some basic optimizer for discrete params, like a GA
# also print memory info using array.nbytes

def timetest(param_file, print_output=False):
	params, G = basin.init(param_file)
	#params['verbose'] = 0 # you really want to hear it every loop?
	reps = 2
	tstart = timer()
	for r in range(reps):
		steadyStates = basin.calc_basin_size(params,G)
		print("\n~~~TIMING CODE: Finished rep ",r+1,'~~~\n')
	tend = timer()

	avg_time = round((tend-tstart)/(60*reps),3)

	print("\n\n\nAverage execution time = ", avg_time,"minutes.\n")
	return avg_time

def timetest_control(param_file, print_output=False):
	params = parse.params(param_file) #for now parsing is not timed
	#params['verbose'] = 0 # you really want to hear it every loop?
	reps = 2
	tstart = timer()
	for r in range(reps):
		control.randomize_experiment(param_file) 
		print("\n\n~~~TIMING CODE: Finished rep ",r+1,'~~~\n')
		tmid = timer()
		print("so far at time=",round((tmid-tstart)/(60*reps),3),'min.\n\n\n')
	tend = timer()

	avg_time = round((tend-tstart)/(60*reps),3)

	print("\n\n\nAverage execution time = ", avg_time,"minutes.\n")
	return avg_time

if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 opt.py PARAMS.yaml optimization_type")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	if sys.argv[2]=='time':
		timetest(sys.argv[1], print_output=True)	
	elif sys.argv[2]=='timeControl':
		timetest_control(sys.argv[1], print_output=True)
	else:
		sys.exit("Unrecognized optimization_type (arguments 3).")