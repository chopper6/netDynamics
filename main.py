import sys, os
from net import Net
from copy import deepcopy
import parse, basin, plot, features

# TODO (as of 11/28)
#	use_inputs_in_phenos appears to be broken
#	continue to clear out any and all refs to the 'always off' node
#	and double check that clause_mapping is squared off given correctly, given this

def main(param_file):
	params = parse.params(param_file)
	G  = Net(params,params['net'])
	steadyStates = find_steadyStates(params,G)
	plot.pie(params, steadyStates,G)

def find_steadyStates(params,G): 
	G.build_Fmapd_and_A(params)
	steadyStates = basin.calc_basin_size(params,G)
	return steadyStates
	# TODO: merge this and '_prebuilt' fn, likely by having copy & mutate as a fn in control instead

def find_steadyStates_prebuilt(params, G_orig):
	G = deepcopy(G_orig)
	G.apply_mutations(params) #TODO: this should be moved out...?
	G.build_Fmapd_and_A(params) 
	steadyStates = basin.calc_basin_size(params,G)
	return steadyStates

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])