import sys, os
import parse, basin, plot, features

def main(param_file):
	params = parse.params(param_file)
	steadyStates, V = find_attractors(params)
	#feats = features.calc_entropy(params,attractors)
	#print(feats)
	plot.pie(params, steadyStates, V)


def find_attractors(params):
	F, F_mapd, A, V  = parse.net(params)
	steadyStates = basin.calc_basin_size(params,F_mapd,V)
	# attractors is a dict {} indexed by the steady state string (or "oscillates")
	#	each element of attractors[i] = {size:% of initial states, pheno:subset of label corresponding to output nodes}
	return steadyStates, V

def find_attractors_prebuilt(params, F, V):
	parse.apply_mutations(params,F) #TODO: this could be moved out...
	F_mapd, A = parse.get_clause_mapping(params, F, V) 
	steadyStates = basin.calc_basin_size(params,F_mapd,V)
	return steadyStates

if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])