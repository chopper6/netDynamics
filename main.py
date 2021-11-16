import sys, os
import parse, basin, plot, features

def main(param_file):
	params = parse.params(param_file)
	attractors,phenos, node_mapping = find_attractors(params)
	#feats = features.calc_entropy(params,attractors)
	#print(feats)
	plot.pie(params,attractors,phenos, node_mapping)
	print(node_mapping)



def find_attractors(params):
	clause_mapping, node_mapping = parse.net(params)
	attractors, phenos = basin.calc_basin_size(params,clause_mapping, node_mapping)
	# attractors is a dict {} indexed by the steady state string (or "oscillates")
	#	each element of attractors[i] = {size:% of initial states, pheno:subset of label corresponding to output nodes}
	return attractors,phenos, node_mapping


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])