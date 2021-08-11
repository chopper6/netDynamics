import sys, os
import parse, basin, plot


def main(param_file):
	params = parse.params(param_file)
	clause_index, node_num_to_name, node_name_to_num, num_nodes, num_neg_nodes = parse.net(params['net_file'])
	# note that num_nodes does not include negative copies, and num_neg_nodes <= num_nodes
	catch_errs(params, clause_index)
	steady_states = basin.calc_basin_size(params, clause_index,num_nodes, num_neg_nodes)
	plot.pie(params,steady_states,node_num_to_name,num_nodes)


def catch_errs(params, clause_index):
	if clause_index.shape[0]>10000 and params['exhaustive']:
		sys.exit("Net is far too large to be exhaustively calculating basin")

if __name__ == "__main__":
	if not len(sys.argv) == 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])