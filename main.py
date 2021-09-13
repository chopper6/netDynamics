import sys, os
import parse, basin, plot

def main(param_file, plot_run=True):
	params = parse.params(param_file)
	clause_mapping, node_mapping, num_nodes = parse.net(params)
	# note that num_nodes does not include negative node copies
	catch_errs(params,  clause_mapping, node_mapping, num_nodes)
	attractors = basin.calc_basin_size(params,clause_mapping, node_mapping, num_nodes)
	if plot_run: #can be disabled due to optimization loops, such as timetest
		plot.pie(params,attractors, node_mapping, num_nodes)


def catch_errs(params, clause_mapping, node_mapping, num_nodes):
	if num_nodes>10000 and params['exhaustive']:
		sys.exit("Net is far too large to exhaustively calculate basin, change 'exhaustive' parameter.")


if __name__ == "__main__":
	if len(sys.argv) != 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])