# curr assuming all fns are an AND of their inputs, no nots
# still using cupy
# assumes all init states are premissible
# downside of AND normz is that reqs much larger dtype...
#
# TODO:
#	git
#	make sublime the default
#	not sure how to display when there are many attractors, maybe group into 1 slice?
#	incld nots (expanded net?)
# 	later shift to more complex fns (curr adj normz in PARSE assumes AND)

import sys, os
import parse, basin, plot


def main(param_file):
	params = parse.params(param_file)

	#params = {'num_samples':1000,'max_iters_per_sample': 10, 'exhaustive':0}
	adj, node_num_to_name, node_name_to_num = parse.net(params['net_file'])
	
	if len(adj[0])>10000 and params['exhaustive']:
		sys.exit("Net is far too large to be exhaustively calculating basin")
	
	steady_states = basin.calc_basin_size(params, adj)

	plot.pie(steady_states)



if __name__ == "__main__":
	if not len(sys.argv) == 2:
		sys.exit("Usage: python3 main.py PARAMS.yaml")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	main(sys.argv[1])