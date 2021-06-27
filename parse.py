import os, yaml, util
CUPY, cp = util.import_cp_or_np(try_cupy=1) #should import numpy as cp if cupy not installed


def params(param_file):
	with open(param_file,'r') as f:
		params = yaml.load(f,Loader=yaml.FullLoader)
	return params


def net(net_file):
	node_name_to_num, node_num_to_name, edgelist = {},{},{}

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	with open(net_file,'r') as file:
		i,n=0,0
		while True:
		 
			line = file.readline()
		 
			if not line: #i.e eof
				break
			if i > 10000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 

			line = line.strip().split('\t')
			node_name_to_num[line[0]] = n
			node_num_to_name[n] = line[0]
			edgelist[n] = []
			for j in range(1,len(line)):
				edgelist[n] += line[j]

			i += 1
			n += 1

	adj = cp.zeros((n,n))
	for source in range(n):
		deg = len(edgelist[source])
		for j in range(deg):
			target = node_name_to_num[edgelist[source][j]]
			adj[source][target] = 1/deg  		# THIS NORMZ IS SPC BC NODE OPS = AND

	return adj, node_num_to_name, node_name_to_num