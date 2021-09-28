import main, parse,plot
import sys, os, itertools, datetime


def input_set(param_file):
	params = parse.params(param_file)
	num_inputs = len(params['phenos']['inputs'])

	params['output_dir'] = os.path.join(os.getcwd()+'/'+params['output_dir'], datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
	print(params['output_dir'])
	os.makedirs(params['output_dir'])

	params['savefig'] = True # i assume you don't want an image popping up for every combination of inputs


	input_sets = itertools.product([0,1], repeat=num_inputs) 
	if params['verbose']:
		print("\nRunning",2**num_inputs,'input combinations.')
	j=1
	for input_set in input_sets:
		if params['verbose']:
			print("\n~~~ Starting input set #",j,'~~~\n')
		label=''
		for i in range(num_inputs):
			input_node = params['phenos']['inputs'][i]
			params['phenos']['init'][input_node] = input_set[i]
			if i!=0:
				label+='_'
			label+=input_node +str(input_set[i])
		attractors, node_mapping = main.find_attractors(params)
		plot.pie(params,attractors, node_mapping,external_label=label)
		j+=1




if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 mult.py PARAMS.yaml run_type")
	if not os.path.isfile(sys.argv[1]):
		sys.exit("Can't find parameter file: " + sys.argv[1])
	if os.path.splitext(sys.argv[1])[-1].lower() != '.yaml':
		sys.exit("Parameter file must be yaml format")
	
	if sys.argv[2]=='inputs':
		input_set(sys.argv[1])
	else:
		sys.exit("Unrecognized run_type (arguments 3).")