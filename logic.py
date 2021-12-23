import os, sys, math
import net
from subprocess import run


def DNF_via_QuineMcCluskey(G,parity=False):
	# exhaustively reduces the logic of a net G
	# if parity is true, will explicitly build negative nodes

	# takes a network object G

	# general approach: 
	# python: parse file, call qm.exe as subprocess
	# c (qm.exe): solve QuineMcCluskey, write to c2py.txt
	# python: read from c2py.txt, write new net file, del c2py.txt
	# c python, caveman approach work best
	
	if parity:
		negatives_bin = []

	for node in G.nodes:
		input_nodes = {}	# name -> num
		input_nodes_rev = {}  # num -> name
		unreduced_clauses = []

		for clause in node.F():
			for lit in clause:
				literal_name = str(lit)
				if G.not_string in literal_name:
					literal_name = literal_name.replace(G.not_string,'')
				if literal_name not in input_nodes.keys():
					input_nodes_rev[len(input_nodes)] = literal_name
					input_nodes[literal_name] = len(input_nodes)
		
		for clause in node.F(): # go through again to build int min term representations of the clauses
			expanded_clause = ['x' for i in range(len(input_nodes))]
			clause_int=0
			
			for lit in clause: #build the int min term representation of the clause

				literal_name = str(lit)
				if G.not_string in literal_name:
					sign = '0'
					literal_name = literal_name.replace(G.not_string,'')
				else:
					sign = '1'
				expanded_clause[input_nodes[literal_name]]=sign

			expanded_clauses = [''.join(expanded_clause)] #if a literal is not in a string, need to make two string with each poss value of that literal	
			final_expanded_clauses = []
			while expanded_clauses != []:
				for j in range(len(input_nodes)):
					for ec in expanded_clauses:
						if 'x' not in ec:
							final_expanded_clauses += [ec]
							expanded_clauses.remove(ec)
						elif ec[j]=='x':
							expanded_clauses+= [ec[:j] + "1" + ec[j+1:]]
							expanded_clauses+= [ec[:j] + "0" + ec[j+1:]]
							expanded_clauses.remove(ec)

			for ec in final_expanded_clauses:
				unreduced_clauses += [int('0b'+ec,2)]

		unreduced_clauses = list(set(unreduced_clauses)) #remove duplicates

		if parity:
			neg_clauses = [i for i in range(2**len(input_nodes))]
			for term in unreduced_clauses:
				neg_clauses.remove(term)
			negNodeName = G.not_string + node.name
			negatives_bin += [{'clauses':neg_clauses,'node':negNodeName,'num_inputs':len(input_nodes),'input_nodes_rev':input_nodes_rev}]

		if len(unreduced_clauses) != 1: # run qm and write reduced function back
			fn = run_qm(unreduced_clauses, len(input_nodes),G.encoding,input_nodes_rev)
			node.setF(fn)

	if parity:
		for neg_dict in negatives_bin:
			node = G.nodesByName(neg_dict['node'])

			if len(neg_dict['clauses']) == 1:
				# qm.exe just returns blank, can't reduce anyway, so just put back original line
				bool_str = int2bool(int(neg_dict['clauses'][0]), neg_dict['num_inputs'])
				finished_clauses, clause = bool2clause(bool_str, neg_dict['input_nodes_rev'], G.encoding)
				node.setF([clause]) 
			
			else: # run qm and write reduced function back
				fn = run_qm(neg_dict['clauses'], neg_dict['num_inputs'], G.encoding, neg_dict['input_nodes_rev'])
				node.setF(fn)


def int2bool(x,lng):
	bool_str = ''
	while x>1:
		bool_str+=str(x%2)
		x=math.floor(x/2)
	bool_str +=str(x%2)
	for i in range(lng-len(bool_str)):
		bool_str+='0'
	return ''.join(reversed(bool_str))


def bool2clause(bool_str, input_nodes_rev, format_name):
	# this is very specific to the syntax used for the c program
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = net.get_file_format(format_name)
	
	clause = []

	for j in range(len(bool_str)):
		finished_clauses = False

		if bool_str[j] == '0':
			clause += [not_str + input_nodes_rev[j]]
		elif bool_str[j] == '1':
			clause += [input_nodes_rev[j]]
		elif bool_str[j] == '-2':
			finished_clauses = True
			break
		elif bool_str[j] != '-1': #note that -1 means don't include the clause
			assert(0) #unrecognized digit

	return finished_clauses, clause




def run_qm(unreduced_clauses, num_inputs, format_name, input_nodes_rev):
	# return function of a single node: fn = [reduced_clause1, reduced_clause2,...] where reduced_clause = [lit1,lit2,...]

	# calls a c logic function, MUST build the exe first
	
	# the following two files hold output from calling the c file qm.exe
	with open('./logicSynth/qm_stdout.log','w') as c2py:
		pass 
	with open('./logicSynth/qm_stderr.log','w') as c2py:
		pass 

	# this is only used internally and will be deleted at the end
	with open('c2py.txt','w') as c2py: 
		pass 

	node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = net.get_file_format(format_name)
	devnull = open(os.devnull, 'w') #just to hide output of subprocess 

	cargs = ["./logicSynth/qm.exe", str(num_inputs), str(len(unreduced_clauses))]
	for num in unreduced_clauses:
		cargs += [str(num)]

	with open('./logicSynth/qm_stdout.log','w') as stdout:
		with open('./logicSynth/qm_stderr.log','w') as stderr:
			run(cargs,stdout=stdout, stderr=stderr) 

	fn = []
	with open('c2py.txt','r') as c2py:
		# result is 1 clause per line, comma sep for each input
		# 1 = ON in clause, 0 = OFF in clause, -1 = not in clause, -2 = done
		i = 0
		while True:
			line = c2py.readline().strip()
			if not line: #i.e eof
				break
			if i > 1000000:
				sys.exit("Hit an infinite loop, unless file is monstrously huge") 

			lits = line.split(",")

			finished_clauses, clause = bool2clause(lits, input_nodes_rev, format_name)
			if clause!=[]:
				fn += [clause]
			if finished_clauses:
				break #once reach a clause with a -2, done
			i+=1

	os.remove('c2py.txt') #clean up
	return fn



def DNF_from_cell_collective_TT(folder, output_file):

	format_name = 'bnet'
	node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = net.get_file_format(format_name)
	
	with open(output_file,'w') as ofile:
		pass # write it in case does not exist

	for filename in os.listdir(folder):
		if '.csv' in filename:
			with open(folder + '/' + filename,'r') as file:

				line = file.readline().replace(' ','')
				inputs = line.split(',')
				node = inputs[-1].strip()
				inputs=inputs[:-1]
				inputs_rev = {i:inputs[i] for i in range(len(inputs))}
				clauses = []
				while True:
					line = file.readline()
					if not line: #i.e eof
						break
					line = line.strip()
					if line[-1] == '1': #only care about ON terms
						line = line[:-1].replace(' ','')
						line=line.replace(',','')
						clauses += [int('0b'+line,2)]

				assert(clauses != []) # should not always be off
				fn = run_qm(clauses, len(inputs), format_name, inputs_rev)

				with open(output_file,'a') as ofile:
					ofile.write(node + node_fn_split + fn_str(fn,format_name) + '\n')


if __name__ == "__main__":
	if len(sys.argv) !=4:
		sys.exit("Usage: python3 logic.py input_net_file output_net_file [reduce, parity, tt]")
	
	if sys.argv[3]=='tt':
		DNF_from_cell_collective_TT(sys.argv[1],sys.argv[2])
	else:
		G = net.Net(model_file=sys.argv[1],debug=True)
		if sys.argv[3]=='reduce':
			parity = False
		elif sys.argv[3]=='parity':
			parity =True
		else:
			sys.exit("Unrecognized 3rd arg. Usage: python3 logic.py input_net_file output_net_file [reduce, parity, tt]")
		
		DNF_via_QuineMcCluskey(G,parity=parity)
		G.write_to_file(sys.argv[2],parity=parity)