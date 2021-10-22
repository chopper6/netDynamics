import os, sys, yaml, util, parse
import ctypes
import numpy as np
from subprocess import call


def DNF_via_QuineMcCluskey(net_file, output_file):
	# net file should be in DNF
	# see README for format specifications

	# general approach: 
	# python: parse file, call qm.exe as subprocess
	# c (qm.exe): solve QuineMcCluskey, write to c2py.txt
	# python: read from c2py.txt, write new net file, del c2py.txt
	# c python, caveman approach work best 

	if not os.path.isfile(net_file):
		sys.exit("Can't find network file: " + str(net_file)) 
	
	with open(net_file,'r') as file:
		format_name = file.readline().replace('\n','') # first line is format

		with open(output_file,'w') as ofile:
   			ofile.write(format_name + "\n")

		node_fn_split, clause_split, literal_split, not_str, strip_from_clause, strip_from_node = parse.get_file_format(format_name)

		loop = 0
		while True:
			line = file.readline()
			if not line: #i.e eof
				break
			lineCopy = line #in case just one literal will later just copy back
			line = line.strip().split(node_fn_split)
			node = line[0]

			clauses = line[1].split(clause_split)

			input_nodes = {}	# name -> num
			input_nodes_rev = {}  # num -> name
			unreduced_clauses = []

			for i in range(len(clauses)): # go through once just to build input nodes

				clause = clauses[i]
				clause_int=0
				for symbol in strip_from_clause:
					clause = clause.replace(symbol,'')
				literals = clauses[i].split(literal_split)
				
				for j in range(len(literals)): #build the int min term representation of the clause
					literal_name = literals[j]
					for symbol in strip_from_node:
						literal_name = literal_name.replace(symbol,'')
					if not_str in literal_name:
						literal_name = literal_name.replace(not_str,'')
					if literal_name not in input_nodes.keys():
						input_nodes_rev[len(input_nodes)] = literal_name
						input_nodes[literal_name] = len(input_nodes)
				
			for i in range(len(clauses)): # go through again to build int min term representations of the clauses
				expanded_clause = ['x' for i in range(len(input_nodes))]
				clause = clauses[i]
				clause_int=0
				for symbol in strip_from_clause:
					clause = clause.replace(symbol,'')
				literals = clauses[i].split(literal_split)
				
				for j in range(len(literals)): #build the int min term representation of the clause
					literal_name = literals[j]
					for symbol in strip_from_node:
						literal_name = literal_name.replace(symbol,'')
					if not_str in literal_name:
						sign = '0'
						literal_name = literal_name.replace(not_str,'')
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

			if len(unreduced_clauses) == 1:
				# qm.exe just returns blank, can't reduce anyway, so just put back original line
				with open(output_file,'a') as ofile:
					ofile.write(lineCopy)

			else: # run qm and write reduced function back
				devnull = open(os.devnull, 'w') #just to hide output of subprocess

				cargs = ["./logicSynth/qm.exe", str(len(input_nodes)), str(len(unreduced_clauses))]
				for num in unreduced_clauses:
					cargs += [str(num)]

				call(cargs,stdout=devnull, stderr=devnull)

				reduced_fn = ''
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
						finished_clauses = False
						clause = ''
						if len(strip_from_clause) > 0:
							clause += strip_from_clause[0]
						added_lit = False

						if i!=0 and lits[0] != '-2':
							reduced_fn += clause_split

						for j in range(len(lits)):
							if lits[j] in ['0','1']:
								if added_lit:
									clause += literal_split
								else:
									added_lit = True

							if lits[j] == '0':
								clause += not_str + input_nodes_rev[j]
							elif lits[j] == '1':
								clause += input_nodes_rev[j]
							elif lits[j] == '-2':
								finished_clauses = True
								break
							elif lits[j] != '-1': #note that -1 means don't include the clause
								assert(0) #unrecognized digit

						if not finished_clauses:
							if len(strip_from_clause) > 0:
								clause += strip_from_clause[1]
							reduced_fn += clause 
						else:
							break #once reach a clause with a -2, done
						i+=1


				with open(output_file,'a') as ofile:
				   	ofile.write(node + node_fn_split + reduced_fn + '\n')

				#print('resulting in:',node + node_fn_split + reduced_fn)
				
			if loop > 1000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 
			loop+=1

	os.remove('c2py.txt') #clean up


if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 reduce.py input_net_file output_net_file")
	
	DNF_via_QuineMcCluskey(sys.argv[1],sys.argv[2])