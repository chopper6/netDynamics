import os, sys, yaml, util, parse
import ctypes
import numpy as np
from subprocess import call

# need to check the for loop around line66, since curr fumia.txt doesn't really use it

QM_SOLVER = ctypes.WinDLL('./logicSynth/qm.dll') 
# if you recompile the dll from logicSynth/qm.c, you may need to change the path
# current qm.dll was built on Windows 10
# replace WinDLL with CDLL for linux/mac i think

def DNF_via_QuineMcCluskey(net_file, output_file):
	# net file should be in DNF
	# see README for format specifications

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
			line = line.strip().split(node_fn_split)
			node = line[0]

			clauses = line[1].split(clause_split)

			input_nodes = {}	# name -> num
			input_nodes_rev = {}  # num -> name
			unreduced_clauses = []

			for i in range(len(clauses)):

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
				
				expanded_clause = ['x' for i in range(len(input_nodes))]
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
				for j in range(len(input_nodes)):
					for ec in expanded_clauses:
						if ec[j]=='x':
							expanded_clauses+= [ec[:j] + "1" + ec[j+1:]]
							ec[j] = ec[:j] + "0" + ec[j+1:]

				for ec in expanded_clauses:
					unreduced_clauses += [int('0b'+ec,2)]


			unreduced_clauses = [3,1]
			input_nodes = [1,2]
			print('\npassing unreduced clauses:',unreduced_clauses)
			#if unreduced_clauses == [0]:
			#	result = ['0'] #for some reason doesn't handle this case
			#else:

			# TODO
			# handle case of a single clause
			devnull = open(os.devnull, 'w') #just to hide output of subprocess

			cargs = ["./logicSynth/qm.exe", str(len(input_nodes)), str(len(unreduced_clauses))]
			for num in unreduced_clauses:
				cargs += [str(num)]
			print(cargs)
			call(cargs,stdout=devnull, stderr=devnull)
			# read from c2py.txt
			# parse reduced clauses, append to reduced network
			# del py2c.txt & c2py.txt

			reduced_fn = ''
			with open('c2py.txt','r') as file:
				# result is 1 clause per line, comma sep for each input
				# 1 = ON in clause, 0 = OFF in clause, -1 = not in clause, -2 = done
				i = 0
				while True:
					line = file.readline().strip()
					if not line: #i.e eof
						break
					if i > 1000000:
						sys.exit("Hit an infinite loop, unless file is monstrously huge") 

					lits = line.split(",")
					finished_clauses = False
					clause = ''
					# TODO: add node seperators ect
					for j in range(len(lits)):
						if lits[j] == '0':
							clause += not_str + input_nodes_rev[j]
						elif lits[j] == '1':
							clause += input_nodes_rev[j]
						elif lits[j] == '-2':
							finished_clauses = True
						elif lits[j] != '-1':
							assert(0) #unrecognized digit

					if not finished_clauses:
						reduced_fn += clause # TODO: also need to add delimiter
					i+=1
					#TODO: write to output file, del c2py.txt

			assert(0)
			# TODO: del after this


			#result = qm.qm(ones=unreduced_clauses)
			# result is of the form: ['10x0','xx11','1110'] 
			reduced_clauses = ''
			j=0
			for clause_str in result:
				if j>0:
					reduced_clauses += clause_split
				if len(strip_from_clause) > 0:
					reduced_clauses += strip_from_clause[0]

				skipped,started=False,False
				for i in range(len(clause_str)):
					char = clause_str[i]
					if i!=0 and started and char!='-1': # LEFT OFF HERE: if 0X1 need to not skip btwn the 0 and 1 
						reduced_clauses += literal_split
					skipped=False
					if char == '0':
						reduced_clauses += not_str + input_nodes_rev[i]
						started=True
					elif char == '1':
						reduced_clauses += input_nodes_rev[i]
						started=True
					elif char == '-1':
						skipped=True
					elif char == '-2':
						break #done with clause, empty marker
					else:
						assert(False) #should only get strings of 0,1,-1,-2

				if len(strip_from_clause) > 0:
					reduced_clauses += strip_from_clause[1]
				j+=1

			with open(output_file,'a') as ofile:
			   	ofile.write(node + node_fn_split + reduced_clauses + '\n')

			print('resulting in:',node + node_fn_split + reduced_clauses)
			if loop > 1000000:
				sys.exit("Hit an infinite loop, unless net is monstrously huge") 
			loop+=1



if __name__ == "__main__":
	if len(sys.argv) != 3:
		sys.exit("Usage: python3 reduce.py input_net_file output_net_file")
	
	DNF_via_QuineMcCluskey(sys.argv[1],sys.argv[2])