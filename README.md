
# Simulating Boolean Networks to Infer Basin Size 

Contents:
1. Requirements
2. Usage 
3. Network File
4. Algorithm 
5. To Do


## 1. Requirements

Python3 and the relevant python libraries must be installed. The main required python libraries are matplotlib and numpy.

Instead of numpy, cupy can optionally be installed. This library provides GPU computing using numpy syntax, with a speedup of roughly 20-50x. It requires that CUDA drivers are installed. Note that it works best with Intel GPUs, and is still in beta for AMD GPUs. If both numpy and cupy are installed, cupy can be forced off by changing the top of util.py or lap.py.


## 2. Usage

The program calculates the attractors and their basin sizes of a given network.

Running a single simulation:
	`python3 main.py params/PARAM.yaml`

The parameter file is a YAML file, which is like JSON, but with more minimal syntax and allows comments. See params.yaml for an example and for an explanation of each parameter. The parameter file also specifies the input network file and an optional phenotype file. 

The phenotype file specifies which nodes are output nodes and groups attractors accordingly. See `inputs/fumiaPhenos.txt` for an example. 

If the parameter savefig is false, then an output image will pop up. Otherwise, the output will appear in the output/ directory.

Timing a simulation:  
	`python3 opt.py params/PARAM.yaml time`
	This will time the current simulation. opt.py is a work in progress that will eventually aid parameter selection to optimize running time.


## 3. Network File

The first line of the network file must specify the encoding. There are two options: DNFwords or DNFsymbolic

DNFwords:
One node per line. Each node is separated from its function by "\*= ", each clause separated by " or ", and each element by ' and '. Negative is represented by 'not '. Spaces do matter. Clauses can be optionally enclosed by brackets. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X*= (A and B) or (not C)"`. See `inputs/fumia.txt` file for a complete example.

DNFsymbol syntax:
One node per line. Each node is separated from its function by a tab, each clause separated by a space, and each element by '&'. Negative is represented by '-'. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X\tA&B -C"`. See `inputs/simpleNet.txt` file for a complete example.


## 4. Algorithm 

This program runs a network many times to estimate the attractors landscape. Typically many samples are generated. Parts of the run are grouped into "laps". After each lap the number samples that have finished is checked, others are assumed to be oscillating or are allowed to run again the next lap. Each lap is broken into "steps". Each step is one activation of the network function.  

A matrix is required for GPU/numpy applications. Especially for biological networks, there tend to be a few nodes with an overwhelming number of clauses. To balance the workload, all clauses are generated. They are then iteratively merged and mapped to the corresponding nodes. The approach is heavily dependent on disjunctive normal form. 

Currently estimating the attractor landscape of the fumia network (98 nodes) using 10^6 samples and running each sample for at most 10^4 steps (before assuming it is oscillating), takes about 5 minutes using cupy on a local desktop computer.



## 5. To Do
- entropy measurements
- Distinguish short oscillations as steady states
- Generalized async update option
- Faster algorithm: logscaled bins for merging clauses
- add control support
- add population support: network copies may have different topology
- add dynamic topology support: topology can change between laps
- add PBNs, including ergodic attractor sets
