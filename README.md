
# Simulating Boolean Networks to Infer Phenotype Basins 


Contents:
1. Requirements
2. Basic Usage 
3. Advanced Usage
4. Network File
5. Algorithm 
6. Fumia Network
7. To Do


## 1. Requirements

Python3 and the relevant python libraries must be installed. The main required python libraries are matplotlib and numpy.

Instead of numpy, cupy can optionally be installed. This library provides GPU computing using numpy syntax, with a speedup of roughly 20-50x. It requires that CUDA drivers are installed. Note that it works best with Intel GPUs, and is still in beta for AMD GPUs. If both numpy and cupy are installed, cupy can be forced off by changing the top of util.py or lap.py.


## 2. Basic Usage

The program calculates the attractors and their basin sizes of a given network.

Running a single simulation:
	`python3 main.py PARAMS.yaml`

The parameter file is a YAML file, which is like JSON, but with more minimal syntax and allows comments. See params.yaml for an example and for an explanation of each parameter. The parameter file also specifies the input network file and an optional phenotype file. 

If the parameter savefig is false, then an output image will pop up. Otherwise, the output will appear in the output/ directory.

If you just want the attractors and their basin sizes, run:
	`params = parse.params(param_file)`
	`attractors, node_mapping = main.find_attractors(params)`


## 3. Advanced Usage

The phenotype file specifies the following nodes: `outputs` groups attractors into phenotypes, `statics` always begin in the specified state, `inputs` can be iterated over using `multi.py`. See `inputs/fumiaPhenos.txt` for an example. 

If you want to run all combinations of inputs specified in a phenotype file:
	`python3 multi.py PARAMS.yaml inputs`
	Note that this will make a seperate timestamped folder in the output directory to place the input files.  

Timing a simulation:  
	`python3 opt.py PARAMS.yaml time`
	This will time the current simulation. opt.py is a work in progress that will eventually aid parameter selection to optimize running time.


## 4. Network File

The first line of the network file must specify the encoding. There are two options: DNFwords or DNFsymbolic

DNFwords:
One node per line. Each node is separated from its function by "\*= ", each clause separated by " or ", and each element by ' and '. Negative is represented by 'not '. Spaces do matter. Clauses can be optionally enclosed by brackets. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X*= (A and B) or (not C)"`. See `inputs/fumia.txt` file for a complete example.

DNFsymbol syntax:
One node per line. Each node is separated from its function by a tab, each clause separated by a space, and each element by '&'. Negative is represented by '-'. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X\tA&B -C"`. See `inputs/simpleNet.txt` file for a complete example.


## 5. Algorithm 

This program runs a network many times to estimate the attractor landscape from many initial conditions. Parts of the run are grouped into "laps". Each lap is broken into "steps". Each step is one activation of the network function. Samples that do not finish after 1 lap are assumed to be oscillating. These samples are rerun until their oscillation period is found.  

A matrix is required for GPU/numpy applications. Especially for biological networks, there tend to be a few nodes with an overwhelming number of clauses. To balance the workload, all clauses are generated. They are then iteratively merged and mapped to the corresponding nodes. The approach is heavily dependent on disjunctive normal form. 

Currently estimating the attractor landscape of the fumia network (103 nodes) using 10^5 samples takes less than 20 seconds using cupy on a local desktop computer.


## 6. Fumia Network

The original Fumia network has been modified by adding nodes for cell cycle checkpoints, and proliferation and quiescence (look at the last few lines of `inputs/fumia.txt` for details). Adding these nodes explicitly to the model automatically outputs the phenotype, even under different node update schemes, ect. 

At the moment, most of the behavior in the original Fumia network is faithfully replicated. All input conditions lead to at least 96% of attractors in the same phenotype. However, the exact proportion of apopotic, proliferative and quiescent attractors is somewhat different than the original paper. This could be due to how the proliferation node is defined. The are also noticeably more attractors (about 300) relative to the original Fumia paper (about 60).


## 7. To Do
- entropy measurements
- Generalized async update option
- Faster algorithm: logscaled bins for merging clauses
- add control support
- add population support: network copies may have different topology
- add dynamic topology support: topology can change between laps
- add PBNs, including ergodic attractor sets
