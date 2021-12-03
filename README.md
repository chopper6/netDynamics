
# Monte Carlo Simulation of Boolean Networks 

Contents:
1. Requirements
2. Basic Usage 
3. Advanced Usage
4. Control
5. Logic Minimization
6. Network File
7. Basin Algorithm 
8. Models

## 1. Requirements

Python3 and the relevant python libraries must be installed. The main required python libraries are matplotlib, numpy, and pyyaml. There may be other libraries as well, please look at the error logs.

Instead of numpy, cupy can optionally be installed. This library provides GPU computing using numpy syntax, with a speedup of roughly 20-50x. It requires that CUDA drivers are installed. Note that it works best with Intel GPUs, and is still in beta for AMD GPUs. If both numpy and cupy are installed, cupy can be forced off by changing the top of util.py.


## 2. Basic Usage

The program calculates the attractors and their basin sizes of a given network.

Running a single simulation:
	`python3 main.py params/basic.yaml`

The *parameter file* is any YAML file with the required arguments. YAML is like JSON, but with more minimal syntax and allows comments. See params/basic.yaml for a minimal example and parameter explanations.

The parameter file also specifies a *model file*. The model should be a logical boolean model, typically in the form of a *.bnet* file (following the format used by pyboolnet).  


## 3. Advanced Usage

The parameter file can optionally specify a `settings` file. This gives more detail about the model used. For instance, it can specify *input* and *output* nodes, which are then used to group attractors into phenotypes. See `settings/toy.yaml` for an example and more details on what can be specified.

Timing a simulation:  
	`python3 opt.py PARAMS.yaml time`
	This will time the current simulation. opt.py is a work in progress that will some day (hopfully) aid parameter selection to optimize running time.

If you just want the steady states for some custom code, run:
	`params = parse.params(param_file)`
	`G = net.Net(params)`
	`steadyStates = basin.find_steadyStates(params,G)`
See the SteadyStates class object in basin.py for details.

## 4. Control

To exhaustively find all nodes whose mutation changes the phenotype basins over a threshold: 

Usage: `python3 control.py PARAMS MUT_THRESH CNT_THRESH`

Where `MUT_THRESH` = minimum distance between wild-type and mutated phenotypes for the mutation to be considered relevant.
	`CNT_THRESH` = minimum distance between the mutated and control phenotypes for the control to be considered successful. 

To find the minimum recommended values for `MUT_THRESH` and `CNT_THRESH`, run:
	`python3 control.py PARAMS tune`


## 5. Logic Minimization
Networks can be exhaustively reduced to their minimal DNF form using a C program.
The original Quine-McCluskey implementation in C is done by use bp274: https://github.com/bp274/Tabulation-method-Quine-McCluskey-
Note that `qm.exe` is machine dependent and must be compiled by the user from `logicSynth/qm.c`

To reduce the logic of an existing network: `python3 logic.py INPUT_FILE OUTPUT_FILE reduce`
`INPUT_FILE` is the existing network, which will be reduced and written to `OUTPUT_FILE`

To build the expanded network representation of an existing network: `python3 logic.py INPUT_FILE OUTPUT_FILE exp`
The expanded network includes explicit logical functions for composite nodes.

To build a network from a directory of Cell Collective truth tables: `python3 logic.py INPUT_DIR OUTPUT_FILE tt`
This make one node per CSV file given. Manually take all *named* (not numbered) truth tables and put into INPUT_DIR. Exclude the SPECIES_KEY.csv file. It seems that the input nodes need to be added manually afterwards, by referring to external_components.ALL.txt.



## 6. Network File

The first line of the network file must specify the encoding. There are three options: .bnet, DNFwords or DNFsymbolic.

.bnet:
This format is recommended and may be the only support form at some later time. See pyboolnet package for syntax specifications. 

DNFwords:
One node per line. Each node is separated from its function by "\*= ", each clause separated by " or ", and each element by ' and '. Negative is represented by 'not '. Spaces do matter. Clauses can be optionally enclosed by brackets. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X*= (A and B) or (not C)"`. See `inputs/fumia.txt` file for a complete example.

DNFsymbol syntax:
One node per line. Each node is separated from its function by a tab, each clause separated by a space, and each element by '&'. Negative is represented by '-'. 
For instance: if node X = (A and B) or (not C), then it's line would be `"X\tA&B -C"`. See `inputs/simpleNet.txt` file for a complete example.


## 7. Basin Algorithm 

This program runs a network many times to estimate the attractor landscape from many initial conditions. Parts of the run are grouped into "laps". Each lap is broken into "steps". Each step is one activation of the network function. Samples that do not finish after 1 lap are assumed to be oscillating. These samples are rerun until their oscillation period is found.  

A matrix is required for GPU/numpy applications. Especially for biological networks, there tend to be a few nodes with an overwhelming number of clauses. To balance the workload, all clauses are generated. They are then iteratively merged and mapped to the corresponding nodes. The approach is heavily dependent on disjunctive normal form. 

Currently estimating the attractor landscape of the fumia network (103 nodes) using 10^5 samples takes less than 30 seconds using cupy on a local desktop computer.


## 8. Models 

Most model files will have a corresponding `models/\_Exp.bnet` file for their expanded network representation and a `settings/\_.yaml`
 file for more details regarding their input and output nodes.

#### Fumia Network
Model file: `models/fumia.txt`
Original paper: Fumia, Herman F., and Marcelo L. Martins. "Boolean network model for cancer pathways: predicting carcinogenesis and targeted therapy outcomes." PloS one 8.7 (2013): e69008.

The original Fumia network has been modified by adding nodes for cell cycle checkpoints, and proliferation and quiescence (look at the last few lines for details). Adding these nodes explicitly to the model automatically outputs the phenotype, even under different node update schemes, ect. 

At the moment, most of the behavior in the original Fumia network is faithfully replicated. All input conditions lead to at least 96% of attractors in the same phenotype. However, the exact proportion of apopotic, proliferative and quiescent attractors is somewhat different than the original paper. This could be due to how the proliferation node is defined. The are also noticeably more attractors (about 300) relative to the original Fumia paper (about 60).

#### Grieco Network
Model file: `models/grieco.bnet`
Original paper: Grieco, Luca, et al. "Integrative modelling of the influence of MAPK network on cancer cell fate decision." PLoS computational biology 9.10 (2013): e1003286.

Bladder cancer MAPK signaling network. The outputs have been modified to be mutually exclusive. Growth Arrest only occurs if Apoptosis does not. Proliferation only occurs if neither Growth Arrest nor Apoptosis occur. 

So far an interesting network. 

#### Zhang Network
Model file: `models/zhang.bnet`
Original paper: Zhang, Ranran, et al. "Network model of survival signaling in large granular lymphocyte leukemia." Proceedings of the National Academy of Sciences 105.42 (2008): 16308-16313.

T-LGL leukemia network. Not sure how relevant it is, since almost 100% of wildtype basin is apoptotic. In the original paper the time to apoptosis is of interest.

#### Sahin Network
Model file: `models/sahin.bnet`. Original paper: Sahin, Özgür, et al. "Modeling ERBB receptor-regulated G1/S transition to find novel targets for de novo trastuzumab resistance." BMC systems biology 3.1 (2009): 1-20.

Mammalian cell cycle network. Not recommended. The output node is almost completely independent of the input, and the network has several faults such as redundant edges.

