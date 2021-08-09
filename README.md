
# Simulating Boolean Networks to Infer Basin Size 

Contents:
1. Requirements
2. Basic Usage 
3. Net File
4. Algorithm 
5. ToDo


## 1. Requirements

Python3 and the relevant python libraries must be installed. The main required python libraries are matplotlib and numpy.

Instead of numpy, cupy can optionally be installed. This library provides GPU computing using numpy syntax, and requires that CUDA drivers are installed. Note that it works best with Intel GPUs, and is still in beta for AMD GPUs. 


## 2. Basic Usage

Running a single simulation:
	`python3 main.py params/PARAM.yaml`

The parameter file is a yaml file, which is like JSON, but with more minimal syntax and allows comments. See params.yaml for an example and for an explanation of each parameter. 

If the parameter savefig is false, then an output image will pop up. Otherwise, the output will appear in the output/ directory.  


## 3. Net File

The program calculates the basin size of a given network. I am open to changing the syntax of the network file. 

For now the syntax is in disjunctive normal form, with one node per line. Each node is separated from its function by a tab, each clause separated by a space, and each element by '&'. Negative is represented by '-'. See `test_net` file for an example.

For instance: if node X = (A and B) or (not C), then it's line would be `X	A&B -C`


## 4. Algorithm 

A number of different initial conditions are generated. The net is run for each initial condition, stopping if it reaches a fixed point or the maximum number of iterations. All initial states are assumed to be premissible. Anything that is not a fixed point is grouped into the 'oscilations' attractor.

The main step of each iteration is in `runNet.py`: `x_next = cp.any(cp.all(X[clause_index],axis=2),axis=1)`. `X` is the state vector of the nodes. The function of each node is represented by a `clause_index`. Let a clause be one AND statement of a disjunctive normal form, and a literal one input of a clause. Then `clause_index` is a 3D matrix of shape (number of nodes, maximum literals, maximum clauses). This is used to index the state vector in `X[clause_index]`. A clause is true if all of it's literals are 1, tested by `cp.all()`. Then if any clause is true, the node is on, tested by `cp.any()`.  

Negative literals are handled by forming pseudo nodes, and appending them to X at each step. In other words, these nodes are inputs to X[clause_index], but their functions are not calculated or output.

The most time and space intensive step is likely `X[clause_index]`. I am not sure how it is implemented in numpy/cupy and may involve under the hood copying. An alternative would be to explicitly copy X into a 3D array of shape (n*n*n) and use a boolean `clause_index` to index it.

A matrix is required for GPU/numpy applications. One downside is that `clause_index` may be unnecessarily large, since most nodes do not have the maximum number of literals or the maximum number of clauses. Biological networks could exacerbate this inefficiency, since they tend to be sparse.



## 5. ToDo
Soon:
- time code with larger networks
- check size in memory with array.nbytes
- not sure how to display when there are many attractors, maybe group all smaller into one miscellaneous slice?

Eventually:
- Distinguish short oscillations as steady states
- Asynchronous approximation. Update a fraction of nodes at each step. Probability of unfaithful asynchrony could be bounded by the largest in degree and the fraction used.
- Try alternative method for `clause_index`?

