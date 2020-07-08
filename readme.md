## Path Planning

The implementation of an optimal path planning algorithm for autonomous vehicles is crucial to their ability to successfully traverse through a static or a dynamic environment. Traditional analytical approaches to path planning are often dependent upon the existence of derivatives, continuity, and unimodality in order to find a globally optimal solution. Classical enumeration methods come close to finding optimal solutions, but they become inefficient as the dimensionality of the problem increases. This project examines the applicability of genetic algorithms, as a search and optimization heuristic technique, for the purpose of path planning.

## Algorithms

Though the focus of this project is on genetic algorithms, it is nonetheless valuable to compare the strengths and weaknesses of several different path planning algorithms. The list below outlines the algorithms for which code has been developed. There are quite a few variations that exist for each of the algorithms and command line arguments can be used to customize the behavior. Please refer to the code for more details.

- Genetic Algorithms
- Dijkstra's Algorithm
- A-Star (A*) Algorithms
- D-Star (D*) Algorithms
- Artificial Potential Fields
- Rapidly Exploring Random Trees

## Installation

It is recommended that you create a new virtual environment:

```
conda create -n path-planning python=3.6
```

The environment must first be activated:

```
conda activate path-planning
```

The following dependencies need to be installed:

```
conda install numpy matplotlib scipy shapely
```

The code can then be executed, for example via command line:

```
python genetic-algorithm.py
```

## Visualization

The results from running the genetic algorithm code (with the default arguments) are shown below:

| Initial Population | Convergence | Optimal Path | Average Fitness |
|:------------------:|:-----------:|:------------:|:---------------:|
|    ![I][start]     |![C][middle] |  ![O][stop]  |  ![A][average]  |

[start]: images/start.png "Initial Population"
[middle]: images/middle.png "Convergence"
[stop]: images/stop.png "Optimal Path"
[average]: images/average.png "Average Fitness"
