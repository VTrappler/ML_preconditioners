# State-dependent preconditioners for Variational Data Assimilation using Machine Learning
## Problem formulation
In Variational Data assimilation, we are looking to solve the following minimisation problem.

$$\min_{x \in \mathbb{R}^n}\frac{1}{2} ||\mathcal{G}(x) - y ||^2_{R^{-1}}$$
In an incremental formulation, we proceed by successive linearization of the $J$ and thus of $\mathcal{G}$: this is the outer loop. For each outer loop iteration, we solve the following linear system with respect to $x_{i+1}$
$$(G_{x_{i}}^TR^{-1}G_{x_{i}})x_{i+1} = -G_{x_{i}}^TR^{-1}(\mathcal{G}(x_i) - y)$$
We aim at learning a preconditioner which depends solely on the current state in order to improve the convergence rate of the resolution of the linear system


## Use Cases
### Lorenz System
For the Lorenz system, the data are generated using [`DA_PoC`](https://github.com/VTrappler/DA_PoC) which implements a few data assimilation methods, and dynamical systems along with their TLM.
### Shallow Water
For the Shallow Water model, the data are generated using code stored on AIRSEA's gitlab

## ML experiments using DVC
I chose to use [DVC](https://dvc.org) for the versioning of the data and the different steps of the experiments. The file paths are indicated for the Lorenz experiment.
### Configuration file
The config file is located in [`lorenz/config.yaml`](./lorenz/config.yaml)
### Pipelines
The pipeline and the different stages of the training are defined in [`lorenz/dvc.yaml`](./lorenz/dvc.yaml)
### Model registry
The model are tracked and registered using [MLflow](https://mlflow.org/)