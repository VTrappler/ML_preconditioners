# State-dependent preconditioners for Variational Data Assimilation using Machine Learning
## Problem formulation
In Variational Data assimilation, we are looking to solve the following minimisation problem.

$$\min_{x \in \mathbb{R}^n}\frac{1}{2} ||\mathcal{G}(x) - y ||^2_{R^{-1}} = \min_{x \in \mathbb{R}^n} J(x)$$

In an incremental formulation, we proceed by successive linearization of $J$ and thus of $\mathcal{G}$: this is the outer loop. For each outer loop iteration, we solve a linear system with respect to $\delta x_{i}$.


At step $i$:
+ Linearize $\mathcal{G}$ around $x_i$ to get $G_{x_i}$
+ Solve for the $\delta x_i$ the following linear system:

$$(G_{x_{i}}^TR^{-1}G_{x_{i}})\delta x_{i} = -G_{x_{i}}^TR^{-1}(\mathcal{G}(x_i) - y)$$
+ $x_{i+1} \gets x_i + \delta x_i$ and $i \gets i+1$

We aim at learning a preconditioner which depends solely on the current state in order to improve the convergence rate of the resolution of the linear system

## ML experiments using DVC
I chose to use [DVC](https://dvc.org) for the versioning of the data and the different steps of the experiments. The file paths are indicated for the Lorenz experiment.
### Configuration file
The config file is located in [`lorenz/config.yaml`](./lorenz/config.yaml)
### Pipelines
The pipeline and the different stages of the training are defined in [`lorenz/dvc.yaml`](./lorenz/dvc.yaml)
### Model registry
The model are tracked and registered using [MLflow](https://mlflow.org/)

## Use Cases
### Lorenz System
For the Lorenz system, the data are generated using [`DA_PoC`](https://github.com/VTrappler/DA_PoC) which implements a few data assimilation methods, and dynamical systems along with their TLM.
### Shallow Water
For the Shallow Water model, the data are generated using code stored on AIRSEA's gitlab.

The config file can be found [`SW/config.yaml`](./SW/config.yaml):

    dvc repro SW/dvc.yaml:generate_data # Generate the data
    dvc repro SW/dvc.yaml:train # Train the NN

Or all at once:

    dvc exp run SW/dvc.yaml




## References
- Poster presented at [9th International Symposium on Data Assimilation](https://hal.science/hal-04309242)
- Preprint soon available
<!-- - Poster presented at [54th International Liège Colloquium on ML and Data Analysis for Oceanography](https://hal.science/hal-04087646) -->