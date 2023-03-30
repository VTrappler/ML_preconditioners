# State-dependent preconditioners for Variational Data Assimilation using Machine Learning
## Problem formulation
In Variational Data assimilation, we are looking to solve the following minimisation problem.
$$x_{\text{analysis}} = \mathop{\text{argmin}}_{x \in \mathbb{R}^n} \frac{1}{2} \|\mathcal{G}(x) - y \|^2_{R^{-1}} = \mathop{\text{argmin}}_{x \in \mathbb{R}^n} J(x)$$

In an incremental formulation, we proceed by successive linearization of the $J$ and thus of $\mathcal{G}$: this is the outer loop. For each outer loop iteration, we solve the following linear system with respect to $x_{i+1}$
$$(G_{x_{i}}^TG_{x_{i}})x_{i+1} = -G_{x_{i}}^T(\mathcal{G}(x_i) - y)$$
We aim at learning a preconditioner which depends solely on the current state in order to 

## ML experiments using DVC
I chose to use [DVC](dvc.org) for the versioning of the data and the different steps of the experiments
### Configuration file
The config file is located in [`smoke/config.yaml`](./smoke/config.yaml)

### Pipelines
The pipeline and the different stages of the training is defined in [`smoke/dvc.yaml`](./smoke/dvc.yaml)

### Model registry
The model are tracked and registered using [MLflow](https://mlflow.org/)