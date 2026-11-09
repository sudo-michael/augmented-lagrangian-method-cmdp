# Augmented Lagrangian Method for Last-Iterate Convergence for Constrained MDPs
Code for: [Augmented Lagrangian Method for Last-Iterate Convergence for Constrained MDPs
](https://arxiv.org/pdf/2605.11694)

Code for the continuous-control tasks can be found in the branch `omnisafe`.


## Installation
Assuming `uv` is installed:
```
uv sync
```


## Reproducing Experiments
To reproduce the deterministic experiments:
```
uv run det_experiments.py
```
To reproduce the linear FA experiments:
```
uv run linear_det_experiments.py
```
To plot the resulting experiments:
```
uv run plot.py
```
