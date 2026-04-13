# GPAGD: Geometric Physics‑Aware Gradient Descent

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**GPAGD** is a novel optimizer for physics‑informed neural networks (PINNs) that enhances gradient descent with three inductive biases:

- **Manifold projection** – projects gradients onto the tangent space of the input coordinates (PCA).
- **Physics gate** – scales the step size by `exp(-ρ * PDE_residual)`, automatically damping steps when the solution violates physical laws.
- **Uncertainty gate** – uses local entropy of the model output to reduce step size in noisy regions.

A capacity scaler adjusts the learning rate based on dataset size and noise level.

## Key Results (3k epochs, 3 seeds on Colab T4)

| Benchmark    | Adam (Rel L2) | GPAGD (Rel L2)     | Improvement | p‑value         |
|--------------|---------------|--------------------|-------------|-----------------|
| Darcy 2D     | 2.988 ± 0.198 | **1.0002 ± 0.0002**| **66% ↓**   | **0.005**       |
| Poisson 1D   | 19.95 ± 2.32  | **10.25 ± 1.99**   | 49% ↓       | 0.051 (marginal)|

Full ablation study and convergence plots are in the `results/` directory after running the experiments.

## Installation

```python
git clone https://github.com/mohsenmostafa/GPAGD-Optimizer.git
cd GPAGD-Optimizer
pip install -e .
```

## Quick Start
```python
import torch
from gpagd.optimizers import GeometricPhysicsGD

model = ...  # your PINN
optimizer = GeometricPhysicsGD(model.parameters(), lr=1e-3, rho=0.1, alpha=1.0)

def closure():
    optimizer.zero_grad()
    loss = physics_residual(model)  # your PDE loss
    loss.backward()
    return loss

# You need to provide:
#   manifold_projector: function that projects a gradient tensor
#   physics_residual_fn: returns current PDE residual
#   noise_estimate_fn: returns local entropy
#   dataset_size: int
#   noise_level: float
optimizer.step(closure, projector, physics_residual_fn, noise_estimate_fn, dataset_size, noise_level)
```

## Reproduce Paper Results
```python
cd experiments
python run_benchmarks.py   # runs 3k epochs, 3 seeds (adjustable)
```
## Citation
```python
@misc{mostafa2026gpagd,
  author = {Mostafa, Mohsen},
  title = {GPAGD: Geometric Physics‑Aware Gradient Descent Optimizer},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/mohsenmostafa/GPAGD-Optimizer}}
}
```
