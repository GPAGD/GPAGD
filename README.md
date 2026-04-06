# GPAGD: Geometric Physics‑Aware Gradient Descent

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)

**GPAGD** is a novel optimizer for physics‑informed neural networks (PINNs) that enhances standard gradient descent with three inductive biases:

- **Geometric projection** – keeps gradients aligned with the data manifold (PCA on input coordinates).
- **Physics gate** – scales the step size by `exp(-ρ * PDE_residual)`, automatically damping steps when the solution violates physical laws.
- **Uncertainty gate** – uses local entropy to reduce step size in noisy regions.

A capacity scaler also adjusts the learning rate based on dataset size and noise level.

## Key Results

| Benchmark | Adam (Rel L2) | GPAGD (Rel L2)    | Improvement   | p‑value |
|-----------|---------------|----------------   |---------------|---------|
| Darcy 2D  | 3.16 ± 0.17   | **1.013 ± 0.002** | 68% ↓         | 0.0032  |
| Poisson 1D| 18.6 ± 3.4    | **6.83 ± 3.67**   | 63% ↓         | 0.1235  |
 
Full ablation study and convergence plots are included in the repository.

## Installation

```python
git clone https://github.com/YOUR_USERNAME/GPAGD-Optimizer.git
cd GPAGD-Optimizer
pip install -e .
```

## Quick Start
```python
import torch
from gpagd import GeometricPhysicsGD

model = ...  # your PINN
optimizer = GeometricPhysicsGD(model.parameters(), lr=1e-3, rho=0.1, alpha=1.0)

def closure():
    optimizer.zero_grad()
    loss = physics_residual(model)  # your PDE loss
    loss.backward()
    return loss

optimizer.step(closure, manifold_projector, physics_residual_fn, noise_estimate_fn, dataset_size, noise_level)
```
## Reproduce Paper Results
```python
cd experiments
python run_benchmarks.py
```
All results (convergence plots, bar chart, CSV table) will be saved to the results/ directory.

## Citation
If you use GPAGD in your research, please cite:

```python
@misc{mostafa2026gpagd,
  author = {Mostafa, Mohsen},
  title = {GPAGD: Geometric Physics‑Aware Gradient Descent},
  year = {2026},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/YOUR_USERNAME/GPAGD-Optimizer}}
}
```
