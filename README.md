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

```bash
git clone https://github.com/YOUR_USERNAME/GPAGD-Optimizer.git
cd GPAGD-Optimizer
pip install -e .
