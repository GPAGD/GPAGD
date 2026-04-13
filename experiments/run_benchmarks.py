import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
import os
import sys
import pandas as pd
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpagd.optimizers import GeometricPhysicsGD, PhysicsAwareGD
from gpagd.utils import PCAManifoldProjector, local_entropy_1d, local_entropy_grid
from benchmarks import Poisson1D, Burgers1D_Fixed, Darcy2D, TaylorGreen2D

# -------------------------------
# Neural networks
# -------------------------------
class PINN1D(nn.Module):
    def __init__(self, hidden_layers=[50,50,50]):
        super().__init__()
        layers = []
        prev = 1
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h)); layers.append(nn.Tanh()); prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

class PINN2D(nn.Module):
    def __init__(self, hidden_layers=[100,100,100], output_dim=1):
        super().__init__()
        layers = []
        prev = 2
        for h in hidden_layers:
            layers.append(nn.Linear(prev, h)); layers.append(nn.Tanh()); prev = h
        layers.append(nn.Linear(prev, output_dim))
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x)

# -------------------------------
# Training function
# -------------------------------
def train_pde_run(problem, model, optimizer_type, num_epochs, seed, device,
                  print_every=1000, hyperparams=None):
    torch.manual_seed(seed)
    model.to(device).train()
    inputs_cpu = problem.get_inputs(model).cpu()
    projector = PCAManifoldProjector(inputs_cpu, n_components=5)

    # Default hyperparameters per problem
    if 'Burgers' in str(problem.__class__):
        lr, rho, alpha = 5e-4, 0.05, 0.5
    elif 'TaylorGreen' in str(problem.__class__):
        lr, rho, alpha = 1e-4, 0.02, 0.2
    else:
        lr, rho, alpha = 1e-3, 0.1, 1.0
    if hyperparams:
        lr = hyperparams.get('lr', lr)
        rho = hyperparams.get('rho', rho)
        alpha = hyperparams.get('alpha', alpha)

    if optimizer_type == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'LBFGS':
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, history_size=10, max_iter=20)
    elif optimizer_type == 'PAGD':
        optimizer = PhysicsAwareGD(model.parameters(), lr=lr, rho=rho)
    elif optimizer_type == 'GPAGD_Full':
        optimizer = GeometricPhysicsGD(model.parameters(), lr=lr, rho=rho, alpha=alpha, lam=0.01,
                                       use_manifold=True, use_physics=True, use_uncertainty=True)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_type}")

    def physics_residual():
        return problem.residual(model)
    def noise_estimate():
        inputs = problem.get_inputs(model)
        if hasattr(problem, 'nx') and hasattr(problem, 'ny'):
            return local_entropy_grid(model, inputs, problem.nx, problem.ny, window_size=5)
        else:
            return local_entropy_1d(model, inputs, window_size=5)
    dataset_size = problem.n_colloc
    def noise_level():
        return 0.1

    loss_history = []
    for epoch in range(1, num_epochs+1):
        def closure():
            optimizer.zero_grad()
            loss = problem.residual(model)
            loss.backward()
            return loss

        if optimizer_type == 'LBFGS':
            loss = optimizer.step(closure)
        elif optimizer_type == 'PAGD':
            loss = closure()
            optimizer.step(closure, physics_residual)
        elif optimizer_type.startswith('GPAGD'):
            loss = optimizer.step(closure, projector, physics_residual, noise_estimate,
                                  dataset_size, noise_level)
        else:
            loss = closure()
            optimizer.step()

        # Gradient clipping to avoid explosions (for all optimizers)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        loss_history.append(loss.item())
        if epoch % print_every == 0:
            print(f"      Epoch {epoch}, Loss: {loss.item():.4e}")
            sys.stdout.flush()

    model.eval()
    with torch.no_grad():
        inputs = problem.get_inputs(model)
        pred = model(inputs)
        exact = problem.exact(inputs)
        if exact.shape[1] == pred.shape[1]:
            rel_l2 = torch.norm(pred - exact) / torch.norm(exact)
        else:
            rel_l2 = float('nan')
    return loss_history, rel_l2.item()

# -------------------------------
# Main experiment
# -------------------------------
def run_experiment():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Settings – adjust for stronger results
    NUM_EPOCHS = 3000
    NUM_SEEDS = 3
    BENCHMARKS = {
        'Poisson1D': Poisson1D(),
        'Burgers1D': Burgers1D_Fixed(),
        'Darcy2D': Darcy2D(),
        'TaylorGreen2D': TaylorGreen2D()
    }
    MODEL_BUILDERS = {
        'Poisson1D': lambda: PINN1D([50,50,50]),
        'Burgers1D': lambda: PINN1D([50,50,50]),
        'Darcy2D': lambda: PINN2D([100,100,100], output_dim=1),
        'TaylorGreen2D': lambda: PINN2D([100,100,100], output_dim=3)
    }
    OPTIMIZERS = ['Adam', 'LBFGS', 'GPAGD_Full']

    results = {bench: {opt: [] for opt in OPTIMIZERS} for bench in BENCHMARKS}
    loss_curves = {bench: {opt: [] for opt in OPTIMIZERS} for bench in BENCHMARKS}

    total_runs = len(BENCHMARKS) * len(OPTIMIZERS) * NUM_SEEDS
    run_count = 0
    start_time = time.time()

    for bench_name, problem in BENCHMARKS.items():
        print(f"\n{'='*60}\nBenchmark: {bench_name}\n{'='*60}")
        for opt in OPTIMIZERS:
            print(f"\n  Optimizer: {opt}")
            rel_errors = []
            for seed in range(NUM_SEEDS):
                run_count += 1
                print(f"    Seed {seed+1}/{NUM_SEEDS} (Run {run_count}/{total_runs})")
                model = MODEL_BUILDERS[bench_name]().to(device)
                loss_hist, rel_err = train_pde_run(
                    problem, model, opt, NUM_EPOCHS, seed+42, device,
                    print_every=NUM_EPOCHS//5)
                rel_errors.append(rel_err)
                if seed == 0:
                    loss_curves[bench_name][opt] = loss_hist
                print(f"      Rel L2 error: {rel_err:.4e}")
                torch.cuda.empty_cache()
            results[bench_name][opt] = rel_errors
            mean_rel = np.mean(rel_errors)
            std_rel = np.std(rel_errors)
            print(f"    Summary: Rel L2 = {mean_rel:.4e} ± {std_rel:.4e}")

    elapsed = time.time() - start_time
    print(f"\nTotal runtime: {elapsed/60:.1f} minutes")

    # Save results
    os.makedirs("../results", exist_ok=True)
    df_rows = []
    for bench in BENCHMARKS:
        for opt in OPTIMIZERS:
            rel_errors = results[bench][opt]
            df_rows.append({
                'Benchmark': bench,
                'Optimizer': opt,
                'Mean_Rel_L2': np.mean(rel_errors),
                'Std_Rel_L2': np.std(rel_errors)
            })
    pd.DataFrame(df_rows).to_csv("../results/summary.csv", index=False)

    # Convergence plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        for opt in OPTIMIZERS:
            if loss_curves[bench][opt]:
                ax.plot(loss_curves[bench][opt], label=opt, linewidth=1.5)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(bench)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/convergence_plots.png", dpi=150)
    plt.close()

    # Bar chart
    x = np.arange(len(BENCHMARKS))
    width = 0.25
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i, opt in enumerate(OPTIMIZERS):
        means = [np.mean(results[bench][opt]) for bench in BENCHMARKS]
        stds = [np.std(results[bench][opt]) for bench in BENCHMARKS]
        ax2.bar(x + i*width, means, width, yerr=stds, label=opt, capsize=3)
    ax2.set_xticks(x + width)
    ax2.set_xticklabels(BENCHMARKS.keys())
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_yscale('log')
    ax2.set_title('GPAGD vs Baselines')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("../results/ablation_barchart.png", dpi=150)
    plt.close()

    print("All results saved to ../results/")

if __name__ == "__main__":
    run_experiment()
