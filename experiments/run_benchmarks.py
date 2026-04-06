import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import ttest_rel
from sklearn.decomposition import PCA
import os
import sys
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpagd import GeometricPhysicsGD
from benchmarks import Poisson1D, Burgers1D_Fixed, Darcy2D, TaylorGreen2D

# -------------------------------
# Helper classes (projector, noise, networks)
# -------------------------------
class PCAManifoldProjector:
    def __init__(self, inputs, n_components=5):
        inputs_np = inputs.detach().cpu().numpy()
        self.input_dim = inputs.shape[1]
        self.n_components = min(n_components, self.input_dim)
        self.pca = PCA(n_components=self.n_components)
        self.pca.fit(inputs_np)
        self.components = torch.tensor(self.pca.components_.T, dtype=torch.float32)
        self.mean = torch.tensor(self.pca.mean_, dtype=torch.float32)
    def __call__(self, grad):
        if grad.dim() == 2 and grad.shape[1] == self.input_dim:
            centered = grad - self.mean.to(grad.device)
            projected = torch.mm(centered, self.components.to(grad.device))
            return projected
        else:
            return grad

def local_entropy_noise(model, inputs, window_size=3):
    with torch.no_grad():
        out = model(inputs).detach()
    if out.dim() == 2 and out.shape[1] > 1:
        out = out.mean(dim=1, keepdim=True)
    device = inputs.device
    if inputs.shape[1] == 1:
        x = inputs[:,0]
        sort_idx = torch.argsort(x)
        out_sorted = out[sort_idx]
        kernel = torch.ones(1, 1, window_size, device=device) / window_size
        out_reshaped = out_sorted.view(1, 1, -1)
        local_mean = torch.nn.functional.conv1d(out_reshaped, kernel, padding=window_size//2)
        local_sq_mean = torch.nn.functional.conv1d(out_reshaped**2, kernel, padding=window_size//2)
        local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
        local_var_flat = local_var[0,0]
        inv_idx = torch.argsort(sort_idx)
        local_var_flat = local_var_flat[inv_idx]
        E_local = torch.log(1 + local_var_flat).mean()
    else:
        unique_x = torch.unique(inputs[:,0]); unique_y = torch.unique(inputs[:,1])
        nx, ny = len(unique_x), len(unique_y)
        out_grid = out.view(nx, ny)
        kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size**2)
        out_grid_reshaped = out_grid.view(1, 1, nx, ny)
        local_mean = torch.nn.functional.conv2d(out_grid_reshaped, kernel, padding=window_size//2)
        local_sq_mean = torch.nn.functional.conv2d(out_grid_reshaped**2, kernel, padding=window_size//2)
        local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
        E_local = torch.log(1 + local_var).mean()
    return E_local

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
    def __init__(self, hidden_layers=[50,50,50], output_dim=1):
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
def train_pde_run(problem, model, opt_variant, num_epochs, seed, device, print_every=2000):
    torch.manual_seed(seed)
    model.to(device).train()
    inputs_cpu = problem.get_inputs(model).cpu()
    projector = PCAManifoldProjector(inputs_cpu, n_components=5)

    if 'Burgers' in str(problem.__class__):
        lr, rho, alpha = 5e-4, 0.05, 0.5
    elif 'TaylorGreen' in str(problem.__class__):
        lr, rho, alpha = 1e-4, 0.02, 0.2
    else:
        lr, rho, alpha = 1e-3, 0.1, 1.0

    if opt_variant == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    elif opt_variant == 'GPAGD_Full':
        optimizer = GeometricPhysicsGD(model.parameters(), lr=lr, rho=rho, alpha=alpha, lam=0.01,
                                       use_manifold=True, use_physics=True, use_uncertainty=True)
    elif opt_variant == 'GPAGD_NoMan':
        optimizer = GeometricPhysicsGD(model.parameters(), lr=lr, rho=rho, alpha=alpha, lam=0.01,
                                       use_manifold=False, use_physics=True, use_uncertainty=True)
    elif opt_variant == 'GPAGD_NoPhy':
        optimizer = GeometricPhysicsGD(model.parameters(), lr=lr, rho=0.0, alpha=alpha, lam=0.01,
                                       use_manifold=True, use_physics=False, use_uncertainty=True)
    elif opt_variant == 'GPAGD_NoUnc':
        optimizer = GeometricPhysicsGD(model.parameters(), lr=lr, rho=rho, alpha=0.0, lam=0.01,
                                       use_manifold=True, use_physics=True, use_uncertainty=False)
    else:
        raise ValueError(f"Unknown variant {opt_variant}")

    def physics_residual(): return problem.residual(model)
    def noise_estimate(): return local_entropy_noise(model, problem.get_inputs(model))
    dataset_size = problem.n_colloc
    def noise_level(): return 0.1

    loss_history = []
    for epoch in range(1, num_epochs+1):
        def closure():
            optimizer.zero_grad()
            loss = problem.residual(model)
            loss.backward()
            return loss
        if opt_variant.startswith('GPAGD'):
            loss = optimizer.step(closure, projector, physics_residual, noise_estimate, dataset_size, noise_level)
        else:
            loss = closure(); optimizer.step()
        loss_history.append(loss.item())
        if epoch % print_every == 0:
            print(f"      Epoch {epoch}, Loss: {loss.item():.4e}")
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
def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    NUM_EPOCHS = 2000
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
        'Darcy2D': lambda: PINN2D([50,50,50], output_dim=1),
        'TaylorGreen2D': lambda: PINN2D([50,50,50], output_dim=3)
    }
    VARIANTS = ['Adam', 'GPAGD_Full', 'GPAGD_NoMan', 'GPAGD_NoPhy', 'GPAGD_NoUnc']

    results = {bench: {var: [] for var in VARIANTS} for bench in BENCHMARKS}
    loss_curves = {bench: {var: [] for var in VARIANTS} for bench in BENCHMARKS}

    for bench_name, problem in BENCHMARKS.items():
        print(f"\n{'='*60}\nBenchmark: {bench_name}\n{'='*60}")
        for var in VARIANTS:
            print(f"\n  Variant: {var}")
            rel_errors = []
            for seed in range(NUM_SEEDS):
                print(f"    Seed {seed+1}/{NUM_SEEDS}")
                model = MODEL_BUILDERS[bench_name]().to(device)
                loss_hist, rel_err = train_pde_run(
                    problem, model, var, NUM_EPOCHS, seed+42, device, print_every=NUM_EPOCHS//2)
                rel_errors.append(rel_err)
                if seed == 0:
                    loss_curves[bench_name][var] = loss_hist
                print(f"      Rel L2 error: {rel_err:.4e}")
            results[bench_name][var] = rel_errors
            mean_rel = np.mean(rel_errors)
            std_rel = np.std(rel_errors)
            print(f"    Summary: Rel L2 = {mean_rel:.4e} ± {std_rel:.4e}")

    # Print final table
    print("\n\n" + "="*80)
    print("FINAL RESULTS: Relative L2 Error (mean ± std over seeds)")
    print("="*80)
    for bench in BENCHMARKS:
        print(f"\n{bench}:")
        for var in VARIANTS:
            rel_errors = results[bench][var]
            print(f"  {var:15s}: {np.mean(rel_errors):.4e} ± {np.std(rel_errors):.4e}")

    # Statistical tests
    print("\n" + "="*80)
    print("Statistical Significance (paired t-test, Adam vs GPAGD_Full)")
    print("="*80)
    for bench in BENCHMARKS:
        adam_errors = results[bench]['Adam']
        gpagd_errors = results[bench]['GPAGD_Full']
        if not np.isnan(adam_errors).any() and not np.isnan(gpagd_errors).any():
            t_stat, p_val = ttest_rel(adam_errors, gpagd_errors)
            print(f"{bench:15s}: t = {t_stat:.4f}, p = {p_val:.4f}")

    # Save results
    os.makedirs("results", exist_ok=True)
    df_rows = []
    for bench in BENCHMARKS:
        for var in VARIANTS:
            rel_errors = results[bench][var]
            df_rows.append({'Benchmark': bench, 'Variant': var,
                            'Mean_Rel_L2': np.mean(rel_errors),
                            'Std_Rel_L2': np.std(rel_errors)})
    pd.DataFrame(df_rows).to_csv("results/summary.csv", index=False)
    print("\nResults saved to results/summary.csv")

    # Convergence plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    for idx, bench in enumerate(BENCHMARKS):
        ax = axes[idx]
        for var in VARIANTS:
            if loss_curves[bench][var]:
                ax.plot(loss_curves[bench][var], label=var, linewidth=1.5)
        ax.set_yscale('log')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(bench)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/convergence_plots.png", dpi=150)
    plt.close()
    print("Convergence plots saved to results/convergence_plots.png")

    # Bar chart
    x = np.arange(len(BENCHMARKS))
    width = 0.15
    fig2, ax2 = plt.subplots(figsize=(12, 6))
    for i, var in enumerate(VARIANTS):
        means = [np.mean(results[bench][var]) for bench in BENCHMARKS]
        stds = [np.std(results[bench][var]) for bench in BENCHMARKS]
        ax2.bar(x + i*width, means, width, yerr=stds, label=var, capsize=3)
    ax2.set_xticks(x + width*2)
    ax2.set_xticklabels(BENCHMARKS.keys())
    ax2.set_ylabel('Relative L2 Error')
    ax2.set_yscale('log')
    ax2.set_title('Ablation Study: GPAGD vs Adam')
    ax2.legend()
    ax2.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/ablation_barchart.png", dpi=150)
    plt.close()
    print("Bar chart saved to results/ablation_barchart.png")

if __name__ == "__main__":
    run()
