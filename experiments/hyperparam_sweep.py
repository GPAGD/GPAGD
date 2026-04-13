import torch
import numpy as np
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gpagd.optimizers import GeometricPhysicsGD
from benchmarks.darcy import Darcy2D
from experiments.run_benchmarks import train_pde_run, PINN2D

def sensitivity_sweep(problem, model_builder, base_params, param_name, values, device, num_epochs=1000):
    results = {}
    for val in values:
        params = base_params.copy()
        params[param_name] = val
        rel_errors = []
        for seed in range(2):
            model = model_builder().to(device)
            _, rel_err = train_pde_run(problem, model, 'GPAGD_Full', num_epochs, seed+42, device,
                                       hyperparams=params, print_every=num_epochs//2)
            rel_errors.append(rel_err)
        results[val] = np.mean(rel_errors)
    return results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    problem = Darcy2D()
    model_builder = lambda: PINN2D([100,100,100], output_dim=1)
    base_params = {'lr': 1e-3, 'rho': 0.1, 'alpha': 1.0}
    rho_values = [0.01, 0.05, 0.1, 0.5, 1.0]
    alpha_values = [0.0, 0.2, 0.5, 1.0, 2.0]
    rho_sens = sensitivity_sweep(problem, model_builder, base_params, 'rho', rho_values, device, num_epochs=1000)
    alpha_sens = sensitivity_sweep(problem, model_builder, base_params, 'alpha', alpha_values, device, num_epochs=1000)
    print("rho sensitivity:", rho_sens)
    print("alpha sensitivity:", alpha_sens)
