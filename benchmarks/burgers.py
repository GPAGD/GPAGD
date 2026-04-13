import torch
import numpy as np

class Burgers1D_Fixed:
    def __init__(self, nu=0.01/np.pi, n_colloc=1000):
        self.nu = nu
        self.x_cpu = torch.linspace(0, 1, n_colloc).reshape(-1,1)
        self.n_colloc = n_colloc
    def residual(self, model):
        device = next(model.parameters()).device
        x = self.x_cpu.to(device).requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        r = u * u_x - self.nu * u_xx
        return r.pow(2).mean()
    def exact(self, x):
        return torch.sin(np.pi * x)
    def get_inputs(self, model):
        device = next(model.parameters()).device
        return self.x_cpu.to(device)
