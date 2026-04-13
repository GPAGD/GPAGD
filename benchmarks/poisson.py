import torch
import numpy as np

class Poisson1D:
    def __init__(self, n_colloc=1000):
        self.x_cpu = torch.linspace(0, 1, n_colloc).reshape(-1,1)
        self.n_colloc = n_colloc
    def residual(self, model):
        device = next(model.parameters()).device
        x = self.x_cpu.to(device).requires_grad_(True)
        u = model(x)
        u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
        f = torch.sin(2*np.pi*x)
        r = -u_xx - f
        return r.pow(2).mean()
    def exact(self, x):
        return torch.sin(2*np.pi*x) / (4*np.pi**2)
    def get_inputs(self, model):
        device = next(model.parameters()).device
        return self.x_cpu.to(device)
