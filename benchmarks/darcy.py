import torch
import numpy as np

class Darcy2D:
    def __init__(self, nx=50, ny=50):
        x = torch.linspace(0, 1, nx)
        y = torch.linspace(0, 1, ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.xy_cpu = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.n_colloc = nx*ny

    def residual(self, model):
        device = next(model.parameters()).device
        xy = self.xy_cpu.to(device).requires_grad_(True)
        u = model(xy)
        u_x = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
        u_y = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
        u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        a = 1 + torch.sin(2*np.pi*xy[:,0:1]) * torch.cos(2*np.pi*xy[:,1:2])
        a_x = 2*np.pi*torch.cos(2*np.pi*xy[:,0:1]) * torch.cos(2*np.pi*xy[:,1:2])
        a_y = -2*np.pi*torch.sin(2*np.pi*xy[:,0:1]) * torch.sin(2*np.pi*xy[:,1:2])
        div_a_grad_u = a*(u_xx+u_yy) + a_x*u_x + a_y*u_y
        f = 8*np.pi**2 * a * torch.sin(2*np.pi*xy[:,0:1]) * torch.sin(2*np.pi*xy[:,1:2])
        r = -div_a_grad_u - f
        return r.pow(2).mean()

    def exact(self, xy):
        return torch.sin(2*np.pi*xy[:,0:1]) * torch.sin(2*np.pi*xy[:,1:2])

    def get_inputs(self, model):
        device = next(model.parameters()).device
        return self.xy_cpu.to(device)
