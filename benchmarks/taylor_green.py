import torch
import numpy as np

class TaylorGreen2D:
    def __init__(self, nx=50, ny=50, Re=100):
        x = torch.linspace(0, 2*np.pi, nx)
        y = torch.linspace(0, 2*np.pi, ny)
        xx, yy = torch.meshgrid(x, y, indexing='ij')
        self.xy_cpu = torch.stack([xx.reshape(-1), yy.reshape(-1)], dim=1)
        self.Re = Re
        self.n_colloc = nx*ny

    def residual(self, model):
        device = next(model.parameters()).device
        xy = self.xy_cpu.to(device).requires_grad_(True)
        out = model(xy)
        u = out[:,0:1]; v = out[:,1:2]; p = out[:,2:3]
        u_x = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,0:1]
        u_y = torch.autograd.grad(u, xy, grad_outputs=torch.ones_like(u), create_graph=True)[0][:,1:2]
        v_x = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)[0][:,0:1]
        v_y = torch.autograd.grad(v, xy, grad_outputs=torch.ones_like(v), create_graph=True)[0][:,1:2]
        p_x = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True)[0][:,0:1]
        p_y = torch.autograd.grad(p, xy, grad_outputs=torch.ones_like(p), create_graph=True)[0][:,1:2]
        u_xx = torch.autograd.grad(u_x, xy, grad_outputs=torch.ones_like(u_x), create_graph=True)[0][:,0:1]
        u_yy = torch.autograd.grad(u_y, xy, grad_outputs=torch.ones_like(u_y), create_graph=True)[0][:,1:2]
        v_xx = torch.autograd.grad(v_x, xy, grad_outputs=torch.ones_like(v_x), create_graph=True)[0][:,0:1]
        v_yy = torch.autograd.grad(v_y, xy, grad_outputs=torch.ones_like(v_y), create_graph=True)[0][:,1:2]
        cont = u_x + v_y
        nu = 1.0/self.Re
        res_u = u*u_x + v*u_y + p_x - nu*(u_xx+u_yy)
        res_v = u*v_x + v*v_y + p_y - nu*(v_xx+v_yy)
        return (cont.pow(2).mean() + res_u.pow(2).mean() + res_v.pow(2).mean())

    def exact(self, xy):
        x = xy[:,0:1]; y = xy[:,1:2]
        u = torch.sin(x) * torch.cos(y)
        v = -torch.cos(x) * torch.sin(y)
        p = -0.25*(torch.cos(2*x) + torch.cos(2*y))
        return torch.cat([u, v, p], dim=1)

    def get_inputs(self, model):
        device = next(model.parameters()).device
        return self.xy_cpu.to(device)
