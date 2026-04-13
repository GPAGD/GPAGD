import torch
import torch.nn.functional as F
import numpy as np
from sklearn.decomposition import PCA

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


def local_entropy_1d(model, inputs, window_size=5):
    """1D sliding‑window local entropy."""
    with torch.no_grad():
        out = model(inputs).detach().squeeze()
    x = inputs[:,0]
    sort_idx = torch.argsort(x)
    out_sorted = out[sort_idx]
    kernel = torch.ones(1, 1, window_size, device=inputs.device) / window_size
    out_reshaped = out_sorted.view(1, 1, -1)
    local_mean = F.conv1d(out_reshaped, kernel, padding=window_size//2)
    local_sq_mean = F.conv1d(out_reshaped**2, kernel, padding=window_size//2)
    local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
    local_var_flat = local_var[0,0]
    inv_idx = torch.argsort(sort_idx)
    local_var_flat = local_var_flat[inv_idx]
    return torch.log(1 + local_var_flat).mean()


def local_entropy_grid(model, inputs, nx, ny, window_size=5):
    """2D sliding‑window local entropy for regular grids."""
    with torch.no_grad():
        out = model(inputs).detach()
    if out.dim() == 2 and out.shape[1] > 1:
        out = out.mean(dim=1, keepdim=True)
    out_grid = out.view(nx, ny).unsqueeze(0).unsqueeze(0)
    device = inputs.device
    kernel = torch.ones(1, 1, window_size, window_size, device=device) / (window_size**2)
    local_mean = F.conv2d(out_grid, kernel, padding=window_size//2)
    local_sq_mean = F.conv2d(out_grid**2, kernel, padding=window_size//2)
    local_var = (local_sq_mean - local_mean**2).clamp(min=1e-8)
    return torch.log(1 + local_var).mean()
