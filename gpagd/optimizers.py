import torch
import torch.optim as optim
import numpy as np

class GeometricPhysicsGD(optim.Optimizer):
    """
    Geometric Physics‑Aware Gradient Descent.

    Args:
        params: iterable of parameters to optimize.
        lr: base learning rate.
        rho: physics gate strength.
        alpha: uncertainty gate strength.
        lam: psi function scale.
        N0, gamma, beta: capacity scaling parameters.
        use_manifold, use_physics, use_uncertainty: flags to toggle components.
    """
    def __init__(self, params, lr=1e-3, rho=0.1, alpha=1.0, lam=0.01,
                 N0=5000, gamma=0.5, beta=1.0, eps=1e-8,
                 use_manifold=True, use_physics=True, use_uncertainty=True):
        defaults = dict(lr=lr, rho=rho, alpha=alpha, lam=lam,
                        N0=N0, gamma=gamma, beta=beta, eps=eps,
                        use_manifold=use_manifold, use_physics=use_physics,
                        use_uncertainty=use_uncertainty)
        super(GeometricPhysicsGD, self).__init__(params, defaults)

    @staticmethod
    def psi(t):
        return torch.log(1 + t) - t / (1 + t)

    def step(self, closure, manifold_projector, physics_residual_fn,
             noise_estimate_fn, dataset_size, estimate_noise_level):
        loss = closure()
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grads.append(p.grad)

        use_man = self.param_groups[0]['use_manifold']
        if use_man:
            projected_grads = [manifold_projector(g) for g in grads]
        else:
            projected_grads = grads

        use_phy = self.param_groups[0]['use_physics']
        if use_phy:
            R_ns = physics_residual_fn()
            if not torch.is_tensor(R_ns):
                R_ns = torch.tensor(R_ns)
            if R_ns.numel() > 1:
                R_ns = R_ns.mean()
            physics_factor = np.exp(-self.param_groups[0]['rho'] * R_ns.item())
        else:
            physics_factor = 1.0

        use_unc = self.param_groups[0]['use_uncertainty']
        if use_unc:
            E_local = noise_estimate_fn()
            if not torch.is_tensor(E_local):
                E_local = torch.tensor(E_local)
            if E_local.numel() > 1:
                E_local = E_local.mean()
            t_val = self.param_groups[0]['lam'] * E_local
            psi_val = self.psi(t_val)
            uncertainty_factor = np.exp(-self.param_groups[0]['alpha'] * psi_val.item())
        else:
            uncertainty_factor = 1.0

        gate = physics_factor * uncertainty_factor

        N = dataset_size
        N0 = self.param_groups[0]['N0']
        gamma = self.param_groups[0]['gamma']
        beta = self.param_groups[0]['beta']
        sigma_noise = estimate_noise_level()
        if N < N0:
            small_factor = 1.0 / (1.0 + ((N0 / N) - 1.0) ** beta)
        else:
            small_factor = 1.0
        noise_factor = 1.0 / (1.0 + gamma * (sigma_noise ** 2))
        capacity_scale = small_factor * noise_factor
        effective_lr = self.param_groups[0]['lr'] * capacity_scale

        with torch.no_grad():
            idx = 0
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    update = -effective_lr * gate * projected_grads[idx]
                    p.add_(update)
                    idx += 1
        return loss


class PhysicsAwareGD(optim.Optimizer):
    """Simpler baseline: only physics gate, no manifold or uncertainty."""
    def __init__(self, params, lr=1e-3, rho=0.1):
        defaults = dict(lr=lr, rho=rho)
        super(PhysicsAwareGD, self).__init__(params, defaults)

    def step(self, closure, physics_residual_fn):
        loss = closure()
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                R_ns = physics_residual_fn()
                if torch.is_tensor(R_ns):
                    R_ns = R_ns.item()
                factor = np.exp(-group['rho'] * R_ns)
                p.data.add_(p.grad, alpha=-group['lr'] * factor)
        return loss
