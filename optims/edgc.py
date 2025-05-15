import torch
from torch.optim import Optimizer
from torch.distributed import ReduceOp

class Edgc(Optimizer):
    def __init__(self, params, base_optimizer, model, rho=0.2, lamb=0.5, adaptive=False, perturb_eps=1e-12, grad_reduce='mean', **kwargs):
        defaults = dict(adaptive=adaptive, **kwargs)
        super(Edgc, self).__init__(params, defaults)
        self.model = model
        self.base_optimizer = base_optimizer
        self.param_groups = self.base_optimizer.param_groups  # reuse the same param groups
        self.rho = rho
        self.lamb = lamb
        self.adaptive = adaptive
        self.perturb_eps = perturb_eps
        self.get_grad_reduce(grad_reduce)

    def get_grad_reduce(self, grad_reduce: str):
        if grad_reduce.lower() == 'mean':
            self.grad_reduce = ReduceOp.AVG
            self.manual_average = False

    @torch.no_grad()
    def get_perturb_direction(self, grads, flatness=False):
        perturb = []
        for g in grads:
            if g is None:
                perturb.append(None)
                continue
            scale = g.norm() if self.adaptive else 1.0
            direction = g / (scale + self.perturb_eps)
            if flatness:
                direction = direction.sign()
            perturb.append(direction)
        return perturb

    @torch.no_grad()
    def apply_perturb(self, direction, coeff=1.0):
        for p, d in zip(self.model.parameters(), direction):
            if d is not None:
                p.add_(d, alpha=coeff * self.rho)

    def step(self, closure):
        assert closure is not None, "EDGC requires a closure for reevaluating the model."

        loss = closure()
        grads = [p.grad.detach().clone() if p.grad is not None else None for p in self.model.parameters()]
        descent_dir = self.get_perturb_direction(grads, flatness=False)

        flat_dir = self.get_perturb_direction(grads, flatness=True)
        self.apply_perturb(flat_dir, coeff=+1.0)

        loss_flat = closure()
        grads_flat = [p.grad.detach().clone() if p.grad is not None else None for p in self.model.parameters()]

        self.apply_perturb(flat_dir, coeff=-1.0)

        combined_grad = []
        for g_d, g_f in zip(grads, grads_flat):
            if g_d is None or g_f is None:
                combined_grad.append(None)
            else:
                combined_grad.append((1 - self.lamb) * g_d + self.lamb * g_f)

        for p, g in zip(self.model.parameters(), combined_grad):
            if p.grad is not None and g is not None:
                p.grad.copy_(g)

        return self.base_optimizer.step()
