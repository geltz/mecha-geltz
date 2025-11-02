"""
Iterative Median: robust geometric median merge using iterative refinement.

Idea:
- Find the geometric median of models (point minimizing sum of distances).
- Uses iteratively reweighted least squares with Huber-like weighting.
- More robust to outliers than simple averaging.

Computation:
- Initialize with weighted average of all models.
- Iteratively reweight based on distances from current estimate.
- Converge to geometric median through fixed-point iteration.

Notes:
- alpha (sigma) controls robustness scale.
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor

def _gmedian(models, weights, sigma=1.0, max_iter=20, tol=1e-4):
    """Compute geometric median without stacking all models."""
    device = models[0].device
    dtype = models[0].dtype
    
    # Initialize with weighted average
    theta = torch.zeros_like(models[0])
    total_weight = 0.0
    
    for i, model in enumerate(models):
        theta.add_(weights[i] * model)
        total_weight += weights[i]
    
    theta.div_(total_weight)
    
    eps = 1e-6 * sigma
    N = len(models)
    
    for _ in range(max_iter):
        # Compute distances
        dists = torch.zeros(N, device=device, dtype=dtype)
        
        for i, model in enumerate(models):
            diff = theta - model
            dists[i] = torch.norm(diff, p=2)
        
        # Update estimate
        psi = torch.clamp(1.0 / (dists / sigma + 1e-8), max=1.0)
        w_psi = weights * psi
        
        num = torch.zeros_like(theta)
        den = 0.0
        
        for i in range(N):
            if dists[i] > eps:
                scale = w_psi[i] / dists[i]
                num.add_(scale * models[i])
                den += scale
        
        if den < 1e-8:
            return theta
        
        theta_new = num / den
        
        # Check convergence
        diff_norm = torch.norm(theta_new - theta, p=2)
        if diff_norm < tol:
            break
            
        theta = theta_new
    
    return theta

@merge_method
def iterative_median(
    *models: Parameter(Tensor),
    alpha: Parameter(Tensor) = 1.0,
) -> Return(Tensor):
    weights = torch.ones(len(models), dtype=models[0].dtype, device=models[0].device)
    return _gmedian(models, weights, sigma=alpha)