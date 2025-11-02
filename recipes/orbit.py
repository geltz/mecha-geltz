"""
Orbit: merge via parallel and orthogonal decomposition with trust region.

Idea:
- Decompose model b into components parallel and orthogonal to model a.
- Blend parallel component conservatively, orthogonal component more freely.
- Clamp changes using MAD-based trust region for stability.

Computation:
- Project b onto a to get parallel component (coef * a).
- Orthogonal component is residual (b - parallel).
- Blend: a + alpha_par*(parallel-a) + alpha_orth*orthogonal.
- Apply trust region based on median absolute deviation.

Notes:
- alpha_par controls parallel blend, alpha_orth controls orthogonal.
- trust_k sets trust region size (larger = more permissive).
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor

def _mad(x: Tensor, eps: Tensor) -> Tensor:
    f = x.flatten()
    m = f.median()
    return (f - m).abs().median().clamp_min(eps)

def _trust_clamp(a: Tensor, y: Tensor, trust_k: float, eps: Tensor) -> Tensor:
    r = float(trust_k) * _mad(a, eps)
    return a + (y - a).clamp(-r, r)

def _finite_or_a(a: Tensor, y: Tensor) -> Tensor:
    return torch.where(torch.isfinite(y), y, a)

@merge_method
def orbit(
    a: Parameter(Tensor),
    b: Parameter(Tensor),
    alpha_par: Parameter(float) = 0.20,
    alpha_orth: Parameter(float) = 0.60,
    trust_k: Parameter(float) = 3.0,
    eps: Parameter(float) = 1e-8,
    coef_clip: Parameter(float) = 8.0,
) -> Return(Tensor):
    e = torch.as_tensor(float(eps), device=a.device, dtype=a.dtype)
    wp = torch.as_tensor(float(alpha_par), device=a.device, dtype=a.dtype)
    wo = torch.as_tensor(float(alpha_orth), device=a.device, dtype=a.dtype)
    af, bf = a.flatten(), b.flatten()
    den = (af @ af).clamp_min(e)
    coef = (bf @ af) / den
    if float(coef_clip) > 0.0:
        c = torch.as_tensor(float(coef_clip), device=a.device, dtype=a.dtype)
        coef = coef.clamp(-c, c)
    bp = coef * a
    bo = b - bp
    y = a + wp * (bp - a) + wo * bo
    y = _trust_clamp(a, y, trust_k, e)
    y = _finite_or_a(a, y)
    return y