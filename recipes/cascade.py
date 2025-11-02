"""
Cascade: hierarchical blending with progressive refinement.

Idea:
- Blend models in layers, where each layer refines the previous.
- Start with average of all models as base.
- Progressively add weighted differences from each model.

Computation:
- base = mean of all models
- For each model, compute delta from base
- Add deltas with exponentially decaying weights
- Creates cascade effect where early models have more influence

Notes:
- strength controls decay rate of influence.
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor

@merge_method
def cascade(
    *models: Parameter(Tensor),
    strength: Parameter(float) = 0.7
) -> Return(Tensor):
    n = len(models)
    if n == 0:
        raise ValueError("cascade requires at least one model")
    if n == 1:
        return models[0]
    
    s = max(0.0, min(1.0, float(strength)))

    base = torch.stack(models, dim=0).mean(dim=0)

    result = base.clone()
    for i, model in enumerate(models):
        weight = s ** i
        delta = model - base
        result = result + delta * weight / n
    
    return result