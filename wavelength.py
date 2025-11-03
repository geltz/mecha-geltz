"""
Wavelength: oscillating influence cascade.

Idea:
- Apply sinusoidal weighting pattern across models.
- Creates alternating strong/weak influence waves.
- Produces more varied blending than monotonic decay.

Computation:
- base = mean of all models
- Weight by sine wave phase across model sequence
- Phase shift controls which models peak
- Combines smooth blending with periodic emphasis
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import math

@merge_method
def wavelength(
    *models: Parameter(Tensor),
    strength: Parameter(float) = 0.7,
    frequency: Parameter(float) = 1.0,
    phase: Parameter(float) = 0.0
) -> Return(Tensor):
    n = len(models)
    if n == 0:
        raise ValueError("wavelength_merge requires at least one model")
    if n == 1:
        return models[0]
    
    s = max(0.0, min(1.0, float(strength)))
    freq = max(0.1, min(5.0, float(frequency)))
    ph = float(phase)
    
    base = torch.stack(models, dim=0).mean(dim=0)
    
    result = base.clone()
    
    for i, model in enumerate(models):
        # normalized position
        pos = i / max(n - 1, 1)
        # sine wave weight (0.5 to 1.0 range)
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * freq * pos + ph)
        weight = wave * s / n
        delta = model - base
        result = result + delta * weight
    
    return result