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
        raise ValueError("wavelength requires at least one model")
    if n == 1:
        return models[0]
    
    s = max(0.0, min(1.0, float(strength)))
    freq = max(0.1, min(5.0, float(frequency)))
    ph = float(phase)
    
    # Compute normalized weights
    weights = []
    for i in range(n):
        pos = i / max(n - 1, 1)
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * freq * pos + ph)
        # Interpolate between uniform (1/n) and wave-modulated
        w = (1 - s) / n + s * wave
        weights.append(w)
    
    # Normalize weights to sum to 1
    total = sum(weights)
    weights = [w / total for w in weights]
    
    # Weighted sum
    result = sum(w * m for w, m in zip(weights, models))
    return result
