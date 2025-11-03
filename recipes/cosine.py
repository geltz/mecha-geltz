"""
Cosine: wavelength with dot product alignment weighting.

Idea:
- Apply sinusoidal weighting modulated by geometric alignment.
- Deltas aligned with reference direction get stronger influence.
- Wave pattern combined with cosine similarity.

Computation:
- base = mean of all models
- reference = first model's delta direction
- For each model, compute delta from base
- Weight by sine wave * dot product with reference
- Emphasizes models that align geometrically
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import math

@merge_method
def cosine(
    *models: Parameter(Tensor),
    strength: Parameter(float) = 0.7,
    frequency: Parameter(float) = 1.0,
    phase: Parameter(float) = 0.0
) -> Return(Tensor):
    n = len(models)
    if n == 0:
        raise ValueError("cosine requires at least one model")
    if n == 1:
        return models[0]
    
    s = max(0.0, min(1.0, float(strength)))
    freq = max(0.1, min(5.0, float(frequency)))
    ph = float(phase)
    
    base = torch.stack(models, dim=0).mean(dim=0)
    
    # Get reference direction from first model
    ref_delta = models[0] - base
    ref_norm = torch.norm(ref_delta)
    if ref_norm < 1e-8:
        return base
    ref_dir = ref_delta / ref_norm
    
    result = base.clone()
    for i, model in enumerate(models):
        pos = i / max(n - 1, 1)
        wave = 0.5 + 0.5 * math.sin(2 * math.pi * freq * pos + ph)
        
        delta = model - base
        delta_norm = torch.norm(delta)
        
        if delta_norm > 1e-8:
            # Dot product alignment: [-1, 1] -> [0, 1]
            delta_dir = delta / delta_norm
            alignment = (torch.dot(delta_dir.flatten(), ref_dir.flatten()) + 1) / 2
            
            weight = s * wave * alignment
            result = result + delta * weight / n
    
    return result
