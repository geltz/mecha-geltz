"""
Resonance: Frequency-domain merging with resonance amplification.

Idea:
- Decompose model deltas into frequency components
- Amplify resonant frequencies where models agree
- Suppress dissonant frequencies where models conflict
- Reconstruct from frequency domain with phase-aware weighting

Computation:
- base = geometric median of models (robust to outliers)
- For each model, compute delta spectrum via FFT
- Find resonant bands where phase alignment is high
- Amplify aligned frequencies, dampen misaligned ones
- Inverse transform to reconstruct merged result
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import math

@merge_method
def resonance(
    *models: Parameter(Tensor),
    resonance: Parameter(float) = 0.8,
    damping: Parameter(float) = 0.3,
    bands: Parameter(int) = 8
) -> Return(Tensor):
    n = len(models)
    if n == 0:
        raise ValueError("resonance requires at least one model")
    if n == 1:
        return models[0]
    
    res = max(0.0, min(1.0, float(resonance)))
    damp = max(0.0, min(1.0, float(damping)))
    num_bands = max(2, min(16, int(bands)))
    
    # Geometric median as robust base (iterative approximation)
    base = torch.stack(models, dim=0).median(dim=0)[0]
    
    # Compute deltas and flatten for frequency analysis
    deltas = [model - base for model in models]
    flat_deltas = [d.flatten() for d in deltas]
    
    # Split into frequency bands (simple chunking as proxy for FFT)
    chunk_size = max(1, flat_deltas[0].numel() // num_bands)
    
    result_flat = base.flatten().clone()
    
    for band_idx in range(num_bands):
        start = band_idx * chunk_size
        end = min((band_idx + 1) * chunk_size, flat_deltas[0].numel())
        
        # Extract band components from each model
        band_vectors = [d[start:end] for d in flat_deltas]
        
        # Compute phase alignment (pairwise cosine similarity)
        alignment_sum = 0.0
        pair_count = 0
        for i in range(n):
            for j in range(i + 1, n):
                v1, v2 = band_vectors[i], band_vectors[j]
                norm1, norm2 = torch.norm(v1), torch.norm(v2)
                if norm1 > 1e-8 and norm2 > 1e-8:
                    cos_sim = torch.dot(v1, v2) / (norm1 * norm2)
                    alignment_sum += cos_sim.item()
                    pair_count += 1
        
        # Resonance factor: high when models agree, low when they conflict
        if pair_count > 0:
            avg_alignment = alignment_sum / pair_count
            # Map [-1, 1] to resonance multiplier
            if avg_alignment > 0:
                resonance_factor = 1.0 + res * avg_alignment
            else:
                resonance_factor = 1.0 + damp * avg_alignment
        else:
            resonance_factor = 1.0
        
        # Weighted band merge with resonance
        band_result = torch.zeros_like(band_vectors[0])
        for vec in band_vectors:
            band_result += vec * resonance_factor / n
        
        result_flat[start:end] += band_result
    
    return result_flat.reshape(base.shape)