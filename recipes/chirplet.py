import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor
import math


@merge_method
def chirplet(
    *models: Parameter(Tensor),
    chirp_rate: Parameter(float) = 1.0,
    sigma: Parameter(float) = 0.2,
    phase_scale: Parameter(float) = 1.0,
) -> Return(Tensor):
    """
    Chirplet-style merge:
    - Take first model as base.
    - For every subsequent model, take its delta to base.
    - Modulate that delta with a chirplet (Gaussian envelope centered farther to the right
      for later models, plus an increasing-frequency chirp).
    - Sum all modulated deltas back onto the base.

    Args:
        chirp_rate: controls how fast frequency increases to the right (t^2 term).
        sigma: width (in [0,1]) of the Gaussian envelope along the flattened parameter axis.
        phase_scale: overall scaling of the starting frequency per model.

    Notes:
        - A "chirp towards the right" is achieved by shifting the Gaussian center of
          later models further to the right, so later models dominate later params.
        - This is intentionally different from the resonance method: no band splitting,
          no pairwise alignment, no geometric median.
    """
    n = len(models)
    if n == 0:
        raise ValueError("chirplet requires at least one model")
    if n == 1:
        return models[0]

    base = models[0]
    base_flat = base.flatten()
    numel = base_flat.numel()

    # timeline over flattened params: 0 ... 1
    t = torch.linspace(
        0.0,
        1.0,
        numel,
        device=base.device,
        dtype=base.dtype,
    )

    # clamp sigma to sane range
    sigma = float(sigma)
    sigma = max(0.01, min(0.5, sigma))

    out_flat = base_flat.clone()

    # each extra model gets its own chirplet
    num_chirps = n - 1
    for idx, model in enumerate(models[1:], start=1):
        delta = (model - base).flatten()

        # center moves right as idx increases
        # idx=1 -> near left, idx=num_chirps -> right
        center = idx / max(1, num_chirps)  # in (0,1]

        # Gaussian envelope around the center
        gauss = torch.exp(-0.5 * ((t - center) / sigma) ** 2)

        # starting frequency grows with idx so later models are "higher"
        f0 = phase_scale * (idx / max(1, num_chirps)) * 0.5  # base frequency
        # linear chirp in t: φ(t) = 2π (f0 t + 0.5 * chirp_rate * t^2)
        phase = 2.0 * math.pi * (f0 * t + 0.5 * float(chirp_rate) * t * t)
        carrier = torch.cos(phase)

        chirplet_win = gauss * carrier

        # apply to delta and add
        out_flat = out_flat + delta * chirplet_win

    return out_flat.reshape(base.shape)
