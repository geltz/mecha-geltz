"""
Consistency: consistency-filtered delta merge.

Idea:
- Take the first model as base.
- For every other model, compute its delta to the base.
- Estimate how "consistent" that delta is by looking at its magnitude distribution.
- Large, spiky deltas are treated as low-confidence and shrunk.
- Add the cleaned deltas back to the base.

This is meant for the 2-checkpoint case (base, finetune) where the finetune may contain
some overfitted or noisy changes, and you want a version that stays close to base while
keeping the smoother parts of the finetune.

Params:
- alpha: global strength of how much cleaned delta to add.
- k: how aggressively to treat big deviations as suspicious.
- eps: numerical floor.
"""

import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor


def _consistency_mask(delta: Tensor, k: float, eps: float) -> Tensor:
    """
    Build a per-tensor mask in [0,1] that downweights unusually large deltas.

    Steps:
    - take absolute delta
    - get median and MAD of magnitudes
    - define a scale = median + k * MAD
    - score = |delta| / scale
    - mask = 1 / (1 + score)

    So small/typical changes -> mask ~ 1
    Large/unusual changes   -> mask < 1
    """
    d_abs = delta.abs()
    flat = d_abs.flatten()

    # median(|delta|)
    med = flat.median()

    # MAD of |delta| around its median
    mad = (flat - med).abs().median()

    scale = med + k * mad
    scale = scale.clamp_min(eps)

    score = d_abs / scale
    mask = 1.0 / (1.0 + score)
    return mask


@merge_method
def consistency(
    *models: Parameter(Tensor),
    alpha: Parameter(float) = 1.0,
    k: Parameter(float) = 2.0,
    eps: Parameter(float) = 1e-8,
) -> Return(Tensor):
    """
    Consistency-filtered delta merge.

    Usage patterns:
    - consistency(base, finetune): produce a version close to base but with the
      smooth parts of finetune.
    - consistency(base, ft1, ft2, ...): average multiple finetunes but suppress
      any one of them that makes very inconsistent edits.

    alpha: overall blend strength for cleaned deltas.
    k: higher = more tolerant, lower = more aggressive.
    eps: numerical stability.
    """
    n = len(models)
    if n == 0:
        raise ValueError("consistency requires at least one model")
    if n == 1:
        return models[0]

    base = models[0]
    out = base.clone()

    # normalize by number of extra models, like other merges in this set
    denom = max(1, n - 1)
    a = float(alpha)
    kk = float(k)
    ee = float(eps)

    for model in models[1:]:
        delta = model - base
        mask = _consistency_mask(delta, kk, ee)
        delta_clean = delta * mask
        out = out + (a * delta_clean) / denom

    return out
