import torch
from sd_mecha import merge_method, Parameter, Return
from torch import Tensor

def _get_svd_approx(
    diff: Tensor, 
    rank: int, 
    svd_mode: str, 
    oversample: int, 
    device: str, 
    quantile: float
) -> Tensor:
    """
    Internal helper to compute SVD approximation of a tensor difference.
    Returns the reconstructed tensor (U @ S @ Vh) shaped like the input.
    """
    # 1. Handle shapes (Linear vs Conv2d)
    orig_shape = diff.shape
    conv2d = (diff.ndim == 4)
    
    # If it's a bias or 1D/0D tensor, we cannot really SVD it effectively 
    # in the context of LoRA (usually passed through as full diff).
    if diff.ndim < 2:
        return diff

    kernel_size = None if not conv2d else diff.size()[2:4]
    conv2d_3x3 = conv2d and kernel_size != (1, 1)
    
    out_dim, in_dim = diff.size()[:2]
    
    # Reshape to Matrix
    x = diff.detach()
    if conv2d:
        if conv2d_3x3:
            x = x.flatten(start_dim=1)
        else:
            x = x.squeeze()
    
    # 2. Move to SVD device
    # Note: sd_mecha input tensors might be on various devices; 
    # we force the SVD computation to the requested device (e.g. 'cpu' to save vram).
    x = x.to(device=device, dtype=torch.float32)

    # Sanitize
    if not torch.isfinite(x).all():
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

    # 3. Perform SVD
    current_rank = min(rank, x.shape[0], x.shape[1])
    
    try:
        if svd_mode == "lowrank" and hasattr(torch.linalg, "svd_lowrank"):
            q = min(current_rank + oversample, min(x.shape))
            U, S, V = torch.linalg.svd_lowrank(x, q=q, niter=2)
            Vh = V.T
            U, S, Vh = U[:, :current_rank], S[:current_rank], Vh[:current_rank, :]
        else:
            # Fallback or exact mode
            kwargs = {"driver": "gesvd"} if device == "cuda" else {}
            U, S, Vh = torch.linalg.svd(x, **kwargs)
            U, S, Vh = U[:, :current_rank], S[:current_rank], Vh[:current_rank, :]
    except Exception:
        # Final fallback
        U, S, Vh = torch.linalg.svd(x)
        U, S, Vh = U[:, :current_rank], S[:current_rank], Vh[:current_rank, :]

    # Merge S into U for U*Vh reconstruction style (standard LoRA style)
    # U becomes (A, r), Vh becomes (r, B)
    U = U * S.unsqueeze(0)

    # 4. Clamp outliers (Quantile)
    if quantile < 1.0:
        hi_u = torch.quantile(U.abs().reshape(-1), quantile)
        hi_v = torch.quantile(Vh.abs().reshape(-1), quantile)
        hi_val = float(max(hi_u, hi_v))
        low_val = -hi_val
        U = U.clamp_(low_val, hi_val)
        Vh = Vh.clamp_(low_val, hi_val)

    # 5. Reconstruct Approximation
    # Result = U @ Vh
    approx = torch.mm(U, Vh)

    # 6. Reshape back to original
    if conv2d:
        if conv2d_3x3:
            approx = approx.reshape(out_dim, in_dim, *kernel_size)
        else:
            approx = approx.reshape(out_dim, in_dim, 1, 1)
    
    # Cast back to original dtype and device logic is handled by return, 
    # but we ensure the result is on the same device as the input 'diff' originally was
    # or let sd_mecha handle the transition.
    return approx.to(device=diff.device, dtype=diff.dtype)


@merge_method
def lora_extract(
    model: Parameter(Tensor),
    base: Parameter(Tensor),
    rank: Parameter(int) = 8,
    svd_mode: Parameter(str) = "lowrank",
    oversample: Parameter(int) = 8,
    device: Parameter(str) = "cpu",
    quantile: Parameter(float) = 0.99,
) -> Return(Tensor):
    """
    Calculates the Low-Rank Approximation of the difference between Model and Base.
    
    Math: 
        Diff = Model - Base
        Approx = SVD_Reconstruct(Diff, rank)
        Result = Approx
    
    The output is a tensor representing the LoRA delta. 
    """
    
    # Compute raw difference
    diff = model - base
    
    # Compute LoRA approximation
    # We do this strictly inside the function to ensure it runs during execution
    result = _get_svd_approx(
        diff, 
        rank=int(rank), 
        svd_mode=str(svd_mode), 
        oversample=int(oversample), 
        device=str(device), 
        quantile=float(quantile)
    )
    
    return result