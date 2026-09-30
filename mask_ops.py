"""Mask resampling for small defects at the model decoder resolution."""

import torch
import torch.nn.functional as F


def preserve_defects_at_decoder_scale(mask: torch.Tensor, size: int) -> torch.Tensor:
    """Keep a decoder cell positive if any source mask pixel in it is positive.

    Bilinear interpolation followed by binarization can erase sub-cell scratches.
    This intentionally expands a fine mask to its intersecting decoder cells.
    """
    if mask.ndim != 3 or mask.shape[0] != 1:
        raise ValueError("Expected a single-channel CHW mask")
    if size <= 0:
        raise ValueError("size must be positive")
    return F.adaptive_max_pool2d(mask, (size, size))


def masked_mean(loss: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
    """Mean pixel loss only over trustworthy labels, preserving old all-valid behavior."""
    if loss.shape != valid_mask.shape:
        raise ValueError("Loss and valid mask must have the same shape")
    if not torch.isfinite(valid_mask).all() or (valid_mask < 0).any() or (valid_mask > 1).any():
        raise ValueError("Valid mask must contain finite values in [0, 1]")
    return (loss * valid_mask).sum() / valid_mask.sum().clamp_min(1)
