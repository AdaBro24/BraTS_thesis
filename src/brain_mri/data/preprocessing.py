"""Preprocessing utilities for brain MRI and segmentation masks."""

import numpy as np


def z_score_normalise(volume: np.ndarray, epsilon: float = 1e-8) -> np.ndarray:
    mask = volume > 0
    normalised = np.zeros_like(volume, dtype=np.float32)

    if not np.any(mask):
        return normalised

    mean = volume[mask].mean()
    std = volume[mask].std()

    normalised[mask] = (volume[mask] - mean) / (std + epsilon)
    # Background outside the brain mask remains 0.0 (from np.zeros_like)
    return normalised

# Not sure if i need that
def clip_percentiles(volume: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    mask = volume > 0

    if not np.any(mask):
        return volume.astype(np.float32)

    lower, upper = np.percentile(volume[mask], [low, high])

    clipped = volume.copy()
    clipped[mask] = np.clip(volume[mask], lower, upper)

    return clipped.astype(np.float32) 

def cut_volume(volume: np.ndarray, lower_idx: int, greater_idx: int) -> np.ndarray: 
    return volume[..., lower_idx : greater_idx + 1]

def get_slice(volume: np.ndarray, z_idx: int):
    return volume[..., z_idx]

def crop_or_pad_slice(
    slice2d: np.ndarray,
    target_size: tuple[int, int] = (224, 224),
    pad_value: float = 0.0,
) -> np.ndarray:
    target_h, target_w = target_size
    h, w = slice2d.shape[-2:]

    if h > target_h:
        start_h = (h - target_h) // 2
        slice2d = slice2d[..., start_h:start_h + target_h, :]

    if w > target_w:
        start_w = (w - target_w) // 2
        slice2d = slice2d[..., :, start_w:start_w + target_w]

    h, w = slice2d.shape[-2:]

    pad_h = target_h - h
    pad_w = target_w - w

    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top

    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    pad_widths = (
        [(0, 0)] * (slice2d.ndim - 2)
        + [(pad_top, pad_bottom), (pad_left, pad_right)]
    )

    return np.pad(
        slice2d,
        pad_widths,
        mode="constant",
        constant_values=pad_value,
    )

def validate_slice(
    slice2d: np.ndarray,
    mask_slice: np.ndarray,
    expected_modalities: int = 4,
) -> bool:
    if slice2d.ndim != 3:
        return False

    if slice2d.shape[0] != expected_modalities:
        return False

    if mask_slice.ndim != 2:
        return False

    if slice2d.shape[1:] != mask_slice.shape:
        return False

    if not np.isfinite(slice2d).all():
        return False

    return bool(np.isfinite(mask_slice).all())
