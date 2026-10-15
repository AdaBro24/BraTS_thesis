from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from brain_mri.config import DataConfig, load_data_config
from brain_mri.data.nifti_loader import load_patient
from brain_mri.data.preprocessing import (
    crop_or_pad_slice,
    validate_slice,
)


def process_single_slice(
    images: np.ndarray,
    segmentation: np.ndarray,
    z_idx: int,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray] | None:
    image_slice = images[..., z_idx]
    mask_slice = segmentation[..., z_idx]

    # Background and padding are consistently 0.0 for images and class 0 for masks
    image_slice = crop_or_pad_slice(image_slice, target_size, pad_value=0.0)
    mask_slice = crop_or_pad_slice(mask_slice, target_size, pad_value=0)

    if not validate_slice(image_slice, mask_slice):
        return None

    return image_slice.astype(np.float32), mask_slice.astype(np.int16)


def get_slice_indices(patient_id: str, boundaries: dict[str, list[int]]) -> range:
    if patient_id not in boundaries:
        raise KeyError(f"Missing Z boundaries for {patient_id}")

    first_z, last_z = boundaries[patient_id]
    if first_z > last_z:
        raise ValueError(f"Invalid boundaries for {patient_id}, {first_z} > {last_z}")
    return range(first_z, last_z + 1)


import functools


@functools.lru_cache(maxsize=4)
def _load_z_boundaries(json_file: Path | str) -> dict[str, list[int]]:
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_patient_slices(
    patient_dir: Path | str,
    boundaries_path: Path | str | None = None,
    target_size: tuple[int, int] | None = None,
    config: DataConfig | None = None,
) -> list[dict]:
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name

    if config is not None:
        cfg = config
    elif boundaries_path is None or target_size is None:
        try:
            cfg = load_data_config()
        except (FileNotFoundError, ValueError):
            cfg = None
    else:
        cfg = None

    resolved_boundaries_path = (
        Path(boundaries_path)
        if boundaries_path is not None
        else (cfg.paths.z_boundaries if cfg else None)
    )
    if resolved_boundaries_path is None:
        raise ValueError("boundaries_path must be provided if no config is available.")

    resolved_target_size = (
        target_size
        if target_size is not None
        else (cfg.target_size if cfg else (224, 224))
    )

    modalities = cfg.modalities if cfg else None
    clip_range = (cfg.clip_percentiles.low, cfg.clip_percentiles.high) if cfg else (1.0, 99.0)

    images, segmentation = load_patient(
        patient_dir,
        modalities=modalities,
        clip_percentiles_range=clip_range,
    )
    boundaries = _load_z_boundaries(resolved_boundaries_path)
    slice_indices = get_slice_indices(patient_id, boundaries)

    extracted_slices = []

    for z_idx in slice_indices:
        result = process_single_slice(
            images=images,
            segmentation=segmentation,
            z_idx=z_idx,
            target_size=resolved_target_size,
        )

        if result is None:
            continue

        image_slice, mask_slice = result
        extracted_slices.append(
            {
                "patient_id": patient_id,
                "image": image_slice,
                "mask": mask_slice,
                "z_idx": z_idx,
            }
        )
    return extracted_slices
