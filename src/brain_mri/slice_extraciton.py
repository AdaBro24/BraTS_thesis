from pathlib import Path
import json

import numpy as np

from brain_mri.data.nifti_loader import load_patient
from brain_mri.data.preprocessing import (
    crop_or_pad_slice,
    validate_slice
)

def process_single_slice(
        images: np.ndarray,
        segmentation: np.ndarray,
        z_idx: int,
        target_size: tuple[int,int] = (224,224)
    ) -> tuple[np.ndarray, np.ndarray] | None:

    image_slice = images[..., z_idx]
    mask_slice = segmentation[..., z_idx]

    image_slice = crop_or_pad_slice(image_slice, target_size)
    mask_slice = crop_or_pad_slice(mask_slice, target_size)

    if not validate_slice(image_slice, mask_slice):
        return None

    return image_slice.astype(np.float32), mask_slice.astype(np.int16)


def get_slice_indices(patient_id: str, boundaries: dict[str, list[int]]) -> range[int]:
    if patient_id not in boundaries:
        raise KeyError(f"Missing Z boundaries for {patient_id}")

    first_z, last_z = boundaries[patient_id]
    if first_z > last_z:
        raise ValueError(f"Invalid boundaries for {patient_id}, {first_z} > {last_z}")
    return range([first_z, last_z + 1])

def _load_z_boundaries(json_file: Path) -> dict[str, list[int]]:
    with open(json_file, "r", encoding="uft-8") as file:
        return json.load(file)

def extract_patient_slices(patient_dir: str, 
                           boundaries_path: Path,
                           target_size: tuple[int,int] = (224,224)
                           ) -> list[dict[str, np.ndarray]]:
    patient_id = patient_dir.name
    boundaries = _load_z_boundaries(boundaries_path)

    images, segmentation = load_patient(patient_id)
    slice_indices = get_slice_indices(patient_id, boundaries)

    extracted_slices = []

    for z_idx in slice_indices:
        result = process_single_slice(
            images=images,
            segmentation=segmentation,
            z_idx=z_idx,
            target_size=target_size
        )

        if result is None:
            continue
        image_slice, mask_slice = result
        extracted_slices.append(
            {
                "patient_id" : patient_id,
                "image" : image_slice,
                "mask" : mask_slice,
                "z_idx" : z_idx
            }
        )
    return extracted_slices
