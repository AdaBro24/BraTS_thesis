import json
from pathlib import Path

import numpy as np

from brain_mri.data.nifti_loader import load_patient
from brain_mri.data.preprocessing import crop_or_pad_slice, validate_slice


def process_single_slice(
    images: np.ndarray,
    segmentation: np.ndarray,
    z_idx: int,
    target_size: tuple[int, int] = (224, 224),
) -> tuple[np.ndarray, np.ndarray] | None:
    """Extracts, crops/pads, and validates a single 2D axial slice.

    Args:
        images: 4D MRI volume with shape (C, H, W, Z).
        segmentation: 3D segmentation mask with shape (H, W, Z).
        z_idx: Slice index along the axial (Z) axis.
        target_size: Target spatial dimensions (H, W). Defaults to (224, 224).

    Returns:
        tuple: (image_slice, mask_slice) as float32 and int16, or None if validation fails.
    """
    image_slice = images[..., z_idx]
    mask_slice = segmentation[..., z_idx]

    image_slice = crop_or_pad_slice(image_slice, target_size)
    mask_slice = crop_or_pad_slice(mask_slice, target_size, pad_value=0)

    if not validate_slice(image_slice, mask_slice):
        return None

    return image_slice.astype(np.float32), mask_slice.astype(np.int16)


def get_slice_indices(patient_id: str, boundaries: dict[str, list[int]]) -> range:
    """Returns the valid axial slice index range for a given patient.

    Args:
        patient_id: Patient identifier string.
        boundaries: Dictionary mapping patient IDs to [first_z, last_z] boundary pairs.

    Returns:
        range: Inclusive range of indices from first_z to last_z.

    Raises:
        KeyError: If patient_id is not found in boundaries.
        ValueError: If first_z is greater than last_z.
    """
    if patient_id not in boundaries:
        raise KeyError(f"Missing Z boundaries for {patient_id}")

    first_z, last_z = boundaries[patient_id]
    if first_z > last_z:
        raise ValueError(f"Invalid boundaries for {patient_id}, {first_z} > {last_z}")
    return range(first_z, last_z + 1)


def _load_z_boundaries(json_file: Path) -> dict[str, list[int]]:
    """Loads precomputed axial slice boundaries from a JSON file.

    Args:
        json_file: Path to the JSON boundaries file.

    Returns:
        dict: Mapping of patient IDs to [first_z, last_z] pairs.
    """
    with open(json_file, "r", encoding="utf-8") as file:
        return json.load(file)


def extract_patient_slices(
    patient_dir: Path,
    boundaries_path: Path,
    target_size: tuple[int, int] = (224, 224),
) -> list[dict]:
    """Loads a patient's volume and extracts all valid processed 2D slices.

    Args:
        patient_dir: Directory containing the patient's NIfTI files.
        boundaries_path: Path to the JSON file with axial slice bounds.
        target_size: Target spatial dimensions (H, W). Defaults to (224, 224).

    Returns:
        list[dict]: List of dictionaries containing patient_id, image, mask, and z_idx.
    """
    patient_id = patient_dir.name
    boundaries = _load_z_boundaries(boundaries_path)

    images, segmentation = load_patient(patient_dir)
    slice_indices = get_slice_indices(patient_id, boundaries)

    extracted_slices = []

    for z_idx in slice_indices:
        result = process_single_slice(
            images=images,
            segmentation=segmentation,
            z_idx=z_idx,
            target_size=target_size,
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
