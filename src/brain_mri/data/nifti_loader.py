from pathlib import Path
import numpy as np
import nibabel as nib

from brain_mri.data.preprocessing import z_score_normalise, clip_percentiles

MRI_MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]

MODALITY_SUFFIXES = {
    "T1": "t1n",
    "T1CE": "t1c",
    "T2": "t2w",
    "FLAIR": "t2f",
    "SEG": "seg",
}


def find_patient_files(patient_dir: Path) -> dict:
    """Discovers and maps MRI modality and segmentation files for a patient.

    Scans the given directory for NIfTI files (.nii, .nii.gz) and associates
    each file path with its corresponding modality identifier based on predefined
    filename suffixes.

    Args:
        patient_dir: Path to the directory containing a single patient's scans.

    Returns:
        A dictionary mapping modality names (e.g., 'T1', 'T1CE', 'T2', 'FLAIR', 'SEG')
        to their corresponding Path objects.
    """
    patient_files = {}

    for file_path in patient_dir.iterdir():
        if not file_path.is_file():
            continue

        name_without_ext = (
            file_path.name.replace(".nii.gz", "").replace(".nii", "").lower()
        )

        for modality, suffix in MODALITY_SUFFIXES.items():
            if name_without_ext.endswith(suffix):
                patient_files[modality] = file_path
                break

    return patient_files


def load_patient(patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """Loads, normalises, and stacks all MRI modalities and segmentation for a patient.

    Validates that all required modality files and the segmentation file exist in
    the patient directory. Applies percentile clipping and foreground Z-score
    normalisation to each modality, then stacks them into a single 4D array.

    Args:
        patient_dir: Path to directory containing patient NIfTI files.

    Returns:
        tuple:
            - images (np.ndarray): Normalised modalities stacked as (C, H, W, Z) float32.
            - segmentation (np.ndarray): Segmentation mask volume as (H, W, Z) int16.

    Raises:
        FileNotFoundError: If patient_dir does not exist.
        ValueError: If required files are missing or image/mask dimensions mismatch.
    """
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Invalid patient directory: {patient_dir}")

    patient_files = find_patient_files(patient_dir)
    required_files = set(MRI_MODALITIES + ["SEG"])
    missing_files = required_files - set(patient_files.keys())

    if missing_files:
        raise ValueError(f"Missing required files for patient: {missing_files}")

    modalities = []
    for modality in MRI_MODALITIES:
        volume = nib.load(patient_files[modality]).get_fdata(dtype=np.float32)
        volume = clip_percentiles(volume)
        volume = z_score_normalise(volume)
        modalities.append(volume)

    segmentation = nib.load(patient_files["SEG"]).get_fdata(dtype=np.float32)
    segmentation = segmentation.astype(np.int16)

    images = np.stack(modalities, axis=0)
    if images.shape[1:] != segmentation.shape:
        raise ValueError(
            f"Image and segmentation shapes do not match: {images.shape[1:]} vs {segmentation.shape}"
        )

    return images, segmentation
