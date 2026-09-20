from pathlib import Path

import nibabel as nib
import numpy as np

from brain_mri.data.preprocessing import clip_percentiles, z_score_normalise

MRI_MODALITIES = ["T1", "T1CE", "T2", "FLAIR"]

MODALITY_SUFFIXES = {
    "T1" : "t1n",
    "T1CE" : "t1c",
    "T2" : "t2w",
    "FLAIR" : "t2f",
    "SEG" : "seg"
}

def find_patient_files(patient_dir : Path) -> dict:
    patient_files = {}

    for file_path in patient_dir.iterdir():
        if not file_path.is_file():
            continue

        name_without_ext = file_path.name.replace(".nii.gz", "").replace(".nii", "").lower()

        for modality, suffix in MODALITY_SUFFIXES.items():
            if name_without_ext.endswith(suffix):
                patient_files[modality] = file_path
                break

    return patient_files

def load_patient(patient_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    if not patient_dir.is_dir():
        raise FileNotFoundError(f"Invalid patient directory: {patient_dir}")
    
    patient_files = find_patient_files(patient_dir)
    required_files = set(MRI_MODALITIES + ["SEG"])
    missing_files = required_files - set(patient_files.keys())

    if missing_files:
        raise ValueError(f"Missing required files for patient: {missing_files}")

    modalities = []
    brain = None
    for modality in MRI_MODALITIES:
        volume = nib.load(patient_files[modality]).get_fdata(dtype=np.float32)

        tissue_mask = (volume > 0).astype(np.uint8)
        brain = tissue_mask if brain is None else brain | tissue_mask

        volume = clip_percentiles(volume)
        volume = z_score_normalise(volume)
        modalities.append(volume)

    segmentation = nib.load(patient_files["SEG"]).get_fdata(dtype=np.float32)
    segmentation = segmentation.astype(np.int16)

    images = np.stack(modalities, axis=0)
    if images.shape[1:] != segmentation.shape:
        raise ValueError(f"Image and segmentation shapes do not match: {images.shape[1:]} vs {segmentation.shape}")

    return images, segmentation, brain


