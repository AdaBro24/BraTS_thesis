from pathlib import Path
import nibabel as nib
import numpy as np

from brain_mri.data.preprocessing import z_score_normalise

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
    patient_files = find_patient_files(patient_dir)
    modalities = []

    for modality in MRI_MODALITIES:
        volume = nib.load(patient_files[modality]).get_fdata(dtype=np.float32)
        volume = z_score_normalise(volume)
        modalities.append(volume)

    segmentation = nib.load(patient_files["SEG"]).get_fdata(dtype=np.int16)
    return np.stack(modalities, axis=0), segmentation


