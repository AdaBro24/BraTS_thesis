import json
import os
from pathlib import Path

from sklearn.model_selection import train_test_split


def split_patients(
    patient_dirs: list[Path], seed: int = 42
) -> tuple[list[Path], list[Path], list[Path]]:
    """Splits a list of patient directories into train, validation, and test sets.

    Divides patients using a 70/15/15 ratio (70% train, 15% validation, 15% test)
    based on patient-level granularity to avoid data leakage between splits.

    Args:
        patient_dirs: List of directory paths, each corresponding to one patient.
        seed: Random seed for reproducible splitting. Defaults to 42.

    Returns:
        tuple: (train_patients, val_patients, test_patients) as lists of Paths.
    """
    train_patients, remaining_patients = train_test_split(
        patient_dirs, train_size=0.7, random_state=seed
    )

    val_patients, test_patients = train_test_split(
        remaining_patients, train_size=0.5, random_state=seed
    )

    return train_patients, val_patients, test_patients


ZIP_DIR = Path("/content/drive/MyDrive/engineering_thesis/brats")
GLI_ZIP = ZIP_DIR / "BraTS2024-BraTS-GLI-TrainingData.zip"
MEN_ZIP = ZIP_DIR / "BraTS2024-MEN-RT-TrainingData.zip"
DATA_ROOT = ZIP_DIR / "unpacked"
GLI_ROOT = DATA_ROOT / "BraTS-GLI" / "training_data1_v2"


def get_or_create_splits(
    folder_path: Path = GLI_ROOT,
    json_path: str
    | Path = "/content/drive/MyDrive/engineering_thesis/brats/data_splits.json",
) -> dict[str, list[str]]:
    """Loads existing patient data splits from JSON or computes and caches new ones.

    If the split file already exists, it loads the saved patient IDs. Otherwise,
    it discovers all patient directories in folder_path, generates a reproducible
    70/15/15 split, saves it to json_path, and returns the mappings.

    Args:
        folder_path: Path to the directory containing unpacked patient folders.
        json_path: Destination path where the split mapping JSON is read or saved.

    Returns:
        Dictionary with keys 'train', 'validate', and 'test', each containing
        a list of patient ID strings.
    """
    json_path = Path(json_path)
    splitted = {}

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            splitted["train"] = data["train"]
            splitted["validate"] = data["validate"]
            splitted["test"] = data["test"]
    else:
        patient_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
        train_patients, val_patients, test_patients = split_patients(patient_dirs)

        splitted["train"] = [p.name for p in train_patients]
        splitted["validate"] = [p.name for p in val_patients]
        splitted["test"] = [p.name for p in test_patients]

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(splitted, f, indent=4)

    return splitted
