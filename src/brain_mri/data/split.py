from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import json

def split_patients(patient_dirs: list[Path], seed: int = 42) -> tuple[list[Path], list[Path], list[Path]]:
    train_patients, remaining_patients = train_test_split(
        patient_dirs, 
        train_size=0.7, 
        random_state=seed
    )

    val_patients, test_patients = train_test_split(
        remaining_patients,
        train_size=0.5,
        random_state=seed
    )

    return train_patients, val_patients, test_patients

ZIP_DIR = Path("/content/drive/MyDrive/engineering_thesis/brats")
GLI_ZIP = ZIP_DIR / "BraTS2024-BraTS-GLI-TrainingData.zip"
MEN_ZIP = ZIP_DIR / "BraTS2024-MEN-RT-TrainingData.zip"
DATA_ROOT = ZIP_DIR / "unpacked"
GLI_ROOT = DATA_ROOT / "BraTS-GLI" / "training_data1_v2"


def get_or_create_splits(folder_path : Path = GLI_ROOT, json_path : str = "/content/drive/MyDrive/engineering_thesis/brats/data_splits.json"):
    splitted = {}

    if os.path.exists(json_path):
        with open(json_path) as f:
            data = json.load(f)
            splitted["train"] = data['train']
            splitted["validate"]  = data["validate"]
            splitted["test"] = data["test"]
    else:
        patient_dirs = [p for p in folder_path.iterdir() if p.is_dir()]
        train_patients, val_patients, test_patients = split_patients(patient_dirs)
        splitted["train"] = [p.name for p in train_patients]
        splitted["val"] = [p.name for p in val_patients]
        splitted["test"] = [p.name for p in test_patients]

        if json_path:
            with open(json_path, "w") as f:
                json.dump(splitted, f)

    return splitted