import json
from pathlib import Path

from sklearn.model_selection import train_test_split

# Default local paths (can be overridden in function calls)
DEFAULT_SPLIT_JSON = Path("data/data_splits.json")


def split_patients(
    patient_dirs: list[Path],
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    train_patients, remaining_patients = train_test_split(
        patient_dirs,
        train_size=0.7,
        random_state=seed,
    )

    val_patients, test_patients = train_test_split(
        remaining_patients,
        train_size=0.5,
        random_state=seed,
    )

    return train_patients, val_patients, test_patients


def get_or_create_splits(
    folder_path: Path | str | None = None,
    json_path: Path | str = DEFAULT_SPLIT_JSON,
    seed: int = 42,
) -> dict[str, list[str]]:
    json_path = Path(json_path)
    splitted: dict[str, list[str]] = {}

    if json_path.exists():
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
            splitted["train"] = data["train"]
            splitted["validate"] = data["validate"]
            splitted["test"] = data["test"]
    else:
        if folder_path is None:
            raise ValueError("folder_path must be provided when json_path does not exist.")
        folder_path = Path(folder_path)
        if not folder_path.exists():
            raise FileNotFoundError(f"Folder path for patient data does not exist: {folder_path}")

        patient_dirs = sorted([p for p in folder_path.iterdir() if p.is_dir()])
        train_patients, val_patients, test_patients = split_patients(patient_dirs, seed=seed)
        splitted["train"] = [p.name for p in train_patients]
        splitted["validate"] = [p.name for p in val_patients]
        splitted["test"] = [p.name for p in test_patients]

        json_path.parent.mkdir(parents=True, exist_ok=True)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(splitted, f, indent=2)

    return splitted