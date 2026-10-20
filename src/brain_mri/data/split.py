import json
from pathlib import Path

from sklearn.model_selection import train_test_split

# Default local paths (can be overridden in function calls)
DEFAULT_SPLIT_JSON = Path("data/data_splits.json")


def subject_id(scan_name: str) -> str:
    """BraTS-GLI-00463-101 -> BraTS-GLI-00463 (one subject can have several timepoint scans)."""
    return scan_name.rsplit("-", 1)[0]


def split_patients(
    patient_dirs: list[Path],
    seed: int = 42,
) -> tuple[list[Path], list[Path], list[Path]]:
    """Splits scans 70/15/15 by subject, so all timepoints of a subject land in the same split."""
    subjects = sorted({subject_id(p.name) for p in patient_dirs})

    train_subjects, remaining_subjects = train_test_split(
        subjects,
        train_size=0.7,
        random_state=seed,
    )

    val_subjects, test_subjects = train_test_split(
        remaining_subjects,
        train_size=0.5,
        random_state=seed,
    )

    def scans_of(split_subjects: list[str]) -> list[Path]:
        split_subjects = set(split_subjects)
        return [p for p in patient_dirs if subject_id(p.name) in split_subjects]

    return scans_of(train_subjects), scans_of(val_subjects), scans_of(test_subjects)


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