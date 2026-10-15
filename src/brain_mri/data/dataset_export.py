import sys
from pathlib import Path

import torch
from tqdm import tqdm

from brain_mri.config import DataConfig, load_data_config
from brain_mri.data.slice_extraction import extract_patient_slices


def save_patients_to_pt(slices: list[dict], output_dir: Path) -> list[Path]:
    """Saves list of patient's slices as .pt files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths = []

    for item in slices:
        patient_id = item["patient_id"]
        z_idx = item["z_idx"]

        # NumPy -> PyTorch conversion
        image_tensor = torch.from_numpy(item["image"]).float()  # (4, 224, 224), float32
        mask_tensor = torch.from_numpy(item["mask"]).long()     # (224, 224), int64

        file_path = output_dir / f"{patient_id}_z{z_idx:03d}.pt"

        # Using Python's file handler prevents Google Drive FUSE driver C++ open errors
        with open(file_path, "wb") as f:
            torch.save(
                {
                    "image": image_tensor,
                    "mask": mask_tensor,
                    "patient_id": patient_id,
                    "z_idx": z_idx,
                },
                f,
            )
        saved_paths.append(file_path)
    return saved_paths


def export_dataset_split(
    data_root: Path,
    splits: dict[str, list[str]],
    output_root: Path = Path("data/processed"),
    config: DataConfig | None = None,
) -> dict[str, int]:
    """Exports slices to subsets (train, val, test) with progress bar and ETA."""
    data_root = Path(data_root)
    output_root = Path(output_root)
    cfg = config or load_data_config()
    stats = {}

    for split_name, patient_ids in splits.items():
        split_dir = output_root / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        total_slices = 0
        total_patients = len(patient_ids)

        print(f"\n[INFO] Starting export for split '{split_name}': {total_patients} patients...", flush=True)

        pbar = tqdm(
            enumerate(patient_ids, 1),
            total=total_patients,
            desc=f"Exporting {split_name}",
            unit="patient",
            file=sys.stdout,
            dynamic_ncols=True,
        )
        for idx, patient_id in pbar:
            patient_dir = data_root / patient_id
            if not patient_dir.exists():
                pbar.write(f"Warning: Missing patient directory {patient_dir}")
                continue

            slices = extract_patient_slices(patient_dir, config=cfg)
            patient_out_dir = split_dir / patient_id
            saved = save_patients_to_pt(slices, patient_out_dir)
            total_slices += len(saved)
            pbar.set_postfix(slices=total_slices)

            # Periodic log every 50 patients to ensure progress is visible even if \r is buffered
            if idx % 50 == 0 or idx == total_patients:
                print(f" -> [{split_name}] Completed {idx}/{total_patients} patients ({total_slices} total slices saved)", flush=True)

        stats[split_name] = total_slices
        print(f"[INFO] Finished split '{split_name}': {total_slices} slices saved.", flush=True)

    return stats