from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ClipPercentilesConfig:
    low: float = 1.0
    high: float = 99.0


@dataclass
class PathsConfig:
    z_boundaries: Path = Path("data/patients_z_boundaries.json")
    splits: Path = Path("data/data_splits.json")


@dataclass
class DataConfig:
    target_size: tuple[int, int] = (224, 224)
    clip_percentiles: ClipPercentilesConfig = field(default_factory=ClipPercentilesConfig)
    modalities: list[str] = field(default_factory=lambda: ["T1", "T1CE", "T2", "FLAIR"])
    paths: PathsConfig = field(default_factory=PathsConfig)

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)


def get_default_config_path() -> Path:
    """Find configs/data_config.yaml relative to current working directory or package root."""
    cwd_path = Path.cwd() / "configs" / "data_config.yaml"
    if cwd_path.exists():
        return cwd_path

    # Try 3 levels up from this file (src/brain_mri/config.py -> root)
    root_path = Path(__file__).resolve().parents[2] / "configs" / "data_config.yaml"
    if root_path.exists():
        return root_path

    return cwd_path


def load_data_config(config_path: str | Path | None = None) -> DataConfig:
    """Load data configuration from a YAML file.

    If config_path is None, attempts to locate configs/data_config.yaml automatically.
    """
    path = Path(config_path) if config_path is not None else get_default_config_path()

    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    target_size = tuple(raw.get("target_size", [224, 224]))
    if len(target_size) != 2:
        raise ValueError(f"target_size must have 2 elements (H, W), got {target_size}")

    clip_dict = raw.get("clip_percentiles", {})
    clip_percentiles = ClipPercentilesConfig(
        low=float(clip_dict.get("low", 1.0)),
        high=float(clip_dict.get("high", 99.0)),
    )

    modalities = list(raw.get("modalities", ["T1", "T1CE", "T2", "FLAIR"]))

    paths_dict = raw.get("paths", {})
    paths = PathsConfig(
        z_boundaries=Path(paths_dict.get("z_boundaries", "data/patients_z_boundaries.json")),
        splits=Path(paths_dict.get("splits", "data/data_splits.json")),
    )

    return DataConfig(
        target_size=(int(target_size[0]), int(target_size[1])),
        clip_percentiles=clip_percentiles,
        modalities=modalities,
        paths=paths,
    )

