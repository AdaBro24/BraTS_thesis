"""Re-export slice extraction utilities for backward compatibility."""

from brain_mri.data.slice_extraction import (
    extract_patient_slices,
    get_slice_indices,
    process_single_slice,
)

__all__ = [
    "extract_patient_slices",
    "get_slice_indices",
    "process_single_slice",
]
