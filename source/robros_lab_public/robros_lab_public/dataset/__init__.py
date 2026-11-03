"""Shared canonical motion datasets."""

from __future__ import annotations

from pathlib import Path

DATASET_ROOT = Path(__file__).resolve().parent
IGRIS_C_DATASET_ROOT = DATASET_ROOT / "igris_c"


def igris_c_motion_dir(category: str) -> Path:
    """Return one flat IGRIS-C motion-category directory.

    Categories are deliberately explicit (for example ``amp_locomotion``) so
    callers cannot accidentally combine unrelated motion usages by pointing at
    the dataset root.
    """
    category_path = Path(category)
    if category_path.is_absolute() or ".." in category_path.parts or len(category_path.parts) != 1:
        raise ValueError(f"IGRIS-C dataset category must be one relative name, got {category!r}.")
    return IGRIS_C_DATASET_ROOT / category_path


__all__ = [
    "DATASET_ROOT",
    "IGRIS_C_DATASET_ROOT",
    "igris_c_motion_dir",
]
