"""Locate the integrated MEDUSA historical runtime without a working-directory dependency."""
from pathlib import Path


def bundled_medusa_root() -> Path:
    return Path(__file__).resolve().parent / '_vendor' / 'medusa'
