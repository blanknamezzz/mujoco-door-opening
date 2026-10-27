"""Resolve case folders while retaining support for legacy flat output directories."""
from pathlib import Path


def case_directory(output, name):
    """Use an existing case subfolder; explicit flat/custom outputs stay flat."""
    output = Path(output)
    nested = output / name
    return nested if nested.is_dir() else output
