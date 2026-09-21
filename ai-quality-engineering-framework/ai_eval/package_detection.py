from __future__ import annotations

import importlib.util
from importlib import metadata


def is_package_available(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def package_version(distribution_name: str) -> str | None:
    try:
        return metadata.version(distribution_name)
    except metadata.PackageNotFoundError:
        return None
