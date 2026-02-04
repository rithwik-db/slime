"""
Databricks Unity Catalog downloader for SLIME training.

Downloads datasets from Databricks Unity Catalog Volumes to local filesystem.
"""

import logging
import os
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

SUPPORTED_PREFIXES = ("dbfs:/Volumes/", "/Volumes/")


def _normalize_path(path: str) -> str:
    """Normalize path to the format expected by Databricks SDK."""
    if path.startswith("dbfs:"):
        path = path[5:]
    return path


def _is_unity_catalog_path(path: str) -> bool:
    """Check if path is a Databricks Unity Catalog volume path."""
    return any(path.startswith(prefix) for prefix in SUPPORTED_PREFIXES)


def download_from_unity_catalog(source: str, destination: str) -> None:
    """
    Download a file from Databricks Unity Catalog to local filesystem.
    """
    dest_path = Path(destination)

    if dest_path.exists():
        logger.info(f"Skipping download, file already exists: {destination}")
        return

    if not _is_unity_catalog_path(source):
        raise ValueError(
            f"Unsupported path: {source}. "
            f"Expected path starting with one of: {SUPPORTED_PREFIXES}"
        )

    try:
        from databricks.sdk import WorkspaceClient
    except ImportError:
        raise ImportError(
            "databricks-sdk is required for Unity Catalog downloads. "
            "Please install it with: pip install databricks-sdk"
        )

    host = os.environ.get("DATABRICKS_HOST")
    token = os.environ.get("DATABRICKS_TOKEN")

    if not host or not token:
        raise EnvironmentError(
            "DATABRICKS_HOST and DATABRICKS_TOKEN environment variables "
            "must be set for Unity Catalog downloads."
        )

    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {source} -> {destination}")

    client = WorkspaceClient()
    normalized_source = _normalize_path(source)

    temp_fd, temp_path = tempfile.mkstemp(
        dir=dest_path.parent,
        prefix=".download_",
        suffix=".tmp"
    )

    try:
        response = client.files.download(normalized_source)

        with os.fdopen(temp_fd, 'wb') as f:
            for chunk in iter(lambda: response.contents.read(64 * 1024 * 1024), b''):
                f.write(chunk)

        shutil.move(temp_path, destination)
        logger.info(f"Successfully downloaded: {destination}")

    except Exception as e:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

        error_msg = str(e)
        if "NOT_FOUND" in error_msg:
            raise FileNotFoundError(f"Source file not found: {source}") from e
        raise IOError(f"Failed to download {source}: {e}") from e


def download_datasets(downloads: list[dict]) -> None:
    """
    Download multiple datasets from Databricks Unity Catalog.
    """
    if not downloads:
        return

    logger.info(f"Starting download of {len(downloads)} dataset(s) from Unity Catalog...")

    for i, download_spec in enumerate(downloads, 1):
        source = download_spec.get("source")
        destination = download_spec.get("destination")

        if not source or not destination:
            raise ValueError(
                f"Download entry {i} missing required 'source' or 'destination' key: {download_spec}"
            )

        download_from_unity_catalog(source, destination)

    logger.info("All dataset downloads complete.")
