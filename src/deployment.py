"""Safe, optional artifact bootstrap for the hosted Streamlit dashboard."""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Callable


DEFAULT_ARTIFACT_FILE_ID = "1k7lDS-B8VNxiRTTtxwL1Slk_MhgG7ivT"
REQUIRED_ARTIFACTS = (
    "models/baseline_model.joblib",
    "models/hardened_model.joblib",
    "models/baseline_preprocessor.joblib",
    "models/feature_names.joblib",
    "metrics/phase1_baseline_metrics.json",
)


def _has_required_artifacts(output_dir: Path) -> bool:
    return all((output_dir / relative_path).is_file() for relative_path in REQUIRED_ARTIFACTS)


def _is_streamlit_cloud(project_root: Path) -> bool:
    return bool(os.getenv("STREAMLIT_SHARING_MODE")) or str(project_root).startswith(
        "/mount/src/"
    )


def extract_artifact_archive(archive_path: str | Path, destination: str | Path) -> Path:
    """Extract a ZIP after rejecting traversal paths and symbolic links."""
    archive = Path(archive_path)
    target = Path(destination)
    target.mkdir(parents=True, exist_ok=True)
    resolved_target = target.resolve()

    with zipfile.ZipFile(archive) as bundle:
        for member in bundle.infolist():
            member_path = (target / member.filename).resolve()
            if os.path.commonpath((str(resolved_target), str(member_path))) != str(
                resolved_target
            ):
                raise ValueError(f"Unsafe path in deployment archive: {member.filename}")
            file_type = (member.external_attr >> 16) & 0o170000
            if file_type == 0o120000:
                raise ValueError(
                    f"Symbolic links are not allowed in deployment archive: {member.filename}"
                )
        bundle.extractall(target)
    return target


def prepare_deployment_artifacts(
    project_root: str | Path,
    *,
    file_id: str | None = None,
    cache_root: str | Path | None = None,
    downloader: Callable[[str, Path], object] | None = None,
) -> Path | None:
    """Return hosted outputs, downloading the public bundle only when required.

    Local and Colab runs continue to use their configured output directories. The
    automatic download is enabled on Streamlit Community Cloud, or explicitly by
    setting ``DEPLOYMENT_ARTIFACT_FILE_ID``.
    """
    root = Path(project_root)
    repository_outputs = root / "outputs"
    if _has_required_artifacts(repository_outputs):
        return repository_outputs

    configured_id = file_id or os.getenv("DEPLOYMENT_ARTIFACT_FILE_ID")
    if not configured_id and _is_streamlit_cloud(root):
        configured_id = DEFAULT_ARTIFACT_FILE_ID
    if not configured_id:
        return None

    cache_base = Path(cache_root or tempfile.gettempdir())
    cache_key = hashlib.sha256(configured_id.encode("utf-8")).hexdigest()[:12]
    cache = cache_base / f"ai_fraud_artifacts_{cache_key}"
    output_dir = cache / "outputs"
    if _has_required_artifacts(output_dir):
        return output_dir

    cache.mkdir(parents=True, exist_ok=True)
    archive_path = cache / "deployment_artifacts.zip"
    if not archive_path.is_file():
        if downloader is not None:
            downloader(configured_id, archive_path)
        else:
            try:
                import gdown
            except ImportError as error:
                raise RuntimeError(
                    "Hosted artifact download requires the 'gdown' package."
                ) from error
            result = gdown.download(
                id=configured_id, output=str(archive_path), quiet=True
            )
            if not result:
                raise RuntimeError("Google Drive did not return the artifact bundle.")

    if not archive_path.is_file() or archive_path.stat().st_size == 0:
        raise RuntimeError("The downloaded deployment artifact bundle is empty.")

    staging = Path(tempfile.mkdtemp(prefix="extract_", dir=cache))
    try:
        extract_artifact_archive(archive_path, staging)
        staged_outputs = staging / "outputs"
        if not _has_required_artifacts(staged_outputs):
            missing = [
                path
                for path in REQUIRED_ARTIFACTS
                if not (staged_outputs / path).is_file()
            ]
            raise RuntimeError(
                "Deployment bundle is missing required artifacts: " + ", ".join(missing)
            )
        if output_dir.exists():
            shutil.rmtree(output_dir)
        shutil.move(str(staged_outputs), str(output_dir))
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return output_dir
