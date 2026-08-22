from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from src.deployment import REQUIRED_ARTIFACTS, extract_artifact_archive, prepare_deployment_artifacts


def _write_bundle(path: Path, *, unsafe_name: str | None = None) -> None:
    with zipfile.ZipFile(path, "w") as bundle:
        if unsafe_name:
            bundle.writestr(unsafe_name, "unsafe")
            return
        for relative_path in REQUIRED_ARTIFACTS:
            bundle.writestr(f"outputs/{relative_path}", "artifact")


def test_prepare_deployment_artifacts_downloads_and_reuses_cache(tmp_path):
    bundle = tmp_path / "source.zip"
    _write_bundle(bundle)
    calls = []

    def downloader(file_id, destination):
        calls.append(file_id)
        destination.write_bytes(bundle.read_bytes())

    project = tmp_path / "project"
    project.mkdir()
    first = prepare_deployment_artifacts(
        project, file_id="example", cache_root=tmp_path / "cache", downloader=downloader
    )
    second = prepare_deployment_artifacts(
        project, file_id="example", cache_root=tmp_path / "cache", downloader=downloader
    )

    assert first == second
    assert all((first / relative_path).is_file() for relative_path in REQUIRED_ARTIFACTS)
    assert calls == ["example"]


def test_extract_artifact_archive_rejects_path_traversal(tmp_path):
    bundle = tmp_path / "unsafe.zip"
    _write_bundle(bundle, unsafe_name="../outside.txt")

    with pytest.raises(ValueError, match="Unsafe path"):
        extract_artifact_archive(bundle, tmp_path / "destination")
    assert not (tmp_path / "outside.txt").exists()
