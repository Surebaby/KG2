import hashlib

import pytest

from scripts.prepare.verify_sourcegate_deployment_v1 import verify_files


def test_transfer_verifies_content_not_only_sizes(tmp_path):
    path = tmp_path / "asset.json"
    path.write_bytes(b"good")
    files = {"asset.json": {"size_bytes": 4, "sha256": hashlib.sha256(b"good").hexdigest()}}
    assert verify_files(tmp_path, files)["all_sha256_match"]
    path.write_bytes(b"evil")
    with pytest.raises(ValueError, match="SHA256"):
        verify_files(tmp_path, files)


@pytest.mark.parametrize("name", ["../outside", "/tmp/outside", "a/../outside"])
def test_transfer_rejects_nonportable_paths(tmp_path, name):
    with pytest.raises(ValueError, match="nonportable"):
        verify_files(tmp_path, {name: {"sha256": "x", "size_bytes": 1}})


def test_transfer_checks_real_bytes_through_model_symlink(tmp_path):
    asset = tmp_path / "weights"
    asset.write_bytes(b"weights")
    (tmp_path / "model").symlink_to(asset)
    files = {"model": {"size_bytes": 7, "sha256": hashlib.sha256(b"weights").hexdigest()}}
    assert verify_files(tmp_path, files)["files"] == 1
    asset.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA256"):
        verify_files(tmp_path, files)
