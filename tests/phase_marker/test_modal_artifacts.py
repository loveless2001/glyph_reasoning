from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import shlex
import socket
import subprocess
from types import SimpleNamespace

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json, sha256_json
from phase_marker.modal_artifacts import (
    INPUT_ALLOWLIST,
    SOURCE_INCLUDE_PATHS,
    AttemptReceipt,
    BundleFile,
    ModelCacheManifest,
    build_input_bundle,
    build_model_cache_manifest,
    create_attempt_id,
    hash_source_tree,
    prepare_ephemeral_workspace,
    promote_validated_output,
    require_clean_tracked_status,
    run_exact_command,
    validate_bundle_at_root,
    validate_model_cache_manifest,
    write_attempt_receipt,
)
from phase_marker.token_audit import (
    QWEN25_7B_TOKENIZER_REVISION,
    _pinned_tokenizer_snapshot_path,
)
import phase_marker.modal_plan as modal_plan
import phase_marker.modal_artifacts as modal_artifacts
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
SOURCE_HASH = "1" * 64
LOCK_HASH = "2" * 64


class CommitVolume:
    def __init__(self, on_commit: object | None = None) -> None:
        self.commit_count = 0
        self.on_commit = on_commit

    def commit(self) -> None:
        if self.on_commit is not None:
            self.on_commit()
        self.commit_count += 1


class FailingCommitVolume(CommitVolume):
    def commit(self) -> None:
        self.commit_count += 1
        raise RuntimeError("simulated receipt commit failure")


class FailFirstCommitVolume(CommitVolume):
    def commit(self) -> None:
        self.commit_count += 1
        if self.commit_count == 1:
            raise RuntimeError("injected first commit failure")


class FailNthCommitVolume(CommitVolume):
    def __init__(self, fail_on: int) -> None:
        super().__init__()
        self.fail_on = fail_on

    def commit(self) -> None:
        self.commit_count += 1
        if self.commit_count == self.fail_on:
            raise RuntimeError(f"injected commit {self.fail_on} failure")


class FailFirstCommitWithFileExistsVolume(CommitVolume):
    def commit(self) -> None:
        self.commit_count += 1
        if self.commit_count == 1:
            raise FileExistsError("injected first commit file-exists failure")


def _pinned_qwen_model_config() -> dict[str, object]:
    return {
        "architectures": ["Qwen2ForCausalLM"],
        "hidden_size": 3584,
        "intermediate_size": 18944,
        "model_type": "qwen2",
        "num_attention_heads": 28,
        "num_hidden_layers": 28,
        "num_key_value_heads": 4,
        "vocab_size": 152064,
    }


def _pinned_qwen_tokenizer_json() -> dict[str, object]:
    return {
        "version": "1.0",
        "added_tokens": [{"id": 0, "content": "<|endoftext|>"}],
        "normalizer": {"type": "NFC"},
        "pre_tokenizer": {"type": "Sequence", "pretokenizers": []},
        "post_processor": {"type": "ByteLevel"},
        "decoder": {"type": "ByteLevel"},
        "model": {
            "type": "BPE",
            "vocab": {"<|endoftext|>": 0, "t": 1, "o": 2, "to": 3},
            "merges": ["t o"],
        },
    }


@pytest.fixture
def qwen_snapshot(tmp_path: Path) -> Path:
    snapshot = (
        tmp_path
        / "models--Qwen--Qwen2.5-7B-Instruct"
        / "snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    snapshot.mkdir(parents=True)
    for name, payload in (
        ("config.json", _pinned_qwen_model_config()),
        (
            "generation_config.json",
            {
                "bos_token_id": 151643,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
            },
        ),
        ("tokenizer.json", _pinned_qwen_tokenizer_json()),
        (
            "tokenizer_config.json",
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
                "model_max_length": 131072,
            },
        ),
        (
            "model.safetensors.index.json",
            {
                "metadata": {"total_size": 8},
                "weight_map": {
                    "model.layers.0.weight": "model-00001-of-00002.safetensors",
                    "model.layers.1.weight": "model-00002-of-00002.safetensors",
                },
            },
        ),
    ):
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first\n")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second\n")
    return snapshot


def test_model_cache_manifest_binds_every_index_shard(qwen_snapshot: Path) -> None:
    """Would fail if a pinned cache omitted a shard named by its model index."""
    manifest = build_model_cache_manifest(qwen_snapshot)

    assert manifest.model_revision == QWEN25_7B_TOKENIZER_REVISION
    assert {item.path for item in manifest.files} >= {
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00002.safetensors",
        "model-00002-of-00002.safetensors",
    }
    validate_model_cache_manifest(qwen_snapshot, manifest)


def test_model_cache_rejects_missing_or_unindexed_shards(qwen_snapshot: Path) -> None:
    """Would fail if a cache could certify incomplete or surplus weight files."""
    (qwen_snapshot / "model-00002-of-00002.safetensors").unlink()
    with pytest.raises(ValueError, match="model shard"):
        build_model_cache_manifest(qwen_snapshot)

    (qwen_snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second\n")
    (qwen_snapshot / "surplus.safetensors").write_bytes(b"surplus\n")
    with pytest.raises(ValueError, match="unindexed"):
        build_model_cache_manifest(qwen_snapshot)


@pytest.mark.parametrize(
    "index_payload",
    (
        [],
        {"metadata": {"total_size": 8}, "weight_map": {}},
        {"metadata": [], "weight_map": {"weight": "model-00001-of-00002.safetensors"}},
        {"metadata": {"total_size": 0}, "weight_map": {"weight": "model-00001-of-00002.safetensors"}},
        {"metadata": {"total_size": 8}, "weight_map": {"weight": "../escape.safetensors"}},
    ),
)
def test_model_cache_rejects_invalid_index_metadata_and_paths(
    qwen_snapshot: Path, index_payload: object,
) -> None:
    """Would fail if malformed index metadata or traversal named a cache file."""
    (qwen_snapshot / "model.safetensors.index.json").write_text(
        json.dumps(index_payload), encoding="utf-8"
    )

    with pytest.raises(ValueError, match="model index|model shard"):
        build_model_cache_manifest(qwen_snapshot)


def test_model_cache_rejects_empty_shard_and_wrong_snapshot_revision(qwen_snapshot: Path) -> None:
    """Would fail if an empty shard or a same-shaped unpinned snapshot passed."""
    (qwen_snapshot / "model-00001-of-00002.safetensors").write_bytes(b"")
    with pytest.raises(ValueError, match="model shard"):
        build_model_cache_manifest(qwen_snapshot)

    wrong_snapshot = qwen_snapshot.parent / ("0" * 40)
    qwen_snapshot.rename(wrong_snapshot)
    with pytest.raises(ValueError, match="pinned Qwen snapshot"):
        build_model_cache_manifest(wrong_snapshot)


def test_model_cache_requires_generation_metadata(qwen_snapshot: Path) -> None:
    """Would fail if a model cache could omit generation configuration bytes."""
    (qwen_snapshot / "generation_config.json").unlink()

    with pytest.raises(ValueError, match="pinned Qwen generation metadata"):
        build_model_cache_manifest(qwen_snapshot)


@pytest.mark.parametrize(
    ("filename", "payload"),
    (
        ("config.json", "not-json"),
        ("config.json", []),
        ("config.json", {**_pinned_qwen_model_config(), "model_type": "qwen3"}),
        ("config.json", {**_pinned_qwen_model_config(), "architectures": ["LlamaForCausalLM"]}),
        ("generation_config.json", "not-json"),
        ("generation_config.json", []),
        ("generation_config.json", {"bos_token_id": 0, "eos_token_id": 1, "pad_token_id": 0}),
    ),
)
def test_model_cache_rejects_invalid_pinned_model_metadata(
    qwen_snapshot: Path, filename: str, payload: object,
) -> None:
    """Would fail if arbitrary metadata could be certified as the pinned Qwen model."""
    content = payload if isinstance(payload, str) else json.dumps(payload)
    (qwen_snapshot / filename).write_text(content, encoding="utf-8")

    with pytest.raises(ValueError, match="pinned Qwen (model|generation) metadata"):
        build_model_cache_manifest(qwen_snapshot)


def test_model_cache_rejects_malformed_tokenizer_layout_on_build_and_validation(
    qwen_snapshot: Path,
) -> None:
    """Would fail if cache certification skipped the public tokenizer preflight."""
    manifest = build_model_cache_manifest(qwen_snapshot)
    (qwen_snapshot / "tokenizer.json").write_text("[]", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="pinned tokenizer snapshot"):
        validate_model_cache_manifest(qwen_snapshot, manifest)
    with pytest.raises(FileNotFoundError, match="pinned tokenizer snapshot"):
        build_model_cache_manifest(qwen_snapshot)


def test_model_cache_validation_rejects_changed_valid_generation_metadata(
    qwen_snapshot: Path,
) -> None:
    """Would fail if manifest validation accepted changed metadata with valid identity fields."""
    manifest = build_model_cache_manifest(qwen_snapshot)
    (qwen_snapshot / "generation_config.json").write_text(
        json.dumps(
            {
                "bos_token_id": 151643,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
                "additional_stable_note": "changed bytes",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="model cache file hash mismatch"):
        validate_model_cache_manifest(qwen_snapshot, manifest)


def test_model_cache_validation_rejects_changed_bytes_and_manifest_metadata(
    qwen_snapshot: Path,
) -> None:
    """Would fail if validation trusted stale manifest hashes or its declared identity."""
    manifest = build_model_cache_manifest(qwen_snapshot)
    (qwen_snapshot / "model-00001-of-00002.safetensors").write_bytes(b"changed\n")

    with pytest.raises(ValueError, match="model cache file hash mismatch"):
        validate_model_cache_manifest(qwen_snapshot, manifest)
    with pytest.raises(ValueError, match="schema version"):
        validate_model_cache_manifest(
            qwen_snapshot,
            ModelCacheManifest(
                schema_version=2,
                model_id=manifest.model_id,
                model_revision=manifest.model_revision,
                files=manifest.files,
                artifact_id=manifest.artifact_id,
            ),
        )


@pytest.mark.parametrize(
    "bad_file",
    (
        {"path": {"not": "a path"}, "size": 1, "sha256": "1" * 64},
        {"path": "config.json", "size": {"not": "an int"}, "sha256": "1" * 64},
    ),
)
def test_model_cache_manifest_parser_normalizes_malformed_file_shapes_to_value_error(
    qwen_snapshot: Path,
    bad_file: object,
) -> None:
    """Would fail if untrusted manifest shapes leaked TypeError or AttributeError."""
    payload = asdict(build_model_cache_manifest(qwen_snapshot))
    payload["files"] = [bad_file]

    with pytest.raises(ValueError, match="model cache"):
        modal_artifacts.parse_model_cache_manifest_payload(payload)


def test_model_cache_manifest_parser_requires_the_minimum_pinned_snapshot_file_set(
    qwen_snapshot: Path,
) -> None:
    """Would fail if Stage A accepted a self-consistent but incomplete cache manifest."""
    payload = asdict(build_model_cache_manifest(qwen_snapshot))
    payload["files"] = [
        item for item in payload["files"] if item["path"] != "config.json"
    ]
    payload["artifact_id"] = modal_artifacts.sha256_json(
        {key: value for key, value in payload.items() if key != "artifact_id"}
    )

    with pytest.raises(ValueError, match="required files"):
        modal_artifacts.parse_model_cache_manifest_payload(payload)


def test_model_cache_hashes_symlink_target_bytes(qwen_snapshot: Path, tmp_path: Path) -> None:
    """Would fail if a snapshot symlink name, rather than its bytes, became identity."""
    target = qwen_snapshot.parent.parent / "blobs/blob-a"
    target.parent.mkdir()
    target.write_bytes(b"first\n")
    shard = qwen_snapshot / "model-00001-of-00002.safetensors"
    shard.unlink()
    shard.symlink_to(target)
    first = build_model_cache_manifest(qwen_snapshot)
    target.write_bytes(b"mutated\n")

    with pytest.raises(ValueError, match="model cache file hash mismatch"):
        validate_model_cache_manifest(qwen_snapshot, first)


def test_model_cache_rejects_shard_symlink_outside_pinned_cache(
    qwen_snapshot: Path, tmp_path: Path,
) -> None:
    """Would fail if a pinned cache shard could resolve to unrelated host bytes."""
    target = tmp_path / "outside-blob"
    target.write_bytes(b"first\n")
    shard = qwen_snapshot / "model-00001-of-00002.safetensors"
    shard.unlink()
    shard.symlink_to(target)

    with pytest.raises(ValueError, match="model shard"):
        build_model_cache_manifest(qwen_snapshot)


def test_model_cache_rejects_a_snapshot_directory_symlink(qwen_snapshot: Path) -> None:
    """Would fail if a pinned revision path could silently point at another snapshot."""
    source = qwen_snapshot.parent / "other-snapshot"
    qwen_snapshot.rename(source)
    qwen_snapshot.symlink_to(source, target_is_directory=True)

    with pytest.raises(ValueError, match="pinned Qwen snapshot"):
        build_model_cache_manifest(qwen_snapshot)


def test_real_pinned_qwen_model_cache_is_valid_when_full_offline_snapshot_exists() -> None:
    """Probe only an already-local full cache; never ask Hugging Face for a shard."""
    snapshot = _pinned_tokenizer_snapshot_path("Qwen/Qwen2.5-7B-Instruct")
    try:
        index = json.loads((snapshot / "model.safetensors.index.json").read_text(encoding="utf-8"))
        weight_map = index["weight_map"]
        shards = sorted(set(weight_map.values()))
    except (OSError, KeyError, TypeError, json.JSONDecodeError):
        pytest.skip("pinned Qwen model index is not locally cached")
    if len(shards) != 4 or not all(
        isinstance(name, str) and (snapshot / name).is_file() for name in shards
    ):
        pytest.skip("all four pinned Qwen model shards are not locally cached")
    if not all(
        (snapshot / name).is_file()
        for name in modal_artifacts._MODEL_CACHE_REQUIRED_FILES
    ):
        pytest.skip("full pinned Qwen model metadata is not locally cached")

    manifest = build_model_cache_manifest(snapshot)

    validate_model_cache_manifest(snapshot, manifest)


def _cache_plan_payload(repo_fixture: Path) -> dict[str, object]:
    bundle = build_input_bundle(repo_fixture)
    plan = modal_plan.build_pilot_plan(
        repo_fixture / CONFIG_PATH,
        repo_fixture / "artifacts/phase-marker",
        bundle=bundle,
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )
    return modal_plan.pilot_plan_payload(plan)


def _install_fake_snapshot_download(
    monkeypatch: pytest.MonkeyPatch,
    source: Path,
    calls: list[dict[str, object]],
) -> None:
    def snapshot_download(**kwargs: object) -> str:
        calls.append(dict(kwargs))
        destination = (
            Path(str(kwargs["cache_dir"]))
            / "models--Qwen--Qwen2.5-7B-Instruct"
            / "snapshots"
            / QWEN25_7B_TOKENIZER_REVISION
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, destination)
        return str(destination)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot_download)


def test_cache_model_downloads_pinned_revision_validates_then_promotes_once(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if cache publication preceded full validation or used a floating revision."""
    cache_root = tmp_path / "model-cache"
    calls: list[dict[str, object]] = []
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, calls)
    observed_at_commit: list[tuple[Path, Path]] = []

    def assert_complete_at_commit() -> None:
        canonical = (
            cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
            / QWEN25_7B_TOKENIZER_REVISION
        )
        manifest_path = canonical.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
        manifest = modal_artifacts.load_model_cache_manifest(manifest_path)
        validate_model_cache_manifest(canonical, manifest)
        observed_at_commit.append((canonical, manifest_path))

    volume = CommitVolume(assert_complete_at_commit)
    result = modal_artifacts.cache_model_to_volume(
        plan_payload=_cache_plan_payload(repo_fixture),
        cache_root=cache_root,
        volume=volume,
    )

    assert calls == [{
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": QWEN25_7B_TOKENIZER_REVISION,
        "cache_dir": str(cache_root / "hub"),
    }]
    assert volume.commit_count == 1
    assert len(observed_at_commit) == 1
    canonical, manifest_path = observed_at_commit[0]
    assert Path(str(result["snapshot_path"])) == canonical
    assert Path(str(result["manifest_path"])) == manifest_path
    assert not manifest_path.is_relative_to(canonical)
    assert result["cached"] is True
    assert result["artifact_id"] == modal_artifacts.load_model_cache_manifest(
        manifest_path
    ).artifact_id


def test_cache_publication_does_not_require_hard_links(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail on Modal Volumes, which reject hard-link creation."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])

    def reject_hard_link(*args: object, **kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(os, "link", reject_hard_link)

    result = modal_artifacts.cache_model_to_volume(
        plan_payload=_cache_plan_payload(repo_fixture),
        cache_root=cache_root,
        volume=CommitVolume(),
    )

    assert result["cached"] is True
    assert Path(str(result["manifest_path"])).is_file()


def test_exclusive_bytes_write_failure_removes_partial_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a caught short-write error could become durable evidence."""
    destination = tmp_path / "provenance.json"
    real_write = os.write
    calls = 0

    def fail_second_write(descriptor: int, content: bytes) -> int:
        nonlocal calls
        calls += 1
        if calls == 1:
            return real_write(descriptor, content[:3])
        raise OSError("injected write failure")

    monkeypatch.setattr(os, "write", fail_second_write)

    with pytest.raises(OSError, match="injected write failure"):
        modal_artifacts._write_bytes_exclusive_or_equal(
            destination, b"complete provenance"
        )

    assert not destination.exists()


def test_exclusive_bytes_fsync_failure_removes_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if unsynced evidence remained visible after publication failed."""
    destination = tmp_path / "receipt.json"

    def fail_fsync(_descriptor: int) -> None:
        raise OSError("injected fsync failure")

    monkeypatch.setattr(os, "fsync", fail_fsync)

    with pytest.raises(OSError, match="injected fsync failure"):
        modal_artifacts._write_bytes_exclusive_or_equal(
            destination, b"complete receipt"
        )

    assert not destination.exists()


def test_exclusive_bytes_close_failure_does_not_retry_close(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a failed close could target a subsequently reused descriptor."""
    destination = tmp_path / "receipt.json"
    real_close = os.close
    close_calls = 0

    def close_then_report_failure(descriptor: int) -> None:
        nonlocal close_calls
        if os.readlink(f"/proc/self/fd/{descriptor}") != str(destination):
            real_close(descriptor)
            return
        close_calls += 1
        if close_calls > 1:
            raise AssertionError("descriptor close was retried")
        real_close(descriptor)
        raise OSError("injected close failure")

    monkeypatch.setattr(os, "close", close_then_report_failure)

    with pytest.raises(OSError, match="injected close failure"):
        modal_artifacts._write_bytes_exclusive_or_equal(
            destination, b"complete receipt"
        )

    assert close_calls == 1
    assert not destination.exists()


def test_exclusive_bytes_refuses_to_unlink_replaced_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if cleanup could unlink a concurrent writer's replacement."""
    destination = tmp_path / "evidence.json"
    real_stat = os.stat

    def fail_write(_descriptor: int, _content: bytes) -> int:
        raise OSError("injected write failure")

    def report_replaced_identity(
        path: object, *args: object, **kwargs: object,
    ) -> object:
        if path == destination.name and kwargs.get("dir_fd") is not None:
            return SimpleNamespace(st_dev=-1, st_ino=-1)
        return real_stat(path, *args, **kwargs)

    monkeypatch.setattr(os, "write", fail_write)
    monkeypatch.setattr(os, "stat", report_replaced_identity)

    with pytest.raises(
        modal_artifacts._EvidencePublicationCleanupError,
        match="evidence cleanup failed",
    ):
        modal_artifacts._write_bytes_exclusive_or_equal(
            destination, b"complete evidence"
        )

    assert destination.exists()


def test_exclusive_bytes_retries_transient_created_identity_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a transient identity read poisoned an uncommitted destination."""
    destination = tmp_path / "evidence.json"
    real_fstat = os.fstat
    identity_calls = 0

    def fail_first_target_fstat(descriptor: int) -> os.stat_result:
        nonlocal identity_calls
        if os.readlink(f"/proc/self/fd/{descriptor}") == str(destination):
            identity_calls += 1
            if identity_calls == 1:
                raise OSError("injected transient fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", fail_first_target_fstat)

    modal_artifacts._write_bytes_exclusive_or_equal(
        destination, b"complete evidence"
    )

    assert identity_calls == 2
    assert destination.read_bytes() == b"complete evidence"


@pytest.mark.parametrize(
    "failure_stage",
    ("during-manifest-publication", "after-publication", "during-final-validation"),
)
def test_cache_publication_failure_rolls_back_before_durable_receipt(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_stage: str,
) -> None:
    """Would fail if any publication failure could durably poison canonical cache state."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])

    def inject(stage: str, **context: object) -> None:
        if stage == failure_stage:
            raise RuntimeError(f"injected {failure_stage} failure")

    monkeypatch.setattr(
        modal_artifacts, "_cache_publication_hook", inject, raising=False
    )
    volume = CommitVolume()

    with pytest.raises(RuntimeError, match=f"injected {failure_stage} failure"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture),
            cache_root=cache_root,
            volume=volume,
        )

    canonical_model = cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct"
    canonical_snapshot = canonical_model / "snapshots" / QWEN25_7B_TOKENIZER_REVISION
    canonical_manifest = canonical_snapshot.parent / (
        f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
    )
    assert not canonical_model.exists()
    assert not canonical_snapshot.exists()
    assert not canonical_manifest.exists()
    attempt_roots = list(cache_root.glob("attempts/cache-model/*"))
    assert len(attempt_roots) == 1
    assert list(attempt_roots[0].rglob("config.json"))
    receipts = list(attempt_roots[0].glob("receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is False
    assert f"injected {failure_stage} failure" in receipt["failure_reason"]
    assert volume.commit_count == 1


def test_cache_rollback_failure_never_commits_poisoned_canonical_state(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a failed rollback committed visible canonical bytes or hid the cause."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])

    def inject(stage: str, **context: object) -> None:
        if stage == "after-publication":
            raise RuntimeError("injected post-publication failure")

    def fail_rollback(*args: object, **kwargs: object) -> None:
        raise OSError("injected rollback failure")

    monkeypatch.setattr(
        modal_artifacts, "_cache_publication_hook", inject, raising=False
    )
    monkeypatch.setattr(
        modal_artifacts, "_rollback_cache_publication", fail_rollback, raising=False
    )
    volume = CommitVolume()

    with pytest.raises(RuntimeError, match="rollback failed") as raised:
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture),
            cache_root=cache_root,
            volume=volume,
        )

    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "post-publication" in str(raised.value.__cause__)
    assert any("injected rollback failure" in note for note in raised.value.__notes__)
    assert (
        cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct"
    ).exists()
    assert volume.commit_count == 0


def test_cache_lock_cleanup_failure_rolls_back_and_quarantines_lock(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if lock cleanup escaped rollback or left a durable stale lock."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])
    monkeypatch.setattr(
        modal_artifacts,
        "_release_cache_promotion_lock",
        lambda path: (_ for _ in ()).throw(OSError("injected lock cleanup failure")),
        raising=False,
    )
    volume = CommitVolume()

    with pytest.raises(OSError, match="injected lock cleanup failure"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture),
            cache_root=cache_root,
            volume=volume,
        )

    assert not (
        cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct"
    ).exists()
    assert not (cache_root / "canonical/.cache-promotion.lock").exists()
    attempt_roots = list(cache_root.glob("attempts/cache-model/*"))
    assert len(attempt_roots) == 1
    assert list(attempt_roots[0].rglob("config.json"))
    assert (attempt_roots[0] / "failed-promotion.lock").is_file()
    assert (attempt_roots[0] / "receipt.json").is_file()
    assert volume.commit_count == 1


def test_cache_first_commit_failure_rolls_back_before_receipt_commit(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if retrying a failed commit durably published canonical cache bytes."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])
    volume = FailFirstCommitVolume()

    with pytest.raises(RuntimeError, match="injected first commit failure"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture),
            cache_root=cache_root,
            volume=volume,
        )

    assert not (
        cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct"
    ).exists()
    attempt_roots = list(cache_root.glob("attempts/cache-model/*"))
    assert len(attempt_roots) == 1
    assert list(attempt_roots[0].rglob("config.json"))
    assert (attempt_roots[0] / "receipt.json").is_file()
    assert volume.commit_count == 2


def test_cache_first_commit_file_exists_rolls_back_before_receipt_commit(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if commit FileExistsError were mistaken for lock contention."""
    cache_root = tmp_path / "model-cache"
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, [])
    volume = FailFirstCommitWithFileExistsVolume()

    with pytest.raises(
        FileExistsError, match="injected first commit file-exists failure"
    ):
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture),
            cache_root=cache_root,
            volume=volume,
        )

    assert not (
        cache_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct"
    ).exists()
    attempt_roots = list(cache_root.glob("attempts/cache-model/*"))
    assert len(attempt_roots) == 1
    assert list(attempt_roots[0].rglob("config.json"))
    assert (attempt_roots[0] / "receipt.json").is_file()
    assert volume.commit_count == 2


def test_cache_model_identical_repeat_is_noop_and_conflict_never_overwrites(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a cache retry redownloaded or overwrote canonical model bytes."""
    cache_root = tmp_path / "model-cache"
    calls: list[dict[str, object]] = []
    _install_fake_snapshot_download(monkeypatch, qwen_snapshot, calls)
    first_volume = CommitVolume()
    first = modal_artifacts.cache_model_to_volume(
        plan_payload=_cache_plan_payload(repo_fixture), cache_root=cache_root,
        volume=first_volume,
    )
    canonical = Path(str(first["snapshot_path"]))
    original = (canonical / "config.json").read_bytes()

    second_volume = CommitVolume()
    second = modal_artifacts.cache_model_to_volume(
        plan_payload=_cache_plan_payload(repo_fixture), cache_root=cache_root,
        volume=second_volume,
    )
    assert second == {**first, "cached": False}
    assert len(calls) == 1
    assert second_volume.commit_count == 0

    (canonical / "config.json").write_bytes(b"conflicting canonical bytes")
    with pytest.raises(ValueError, match="canonical model cache conflicts"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=_cache_plan_payload(repo_fixture), cache_root=cache_root,
            volume=CommitVolume(),
        )
    assert (canonical / "config.json").read_bytes() != original
    assert len(calls) == 1


def test_cache_model_validates_plan_before_download_and_records_failed_attempt(
    repo_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if invalid approval data reached Hugging Face or failures lacked evidence."""
    cache_root = tmp_path / "model-cache"
    invalid = _cache_plan_payload(repo_fixture)
    invalid["model_revision"] = "main"
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(dict(kwargs)) or pytest.fail("downloaded invalid plan"),
    )
    volume = CommitVolume()

    with pytest.raises(ValueError, match="plan payload"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=invalid, cache_root=cache_root, volume=volume,
        )
    assert calls == []
    receipts = list(cache_root.glob("attempts/cache-model/*/receipt.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is False
    assert receipt["stage"] == "cache-model"
    assert receipt["artifact_id"] == sha256_json({
        key: value for key, value in receipt.items() if key != "artifact_id"
    })
    assert volume.commit_count == 1


@pytest.mark.parametrize("mutation", ("extra-resource", "empty-command"))
def test_cache_model_rejects_nested_plan_drift_before_download(
    repo_fixture: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    """Would fail if nested resource or job drift reached the download boundary."""
    payload = deepcopy(_cache_plan_payload(repo_fixture))
    if mutation == "extra-resource":
        payload["resources"]["unapproved"] = True
    else:
        payload["jobs"][0]["training_command"] = ""
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "huggingface_hub.snapshot_download",
        lambda **kwargs: calls.append(dict(kwargs)) or pytest.fail("downloaded drifted plan"),
    )

    with pytest.raises(ValueError, match="plan payload"):
        modal_artifacts.cache_model_to_volume(
            plan_payload=payload,
            cache_root=tmp_path / "model-cache",
            volume=CommitVolume(),
        )
    assert calls == []


def test_cache_failure_preserves_original_error_when_receipt_commit_fails(
    repo_fixture: Path, tmp_path: Path,
) -> None:
    """Would fail if evidence persistence masked the cache failure operators need."""
    payload = _cache_plan_payload(repo_fixture)
    payload["model_revision"] = "main"
    volume = FailingCommitVolume()

    with pytest.raises(ValueError, match="plan payload") as raised:
        modal_artifacts.cache_model_to_volume(
            plan_payload=payload,
            cache_root=tmp_path / "model-cache",
            volume=volume,
        )

    assert volume.commit_count == 1
    assert any("receipt persistence also failed" in note for note in raised.value.__notes__)


def test_cpu_smoke_invalid_run_id_receipt_cannot_escape_run_root(
    repo_fixture: Path, tmp_path: Path,
) -> None:
    """Would fail if malformed remote payload identity controlled a receipt path."""
    payload = _cache_plan_payload(repo_fixture)
    payload["run_id"] = "../../escaped"
    run_root = tmp_path / "run-volume"
    volume = CommitVolume()

    with pytest.raises(ValueError, match="plan payload"):
        modal_artifacts.run_cpu_smoke(
            plan_payload=payload,
            code_root=repo_fixture,
            input_root=tmp_path / "inputs",
            model_root=tmp_path / "model",
            run_root=run_root,
            volume=volume,
            runtime_imports=("json",),
            execution_provenance=_execution_provenance("smoke", "invalid-plan"),
        )

    receipts = list((run_root / "runs/invalid-plan/receipts/smoke").glob("*.json"))
    assert len(receipts) == 1
    assert not (tmp_path / "escaped").exists()
    assert volume.commit_count == 1


def receipt_fixture(**changes: object) -> AttemptReceipt:
    plan_digest = "1" * 64
    cache_artifact_id = "e" * 64
    smoke_receipt_artifact_id = "9" * 64
    stage_a_resume = False
    started_at = datetime.now(timezone.utc)
    materialization_ids = tuple(f"{value:x}" * 64 for value in range(3, 9))
    bundle_files = tuple(
        modal_artifacts.BundleFile(path, 1, "a" * 64)
        for path in modal_artifacts.INPUT_ALLOWLIST
    )
    bundle_id = modal_artifacts.sha256_json(
        {
            "schema_version": 1,
            "files": [asdict(item) for item in bundle_files],
            "artifact_ids": ["2" * 64, *materialization_ids],
        }
    )
    bundle_manifest_artifact_id = hashlib.sha256(
        (
            canonical_json(
                {
                    "schema_version": 1,
                    "bundle_id": bundle_id,
                    "files": [asdict(item) for item in bundle_files],
                    "artifact_ids": ["2" * 64, *materialization_ids],
                }
            )
            + "\n"
        ).encode()
    ).hexdigest()
    fields: dict[str, object] = {
        "schema_version": 1,
        "run_id": (
            "pilot-s42-cfg-12345678-split-22222222-src-cccccccccccc-plan-"
            + "1" * 64
        ),
        "bundle_id": bundle_id,
        "bundle_manifest_artifact_id": bundle_manifest_artifact_id,
        "bundle_files": bundle_files,
        "modal_environment": "main",
        "stage": "train",
        "arm": "glyph",
        "seed": 42,
        "attempt_id": "f41aa3e7-f05f-48e3-87d6-5877fcce21d1",
        "command": "./.venv/bin/python -m phase_marker.training train --config configs/phase-marker-qwen25-7b.toml --arm glyph --seed 42 --data artifacts/phase-marker/training-data/glyph.jsonl --output-dir artifacts/phase-marker/checkpoints/pilot/seed-42/glyph --manifest artifacts/phase-marker/checkpoints/pilot/seed-42/glyph/run-manifest.json",
        "command_hash": "b" * 64,
        "source_hash": "c" * 64,
        "dependency_lock_hash": "d" * 64,
        "model_cache_artifact_id": cache_artifact_id,
        "stage_a_action_digest": modal_artifacts._stage_a_action_digest(
            plan_digest=plan_digest,
            resume=stage_a_resume,
            smoke_receipt_artifact_id=smoke_receipt_artifact_id,
            model_cache_artifact_id=cache_artifact_id,
            bundle_manifest_artifact_id=bundle_manifest_artifact_id,
            modal_environment="main",
        ),
        "stage_a_resume": stage_a_resume,
        "smoke_receipt_artifact_id": smoke_receipt_artifact_id,
        "requested_gpu": "H100",
        "observed_gpu": None,
        "started_at": started_at.isoformat(),
        "finished_at": (started_at + timedelta(seconds=1)).isoformat(),
        "elapsed_seconds": 1.0,
        "timeout_seconds": 60,
        "exit_status": 0,
        "validated": True,
        "promoted": False,
        "expected_outputs": ("adapter_config.json",),
        "output_hashes": ("f" * 64,),
        "failure_reason": None,
        "plan_digest": plan_digest,
        "config_hash": "1" * 64,
        "split_artifact_id": "2" * 64,
        "materialization_artifact_ids": materialization_ids,
        "model_revision": QWEN25_7B_TOKENIZER_REVISION,
        "modal_app_id": "ap-test",
        "modal_app_name": "phase-marker-pilot-stage-a",
        "modal_function_name": "run_training_job",
        "modal_function_call_id": "fc-test",
        "modal_input_id": "in-test",
        "python_version": "3.12.test",
        "torch_version": "2.7.test",
        "cuda_runtime_version": "12.8.test",
        "cuda_driver_version": "570.test",
        "runtime_versions": tuple(
            (module, "locked-test-version")
            for module in modal_artifacts.LOCKED_RUNTIME_MODULES
        ),
        "artifact_id": "",
        "failure_stage": None,
    }
    fields.update(changes)
    if fields["validated"] is False:
        if "failure_reason" not in changes:
            fields["failure_reason"] = "RuntimeError: failed fixture attempt"
        if "failure_stage" not in changes:
            fields["failure_stage"] = (
                "producer-validation" if fields["exit_status"] == 0 else "command"
            )
    receipt = AttemptReceipt(**fields)
    if "artifact_id" not in changes:
        return replace(receipt, artifact_id=receipt.recomputed_artifact_id())
    return receipt


@pytest.fixture
def repo_fixture(tmp_path: Path) -> Path:
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(CONFIG_PATH, config_path)
    config = ExperimentConfig.load(config_path)
    artifact_root = tmp_path / "artifacts/phase-marker"
    _write_split(artifact_root, config)
    _write_materializations(artifact_root, config)
    return tmp_path


def test_input_bundle_is_exact_and_content_addressed(repo_fixture: Path) -> None:
    bundle = build_input_bundle(repo_fixture)

    assert tuple(item.path for item in bundle.files) == INPUT_ALLOWLIST
    assert all(len(item.sha256) == 64 and item.size > 0 for item in bundle.files)
    assert bundle.bundle_id == sha256_json({
        "schema_version": 1,
        "files": [asdict(item) for item in bundle.files],
        "artifact_ids": list(bundle.artifact_ids),
    })


def test_input_allowlist_has_frozen_protocol_order() -> None:
    assert INPUT_ALLOWLIST == (
        "configs/phase-marker-qwen25-7b.toml",
        "artifacts/phase-marker/splits/train.jsonl",
        "artifacts/phase-marker/splits/validation.jsonl",
        "artifacts/phase-marker/splits/test.jsonl",
        "artifacts/phase-marker/splits/exclusions.jsonl",
        "artifacts/phase-marker/splits/manifest.json",
        "artifacts/phase-marker/training-data/semantic.jsonl",
        "artifacts/phase-marker/training-data/semantic.manifest.json",
        "artifacts/phase-marker/training-data/glyph.jsonl",
        "artifacts/phase-marker/training-data/glyph.manifest.json",
        "artifacts/phase-marker/training-data/dot.jsonl",
        "artifacts/phase-marker/training-data/dot.manifest.json",
        "artifacts/phase-marker/training-data/random.jsonl",
        "artifacts/phase-marker/training-data/random.manifest.json",
        "artifacts/phase-marker/training-data/direct.jsonl",
        "artifacts/phase-marker/training-data/direct.manifest.json",
        "artifacts/phase-marker/training-data/filler.jsonl",
        "artifacts/phase-marker/training-data/filler.manifest.json",
    )


def test_bundle_rejects_changed_files(repo_fixture: Path) -> None:
    bundle = build_input_bundle(repo_fixture)
    target = repo_fixture / "artifacts/phase-marker/training-data/glyph.jsonl"
    target.write_text(target.read_text() + "{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="bundle file hash mismatch"):
        validate_bundle_at_root(bundle, repo_fixture)


def test_bundle_build_and_revalidation_reject_symlinked_allowlist_files(
    repo_fixture: Path, tmp_path: Path,
) -> None:
    """Would fail if an approved pathname could redirect a later trusted read."""
    target = repo_fixture / "artifacts/phase-marker/training-data/glyph.jsonl"
    original = target.read_bytes()
    external = tmp_path / "same-bytes.jsonl"
    external.write_bytes(original)
    bundle = build_input_bundle(repo_fixture)
    target.unlink()
    target.symlink_to(external)

    with pytest.raises(ValueError, match="symlinked|regular"):
        validate_bundle_at_root(bundle, repo_fixture)
    with pytest.raises(ValueError, match="symlinked|regular"):
        build_input_bundle(repo_fixture)


def test_bundle_rejects_extra_and_unsafe_file_entries(repo_fixture: Path) -> None:
    bundle = build_input_bundle(repo_fixture)
    extra = BundleFile("extra.json", 1, "0" * 64)

    with pytest.raises(ValueError, match="bundle file paths"):
        validate_bundle_at_root(replace(bundle, files=(*bundle.files, extra)), repo_fixture)
    with pytest.raises(ValueError, match="relative"):
        validate_bundle_at_root(
            replace(bundle, files=(replace(bundle.files[0], path="../secret"), *bundle.files[1:])),
            repo_fixture,
        )
    with pytest.raises(ValueError, match="relative"):
        validate_bundle_at_root(
            replace(bundle, files=(replace(bundle.files[0], path="/secret"), *bundle.files[1:])),
            repo_fixture,
        )


@pytest.mark.parametrize(
    "status",
    [
        " M phase_marker/modal_plan.py\n",
        "D  README.md\n",
        "A  phase_marker/new.py\n",
        "R  old.py -> phase_marker/new.py\n",
        "UU phase_marker/conflict.py\n",
        "?? paper/\n M phase_marker/modal_plan.py\n",
    ],
)
def test_tracked_dirty_status_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match="tracked source changes"):
        require_clean_tracked_status(status)


def test_untracked_approved_artifacts_are_ignored() -> None:
    require_clean_tracked_status("?? artifacts/\n")


def test_untracked_nonbundle_worktrees_are_ignored() -> None:
    require_clean_tracked_status("?? model_cards/\n?? paper/\n")


def test_source_hash_includes_only_approved_python_sources(tmp_path: Path) -> None:
    (tmp_path / "phase_marker").mkdir()
    included = tmp_path / "phase_marker/planner.py"
    included.write_text("VALUE = 1\n", encoding="utf-8")
    legacy = tmp_path / "modal_app.py"
    legacy.write_text("legacy\n", encoding="utf-8")
    first = hash_source_tree(tmp_path)

    legacy.write_text("changed legacy\n", encoding="utf-8")
    assert hash_source_tree(tmp_path) == first
    included.write_text("VALUE = 2\n", encoding="utf-8")
    assert hash_source_tree(tmp_path) != first
    assert SOURCE_INCLUDE_PATHS == ("phase_marker/**/*.py", "modal_phase_marker.py")


def test_source_hash_rejects_symlinked_or_nonregular_python_sources(
    tmp_path: Path,
) -> None:
    """Would fail if unhashed indirection or a special source entered the image set."""
    package = tmp_path / "phase_marker"
    package.mkdir()
    external = tmp_path / "external.py"
    external.write_text("VALUE = 1\n", encoding="utf-8")
    linked = package / "linked.py"
    linked.symlink_to(external)
    with pytest.raises(ValueError, match="source path is a symlink"):
        hash_source_tree(tmp_path)

    linked.unlink()
    fifo = package / "special.py"
    os.mkfifo(fifo)
    with pytest.raises(ValueError, match="not a regular file"):
        hash_source_tree(tmp_path)


def test_plan_uses_bound_bundle_ids_and_identity_hashes(repo_fixture: Path) -> None:
    artifact_root = repo_fixture / "artifacts/phase-marker"
    bundle = build_input_bundle(repo_fixture)
    plan = modal_plan.build_pilot_plan(
        repo_fixture / CONFIG_PATH,
        artifact_root,
        bundle=bundle,
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )

    manifests = [
        artifact_root / "splits/manifest.json",
        *(artifact_root / f"training-data/{arm}.manifest.json" for arm in (
            "semantic", "glyph", "dot", "random", "direct", "filler",
        )),
    ]
    assert (plan.split_artifact_id, *plan.materialization_artifact_ids) == tuple(
        json.loads(path.read_text(encoding="utf-8"))["artifact_id"] for path in manifests
    )
    changed_source = modal_plan.build_pilot_plan(
        repo_fixture / CONFIG_PATH, artifact_root, bundle=bundle,
        source_hash="3" * 64, dependency_lock_hash=LOCK_HASH,
    )
    changed_lock = modal_plan.build_pilot_plan(
        repo_fixture / CONFIG_PATH, artifact_root, bundle=bundle,
        source_hash=SOURCE_HASH, dependency_lock_hash="4" * 64,
    )
    assert changed_source.source_hash != plan.source_hash
    assert changed_source.dependency_lock_hash == plan.dependency_lock_hash
    assert changed_source.run_id != plan.run_id
    assert changed_lock.source_hash == plan.source_hash
    assert changed_lock.dependency_lock_hash != plan.dependency_lock_hash
    assert changed_lock.run_label == plan.run_label
    assert changed_lock.run_id != plan.run_id

    config = ExperimentConfig.load(repo_fixture / CONFIG_PATH)
    changed_config = replace(config, confirmatory_seeds=(102, 202, 303))
    assert sha256_json(asdict(changed_config)) != plan.config_hash
    assert plan.source_hash == SOURCE_HASH
    assert plan.dependency_lock_hash == LOCK_HASH


def test_plan_rejects_missing_or_duplicate_bundle_artifact_ids(repo_fixture: Path) -> None:
    bundle = build_input_bundle(repo_fixture)
    artifact_root = repo_fixture / "artifacts/phase-marker"
    with pytest.raises(ValueError, match="duplicate artifact ID"):
        modal_plan.build_pilot_plan(
            repo_fixture / CONFIG_PATH, artifact_root,
            bundle=replace(bundle, artifact_ids=(bundle.artifact_ids[0],) * 7),
            source_hash=SOURCE_HASH, dependency_lock_hash=LOCK_HASH,
        )
    missing = artifact_root / "training-data/glyph.manifest.json"
    payload = json.loads(missing.read_text(encoding="utf-8"))
    payload.pop("artifact_id")
    missing.write_text(canonical_json(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="artifact_id is missing"):
        build_input_bundle(repo_fixture)
    with pytest.raises(ValueError, match="bundle file hash mismatch"):
        modal_plan.build_pilot_plan(
            repo_fixture / CONFIG_PATH, artifact_root, bundle=bundle,
            source_hash=SOURCE_HASH, dependency_lock_hash=LOCK_HASH,
        )


def test_plan_requires_the_bundled_config_and_canonical_artifact_root(
    repo_fixture: Path,
) -> None:
    config_path = repo_fixture / CONFIG_PATH
    artifact_root = repo_fixture / "artifacts/phase-marker"
    bundle = build_input_bundle(repo_fixture)
    alternate_config = repo_fixture / "configs/alternate.toml"
    shutil.copyfile(config_path, alternate_config)
    alternate_artifact_root = repo_fixture / "artifacts/alternate-phase-marker"
    alternate_artifact_root.mkdir()
    config_link = repo_fixture / "configs/alternate-link.toml"
    config_link.symlink_to(alternate_config)
    artifact_link = repo_fixture / "artifacts/alternate-link"
    artifact_link.symlink_to(alternate_artifact_root, target_is_directory=True)

    for path in (
        alternate_config,
        config_link,
        repo_fixture / "configs/../configs/alternate.toml",
    ):
        with pytest.raises(ValueError, match="approved configuration"):
            modal_plan.build_pilot_plan(
                path, artifact_root, bundle=bundle,
                source_hash=SOURCE_HASH, dependency_lock_hash=LOCK_HASH,
            )
    for path in (
        alternate_artifact_root,
        artifact_link,
        repo_fixture / "artifacts/../artifacts/alternate-phase-marker",
    ):
        with pytest.raises(ValueError, match="approved artifact root"):
            modal_plan.build_pilot_plan(
                config_path, path, bundle=bundle,
                source_hash=SOURCE_HASH, dependency_lock_hash=LOCK_HASH,
            )

    canonical_config_alias = repo_fixture / "configs/canonical-link.toml"
    canonical_config_alias.symlink_to(config_path)
    canonical_artifact_alias = repo_fixture / "artifacts/canonical-link"
    canonical_artifact_alias.symlink_to(artifact_root, target_is_directory=True)
    plan = modal_plan.build_pilot_plan(
        repo_fixture / "configs/../configs/canonical-link.toml",
        repo_fixture / "artifacts/../artifacts/canonical-link",
        bundle=bundle,
        source_hash=SOURCE_HASH,
        dependency_lock_hash=LOCK_HASH,
    )
    assert plan.config_hash == sha256_json(asdict(ExperimentConfig.load(config_path)))


def test_cli_requires_the_bundled_config_and_canonical_artifact_root(
    repo_fixture: Path,
) -> None:
    lock = repo_fixture / "requirements-modal-phase-marker.txt"
    lock.write_text("example==1\n", encoding="utf-8")
    alternate_config = repo_fixture / "configs/alternate.toml"
    shutil.copyfile(repo_fixture / CONFIG_PATH, alternate_config)
    alternate_artifact_root = repo_fixture / "artifacts/alternate-phase-marker"
    alternate_artifact_root.mkdir()
    common = ["--repo-root", str(repo_fixture), "--dependency-lock", lock.name]

    with pytest.raises(ValueError, match="approved configuration"):
        modal_plan.main([
            "plan", *common, "--config", "configs/alternate.toml",
            "--artifact-root", "artifacts/phase-marker",
        ])
    with pytest.raises(ValueError, match="approved artifact root"):
        modal_plan.main([
            "run-id", *common, "--config", str(CONFIG_PATH),
            "--artifact-root", "artifacts/alternate-phase-marker",
        ])


def test_cli_rejects_nested_byte_identical_pseudo_root(
    repo_fixture: Path, capsys: pytest.CaptureFixture[str],
) -> None:
    lock = repo_fixture / "requirements-modal-phase-marker.txt"
    lock.write_text("example==1\n", encoding="utf-8")
    nested_root = repo_fixture / "copy"
    nested_config = nested_root / CONFIG_PATH
    nested_config.parent.mkdir(parents=True)
    shutil.copyfile(repo_fixture / CONFIG_PATH, nested_config)
    shutil.copytree(
        repo_fixture / "artifacts/phase-marker",
        nested_root / "artifacts/phase-marker",
    )
    common = ["--repo-root", str(repo_fixture), "--dependency-lock", lock.name]

    with pytest.raises(ValueError, match="--repo-root approved configuration"):
        modal_plan.main([
            "plan", *common, "--config", "copy/configs/phase-marker-qwen25-7b.toml",
            "--artifact-root", "copy/artifacts/phase-marker",
        ])

    config_alias = repo_fixture / "configs/canonical-cli-link.toml"
    config_alias.symlink_to(repo_fixture / CONFIG_PATH)
    artifact_alias = repo_fixture / "artifacts/canonical-cli-link"
    artifact_alias.symlink_to(
        repo_fixture / "artifacts/phase-marker", target_is_directory=True
    )
    modal_plan.main([
        "run-id", *common,
        "--config", "configs/../configs/canonical-cli-link.toml",
        "--artifact-root", "artifacts/../artifacts/canonical-cli-link",
    ])
    assert capsys.readouterr().out.startswith("pilot-s42-cfg-")


def test_cli_prints_plan_or_only_run_id_without_side_effects(
    repo_fixture: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock = repo_fixture / "requirements-modal-phase-marker.txt"
    lock.write_text("example==1\n", encoding="utf-8")
    argv = [
        "plan", "--repo-root", str(repo_fixture), "--config", str(CONFIG_PATH),
        "--artifact-root", "artifacts/phase-marker", "--dependency-lock", lock.name,
    ]
    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("offline planner attempted an external or write side effect")

    monkeypatch.setattr(subprocess, "run", forbidden)
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(Path, "write_text", forbidden)
    modal_plan.main(argv)
    plan_payload = json.loads(capsys.readouterr().out)
    assert plan_payload["run_id"].startswith("pilot-s42-cfg-")

    modal_plan.main(["run-id", *argv[1:]])
    assert capsys.readouterr().out.strip() == plan_payload["run_id"]


def test_workspace_recreates_exact_repository_paths(repo_fixture: Path, tmp_path: Path) -> None:
    """Would fail if a workspace used host paths or omitted a bundled input."""
    bundle = build_input_bundle(repo_fixture)
    code_root = tmp_path / "code"
    (code_root / ".venv/bin").mkdir(parents=True)
    (code_root / ".venv/bin/python").write_text("#!/bin/sh\n", encoding="utf-8")
    (code_root / "phase_marker").mkdir()
    (code_root / "phase_marker/__init__.py").write_text("", encoding="utf-8")

    workspace = prepare_ephemeral_workspace(
        code_root=code_root,
        input_root=repo_fixture,
        run_root=tmp_path / "runs",
        bundle=bundle,
        stage="train",
        arm="glyph",
        attempt_id="attempt-1",
    )

    assert (workspace / ".venv/bin/python").is_symlink()
    assert (workspace / "phase_marker").is_symlink()
    assert (workspace / "configs/phase-marker-qwen25-7b.toml").is_file()
    assert (workspace / "artifacts/phase-marker/training-data/glyph.jsonl").is_file()
    assert workspace.is_relative_to(tmp_path / "runs" / "attempts" / "attempt-1")


def test_workspace_rejects_existing_attempt_directory(repo_fixture: Path, tmp_path: Path) -> None:
    """Would fail if a retry could overwrite a prior attempt workspace."""
    bundle = build_input_bundle(repo_fixture)
    code_root = tmp_path / "code"
    (code_root / ".venv/bin").mkdir(parents=True)
    (code_root / ".venv/bin/python").touch()
    (code_root / "phase_marker").mkdir()
    (tmp_path / "runs/attempts/a1").mkdir(parents=True)

    with pytest.raises(FileExistsError, match="attempt workspace already exists"):
        prepare_ephemeral_workspace(
            code_root=code_root, input_root=repo_fixture, run_root=tmp_path / "runs",
            bundle=bundle, stage="train", arm="glyph", attempt_id="a1",
        )


def test_exact_command_uses_no_shell(
    repo_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if command execution reintroduced a shell boundary."""
    calls: list[tuple[object, dict[str, object]]] = []
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda argv, **kw: calls.append((argv, kw)) or type("Result", (), {"returncode": 0})(),
    )

    code_root = tmp_path / "code"
    (code_root / ".venv/bin").mkdir(parents=True)
    (code_root / ".venv/bin/python").touch()
    (code_root / "phase_marker").mkdir()
    workspace = prepare_ephemeral_workspace(
        code_root=code_root, input_root=repo_fixture, run_root=tmp_path / "runs",
        bundle=build_input_bundle(repo_fixture), stage="train", arm="glyph", attempt_id="a1",
    )
    result = run_exact_command(
        _training_command("glyph"), workspace=workspace,
        log_path=workspace.parent / "logs/train.log", env={"A": "B"},
    )

    assert result == 0
    assert calls[0][0][:4] == ["./.venv/bin/python", "-m", "phase_marker.training", "train"]
    assert calls[0][1]["shell"] is False
    assert calls[0][1]["cwd"] == workspace
    assert (workspace.parent / "logs/train.log").is_file()


@pytest.mark.parametrize(
    "command",
    [
        "python -m phase_marker.training train --arm glyph --seed 42",
        "./.venv/bin/python -m phase_marker.training train --arm glyph --seed 101",
        "./.venv/bin/python -m phase_marker.training train --arm glyph --seed 42; id",
        "./.venv/bin/python -m phase_marker.training train --arm glyph --seed 42 --data ../outside.jsonl",
    ],
)
def test_exact_command_rejects_unapproved_shapes(tmp_path: Path, command: str) -> None:
    """Would fail if the boundary merely blacklisted a few shell characters."""
    with pytest.raises(ValueError, match="approved command"):
        run_exact_command(command, workspace=tmp_path, log_path=tmp_path / "run.log", env={})


def test_failed_attempt_never_promotes(tmp_path: Path) -> None:
    """Would fail if a failed or unvalidated attempt could reach canonical output."""
    receipt = receipt_fixture(exit_status=1, validated=False)
    with pytest.raises(ValueError, match="validated successful receipt"):
        promote_validated_output(
            tmp_path / "output", tmp_path / "attempts/a1",
            tmp_path / "canonical/glyph", receipt,
        )
    assert not (tmp_path / "canonical/glyph").exists()


def test_receipt_and_log_are_outside_hashed_output(tmp_path: Path) -> None:
    """Would fail if receipt persistence were mixed into a checkpoint output tree."""
    receipt_path = write_attempt_receipt(tmp_path / "runs", receipt_fixture())
    assert "/receipts/" in receipt_path.as_posix()
    assert "/checkpoints/" not in receipt_path.as_posix()


def test_receipt_rejects_stale_artifact_id(tmp_path: Path) -> None:
    """Would fail if receipt fields could be changed without changing their identity."""
    receipt = receipt_fixture(artifact_id="0" * 64)
    with pytest.raises(ValueError, match="artifact ID"):
        write_attempt_receipt(tmp_path / "runs", receipt)


def test_promotion_copies_attempt_bytes_once_and_refuses_existing_canonical(tmp_path: Path) -> None:
    """Would fail if promotion overwrote a canonical result or changed output bytes."""
    receipt = _receipt_for_file(
        "adapter.bin", b"frozen adapter bytes", timeout_seconds=14_400
    )
    attempt_root = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt_root / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"frozen adapter bytes")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"

    promoted = promote_validated_output(source, attempt_root, canonical, receipt)

    assert promoted == canonical
    assert (canonical / "adapter.bin").read_bytes() == b"frozen adapter bytes"
    assert (source / "adapter.bin").read_bytes() == b"frozen adapter bytes"
    with pytest.raises(FileExistsError, match="canonical output already exists"):
        promote_validated_output(source, attempt_root, canonical, receipt)


def test_retained_promotion_lease_is_exact_self_hashed_owner_deadline(
    tmp_path: Path,
) -> None:
    """Would fail if resume inferred staleness from an empty or unauthenticated lock."""
    receipt = _receipt_for_file(
        "adapter.bin", b"frozen adapter bytes", timeout_seconds=14_400
    )
    attempt_root = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt_root / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"frozen adapter bytes")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    created = datetime.fromisoformat(receipt.started_at) + timedelta(seconds=1)

    promote_validated_output(
        source,
        attempt_root,
        canonical,
        receipt,
        retain_lease=True,
        lease_created_at=created,
    )

    lock = canonical.parent / ".glyph.promotion.lock"
    lease = modal_artifacts.load_promotion_lease_payload(lock.read_bytes())
    assert lease["owner_attempt_id"] == receipt.attempt_id
    assert lease["owner_receipt_artifact_id"] == receipt.artifact_id
    assert lease["owner_stage_a_action_digest"] == receipt.stage_a_action_digest
    assert lease["producer_path"].endswith("/checkpoints/pilot/seed-42/glyph")
    assert lease["receipt_path"].endswith("/receipts/canonical/train/glyph.json")
    assert lease["lease_path"].endswith("/.glyph.promotion.lock")
    assert datetime.fromisoformat(str(lease["recover_after"])) == (
        datetime.fromisoformat(receipt.started_at)
        + timedelta(seconds=receipt.timeout_seconds + 3_600)
    )
    tampered = dict(lease)
    tampered["arm"] = "dot"
    with pytest.raises(ValueError, match="lease"):
        modal_artifacts.load_promotion_lease_payload(
            (canonical_json(tampered) + "\n").encode("utf-8")
        )


def test_promotion_lock_cleanup_failure_rolls_publication_back_to_quarantine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if post-rename lock cleanup could strand canonical output."""
    receipt = _receipt_for_file("adapter.bin", b"frozen adapter bytes")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"frozen adapter bytes")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    lock = canonical.parent / ".glyph.promotion.lock"
    original_unlink = os.unlink
    failed = False

    def fail_once(path: object, *args: object, **kwargs: object) -> None:
        nonlocal failed
        if path == lock.name and kwargs.get("dir_fd") is not None and not failed:
            failed = True
            raise OSError("injected promotion lock cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "unlink", fail_once)

    with pytest.raises(OSError, match="lock cleanup"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert not canonical.exists()
    assert (attempt / "failed-promotion/adapter.bin").read_bytes() == b"frozen adapter bytes"
    assert (attempt / "failed-promotion.lock").is_file()


def test_rescheduled_executions_get_distinct_uuid_attempt_ids() -> None:
    """Would fail if a reschedule could reuse an attempt's mutable namespace."""
    first = create_attempt_id()
    second = create_attempt_id()
    assert first != second
    assert len(first) == 36
    assert len(second) == 36


def _workspace_code_root(tmp_path: Path) -> Path:
    code_root = tmp_path / "code"
    (code_root / ".venv/bin").mkdir(parents=True)
    (code_root / ".venv/bin/python").write_text("#!/bin/sh\n", encoding="utf-8")
    (code_root / "phase_marker").mkdir()
    (code_root / "phase_marker/__init__.py").write_text("", encoding="utf-8")
    return code_root


def _training_command(arm: str) -> str:
    output = f"artifacts/phase-marker/checkpoints/pilot/seed-42/{arm}"
    return (
        "./.venv/bin/python -m phase_marker.training train "
        "--config configs/phase-marker-qwen25-7b.toml "
        f"--arm {arm} --seed 42 "
        f"--data artifacts/phase-marker/training-data/{arm}.jsonl "
        f"--output-dir {output} --manifest {output}/run-manifest.json"
    )


@pytest.mark.parametrize(
    "command_mutator",
    (
        lambda command: " " + command,
        lambda command: command.replace(" -m ", "  -m "),
        lambda command: command.replace("--arm glyph", "--arm 'glyph'"),
    ),
)
def test_exact_command_rejects_noncanonical_whitespace_and_quoting(
    tmp_path: Path,
    command_mutator: object,
) -> None:
    """Would fail if argv equivalence replaced byte-exact command approval."""
    assert callable(command_mutator)
    with pytest.raises(ValueError, match="approved command"):
        run_exact_command(
            command_mutator(_training_command("glyph")),
            workspace=tmp_path,
            log_path=tmp_path / "run.log",
            env={},
        )


def _receipt_for_file(path: str, content: bytes, **changes: object) -> AttemptReceipt:
    return receipt_fixture(
        expected_outputs=(path,),
        output_hashes=(hashlib.sha256(content).hexdigest(),),
        **changes,
    )


def test_workspace_setup_failure_keeps_partial_attempt_quarantined(
    repo_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if setup errors erase forensic evidence from an attempt."""
    original_write = Path.write_bytes
    interrupted = False

    def write_then_fail(destination: Path, content: bytes) -> int:
        nonlocal interrupted
        written = original_write(destination, content)
        if not interrupted and "workspace" in destination.parts:
            interrupted = True
            raise OSError("simulated copy interruption")
        return written

    monkeypatch.setattr(Path, "write_bytes", write_then_fail)
    with pytest.raises(OSError, match="copy interruption"):
        prepare_ephemeral_workspace(
            code_root=_workspace_code_root(tmp_path), input_root=repo_fixture,
            run_root=tmp_path / "runs", bundle=build_input_bundle(repo_fixture),
            stage="train", arm="glyph", attempt_id="attempt-keep",
        )

    retained = tmp_path / "runs/attempts/attempt-keep/workspace"
    assert (retained / ".venv/bin/python").is_symlink()
    assert (retained / "configs/phase-marker-qwen25-7b.toml").is_file()


def test_failed_output_copy_keeps_staging_quarantined(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a byte-validation failure deleted the attempt copy."""
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    monkeypatch.setattr(
        modal_artifacts,
        "_tree_hashes",
        lambda root: (("adapter.bin", 7, "a" * 64),)
        if Path(root).resolve() == source.resolve()
        else (("adapter.bin", 7, "b" * 64),),
    )

    with pytest.raises(ValueError, match="does not match"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert (attempt / "promotion-staging/adapter.bin").read_bytes() == b"adapter"


def test_promotion_rejects_source_mutation_immediately_before_copy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a raced source copy could escape the immutable receipt."""
    receipt = _receipt_for_file("adapter.bin", b"approved bytes")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"approved bytes")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    real_copytree = shutil.copytree

    def mutate_then_copy(source_root: Path, destination: Path, **kwargs: object) -> Path:
        (Path(source_root) / "adapter.bin").write_bytes(b"unapproved raced bytes")
        return real_copytree(source_root, destination, **kwargs)

    monkeypatch.setattr(shutil, "copytree", mutate_then_copy)

    with pytest.raises(ValueError, match="receipt output hash"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert not canonical.exists()
    assert (
        attempt / "promotion-staging/adapter.bin"
    ).read_bytes() == b"unapproved raced bytes"


def test_promotion_rejects_staged_mutation_immediately_before_rename(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if canonical bytes could change after staged receipt validation."""
    receipt = _receipt_for_file("adapter.bin", b"approved bytes")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"approved bytes")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    real_rename = modal_artifacts._atomic_rename_directory_noreplace_at

    def mutate_then_rename(**kwargs: object) -> None:
        (attempt / "promotion-staging/adapter.bin").write_bytes(
            b"unapproved after validation"
        )
        real_rename(**kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(
        modal_artifacts, "_atomic_rename_directory_noreplace_at", mutate_then_rename
    )

    with pytest.raises(ValueError, match="receipt output hash"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert not canonical.exists()
    assert (
        attempt / "failed-promotion/adapter.bin"
    ).read_bytes() == b"unapproved after validation"


def test_workspace_metadata_rejects_another_arm_command(
    repo_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a glyph workspace could execute a different frozen arm."""
    workspace = prepare_ephemeral_workspace(
        code_root=_workspace_code_root(tmp_path), input_root=repo_fixture,
        run_root=tmp_path / "runs", bundle=build_input_bundle(repo_fixture),
        stage="train", arm="glyph", attempt_id="attempt-glyph",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("ran drifted command"))

    with pytest.raises(ValueError, match="workspace command"):
        run_exact_command(
            _training_command("dot"), workspace=workspace,
            log_path=workspace / "logs/drift.log", env={},
        )


def test_log_must_be_outside_workspace_and_producer_paths(
    repo_fixture: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a producer tree could be polluted with execution logs."""
    workspace = prepare_ephemeral_workspace(
        code_root=_workspace_code_root(tmp_path), input_root=repo_fixture,
        run_root=tmp_path / "runs", bundle=build_input_bundle(repo_fixture),
        stage="train", arm="glyph", attempt_id="attempt-logs",
    )
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: pytest.fail("ran with producer log"))

    with pytest.raises(ValueError, match="outside the ephemeral workspace"):
        run_exact_command(
            _training_command("glyph"), workspace=workspace,
            log_path=workspace / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph/train.log",
            env={},
        )


@pytest.mark.parametrize("attempt_id", (".", ".."))
def test_workspace_rejects_dot_path_identity(
    repo_fixture: Path, tmp_path: Path, attempt_id: str,
) -> None:
    """Would fail if dot components could escape an attempt namespace."""
    with pytest.raises(ValueError, match="attempt ID"):
        prepare_ephemeral_workspace(
            code_root=_workspace_code_root(tmp_path), input_root=repo_fixture,
            run_root=tmp_path / "runs", bundle=build_input_bundle(repo_fixture),
            stage="train", arm="glyph", attempt_id=attempt_id,
        )


def test_promotion_rejects_output_bytes_not_named_by_receipt(tmp_path: Path) -> None:
    """Would fail if a valid receipt could promote different bytes than it records."""
    receipt = _receipt_for_file("adapter.bin", b"expected bytes")
    source = (
        tmp_path / "runs/attempts" / receipt.attempt_id / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"wrong bytes")

    with pytest.raises(ValueError, match="receipt output hash"):
        promote_validated_output(
            source, tmp_path / "runs/attempts" / receipt.attempt_id,
            tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph", receipt,
        )


def test_promotion_binds_receipt_arm_to_canonical_destination(tmp_path: Path) -> None:
    """Would fail if a glyph receipt could publish into another arm's canonical root."""
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    source = (
        tmp_path / "runs/attempts" / receipt.attempt_id / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")

    with pytest.raises(ValueError, match="canonical destination"):
        promote_validated_output(
            source, source.parents[6],
            tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/dot",
            receipt,
        )


def test_promotion_rejects_different_filesystem_before_staging(tmp_path: Path) -> None:
    """Would fail if cross-filesystem promotion staged bytes before refusing rename."""
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    other_filesystem = tmp_path / "runs/artifacts"
    other_filesystem.symlink_to("/proc", target_is_directory=True)

    with pytest.raises(ValueError, match="same filesystem|symlink"):
        promote_validated_output(
            source, attempt,
            other_filesystem / "phase-marker/checkpoints/pilot/seed-42/glyph", receipt,
        )

    assert not (attempt / "promotion-staging").exists()


def test_promotion_rejects_unlisted_regular_output_before_staging(tmp_path: Path) -> None:
    """Would fail if a receipt could omit a regular file copied into canonical output."""
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    (source / "unlisted.txt").write_bytes(b"not in receipt")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"

    with pytest.raises(ValueError, match="complete source file set"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert not (attempt / "promotion-staging").exists()
    assert not canonical.exists()


def test_promotion_rejects_symlink_entry_before_staging(tmp_path: Path) -> None:
    """Would fail if a symlink's followed bytes could evade the receipt file manifest."""
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = (
        attempt / "workspace"
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    )
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"symlink target")
    (source / "linked.bin").symlink_to(outside)
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"

    with pytest.raises(ValueError, match="regular files"):
        promote_validated_output(source, attempt, canonical, receipt)

    assert not (attempt / "promotion-staging").exists()


def _stage_job_inputs(repo_fixture: Path, input_root: Path) -> object:
    bundle = build_input_bundle(repo_fixture)
    bundle_root = input_root / "bundles" / bundle.bundle_id
    for item in bundle.files:
        destination = bundle_root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo_fixture / item.path, destination)
    (bundle_root / "bundle-manifest.json").write_text(
        canonical_json(asdict(bundle)) + "\n", encoding="utf-8"
    )
    return bundle


def _stage_job_model(qwen_snapshot: Path, model_root: Path) -> tuple[Path, ModelCacheManifest]:
    snapshot = (
        model_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    snapshot.parent.mkdir(parents=True)
    shutil.copytree(qwen_snapshot, snapshot)
    manifest = build_model_cache_manifest(snapshot)
    (snapshot.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json").write_text(
        canonical_json(asdict(manifest)) + "\n", encoding="utf-8"
    )
    return snapshot, manifest


def _job_execution_plan(repo_fixture: Path, bundle: object) -> modal_plan.PilotPlan:
    code_root = Path(__file__).resolve().parents[2]
    return modal_plan.build_pilot_plan(
        repo_fixture / CONFIG_PATH,
        repo_fixture / "artifacts/phase-marker",
        bundle=bundle,
        source_hash=hash_source_tree(code_root),
        dependency_lock_hash=hashlib.sha256(
            (code_root / "requirements-modal-phase-marker.txt").read_bytes()
        ).hexdigest(),
    )


def _stage_a_approval(
    plan: modal_plan.PilotPlan,
    *,
    cache_artifact_id: str = "e" * 64,
    smoke_receipt_artifact_id: str = "9" * 64,
    resume: bool = False,
) -> dict[str, object]:
    return modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=cache_artifact_id,
    )


def _stage_a_approval_for_model(
    plan: modal_plan.PilotPlan, model_root: Path, *, resume: bool = False,
) -> dict[str, object]:
    _snapshot, manifest = modal_artifacts._validated_model_cache(model_root)
    return _stage_a_approval(
        plan, cache_artifact_id=manifest.artifact_id, resume=resume
    )


def _execution_provenance(stage: str, suffix: str = "test") -> dict[str, object]:
    function = {
        "train": "run_training_job",
        "selection": "run_selection_job",
        "smoke": "smoke_remote",
        "finalizer": "finalize_stage_a_remote",
    }[stage]
    return {
        "modal_app_id": "ap-test",
        "modal_app_name": "phase-marker-pilot-stage-a",
        "modal_function_name": function,
        "modal_function_call_id": f"fc-{suffix}",
        "modal_input_id": f"in-{suffix}",
        "python_version": "3.12.test",
        "torch_version": "2.7.test",
        "cuda_runtime_version": "12.8.test",
        "cuda_driver_version": (
            "not-observed-cpu" if stage in {"smoke", "finalizer"} else "570.test"
        ),
        "runtime_versions": [
            {"module": module, "version": "locked-test-version"}
            for module in modal_artifacts.LOCKED_RUNTIME_MODULES
        ],
    }


def test_execute_jobs_use_exact_commands_offline_env_and_promote_only_producers(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if Stage A translated a command or mixed logs/receipts into output."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    snapshot, cache_manifest = _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[1]
    commands: list[tuple[list[str], dict[str, object]]] = []
    validated: list[tuple[str, str]] = []
    workspaces: list[Path] = []
    selection_parent_checks: list[tuple[bool, bool]] = []
    device_paths: list[Path] = []

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            assert kwargs["timeout"] == 10
            return SimpleNamespace(returncode=0, stdout="NVIDIA H200\n")
        commands.append((list(argv), dict(kwargs)))
        workspace = Path(str(kwargs["cwd"]))
        workspaces.append(workspace)
        (workspace / "unpublished-scratch.bin").write_bytes(b"scratch")
        output_flag = "--output-dir" if "phase_marker.training" in argv else "--output"
        if output_flag == "--output":
            training_parent = (
                workspace
                / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
            )
            selection_parent_checks.append(
                (
                    training_parent.is_symlink(),
                    all(
                        path.stat().st_mode & 0o222 == 0
                        for path in training_parent.rglob("*")
                        if path.is_file()
                    ),
                )
            )
        output = workspace / argv[argv.index(output_flag) + 1]
        output.mkdir(parents=True)
        if output_flag == "--output-dir":
            (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
            (output / "adapter_model.safetensors").write_bytes(b"adapter")
            (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        else:
            (output / "manifest.json").write_text("{}\n", encoding="utf-8")
            (output / "evidence.jsonl").write_text("{}\n", encoding="utf-8")
        kwargs["stdout"].write(b"fake model log\n")
        return SimpleNamespace(returncode=0)

    def validate(stage: str, producer: Path, *_: object) -> None:
        assert producer.is_dir()
        validated.append((stage, producer.name))

    monkeypatch.setattr(subprocess, "run", fake_subprocess)

    def fake_filesystem_device(path: Path) -> int:
        resolved = Path(path).resolve()
        device_paths.append(resolved)
        return 1 if resolved.is_relative_to(ephemeral_root) else 2

    monkeypatch.setattr(modal_artifacts, "_filesystem_device", fake_filesystem_device)
    volume = CommitVolume()
    common = {
        "plan_payload": modal_plan.pilot_plan_payload(plan),
        "job_payload": asdict(job),
        "code_root": code_root,
        "input_root": input_root,
        "model_root": model_root,
        "run_root": run_root,
        "volume": volume,
        "ephemeral_root": ephemeral_root,
        "environ": {"CUDA_VISIBLE_DEVICES": "0", "PATH": "/usr/bin"},
        "producer_validator": validate,
        "bf16_probe": lambda: True,
        "stage_a_approval": _stage_a_approval(
            plan, cache_artifact_id=cache_manifest.artifact_id
        ),
    }

    training = modal_artifacts.execute_pilot_job(
        stage="train", execution_provenance=_execution_provenance("train", "training"),
        **common,
    )
    selection = modal_artifacts.execute_pilot_job(
        stage="selection",
        execution_provenance=_execution_provenance("selection", "selection"),
        **common,
    )

    assert [argv for argv, _ in commands] == [
        shlex.split(job.training_command), shlex.split(job.selection_command)
    ]
    for _, kwargs in commands:
        assert kwargs["shell"] is False
        env = kwargs["env"]
        assert env["CUDA_VISIBLE_DEVICES"] == "0"
        assert env["HF_HUB_OFFLINE"] == "1"
        assert env["TRANSFORMERS_OFFLINE"] == "1"
        assert env["HF_HUB_CACHE"] == str(snapshot.parents[2])
    assert validated == [("train", "glyph"), ("selection", "glyph")]
    assert all(workspace.is_relative_to(ephemeral_root) for workspace in workspaces)
    assert all(not workspace.is_relative_to(run_root) for workspace in workspaces)
    assert selection_parent_checks == [(False, True)]
    assert all(not path.is_relative_to(ephemeral_root) for path in device_paths)
    assert training["command"] == job.training_command
    assert selection["command"] == job.selection_command
    assert training["model_cache_artifact_id"] == cache_manifest.artifact_id
    assert selection["model_cache_artifact_id"] == cache_manifest.artifact_id
    assert training["requested_gpu"] == "H100"
    assert training["observed_gpu"] == "NVIDIA H200"
    assert selection["requested_gpu"] == "H100"
    assert selection["observed_gpu"] == "NVIDIA H200"
    assert training["plan_digest"] == plan.plan_digest
    assert training["config_hash"] == plan.config_hash
    assert training["split_artifact_id"] == plan.split_artifact_id
    assert training["materialization_artifact_ids"] == list(
        plan.materialization_artifact_ids
    )
    assert training["model_revision"] == plan.model_revision
    assert training["modal_function_name"] == "run_training_job"
    assert training["modal_function_call_id"] == "fc-training"
    assert training["modal_input_id"] == "in-training"
    assert training["cuda_runtime_version"] == "12.8.test"
    assert training["promoted"] is True and selection["promoted"] is True
    assert modal_artifacts.validate_job_receipt_payload(
        receipt_payload=training,
        plan_payload=modal_plan.pilot_plan_payload(plan),
        job_payload=asdict(job),
        stage="train",
    )["artifact_id"] == training["artifact_id"]
    assert modal_artifacts.validate_job_receipt_payload(
        receipt_payload=selection,
        plan_payload=modal_plan.pilot_plan_payload(plan),
        job_payload=asdict(job),
        stage="selection",
    )["artifact_id"] == selection["artifact_id"]
    # Each successful job durably creates its lease, publishes the producer and
    # receipts, then removes the lease.
    assert volume.commit_count == 6

    run = run_root / "runs" / plan.run_id
    checkpoint = run / "artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    selected = run / "artifacts/phase-marker/checkpoint-selections/pilot/seed-42/glyph"
    assert {path.name for path in checkpoint.iterdir()} == {
        "adapter_config.json", "adapter_model.safetensors", "run-manifest.json"
    }
    assert {path.name for path in selected.iterdir()} == {"manifest.json", "evidence.jsonl"}
    assert not any(path.name.endswith(".log") for path in checkpoint.rglob("*"))
    assert not any("receipt" in path.name for path in selected.rglob("*"))
    assert len(list((run / "attempts").glob("*/logs/*.log"))) == 2
    assert not list((run / "attempts").glob("*/workspace"))
    assert not list((run / "attempts").rglob("unpublished-scratch.bin"))
    assert not list((run / "attempts").rglob("bundle-manifest.json"))
    assert not list((run / "attempts").rglob("adapter_model.safetensors"))
    assert len(list((run / "receipts/attempts").glob("*.json"))) == 2
    assert (run / "receipts/canonical/train/glyph.json").is_file()
    assert (run / "receipts/canonical/selection/glyph.json").is_file()


def test_job_validation_failure_persists_unpromoted_receipt_and_reraises(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a rejected producer vanished or became canonical success."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[2]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        workspace = Path(str(kwargs["cwd"]))
        output = workspace / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        kwargs["stdout"].write(b"producer reached validation\n")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = CommitVolume()
    with pytest.raises(RuntimeError, match="rejected dot"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "validation-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            producer_validator=lambda *_: (_ for _ in ()).throw(
                RuntimeError("producer rejected dot")
            ),
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    failed = list((run / "receipts/attempts").glob("*.json"))
    assert len(failed) == 1
    receipt = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt["arm"] == "dot"
    assert receipt["exit_status"] == 0
    assert receipt["failure_stage"] == "producer-validation"
    assert receipt["validated"] is False
    assert receipt["promoted"] is False
    assert "RuntimeError: producer rejected dot" in receipt["failure_reason"]
    assert len(list((run / "attempts").glob("*/logs/train.log"))) == 1
    assert not list((run / "attempts").glob("*/workspace"))
    assert not list((run / "attempts").rglob("adapter_model.safetensors"))
    assert not (run / "artifacts/phase-marker/checkpoints/pilot/seed-42/dot").exists()
    assert not (run / "receipts/canonical/train/dot.json").exists()
    assert volume.commit_count == 1


def test_job_subprocess_failure_without_outputs_still_persists_failed_receipt(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an early model-command crash masked itself while writing evidence."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[3]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        kwargs["stdout"].write(b"model process crashed\n")
        return SimpleNamespace(returncode=7)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = CommitVolume()
    with pytest.raises(RuntimeError, match="random.*status 7"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "command-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            bf16_probe=lambda: True,
        )

    failed = list(
        (run_root / "runs" / plan.run_id / "receipts/attempts").glob("*.json")
    )
    assert len(failed) == 1
    receipt = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt["exit_status"] == 7
    assert receipt["expected_outputs"] == []
    assert receipt["output_hashes"] == []
    assert receipt["validated"] is False and receipt["promoted"] is False
    assert volume.commit_count == 1


def test_job_bf16_preflight_failure_still_persists_failed_receipt(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if hardware rejection happened outside the forensic attempt boundary."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[4]
    commands: list[list[str]] = []

    def fake_subprocess(argv: list[str], **_: object) -> SimpleNamespace:
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = CommitVolume()
    with pytest.raises(RuntimeError, match="BF16"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "bf16-fail"),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            bf16_probe=lambda: False,
        )

    run = run_root / "runs" / plan.run_id
    failed = list((run / "receipts/attempts").glob("*.json"))
    assert len(failed) == 1
    receipt = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt["arm"] == "direct"
    assert receipt["observed_gpu"] == "NVIDIA H100 80GB HBM3"
    assert receipt["validated"] is False and receipt["promoted"] is False
    assert "RuntimeError: pilot job requires BF16 support" in receipt["failure_reason"]
    assert commands == [["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]]
    assert len(list((run / "attempts").glob("*/logs/train.log"))) == 1
    assert volume.commit_count == 1


def test_job_rejects_nonapproved_gpu_before_model_command_and_persists_failure(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a singular non-Hopper GPU could cross the model boundary."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[4]
    commands: list[list[str]] = []

    def fake_subprocess(argv: list[str], **_: object) -> SimpleNamespace:
        commands.append(argv)
        return SimpleNamespace(returncode=0, stdout="NVIDIA A100-SXM4-80GB\n")

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = CommitVolume()
    with pytest.raises(RuntimeError, match="H100 or H200"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "gpu-fail"),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            bf16_probe=lambda: pytest.fail("BF16 probe ran on rejected GPU"),
        )

    run = run_root / "runs" / plan.run_id
    failed = list((run / "receipts/attempts").glob("*.json"))
    assert len(failed) == 1
    receipt = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt["requested_gpu"] == "H100"
    assert receipt["observed_gpu"] == "NVIDIA A100-SXM4-80GB"
    assert receipt["validated"] is False and receipt["promoted"] is False
    assert commands == [["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]]
    assert volume.commit_count == 1


def test_selection_workspace_failure_still_persists_failed_receipt(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if workspace construction escaped the forensic attempt boundary."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[0]
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("hardware or model command ran"),
    )
    volume = CommitVolume()

    with pytest.raises(ValueError, match="canonical training root is missing"):
        modal_artifacts.execute_pilot_job(
            stage="selection",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance(
                "selection", "workspace-fail"
            ),
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    receipts = list((run / "receipts/attempts").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["stage"] == "selection"
    assert receipt["validated"] is False and receipt["promoted"] is False
    assert "canonical training root is missing" in receipt["failure_reason"]
    assert len(list((run / "attempts").glob("*/logs/selection.log"))) == 1
    assert volume.commit_count == 1


def test_job_lease_durability_failure_aborts_before_promotion_and_records_failure(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an undurable lease allowed canonical promotion to proceed."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[5]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        workspace = Path(str(kwargs["cwd"]))
        output = workspace / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = FailFirstCommitVolume()
    with pytest.raises(RuntimeError, match="injected first commit failure"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "commit-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            producer_validator=lambda *_: None,
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    canonical = run / "artifacts/phase-marker/checkpoints/pilot/seed-42/filler"
    canonical_receipt = run / "receipts/canonical/train/filler.json"
    failed = list((run / "receipts/attempts").glob("*.json"))
    assert not canonical.exists()
    assert not canonical_receipt.exists()
    assert len(failed) == 1
    receipt_payload = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt_payload["validated"] is False and receipt_payload["promoted"] is False
    assert "RuntimeError: injected first commit failure" in receipt_payload["failure_reason"]
    failed_attempt = next((run / "attempts").iterdir())
    assert not (failed_attempt / "failed-publication").exists()
    assert not (
        canonical.parent / f".{canonical.name}.promotion.lock"
    ).exists()
    assert not (failed_attempt / "workspace").exists()
    assert not list(failed_attempt.rglob("bundle-manifest.json"))
    assert volume.commit_count == 2


def test_job_post_promotion_mismatch_quarantines_and_commits_failure_receipt(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if outer canonical revalidation could lose rollback evidence."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[5]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        workspace = Path(str(kwargs["cwd"]))
        output = workspace / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    canonical = (
        run_root / "runs" / plan.run_id
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/filler"
    )
    real_validate = modal_artifacts._validate_receipt_outputs
    canonical_checks = 0

    def mutate_before_outer_validation(root: Path, receipt: AttemptReceipt) -> None:
        nonlocal canonical_checks
        if Path(root) == canonical:
            canonical_checks += 1
            if canonical_checks == 2:
                (canonical / "adapter_model.safetensors").write_bytes(
                    b"unapproved after promotion"
                )
        real_validate(root, receipt)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    monkeypatch.setattr(
        modal_artifacts, "_validate_receipt_outputs", mutate_before_outer_validation
    )
    volume = CommitVolume()

    with pytest.raises(ValueError, match="receipt output hash"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "post-promotion-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            producer_validator=lambda *_: None,
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    assert canonical_checks == 2
    assert not canonical.exists()
    assert not (run / "receipts/canonical/train/filler.json").exists()
    attempt = next((run / "attempts").iterdir())
    assert (
        attempt / "failed-publication/producer/adapter_model.safetensors"
    ).read_bytes() == b"unapproved after promotion"
    assert (attempt / "failed-publication/promotion.lock").is_file()
    receipts = list((run / "receipts/attempts").glob("*.json"))
    assert len(receipts) == 1
    failed = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert failed["failure_stage"] == "post-promotion-validation"
    assert failed["validated"] is False and failed["promoted"] is False
    assert volume.commit_count == 2


def test_promotion_never_replaces_a_late_empty_destination_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = attempt / "workspace/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    original_rename = modal_artifacts._renameat2_noreplace

    def race_rename(
        source_parent_fd: int, source_name: str,
        destination_parent_fd: int, destination_name: str,
    ) -> None:
        os.mkdir(destination_name, dir_fd=destination_parent_fd)
        original_rename(
            source_parent_fd, source_name, destination_parent_fd, destination_name
        )

    monkeypatch.setattr(modal_artifacts, "_renameat2_noreplace", race_rename)
    with pytest.raises(FileExistsError):
        promote_validated_output(source, attempt, canonical, receipt)
    assert canonical.is_dir()
    assert not (canonical / "adapter.bin").exists()


def test_promotion_rejects_dangling_destination_symlink_without_redirect(
    tmp_path: Path,
) -> None:
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = attempt / "workspace/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    canonical.parent.mkdir(parents=True)
    outside = tmp_path / "outside/missing"
    canonical.symlink_to(outside, target_is_directory=True)

    with pytest.raises(FileExistsError):
        promote_validated_output(source, attempt, canonical, receipt)
    assert canonical.is_symlink()
    assert not outside.exists()


def test_promotion_rejects_symlinked_canonical_parent_without_external_write(
    tmp_path: Path,
) -> None:
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = attempt / "workspace/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    external = tmp_path / "external"
    external.mkdir()
    redirected = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42"
    redirected.parent.mkdir(parents=True)
    redirected.symlink_to(external, target_is_directory=True)
    canonical = redirected / "glyph"

    with pytest.raises(ValueError, match="symlink|canonical"):
        promote_validated_output(source, attempt, canonical, receipt)
    assert not (external / "glyph").exists()


def test_promotion_detects_post_rename_target_inode_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = _receipt_for_file("adapter.bin", b"adapter")
    attempt = tmp_path / "runs/attempts" / receipt.attempt_id
    source = attempt / "workspace/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    source.mkdir(parents=True)
    (source / "adapter.bin").write_bytes(b"adapter")
    canonical = tmp_path / "runs/artifacts/phase-marker/checkpoints/pilot/seed-42/glyph"
    original_rename = modal_artifacts._renameat2_noreplace

    def replace_after_rename(
        source_parent_fd: int, source_name: str,
        destination_parent_fd: int, destination_name: str,
    ) -> None:
        original_rename(
            source_parent_fd, source_name, destination_parent_fd, destination_name
        )
        os.rename(
            destination_name, "attacker-stolen", src_dir_fd=destination_parent_fd,
            dst_dir_fd=destination_parent_fd,
        )
        os.mkdir(destination_name, dir_fd=destination_parent_fd)

    monkeypatch.setattr(
        modal_artifacts, "_renameat2_noreplace", replace_after_rename
    )
    with pytest.raises(OSError, match="inode changed"):
        promote_validated_output(source, attempt, canonical, receipt)
    assert canonical.is_dir() and not any(canonical.iterdir())
    assert (canonical.parent / "attacker-stolen/adapter.bin").is_file()


def test_job_publication_commit_failure_quarantines_uncommitted_pair_and_lease(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a failed publication commit stranded canonical state or its lease."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[5]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        workspace = Path(str(kwargs["cwd"]))
        output = workspace / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = FailNthCommitVolume(2)
    with pytest.raises(RuntimeError, match="injected commit 2 failure"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "commit-two-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            producer_validator=lambda *_: None,
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    canonical = run / "artifacts/phase-marker/checkpoints/pilot/seed-42/filler"
    canonical_receipt = run / "receipts/canonical/train/filler.json"
    failed = list((run / "receipts/attempts").glob("*.json"))
    assert not canonical.exists()
    assert not canonical_receipt.exists()
    assert len(failed) == 1
    receipt = json.loads(failed[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is False and receipt["promoted"] is False
    assert "RuntimeError: injected commit 2 failure" in receipt["failure_reason"]
    attempt = next((run / "attempts").iterdir())
    assert (attempt / "failed-publication/producer").is_dir()
    assert (attempt / "failed-publication/success-receipt.json").is_file()
    assert (attempt / "failed-publication/canonical-receipt.json").is_file()
    assert (attempt / "failed-publication/promotion.lock").is_file()
    assert not (
        canonical.parent / f".{canonical.name}.promotion.lock"
    ).exists()
    assert volume.commit_count == 3


def test_job_lease_cleanup_commit_failure_preserves_durable_publication(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if cleanup uncertainty rolled back a durably published producer."""
    code_root = Path(__file__).resolve().parents[2]
    input_root = tmp_path / "inputs"
    model_root = tmp_path / "model-cache"
    run_root = tmp_path / "runs"
    ephemeral_root = tmp_path / "ephemeral"
    ephemeral_root.mkdir()
    bundle = _stage_job_inputs(repo_fixture, input_root)
    _stage_job_model(qwen_snapshot, model_root)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[5]

    def fake_subprocess(argv: list[str], **kwargs: object) -> SimpleNamespace:
        if argv[0] == "nvidia-smi":
            return SimpleNamespace(returncode=0, stdout="NVIDIA H100 80GB HBM3\n")
        workspace = Path(str(kwargs["cwd"]))
        output = workspace / argv[argv.index("--output-dir") + 1]
        output.mkdir(parents=True)
        (output / "adapter_config.json").write_text("{}\n", encoding="utf-8")
        (output / "adapter_model.safetensors").write_bytes(b"adapter")
        (output / "run-manifest.json").write_text("{}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(subprocess, "run", fake_subprocess)
    volume = FailNthCommitVolume(3)
    with pytest.raises(RuntimeError, match="injected commit 3 failure"):
        modal_artifacts.execute_pilot_job(
            stage="train",
            plan_payload=modal_plan.pilot_plan_payload(plan),
            job_payload=asdict(job),
            code_root=code_root,
            input_root=input_root,
            model_root=model_root,
            run_root=run_root,
            volume=volume,
            stage_a_approval=_stage_a_approval_for_model(plan, model_root),
            execution_provenance=_execution_provenance("train", "commit-three-fail"),
            ephemeral_root=ephemeral_root,
            environ={"CUDA_VISIBLE_DEVICES": "0"},
            producer_validator=lambda *_: None,
            bf16_probe=lambda: True,
        )

    run = run_root / "runs" / plan.run_id
    canonical = run / "artifacts/phase-marker/checkpoints/pilot/seed-42/filler"
    canonical_receipt = run / "receipts/canonical/train/filler.json"
    attempt_receipts = list((run / "receipts/attempts").glob("*.json"))
    assert canonical.is_dir()
    assert canonical_receipt.is_file()
    assert len(attempt_receipts) == 1
    receipt = json.loads(attempt_receipts[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is True and receipt["promoted"] is True
    assert not list((run / "attempts").glob("*/failed-publication"))
    assert volume.commit_count == 3


def _finalizer_receipt_payload(
    plan: modal_plan.PilotPlan,
    job: modal_plan.PilotJob,
    stage: str,
    *,
    cache_artifact_id: str = "e" * 64,
    resume: bool = False,
) -> dict[str, object]:
    approval = _stage_a_approval(
        plan, cache_artifact_id=cache_artifact_id, resume=resume
    )
    command = job.training_command if stage == "train" else job.selection_command
    paths = (
        ("adapter_config.json", "adapter_model.safetensors", "run-manifest.json")
        if stage == "train"
        else ("manifest.json", "evidence.jsonl")
    )
    receipt = AttemptReceipt(
        schema_version=1,
        run_id=plan.run_id,
        bundle_id=plan.bundle_id,
        bundle_manifest_artifact_id=plan.bundle_manifest_artifact_id,
        bundle_files=plan.bundle_files,
        modal_environment=plan.modal_environment,
        stage=stage,
        arm=job.arm,
        seed=42,
        attempt_id=f"finalizer-{stage}-{job.arm}",
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=plan.source_hash,
        dependency_lock_hash=plan.dependency_lock_hash,
        model_cache_artifact_id=cache_artifact_id,
        stage_a_action_digest=str(approval["approval_digest"]),
        stage_a_resume=resume,
        smoke_receipt_artifact_id=str(approval["smoke_receipt_artifact_id"]),
        requested_gpu="H100",
        observed_gpu="NVIDIA H100 80GB HBM3",
        started_at="2026-08-05T00:00:00+00:00",
        finished_at="2026-08-05T00:01:00+00:00",
        elapsed_seconds=60.0,
        timeout_seconds=14_400,
        exit_status=0,
        validated=True,
        promoted=True,
        expected_outputs=paths,
        output_hashes=tuple(hashlib.sha256(path.encode()).hexdigest() for path in paths),
        failure_reason=None,
        plan_digest=plan.plan_digest,
        config_hash=plan.config_hash,
        split_artifact_id=plan.split_artifact_id,
        materialization_artifact_ids=plan.materialization_artifact_ids,
        model_revision=plan.model_revision,
        modal_app_id="ap-test",
        modal_app_name="phase-marker-pilot-stage-a",
        modal_function_name=(
            "run_training_job" if stage == "train" else "run_selection_job"
        ),
        modal_function_call_id=f"fc-{stage}-{job.arm}",
        modal_input_id=f"in-{stage}-{job.arm}",
        python_version="3.12.test",
        torch_version="2.7.test",
        cuda_runtime_version="12.8.test",
        cuda_driver_version="570.test",
        runtime_versions=tuple(
            (module, "locked-test-version")
            for module in modal_artifacts.LOCKED_RUNTIME_MODULES
        ),
        artifact_id="",
    )
    receipt = replace(receipt, artifact_id=receipt.recomputed_artifact_id())
    payload = asdict(receipt)
    payload["expected_outputs"] = list(receipt.expected_outputs)
    payload["output_hashes"] = list(receipt.output_hashes)
    payload["materialization_artifact_ids"] = list(
        receipt.materialization_artifact_ids
    )
    payload["runtime_versions"] = [
        {"module": module, "version": version}
        for module, version in receipt.runtime_versions
    ]
    payload["bundle_files"] = [asdict(item) for item in receipt.bundle_files]
    return payload


@pytest.mark.parametrize(
    ("field", "changed"),
    (
        ("stage_a_action_digest", "d" * 64),
        ("stage_a_resume", True),
        ("smoke_receipt_artifact_id", "d" * 64),
        ("model_cache_artifact_id", "d" * 64),
    ),
)
def test_attempt_receipt_rejects_self_hashed_approval_field_tampering(
    repo_fixture: Path,
    field: str,
    changed: object,
) -> None:
    """Would fail if a self-hash could hide altered Stage A authorization."""
    bundle = build_input_bundle(repo_fixture)
    plan = _job_execution_plan(repo_fixture, bundle)
    payload = _finalizer_receipt_payload(plan, plan.jobs[0], "train")
    payload[field] = changed
    unsigned = dict(payload)
    unsigned.pop("artifact_id")
    payload["artifact_id"] = modal_artifacts.sha256_json(unsigned)

    with pytest.raises(ValueError, match="receipt"):
        modal_artifacts.load_attempt_receipt_payload(payload)


def test_finalizer_preserves_mixed_fresh_and_resume_approval_history(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
) -> None:
    """Would fail if resumed lineage flattened receipts into one approval claim."""
    bundle = build_input_bundle(repo_fixture)
    plan = _job_execution_plan(repo_fixture, bundle)
    plan_payload = modal_plan.pilot_plan_payload(plan)
    _snapshot, cache_manifest = _stage_job_model(
        qwen_snapshot, tmp_path / "model-cache"
    )
    training = tuple(
        _finalizer_receipt_payload(
            plan,
            job,
            "train",
            cache_artifact_id=cache_manifest.artifact_id,
            resume=index >= 3,
        )
        for index, job in enumerate(plan.jobs)
    )
    selection = tuple(
        _finalizer_receipt_payload(
            plan,
            job,
            "selection",
            cache_artifact_id=cache_manifest.artifact_id,
            resume=True,
        )
        for job in plan.jobs
    )
    approval = _stage_a_approval(
        plan, cache_artifact_id=cache_manifest.artifact_id, resume=True
    )

    result = modal_artifacts.finalize_stage_a(
        plan_payload=plan_payload,
        receipts=(*training, *selection),
        input_root=tmp_path / "inputs",
        model_root=tmp_path / "model-cache",
        run_root=tmp_path / "runs",
        volume=CommitVolume(),
        stage_a_approval=approval,
        execution_provenance=_execution_provenance(
            "finalizer", "mixed-approval-history"
        ),
        behavior_gate=lambda **_: {
            "passed": True,
            "checked_artifact_ids": ["f" * 64],
            "commands": ["./.venv/bin/python -m phase_marker.behavior run"],
        },
    )

    expected_history = modal_artifacts._receipt_approval_history(
        (*training, *selection)
    )
    assert result["stage_a_resume"] is True
    assert result["stage_a_action_digest"] == approval["approval_digest"]
    assert result["receipt_approval_history"] == expected_history
    assert {item["stage_a_resume"] for item in expected_history} == {False, True}
    assert sum(
        len(item["receipt_artifact_ids"]) for item in expected_history
    ) == 12
    assert modal_artifacts.validate_stage_a_summary(
        result,
        plan_payload=plan_payload,
        training_receipts=training,
        selection_receipts=selection,
        stage_a_approval=approval,
    ) == result


def test_cpu_finalizer_runs_read_only_gate_and_publishes_inert_stop_summary(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if finalization executed behavior or published an authorization field."""
    bundle = build_input_bundle(repo_fixture)
    plan = _job_execution_plan(repo_fixture, bundle)
    plan_payload = modal_plan.pilot_plan_payload(plan)
    snapshot, cache_manifest = _stage_job_model(
        qwen_snapshot, tmp_path / "model-cache"
    )
    training = tuple(
        _finalizer_receipt_payload(
            plan, job, "train", cache_artifact_id=cache_manifest.artifact_id
        )
        for job in plan.jobs
    )
    selection = tuple(
        _finalizer_receipt_payload(
            plan, job, "selection", cache_artifact_id=cache_manifest.artifact_id
        )
        for job in plan.jobs
    )
    gate_calls: list[dict[str, object]] = []
    monkeypatch.setenv("HF_HUB_CACHE", "before-cache")
    monkeypatch.setenv("HF_HUB_OFFLINE", "before-hub")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "before-transformers")

    def gate(**kwargs: object) -> dict[str, object]:
        assert os.environ["HF_HUB_CACHE"] == str(snapshot.parents[2])
        assert os.environ["HF_HUB_OFFLINE"] == "1"
        assert os.environ["TRANSFORMERS_OFFLINE"] == "1"
        gate_calls.append(dict(kwargs))
        return {
            "passed": True,
            "checked_artifact_ids": ["f" * 64],
            "commands": ["./.venv/bin/python -m phase_marker.behavior run"],
        }

    volume = CommitVolume()
    result = modal_artifacts.finalize_stage_a(
        plan_payload=plan_payload,
        receipts=(*training, *selection),
        input_root=tmp_path / "inputs",
        model_root=tmp_path / "model-cache",
        run_root=tmp_path / "runs",
        volume=volume,
        stage_a_approval=_stage_a_approval(
            plan, cache_artifact_id=cache_manifest.artifact_id
        ),
        behavior_gate=gate,
        execution_provenance=_execution_provenance("finalizer", "finalizer"),
    )

    assert len(gate_calls) == 1
    assert gate_calls[0]["plan_payload"] is plan_payload
    assert gate_calls[0]["input_root"] == tmp_path / "inputs"
    assert gate_calls[0]["model_root"] == tmp_path / "model-cache"
    assert gate_calls[0]["run_root"] == tmp_path / "runs"
    assert result["stopped_before_behavior"] is True
    assert result["next_command"] == "./.venv/bin/python -m phase_marker.behavior run"
    assert result["elapsed_gpu_seconds"] == {
        "training": 360.0,
        "selection": 360.0,
        "total": 720.0,
    }
    assert result["finalizer_provenance"]["modal_function_name"] == (
        "finalize_stage_a_remote"
    )
    assert "confirmation_seeds" not in result
    assert "mechanism_approval" not in result
    assert "callback" not in result
    summary_path = tmp_path / "runs" / "runs" / plan.run_id / "stage-a-summary.json"
    assert json.loads(summary_path.read_text(encoding="utf-8")) == result
    assert volume.commit_count == 1
    assert os.environ["HF_HUB_CACHE"] == "before-cache"
    assert os.environ["HF_HUB_OFFLINE"] == "before-hub"
    assert os.environ["TRANSFORMERS_OFFLINE"] == "before-transformers"

    resumed = modal_artifacts.finalize_stage_a(
        plan_payload=plan_payload,
        receipts=(*training, *selection),
        input_root=tmp_path / "inputs",
        model_root=tmp_path / "model-cache",
        run_root=tmp_path / "runs",
        volume=volume,
        stage_a_approval=_stage_a_approval(
            plan, cache_artifact_id=cache_manifest.artifact_id
        ),
        behavior_gate=gate,
        execution_provenance=_execution_provenance("finalizer", "finalizer"),
    )
    assert resumed == result
    assert len(gate_calls) == 2
    assert volume.commit_count == 1

    for change in (
        {"next_command": lambda: None},
        {"confirmation_seeds": [101, 202, 303]},
        {"mechanism_approval": True},
    ):
        invalid = {**result, **change}
        with pytest.raises(ValueError, match="summary"):
            modal_artifacts.validate_stage_a_summary(
                invalid,
                plan_payload=plan_payload,
                training_receipts=training,
                selection_receipts=selection,
            )


def test_default_training_producer_validation_rejects_semantically_empty_manifest(
    repo_fixture: Path,
) -> None:
    """Would fail if producer validation only checked that expected filenames exist."""
    bundle = build_input_bundle(repo_fixture)
    plan = _job_execution_plan(repo_fixture, bundle)
    job = plan.jobs[0]
    producer = (
        repo_fixture
        / "artifacts/phase-marker/checkpoints/pilot/seed-42/semantic"
    )
    producer.mkdir(parents=True)
    (producer / "adapter_config.json").write_text("{}\n", encoding="utf-8")
    (producer / "adapter_model.safetensors").write_bytes(b"adapter")
    (producer / "run-manifest.json").write_text("{}\n", encoding="utf-8")
    job_payload = asdict(job)
    job_payload["expected_outputs"] = list(job.expected_outputs)

    with pytest.raises(ValueError, match="manifest|completion|producer"):
        modal_artifacts._validate_job_producer(
            "train", producer, modal_plan.pilot_plan_payload(plan), job_payload
        )


def test_finalizer_commit_failure_quarantines_summary_and_preserves_original_error(
    repo_fixture: Path,
    qwen_snapshot: Path,
    tmp_path: Path,
) -> None:
    """Would fail if an uncommitted summary made explicit resume permanently stuck."""
    bundle = build_input_bundle(repo_fixture)
    plan = _job_execution_plan(repo_fixture, bundle)
    plan_payload = modal_plan.pilot_plan_payload(plan)
    _snapshot, cache_manifest = _stage_job_model(
        qwen_snapshot, tmp_path / "model-cache"
    )
    training = tuple(
        _finalizer_receipt_payload(
            plan, job, "train", cache_artifact_id=cache_manifest.artifact_id
        )
        for job in plan.jobs
    )
    selection = tuple(
        _finalizer_receipt_payload(
            plan, job, "selection", cache_artifact_id=cache_manifest.artifact_id
        )
        for job in plan.jobs
    )
    volume = FailFirstCommitVolume()

    with pytest.raises(RuntimeError, match="injected first commit failure"):
        modal_artifacts.finalize_stage_a(
            plan_payload=plan_payload,
            receipts=(*training, *selection),
            input_root=tmp_path / "inputs",
            model_root=tmp_path / "model-cache",
            run_root=tmp_path / "runs",
            volume=volume,
            stage_a_approval=_stage_a_approval(
                plan, cache_artifact_id=cache_manifest.artifact_id
            ),
            execution_provenance=_execution_provenance(
                "finalizer", "commit-fail"
            ),
            behavior_gate=lambda **_: {
                "passed": True,
                "checked_artifact_ids": ["f" * 64],
                "commands": ["./.venv/bin/python -m phase_marker.behavior run"],
            },
        )

    run = tmp_path / "runs" / "runs" / plan.run_id
    assert not (run / "stage-a-summary.json").exists()
    assert len(list((run / "attempts").glob("*/stage-a-summary.json"))) == 1
    assert volume.commit_count == 2
