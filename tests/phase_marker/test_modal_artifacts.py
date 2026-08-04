from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import shutil
import socket
import subprocess

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
            {"bos_token_id": 151643, "eos_token_id": 151645, "pad_token_id": 151643},
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
                "eos_token_id": 151645,
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
        )

    receipts = list((run_root / "runs/invalid-plan/receipts/smoke").glob("*.json"))
    assert len(receipts) == 1
    assert not (tmp_path / "escaped").exists()
    assert volume.commit_count == 1


def receipt_fixture(**changes: object) -> AttemptReceipt:
    fields: dict[str, object] = {
        "schema_version": 1,
        "run_id": "pilot-s42-cfg-12345678-split-12345678-src-123456789012",
        "bundle_id": "a" * 64,
        "stage": "train",
        "arm": "glyph",
        "seed": 42,
        "attempt_id": "f41aa3e7-f05f-48e3-87d6-5877fcce21d1",
        "command": "./.venv/bin/python -m phase_marker.training train --config configs/phase-marker-qwen25-7b.toml --arm glyph --seed 42 --data artifacts/phase-marker/training-data/glyph.jsonl --output-dir artifacts/phase-marker/checkpoints/pilot/seed-42/glyph --manifest artifacts/phase-marker/checkpoints/pilot/seed-42/glyph/run-manifest.json",
        "command_hash": "b" * 64,
        "source_hash": "c" * 64,
        "dependency_lock_hash": "d" * 64,
        "model_cache_artifact_id": "e" * 64,
        "requested_gpu": "H100",
        "observed_gpu": None,
        "started_at": "2026-08-05T00:00:00+00:00",
        "finished_at": "2026-08-05T00:00:01+00:00",
        "elapsed_seconds": 1.0,
        "timeout_seconds": 60,
        "exit_status": 0,
        "validated": True,
        "promoted": False,
        "expected_outputs": ("adapter_config.json",),
        "output_hashes": ("f" * 64,),
        "failure_reason": None,
        "artifact_id": "",
    }
    fields.update(changes)
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


@pytest.mark.parametrize("status", [" M phase_marker/modal_plan.py\n", "D  README.md\n"])
def test_tracked_dirty_status_is_rejected(status: str) -> None:
    with pytest.raises(ValueError, match="tracked source changes"):
        require_clean_tracked_status(status)


def test_untracked_approved_artifacts_are_ignored() -> None:
    require_clean_tracked_status("?? artifacts/\n")


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
    assert changed_lock.run_id == plan.run_id

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
    receipt = _receipt_for_file("adapter.bin", b"frozen adapter bytes")
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
    original_copy = modal_artifacts.shutil.copyfile

    def copy_then_fail(source: Path, destination: Path) -> str:
        original_copy(source, destination)
        raise OSError("simulated copy interruption")

    monkeypatch.setattr(modal_artifacts.shutil, "copyfile", copy_then_fail)
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

    with pytest.raises(ValueError, match="same filesystem"):
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
