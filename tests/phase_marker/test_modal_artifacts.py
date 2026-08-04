from __future__ import annotations

from dataclasses import asdict, replace
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
    BundleFile,
    build_input_bundle,
    hash_source_tree,
    require_clean_tracked_status,
    validate_bundle_at_root,
)
import phase_marker.modal_plan as modal_plan
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
SOURCE_HASH = "1" * 64
LOCK_HASH = "2" * 64


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
