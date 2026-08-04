from __future__ import annotations

import builtins
from contextlib import contextmanager
from dataclasses import asdict, replace
import hashlib
import io
import importlib
import importlib.util
import json
from pathlib import Path
import re
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.modal_artifacts import (
    build_input_bundle,
    build_model_cache_manifest,
    hash_source_tree,
)
import phase_marker.modal_artifacts as modal_artifacts
import phase_marker.modal_plan as modal_plan
from phase_marker.token_audit import QWEN25_7B_TOKENIZER_REVISION
from tests.phase_marker.test_pipeline import _write_materializations, _write_split


REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "requirements-modal-phase-marker.txt"
DIRECT_PATH = REPO_ROOT / "requirements-modal-phase-marker.in"
CONFIG_PATH = Path("configs/phase-marker-qwen25-7b.toml")
DIRECT_PINS = (
    "accelerate==1.12.0",
    "datasets==4.4.2",
    "einops==0.8.1",
    "huggingface-hub==0.36.0",
    "modal==1.3.5",
    "numpy==2.2.6",
    "peft==0.18.0",
    "protobuf==6.33.2",
    "safetensors==0.7.0",
    "sentencepiece==0.2.1",
    "statsmodels==0.14.6",
    "tokenizers==0.22.2",
    "torch==2.9.0",
    "transformers==4.57.3",
    "vllm==0.13.0",
)
COMPILE_HEADER = (
    "#    uv pip compile requirements-modal-phase-marker.in "
    "--output-file requirements-modal-phase-marker.txt --python-version 3.12 "
    "--python-platform x86_64-manylinux_2_28 --generate-hashes"
)


class FakeRemoteFunction:
    def __init__(self, modal: FakeModal, function: Callable[..., object]) -> None:
        self._modal = modal
        self._function = function
        self.name = function.__name__

    def remote(self, *args: object, **kwargs: object) -> object:
        self._modal.rpc_calls.append(("remote", self.name, args, kwargs))
        raise AssertionError("adapter attempted a remote call")

    def local(self, *args: object, **kwargs: object) -> object:
        return self._function(*args, **kwargs)


class FakeImage:
    def __init__(self, modal: FakeModal, base: str, add_python: str) -> None:
        self._modal = modal
        self.operations: list[tuple[object, ...]] = [
            ("from_registry", base, {"add_python": add_python})
        ]

    def pip_install_from_requirements(self, path: str) -> FakeImage:
        self.operations.append(("pip_install_from_requirements", path))
        return self

    def add_local_dir(
        self, local_path: str, remote_path: str, *, copy: bool
    ) -> FakeImage:
        self.operations.append(("add_local_dir", local_path, remote_path, {"copy": copy}))
        return self

    def add_local_file(
        self, local_path: str, remote_path: str, *, copy: bool
    ) -> FakeImage:
        self.operations.append(("add_local_file", local_path, remote_path, {"copy": copy}))
        return self

    def run_commands(self, *commands: str) -> FakeImage:
        self.operations.append(("run_commands", *commands))
        return self


class FakeVolumeMount:
    def __init__(self, volume: FakeVolume) -> None:
        self.volume = volume
        self.read_only = True


class FakeVolume:
    def __init__(self, modal: FakeModal, name: str, create_if_missing: bool) -> None:
        self._modal = modal
        self.name = name
        self.create_if_missing = create_if_missing
        self.read_only_calls = 0

    def read_only(self) -> FakeVolumeMount:
        self.read_only_calls += 1
        return FakeVolumeMount(self)

    def __getattr__(self, name: str) -> Any:
        if name in {"commit", "reload"}:
            raise AssertionError(f"adapter attempted volume client RPC: {name}")
        raise AttributeError(name)


class RecordingVolume:
    """Small local stand-in for the Modal volume methods used by staging."""

    def __init__(self, files: dict[str, bytes] | None = None) -> None:
        self.files = dict(files or {})
        self.put_calls: list[SimpleNamespace] = []

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        assert recursive is True
        prefix = path.rstrip("/") + "/"
        return [
            SimpleNamespace(path=remote_path, type="file")
            for remote_path in sorted(self.files)
            if remote_path == path or remote_path.startswith(prefix)
        ]

    def read_file(self, path: str) -> list[bytes]:
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    @contextmanager
    def batch_upload(self) -> object:
        pending: list[tuple[str, bytes]] = []

        class Batch:
            def put_file(inner_self, local_file: object, remote_path: str) -> None:
                if isinstance(local_file, (str, Path)):
                    content = Path(local_file).read_bytes()
                else:
                    assert isinstance(local_file, io.BytesIO)
                    content = local_file.getvalue()
                self.put_calls.append(SimpleNamespace(remote_path=remote_path, content=content))
                pending.append((remote_path, content))

        yield Batch()
        self.files.update(pending)


class CommitOnlyVolume:
    def __init__(self) -> None:
        self.commit_count = 0

    def commit(self) -> None:
        self.commit_count += 1


class FakeApp:
    def __init__(
        self,
        modal: FakeModal,
        name: str,
        *,
        tags: dict[str, str],
        include_source: bool,
    ) -> None:
        self._modal = modal
        self.name = name
        self.tags = dict(tags)
        self.include_source = include_source
        self.remote_functions: dict[str, FakeRemoteFunction] = {}

    def function(self, **options: object) -> Callable[[Callable[..., object]], FakeRemoteFunction]:
        self._modal.declaration_calls.append(("function", dict(options)))

        def decorate(function: Callable[..., object]) -> FakeRemoteFunction:
            remote = FakeRemoteFunction(self._modal, function)
            self.remote_functions[function.__name__] = remote
            self._modal.declaration_calls.append(("function_decorated", function.__name__))
            return remote

        return decorate

    def local_entrypoint(
        self, **options: object
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        self._modal.declaration_calls.append(("local_entrypoint", dict(options)))

        def decorate(function: Callable[..., object]) -> Callable[..., object]:
            self._modal.declaration_calls.append(("local_entrypoint_decorated", function.__name__))
            return function

        return decorate

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._modal.importing:
            raise AssertionError("adapter attempted a client RPC during import")
        copied = dict(tags)
        self._modal.rpc_calls.append(("set_tags", copied))
        self.tags = copied

    def __getattr__(self, name: str) -> Any:
        if name in {"deploy", "hydrate", "run", "lookup"}:
            raise AssertionError(f"adapter attempted app client operation: {name}")
        raise AttributeError(name)


class FakeModal(ModuleType):
    def __init__(self) -> None:
        super().__init__("modal")
        self.importing = True
        self.declaration_calls: list[tuple[object, ...]] = []
        self.rpc_calls: list[tuple[object, ...]] = []
        self.images: list[FakeImage] = []
        self.volumes: list[FakeVolume] = []
        self.apps: list[FakeApp] = []
        self.App = self._app
        self.Image = SimpleNamespace(from_registry=self._from_registry)
        self.Volume = SimpleNamespace(from_name=self._from_name)

    def _app(
        self,
        name: str,
        *,
        tags: dict[str, str],
        include_source: bool,
    ) -> FakeApp:
        app = FakeApp(self, name, tags=tags, include_source=include_source)
        self.apps.append(app)
        self.declaration_calls.append(
            ("App", name, {"tags": dict(tags), "include_source": include_source})
        )
        return app

    def _from_registry(self, base: str, *, add_python: str) -> FakeImage:
        image = FakeImage(self, base, add_python)
        self.images.append(image)
        self.declaration_calls.append(("Image.from_registry", base, {"add_python": add_python}))
        return image

    def _from_name(self, name: str, *, create_if_missing: bool) -> FakeVolume:
        volume = FakeVolume(self, name, create_if_missing)
        self.volumes.append(volume)
        self.declaration_calls.append(
            ("Volume.from_name", name, {"create_if_missing": create_if_missing})
        )
        return volume

    def __getattr__(self, name: str) -> Any:
        if name in {"Client", "deploy", "enable_output", "runner"}:
            raise AssertionError(f"adapter attempted Modal client operation: {name}")
        raise AttributeError(name)


def _load_adapter(monkeypatch: pytest.MonkeyPatch, fake_modal: FakeModal) -> ModuleType:
    path = REPO_ROOT / "modal_phase_marker.py"
    spec = importlib.util.spec_from_file_location("modal_phase_marker_under_test", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
    finally:
        fake_modal.importing = False
    module.source_text = path.read_text(encoding="utf-8")
    module.fake_modal = fake_modal
    return module


@pytest.fixture
def imported_adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_adapter(monkeypatch, FakeModal())


@pytest.fixture
def pilot_repo(tmp_path: Path) -> Path:
    config_path = tmp_path / CONFIG_PATH
    config_path.parent.mkdir(parents=True)
    shutil.copyfile(REPO_ROOT / CONFIG_PATH, config_path)
    config = ExperimentConfig.load(config_path)
    artifact_root = tmp_path / "artifacts/phase-marker"
    _write_split(artifact_root, config)
    _write_materializations(artifact_root, config)
    shutil.copyfile(LOCK_PATH, tmp_path / LOCK_PATH.name)
    return tmp_path


def _build_plan(pilot_repo: Path, dependency_lock_hash: str) -> modal_plan.PilotPlan:
    return modal_plan.build_pilot_plan(
        pilot_repo / CONFIG_PATH,
        pilot_repo / "artifacts/phase-marker",
        bundle=build_input_bundle(pilot_repo),
        source_hash="1" * 64,
        dependency_lock_hash=dependency_lock_hash,
    )


def _locked_requirements() -> dict[str, tuple[str, tuple[str, ...]]]:
    records: dict[str, tuple[str, tuple[str, ...]]] = {}
    lines = LOCK_PATH.read_text(encoding="utf-8").splitlines()
    starts = [index for index, line in enumerate(lines) if line and not line[0].isspace() and not line.startswith("#")]
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(lines)
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s\\]+) \\", lines[start])
        assert match is not None, f"unlocked requirement line: {lines[start]}"
        name, version = match.groups()
        canonical_name = re.sub(r"[-_.]+", "-", name).lower()
        assert canonical_name not in records, f"duplicate package: {canonical_name}"
        hashes = tuple(
            re.findall(r"--hash=sha256:([0-9a-f]{64})(?:\s|\\|$)", "\n".join(lines[start:end]))
        )
        assert hashes, f"requirement lacks sha256 hashes: {canonical_name}"
        records[canonical_name] = (version, hashes)
    return records


def test_compiled_lock_is_complete_hashed_and_reproducibly_targeted() -> None:
    lines = LOCK_PATH.read_text(encoding="utf-8").splitlines()
    assert lines[:2] == [
        "# This file was autogenerated by uv via the following command:",
        COMPILE_HEADER,
    ]
    assert tuple(DIRECT_PATH.read_text(encoding="utf-8").splitlines()) == DIRECT_PINS

    records = _locked_requirements()
    assert len(records) > len(DIRECT_PINS)
    assert all(len(hashes) >= 1 for _, hashes in records.values())
    for pin in DIRECT_PINS:
        name, version = pin.split("==", maxsplit=1)
        assert records[re.sub(r"[-_.]+", "-", name).lower()][0] == version


def test_dependency_lock_hash_tracks_only_the_compiled_lock_bytes(pilot_repo: Path) -> None:
    lock = pilot_repo / LOCK_PATH.name
    original_bytes = lock.read_bytes()
    original_hash = modal_plan._file_sha256(lock)
    original = _build_plan(pilot_repo, original_hash)

    lock.write_bytes(original_bytes + b"# byte mutation\n")
    changed_hash = modal_plan._file_sha256(lock)
    changed = _build_plan(pilot_repo, changed_hash)

    assert original.dependency_lock_hash == hashlib.sha256(original_bytes).hexdigest()
    assert changed.dependency_lock_hash != original.dependency_lock_hash
    assert changed.source_hash == original.source_hash
    assert changed.config_hash == original.config_hash
    assert changed.run_id == original.run_id


def test_modal_graph_is_dedicated_and_bounded(imported_adapter: ModuleType) -> None:
    assert imported_adapter.app.name == "phase-marker-pilot-stage-a"
    assert imported_adapter.BASE_IMAGE == (
        "nvidia/cuda@sha256:61f6c08f2b59036cb935e56d1e31a6b64e3ae2c7ddb86d33fa0b044c7917b719"
    )
    assert imported_adapter.VOLUME_NAMES == (
        "phase-marker-pilot-inputs-v1",
        "phase-marker-pilot-model-cache-v1",
        "phase-marker-pilot-runs-v1",
    )
    assert imported_adapter.GPU == "H100"
    assert imported_adapter.GPU_TIMEOUT_SECONDS == 14_400
    assert imported_adapter.MAX_GPU_CONTAINERS == 2
    assert "glyph-reasoning-vol" not in imported_adapter.source_text
    assert "/vol/work" not in imported_adapter.source_text

    fake = imported_adapter.fake_modal
    assert fake.rpc_calls == []
    assert len(fake.apps) == 1
    assert imported_adapter.app.include_source is False
    assert imported_adapter.app.tags == {
        "experiment": "phase-marker",
        "run-kind": "pilot",
        "seed": "42",
    }
    assert [(volume.name, volume.create_if_missing) for volume in fake.volumes] == [
        (name, True) for name in imported_adapter.VOLUME_NAMES
    ]

    assert len(fake.images) == 1
    assert imported_adapter.cpu_image is imported_adapter.gpu_image
    assert imported_adapter.gpu_image.operations == [
        ("from_registry", imported_adapter.BASE_IMAGE, {"add_python": "3.12"}),
        ("pip_install_from_requirements", "requirements-modal-phase-marker.txt"),
        ("add_local_dir", "phase_marker", "/opt/glyph_reasoning/phase_marker", {"copy": True}),
        (
            "add_local_file",
            "modal_phase_marker.py",
            "/opt/glyph_reasoning/modal_phase_marker.py",
            {"copy": True},
        ),
        (
            "add_local_file",
            "requirements-modal-phase-marker.txt",
            "/opt/glyph_reasoning/requirements-modal-phase-marker.txt",
            {"copy": True},
        ),
        (
            "run_commands",
            "mkdir -p /opt/glyph_reasoning/.venv/bin",
            "ln -sf /usr/local/bin/python /opt/glyph_reasoning/.venv/bin/python",
        ),
    ]
    assert ":12.8.1-cudnn-devel-ubuntu22.04" not in imported_adapter.BASE_IMAGE

    gpu_options = next(
        call[1] for call in fake.declaration_calls
        if call[0] == "function" and call[1].get("gpu") == "H100"
    )
    assert gpu_options["image"] is imported_adapter.gpu_image
    assert gpu_options["timeout"] == 14_400
    assert gpu_options["max_containers"] == 2
    assert gpu_options["retries"] == 0
    assert gpu_options["volumes"]["/mnt/inputs"].volume is imported_adapter.inputs_volume
    assert gpu_options["volumes"]["/mnt/model"].volume is imported_adapter.model_volume
    assert gpu_options["volumes"]["/mnt/runs"] is imported_adapter.runs_volume
    assert gpu_options["volumes"]["/mnt/inputs"].read_only is True
    assert gpu_options["volumes"]["/mnt/model"].read_only is True

    status_options = next(
        call[1] for call in fake.declaration_calls
        if call[0] == "function" and set(call[1]["volumes"]) == {"/mnt/runs"}
    )
    assert "gpu" not in status_options

    cache_options = next(
        call[1] for call in fake.declaration_calls
        if call[0] == "function" and set(call[1]["volumes"]) == {"/model-cache"}
    )
    assert cache_options == {
        "image": imported_adapter.cpu_image,
        "cpu": 4.0,
        "memory": 32_768,
        "timeout": 7_200,
        "retries": 0,
        "volumes": {"/model-cache": imported_adapter.model_volume},
    }
    smoke_options = next(
        call[1] for call in fake.declaration_calls
        if call[0] == "function" and set(call[1]["volumes"]) == {
            "/mnt/inputs", "/mnt/model", "/mnt/runs",
        } and "gpu" not in call[1]
    )
    assert smoke_options["image"] is imported_adapter.cpu_image
    assert smoke_options["cpu"] == 2.0
    assert smoke_options["memory"] == 8_192
    assert smoke_options["timeout"] == 900
    assert smoke_options["retries"] == 0
    assert smoke_options["volumes"]["/mnt/inputs"].read_only is True
    assert smoke_options["volumes"]["/mnt/model"].read_only is True
    assert smoke_options["volumes"]["/mnt/runs"] is imported_adapter.runs_volume

    assert set(imported_adapter.app.remote_functions) == {
        "cache_model_remote", "smoke_remote", "gpu_resources", "status_resources",
    }
    assert isinstance(imported_adapter.cache_model_remote, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.smoke_remote, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.gpu_resources, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.status_resources, imported_adapter.RemoteFunction)
    assert [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ] == ["status", "stage_inputs", "cache_model", "smoke"]
    assert "plan" not in [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ]


def test_apply_approved_app_tags_validates_before_the_client_rpc(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    lock_hash = modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    plan = _build_plan(pilot_repo, lock_hash)
    fake = imported_adapter.fake_modal

    with pytest.raises(ValueError, match="run ID"):
        imported_adapter.apply_approved_app_tags(replace(plan, run_id="noncanonical"))
    assert fake.rpc_calls == []

    imported_adapter.apply_approved_app_tags(plan)

    expected_tags = {
        "experiment": "phase-marker",
        "run-kind": "pilot",
        "seed": "42",
        "run-id": plan.run_id,
    }
    assert fake.rpc_calls == [("set_tags", expected_tags)]
    assert imported_adapter.app.tags == expected_tags


def test_status_is_local_read_only_and_never_mutates_app_tags(imported_adapter: ModuleType) -> None:
    fake = imported_adapter.fake_modal

    assert imported_adapter.status() == {
        "app": "phase-marker-pilot-stage-a",
        "gpu": "H100",
        "max_gpu_containers": 2,
        "volumes": list(imported_adapter.VOLUME_NAMES),
    }
    assert fake.rpc_calls == []


def test_pure_python_plan_cli_does_not_import_or_call_modal(
    pilot_repo: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeModal()
    fake.importing = False
    monkeypatch.setitem(sys.modules, "modal", fake)
    modal_imports: list[str] = []
    real_import = builtins.__import__

    def track_modal_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "modal" or name.startswith("modal."):
            modal_imports.append(name)
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", track_modal_import)
    argv = [
        "plan",
        "--repo-root", str(pilot_repo),
        "--config", str(CONFIG_PATH),
        "--artifact-root", "artifacts/phase-marker",
        "--dependency-lock", LOCK_PATH.name,
    ]

    modal_plan.main(argv)

    output = capsys.readouterr().out
    payload = json.loads(output)
    assert output == canonical_json(payload) + "\n"
    assert payload["run_id"].startswith("pilot-s42-cfg-")
    assert modal_imports == []
    assert fake.declaration_calls == []
    assert fake.rpc_calls == []

    adapter = _load_adapter(monkeypatch, fake)
    assert adapter.app.name == "phase-marker-pilot-stage-a"
    assert modal_imports == ["modal"]
    assert fake.declaration_calls
    assert fake.rpc_calls == []


def _bundle_volume_files(bundle: object, repo_root: Path) -> dict[str, bytes]:
    bundle_id = bundle.bundle_id
    files = {
        f"/bundles/{bundle_id}/{item.path}": (repo_root / item.path).read_bytes()
        for item in bundle.files
    }
    files[f"/bundles/{bundle_id}/bundle-manifest.json"] = (
        canonical_json(asdict(bundle)) + "\n"
    ).encode("utf-8")
    return files


def test_stage_inputs_uploads_only_allowlisted_bundle(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if staging omitted an input or uploaded an unapproved local file."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume = RecordingVolume()
    monkeypatch.chdir(pilot_repo)

    result = imported_adapter.stage_inputs_local(
        bundle,
        volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )

    assert [call.remote_path for call in volume.put_calls] == [
        f"/bundles/{bundle.bundle_id}/{item.path}" for item in bundle.files
    ] + [f"/bundles/{bundle.bundle_id}/bundle-manifest.json"]
    assert result == {"bundle_id": bundle.bundle_id, "uploaded": True}
    assert volume.files == _bundle_volume_files(bundle, pilot_repo)


def test_byte_identical_restaging_is_noop(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an identical bundle could be overwritten or uploaded twice."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume = RecordingVolume(_bundle_volume_files(bundle, pilot_repo))
    monkeypatch.chdir(pilot_repo)

    result = imported_adapter.stage_inputs_local(
        bundle,
        volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )

    assert result == {"bundle_id": bundle.bundle_id, "uploaded": False}
    assert volume.put_calls == []


def test_stage_inputs_reads_the_repository_bound_into_the_plan(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an explicit repository plan silently staged the process CWD."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    unrelated_cwd = tmp_path / "unrelated"
    unrelated_cwd.mkdir()
    monkeypatch.chdir(unrelated_cwd)
    volume = RecordingVolume()

    result = imported_adapter.stage_inputs_local(
        bundle,
        volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )

    assert result["uploaded"] is True
    assert volume.files == _bundle_volume_files(bundle, pilot_repo)


def test_stage_inputs_rejects_conflicting_or_out_of_scope_remote_bytes(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if staging trusted a conflicting or escaped remote listing."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    expected = _bundle_volume_files(bundle, pilot_repo)
    first_path = next(iter(expected))
    monkeypatch.chdir(pilot_repo)

    with pytest.raises(FileExistsError, match="conflicting remote bundle"):
        imported_adapter.stage_inputs_local(
            bundle,
            RecordingVolume({first_path: b"wrong"}),
            approved_run_id=plan.run_id,
            plan=plan,
            budget_acknowledged=True,
        )

    escaped = RecordingVolume()
    escaped.listdir = lambda path, recursive=False: [
        SimpleNamespace(path="/bundles/not-this-bundle/secret", type="file")
    ]
    with pytest.raises(ValueError, match="outside the bundle ID"):
        imported_adapter.stage_inputs_local(
            bundle,
            escaped,
            approved_run_id=plan.run_id,
            plan=plan,
            budget_acknowledged=True,
        )

    file_at_directory = RecordingVolume()
    file_at_directory.listdir = lambda path, recursive=False: [
        SimpleNamespace(path=f"/bundles/{bundle.bundle_id}/configs", type="file")
    ]
    with pytest.raises(FileExistsError, match="conflicting remote bundle path"):
        imported_adapter.stage_inputs_local(
            bundle,
            file_at_directory,
            approved_run_id=plan.run_id,
            plan=plan,
            budget_acknowledged=True,
        )
    assert file_at_directory.put_calls == []


def test_stage_inputs_requires_budget_and_matching_full_identities(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a partial approval or unrelated bundle could mutate the volume."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    monkeypatch.chdir(pilot_repo)

    for approved_run_id, approved_plan, acknowledged, message in (
        (plan.run_id[:-1], plan, True, "full approved run ID"),
        (plan.run_id, plan, False, "USD 1000"),
        (plan.run_id, replace(plan, bundle_id="0" * 64), True, "bundle identity"),
    ):
        volume = RecordingVolume()
        with pytest.raises(ValueError, match=message):
            imported_adapter.stage_inputs_local(
                bundle,
                volume,
                approved_run_id=approved_run_id,
                plan=approved_plan,
                budget_acknowledged=acknowledged,
            )
        assert volume.put_calls == []


def _write_smoke_model_cache(model_root: Path) -> tuple[Path, Path]:
    snapshot = (
        model_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
        / QWEN25_7B_TOKENIZER_REVISION
    )
    snapshot.mkdir(parents=True)
    metadata = (
        (
            "config.json",
            {
                "architectures": ["Qwen2ForCausalLM"],
                "hidden_size": 3584,
                "intermediate_size": 18944,
                "model_type": "qwen2",
                "num_attention_heads": 28,
                "num_hidden_layers": 28,
                "num_key_value_heads": 4,
                "vocab_size": 152064,
            },
        ),
        (
            "generation_config.json",
            {"bos_token_id": 151643, "eos_token_id": 151645, "pad_token_id": 151643},
        ),
        (
            "tokenizer.json",
            {
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
            },
        ),
        (
            "tokenizer_config.json",
            {
                "tokenizer_class": "Qwen2Tokenizer",
                "chat_template": "{{ message['content'] }}",
                "model_max_length": 131072,
            },
        ),
        (
            "model.safetensors.index.json",
            {
                "metadata": {"total_size": 4},
                "weight_map": {"model.weight": "model-00001-of-00001.safetensors"},
            },
        ),
    )
    for name, payload in metadata:
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"fake weights\n")
    manifest = build_model_cache_manifest(snapshot)
    manifest_path = snapshot.parent / f"{QWEN25_7B_TOKENIZER_REVISION}.manifest.json"
    manifest_path.write_text(canonical_json(asdict(manifest)) + "\n", encoding="utf-8")
    return snapshot, manifest_path


def _prepare_smoke_roots(
    pilot_repo: Path, tmp_path: Path,
) -> tuple[modal_plan.PilotPlan, Path, Path, Path]:
    source_package = pilot_repo / "phase_marker"
    source_package.mkdir()
    (source_package / "__init__.py").write_text("VALUE = 1\n", encoding="utf-8")
    (pilot_repo / "modal_phase_marker.py").write_text("APP = 'smoke'\n", encoding="utf-8")
    source_hash = hash_source_tree(pilot_repo)
    lock_hash = modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    bundle = build_input_bundle(pilot_repo)
    plan = modal_plan.build_pilot_plan(
        pilot_repo / CONFIG_PATH,
        pilot_repo / "artifacts/phase-marker",
        bundle=bundle,
        source_hash=source_hash,
        dependency_lock_hash=lock_hash,
    )

    input_root = tmp_path / "inputs"
    for remote_path, content in _bundle_volume_files(bundle, pilot_repo).items():
        destination = input_root / remote_path.lstrip("/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
    model_root = tmp_path / "model"
    _write_smoke_model_cache(model_root)
    run_root = tmp_path / "runs"
    return plan, input_root, model_root, run_root


def test_cpu_smoke_validates_imports_source_bundle_and_cache_without_loading_model(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if smoke skipped a preflight input or instantiated model weights."""
    plan, input_root, model_root, run_root = _prepare_smoke_roots(pilot_repo, tmp_path)
    imported: list[str] = []

    class ImportProbe:
        __version__ = "locked-test-version"

        def __getattr__(self, name: str) -> object:
            raise AssertionError(f"smoke accessed runtime API while importing: {name}")

    def import_module(name: str) -> ImportProbe:
        imported.append(name)
        return ImportProbe()

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "runs_volume", run_volume)

    result = imported_adapter.smoke_remote.local(modal_plan.pilot_plan_payload(plan))

    assert imported == list(imported_adapter.LOCKED_RUNTIME_IMPORTS)
    assert run_volume.commit_count == 1
    receipt_path = Path(str(result["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["validated"] is True
    assert receipt["run_id"] == plan.run_id
    assert receipt["source_hash"] == plan.source_hash
    assert receipt["bundle_id"] == plan.bundle_id
    assert receipt["model_revision"] == QWEN25_7B_TOKENIZER_REVISION
    assert receipt_path.stem == receipt["artifact_id"]
    assert receipt["artifact_id"] == modal_artifacts.sha256_json({
        key: value for key, value in receipt.items() if key != "artifact_id"
    })


def test_cpu_smoke_persists_content_addressed_failure_and_reraises(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an import failure disappeared or returned a successful receipt."""
    plan, input_root, model_root, run_root = _prepare_smoke_roots(pilot_repo, tmp_path)

    def fail_import(name: str) -> SimpleNamespace:
        if name == "vllm":
            raise ImportError("simulated locked import failure")
        return SimpleNamespace(__version__="locked-test-version")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "runs_volume", run_volume)

    with pytest.raises(ImportError, match="simulated locked import failure"):
        imported_adapter.smoke_remote.local(modal_plan.pilot_plan_payload(plan))

    receipts = list((run_root / f"runs/{plan.run_id}/receipts/smoke").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is False
    assert "ImportError" in receipt["failure_reason"]
    assert receipts[0].stem == receipt["artifact_id"]
    assert run_volume.commit_count == 1


def test_operator_entrypoints_print_exact_envelopes_tag_then_cross_one_boundary(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if an operator action crossed its boundary before attribution and disclosure."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    monkeypatch.setattr(imported_adapter, "_build_operator_context", lambda root: (bundle, plan))
    boundaries: list[tuple[str, object]] = []

    def stage_boundary(*args: object, **kwargs: object) -> dict[str, object]:
        assert imported_adapter.fake_modal.rpc_calls[-1][0] == "set_tags"
        boundaries.append(("stage-inputs", kwargs))
        return {"bundle_id": bundle.bundle_id, "uploaded": True}

    class RemoteBoundary:
        def __init__(self, name: str, result: dict[str, object]) -> None:
            self.name = name
            self.result = result

        def remote(self, payload: object) -> dict[str, object]:
            assert imported_adapter.fake_modal.rpc_calls[-1][0] == "set_tags"
            boundaries.append((self.name, payload))
            return self.result

    monkeypatch.setattr(imported_adapter, "stage_inputs_local", stage_boundary)
    monkeypatch.setattr(
        imported_adapter,
        "cache_model_remote",
        RemoteBoundary("cache-model", {"artifact_id": "a" * 64}),
    )
    monkeypatch.setattr(
        imported_adapter,
        "smoke_remote",
        RemoteBoundary(
            "smoke",
            {"artifact_id": "b" * 64, "receipt_path": f"/runs/{plan.run_id}/smoke.json"},
        ),
    )

    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
    )
    stage_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert stage_output[0] == {
        "operation": "stage-inputs",
        "run_id": plan.run_id,
        "bundle_id": bundle.bundle_id,
        "file_count": len(bundle.files) + 1,
        "destination": f"{imported_adapter.VOLUME_NAMES[0]}:/bundles/{bundle.bundle_id}",
        "budget_acknowledged_usd": 1_000.0,
    }

    imported_adapter.cache_model(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
    )
    cache_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert cache_output[0] == {
        "operation": "cache-model",
        "run_id": plan.run_id,
        "model_revision": QWEN25_7B_TOKENIZER_REVISION,
        "cpu": 4.0,
        "memory_mib": 32_768,
        "timeout_seconds": 7_200,
        "destination": f"{imported_adapter.VOLUME_NAMES[1]}:/model-cache/canonical",
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "budget_acknowledged_usd": 1_000.0,
    }

    imported_adapter.smoke(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
    )
    smoke_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert smoke_output[0] == {
        "operation": "smoke",
        "run_id": plan.run_id,
        "hardware": "CPU",
        "cpu": 2.0,
        "memory_mib": 8_192,
        "timeout_seconds": 900,
        "checks": ["locked-imports", "source-hash", "dependency-lock-hash", "input-bundle", "model-cache"],
        "budget_acknowledged_usd": 1_000.0,
    }
    assert smoke_output[1]["receipt_path"] == f"/runs/{plan.run_id}/smoke.json"
    assert [name for name, _ in boundaries] == ["stage-inputs", "cache-model", "smoke"]


@pytest.mark.parametrize("entrypoint", ("stage_inputs", "cache_model", "smoke"))
@pytest.mark.parametrize(
    ("approved_run_id", "budget_acknowledged", "message"),
    (("truncated", True, "full approved run ID"), ("valid", False, "USD 1000")),
)
def test_operator_entrypoints_reject_before_tags_or_remote_boundary(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    approved_run_id: str,
    budget_acknowledged: bool,
    message: str,
) -> None:
    """Would fail if an incomplete operator acknowledgement caused any remote action."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    monkeypatch.setattr(imported_adapter, "_build_operator_context", lambda root: (bundle, plan))
    monkeypatch.setattr(
        imported_adapter,
        "stage_inputs_local",
        lambda *args, **kwargs: pytest.fail("crossed staging boundary"),
    )
    forbidden_remote = SimpleNamespace(
        remote=lambda *args, **kwargs: pytest.fail("crossed compute boundary")
    )
    monkeypatch.setattr(imported_adapter, "cache_model_remote", forbidden_remote)
    monkeypatch.setattr(imported_adapter, "smoke_remote", forbidden_remote)
    actual_run_id = plan.run_id if approved_run_id == "valid" else approved_run_id

    with pytest.raises(ValueError, match=message):
        getattr(imported_adapter, entrypoint)(
            repo_root=str(pilot_repo),
            approved_run_id=actual_run_id,
            budget_acknowledged=budget_acknowledged,
        )

    assert imported_adapter.fake_modal.rpc_calls == []
