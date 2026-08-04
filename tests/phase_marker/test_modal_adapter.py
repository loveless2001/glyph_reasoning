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
import shlex
import shutil
import sys
from types import ModuleType, SimpleNamespace
from typing import Any, Callable
import uuid

import pytest

from phase_marker.config import ExperimentConfig
from phase_marker.io import canonical_json
from phase_marker.modal_artifacts import (
    AttemptReceipt,
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

    def map(self, payloads: object) -> object:
        self._modal.rpc_calls.append(("map", self.name, payloads))
        raise AssertionError("adapter attempted a remote map call")

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
        self.events: list[tuple[object, ...]] = []

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        assert recursive is True
        self.events.append(("listdir", path))
        prefix = path.rstrip("/") + "/"
        return [
            SimpleNamespace(path=remote_path, type="file")
            for remote_path in sorted(self.files)
            if remote_path == path or remote_path.startswith(prefix)
        ]

    def read_file(self, path: str) -> list[bytes]:
        self.events.append(("read_file", path))
        if path not in self.files:
            raise FileNotFoundError(path)
        return [self.files[path]]

    @contextmanager
    def batch_upload(self) -> object:
        self.events.append(("batch_upload",))
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
        if call[0] == "function"
        and call[1].get("gpu") == "H100"
        and "/mnt/inputs" in call[1]["volumes"]
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
        "cache_model_remote",
        "smoke_remote",
        "run_training_job",
        "run_selection_job",
        "finalize_stage_a_remote",
        "gpu_resources",
        "status_resources",
    }
    assert isinstance(imported_adapter.cache_model_remote, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.smoke_remote, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.gpu_resources, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.status_resources, imported_adapter.RemoteFunction)
    assert [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ] == ["status", "stage_inputs", "cache_model", "smoke", "run_stage_a"]
    assert "plan" not in [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ]


def test_stage_a_job_resources_and_mount_permissions_are_exact(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if either Stage A job could exceed or bypass the approved H100 envelope."""
    declarations = {
        name: options
        for (kind, options), (_, name) in zip(
            (
                call for call in imported_adapter.fake_modal.declaration_calls
                if call[0] == "function"
            ),
            (
                call for call in imported_adapter.fake_modal.declaration_calls
                if call[0] == "function_decorated"
            ),
            strict=True,
        )
    }

    for name in ("run_training_job", "run_selection_job"):
        options = declarations[name]
        assert {
            key: options[key]
            for key in (
                "gpu", "timeout", "startup_timeout", "max_containers",
                "retries", "ephemeral_disk",
            )
        } == {
            "gpu": "H100",
            "timeout": 14_400,
            "startup_timeout": 1_200,
            "max_containers": 2,
            "retries": 0,
            "ephemeral_disk": 80 * 1024,
        }
        assert set(options["volumes"]) == {"/inputs", "/model-cache", "/runs"}
        assert options["volumes"]["/inputs"].volume is imported_adapter.inputs_volume
        assert options["volumes"]["/inputs"].read_only is True
        assert options["volumes"]["/model-cache"].volume is imported_adapter.model_volume
        assert options["volumes"]["/model-cache"].read_only is True
        assert options["volumes"]["/runs"] is imported_adapter.runs_volume

    finalizer = declarations["finalize_stage_a_remote"]
    assert "gpu" not in finalizer
    assert finalizer["cpu"] == 2.0
    assert finalizer["memory"] == 8_192
    assert finalizer["timeout"] == 900
    assert finalizer["retries"] == 0
    assert finalizer["volumes"]["/inputs"].read_only is True
    assert finalizer["volumes"]["/model-cache"].read_only is True
    assert finalizer["volumes"]["/runs"] is imported_adapter.runs_volume


def test_gpu_job_wrappers_forward_one_approved_payload_to_the_exact_stage(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a remote wrapper changed the payload or crossed the wrong stage."""
    plan = _build_plan(
        pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    )
    payload = {
        "plan": modal_plan.pilot_plan_payload(plan),
        "job": asdict(plan.jobs[0]),
    }
    calls: list[dict[str, object]] = []

    def execute(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"stage": kwargs["stage"]}

    monkeypatch.setattr(imported_adapter, "execute_pilot_job", execute)

    assert imported_adapter.run_training_job.local(payload) == {"stage": "train"}
    assert imported_adapter.run_selection_job.local(payload) == {"stage": "selection"}
    assert [call["stage"] for call in calls] == ["train", "selection"]
    for call in calls:
        assert call["plan_payload"] is payload["plan"]
        assert call["job_payload"] is payload["job"]
        assert call["code_root"] == imported_adapter.CODE_ROOT
        assert call["input_root"] == Path("/inputs")
        assert call["model_root"] == Path("/model-cache")
        assert call["run_root"] == Path("/runs")
        assert call["volume"] is imported_adapter.runs_volume


def test_cpu_finalizer_wrapper_forwards_receipts_without_loading_weights(
    imported_adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if finalization crossed a GPU/model or behavior execution boundary."""
    plan_payload = {"run_id": "pilot"}
    receipts = ({"artifact_id": "a" * 64},)
    calls: list[dict[str, object]] = []

    def finalize(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"stopped_before_behavior": True}

    monkeypatch.setattr(imported_adapter, "finalize_stage_a", finalize, raising=False)

    result = imported_adapter.finalize_stage_a_remote.local(plan_payload, receipts)

    assert result == {"stopped_before_behavior": True}
    assert calls == [
        {
            "plan_payload": plan_payload,
            "receipts": receipts,
            "input_root": Path("/inputs"),
            "model_root": Path("/model-cache"),
            "run_root": Path("/runs"),
            "volume": imported_adapter.runs_volume,
        }
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


@pytest.mark.parametrize(
    ("conflict", "message"),
    (
        ("byte", "conflicting remote bundle byte"),
        ("path", "outside the bundle ID"),
        ("type", "conflicting remote bundle path"),
        ("incomplete", "conflicting remote bundle is incomplete"),
    ),
)
def test_stage_entrypoint_conflicts_fail_before_tags_or_writes(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    conflict: str,
    message: str,
) -> None:
    """Would fail if a known remote conflict were discovered only after app tagging."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    expected = _bundle_volume_files(bundle, pilot_repo)
    if conflict == "byte":
        files = dict(expected)
        files[next(iter(files))] = b"wrong bytes"
        volume = RecordingVolume(files)
    elif conflict == "incomplete":
        first_path = next(iter(expected))
        volume = RecordingVolume({first_path: expected[first_path]})
    else:
        volume = RecordingVolume()
        listed_path = (
            "/bundles/not-this-bundle/secret"
            if conflict == "path"
            else f"/bundles/{bundle.bundle_id}/configs"
        )
        volume.listdir = lambda path, recursive=False: [
            SimpleNamespace(path=listed_path, type="file")
        ]
    monkeypatch.setattr(imported_adapter, "inputs_volume", volume)
    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda root: (bundle, plan)
    )

    with pytest.raises((FileExistsError, ValueError), match=message):
        imported_adapter.stage_inputs(
            repo_root=str(pilot_repo),
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
        )

    assert imported_adapter.fake_modal.rpc_calls == []
    assert volume.put_calls == []
    assert capsys.readouterr().out == ""


def test_stage_entrypoint_identical_noop_needs_no_tag_or_upload(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if proving an identical remote bundle still mutated app or volume state."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume = RecordingVolume(_bundle_volume_files(bundle, pilot_repo))
    monkeypatch.setattr(imported_adapter, "inputs_volume", volume)
    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda root: (bundle, plan)
    )

    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
    )

    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["action"] == "no-op"
    assert output[1] == {"bundle_id": bundle.bundle_id, "uploaded": False}
    assert imported_adapter.fake_modal.rpc_calls == []
    assert volume.put_calls == []
    assert all(event[0] != "batch_upload" for event in volume.events)


def test_stage_entrypoint_tags_only_after_read_only_preflight_then_narrow_apply(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if tagging or upload preceded the complete read-only staging preflight."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    upload_items = tuple(_bundle_volume_files(bundle, pilot_repo).items())
    volume = RecordingVolume()

    def tag(approved_plan: object) -> None:
        assert approved_plan is plan
        volume.events.append(("tags",))

    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda root: (bundle, plan)
    )
    monkeypatch.setattr(imported_adapter, "inputs_volume", volume)
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tag)

    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
    )

    assert [event[0] for event in volume.events] == ["listdir", "tags", "batch_upload"]
    output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert output[0]["action"] == "upload"
    assert output[0]["remote_files"] == [
        {
            "path": path,
            "size": len(content),
            "sha256": hashlib.sha256(content).hexdigest(),
        }
        for path, content in upload_items
    ]
    assert output[1] == {"bundle_id": bundle.bundle_id, "uploaded": True}


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
    upload_items = tuple(_bundle_volume_files(bundle, pilot_repo).items())
    staging_plan = imported_adapter.InputStagingPlan(
        bundle_id=bundle.bundle_id,
        bundle_root=f"/bundles/{bundle.bundle_id}",
        upload_items=upload_items,
        upload_required=True,
    )

    def stage_boundary(preflight_plan: object, volume: object) -> dict[str, object]:
        assert imported_adapter.fake_modal.rpc_calls[-1][0] == "set_tags"
        assert preflight_plan is staging_plan
        assert volume is imported_adapter.inputs_volume
        boundaries.append(("stage-inputs", preflight_plan))
        return {"bundle_id": bundle.bundle_id, "uploaded": True}

    class RemoteBoundary:
        def __init__(self, name: str, result: dict[str, object]) -> None:
            self.name = name
            self.result = result

        def remote(self, payload: object) -> dict[str, object]:
            assert imported_adapter.fake_modal.rpc_calls[-1][0] == "set_tags"
            boundaries.append((self.name, payload))
            return self.result

    monkeypatch.setattr(
        imported_adapter,
        "preflight_inputs_local",
        lambda *args, **kwargs: staging_plan,
    )
    monkeypatch.setattr(imported_adapter, "_apply_input_staging_plan", stage_boundary)
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
        "action": "upload",
        "run_id": plan.run_id,
        "bundle_id": bundle.bundle_id,
        "file_count": len(bundle.files) + 1,
        "destination": f"{imported_adapter.VOLUME_NAMES[0]}:/bundles/{bundle.bundle_id}",
        "remote_files": [
            {
                "path": path,
                "size": len(content),
                "sha256": hashlib.sha256(content).hexdigest(),
            }
            for path, content in upload_items
        ],
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


def test_run_stage_a_entrypoint_forwards_explicit_resume_and_prints_summary(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if the CLI silently enabled resume or bypassed the local orchestrator."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    calls: list[dict[str, object]] = []
    summary = {"stopped_before_behavior": True, "artifact_id": "a" * 64}

    def run(local_plan: object, **kwargs: object) -> dict[str, object]:
        assert local_plan is plan
        calls.append(dict(kwargs))
        return summary

    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda root: (bundle, plan)
    )
    monkeypatch.setattr(imported_adapter, "run_stage_a_local", run)

    imported_adapter.run_stage_a(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=True,
    )

    assert calls == [
        {
            "approved_run_id": plan.run_id,
            "budget_acknowledged": True,
            "resume": True,
            "training_function": imported_adapter.run_training_job,
            "selection_function": imported_adapter.run_selection_job,
            "finalizer_function": imported_adapter.finalize_stage_a_remote,
            "runs_client": imported_adapter.runs_volume,
        }
    ]
    assert capsys.readouterr().out == canonical_json(summary) + "\n"


def _stage_a_receipt(
    plan: modal_plan.PilotPlan, job: modal_plan.PilotJob, stage: str,
) -> dict[str, object]:
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
        stage=stage,
        arm=job.arm,
        seed=job.seed,
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{stage}:{job.arm}")),
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=plan.source_hash,
        dependency_lock_hash=plan.dependency_lock_hash,
        model_cache_artifact_id="e" * 64,
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
        output_hashes=tuple(hashlib.sha256(path.encode("utf-8")).hexdigest() for path in paths),
        failure_reason=None,
        artifact_id="",
    )
    receipt = replace(receipt, artifact_id=receipt.recomputed_artifact_id())
    payload = asdict(receipt)
    payload["expected_outputs"] = list(receipt.expected_outputs)
    payload["output_hashes"] = list(receipt.output_hashes)
    return payload


class StageAMapFunction:
    def __init__(
        self,
        stage: str,
        results: dict[str, object],
        events: list[tuple[object, ...]],
        *,
        before_first: Callable[[], None] | None = None,
        after_result: Callable[[str], None] | None = None,
    ) -> None:
        self.stage = stage
        self.results = results
        self.events = events
        self.calls: list[dict[str, object]] = []
        self.before_first = before_first
        self.after_result = after_result

    def map(self, payloads: object) -> object:
        items = list(payloads)

        def results() -> object:
            for index, payload in enumerate(items):
                if index == 0 and self.before_first is not None:
                    self.before_first()
                assert isinstance(payload, dict)
                job = payload["job"]
                assert isinstance(job, dict)
                arm = str(job["arm"])
                self.calls.append(payload)
                self.events.append((self.stage, arm))
                result = self.results[arm]
                if isinstance(result, Exception):
                    raise RuntimeError(f"{arm}: {result}") from result
                if self.after_result is not None:
                    self.after_result(arm)
                yield result

        return results()


class StageAFinalizer:
    def __init__(self, summary: dict[str, object], events: list[tuple[object, ...]]) -> None:
        self.summary = summary
        self.events = events
        self.calls: list[tuple[object, object]] = []

    def remote(self, plan_payload: object, receipts: object) -> dict[str, object]:
        self.calls.append((plan_payload, receipts))
        self.events.append(("finalizer",))
        return self.summary


class EmptyStageARunsClient:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events
        self.reload_count = 0

    def read_file(self, path: str) -> list[bytes]:
        self.events.append(("read_file", path))
        raise FileNotFoundError(path)

    def listdir(self, path: str, *, recursive: bool = False) -> list[object]:
        assert recursive is True
        self.events.append(("listdir", path))
        raise FileNotFoundError(path)

    def reload(self) -> None:
        self.reload_count += 1
        self.events.append(("reload", self.reload_count))


class StageARunsClient(RecordingVolume):
    def __init__(
        self, files: dict[str, bytes], events: list[tuple[object, ...]],
    ) -> None:
        super().__init__(files)
        self.events = events
        self.reload_count = 0

    def reload(self) -> None:
        self.reload_count += 1
        self.events.append(("reload", self.reload_count))


def _canonical_stage_a_files(
    plan: modal_plan.PilotPlan,
    job: modal_plan.PilotJob,
    stage: str,
) -> tuple[dict[str, bytes], dict[str, object]]:
    kind = "checkpoints" if stage == "train" else "checkpoint-selections"
    producer = (
        f"/runs/{plan.run_id}/artifacts/phase-marker/{kind}/pilot/seed-42/{job.arm}"
    )
    if stage == "train":
        data_path = f"artifacts/phase-marker/training-data/{job.arm}.jsonl"
        data = (Path(plan.local_repo_root) / data_path).read_bytes()
        materialization_id = dict(
            zip(
                (candidate.arm for candidate in plan.jobs),
                plan.materialization_artifact_ids,
                strict=True,
            )
        )[job.arm]
        producer_files = {
            "adapter_config.json": (
                canonical_json(
                    {
                        "base_model_name_or_path": "Qwen/Qwen2.5-7B-Instruct",
                        "revision": plan.model_revision,
                    }
                )
                + "\n"
            ).encode("utf-8"),
            "adapter_model.safetensors": f"adapter:{job.arm}".encode("utf-8"),
            "tokenizer_config.json": b"{}\n",
            "trainer_state.json": b"{}\n",
        }
        output_hash = modal_artifacts.sha256_json(
            [
                {
                    "path": path,
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for path, content in sorted(producer_files.items())
            ]
        )
        manifest = {
            "kind": "phase_marker_training_run",
            "arm": job.arm,
            "seed": 42,
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": plan.model_revision,
            "tokenizer_revision": plan.model_revision,
            "config_hash": plan.config_hash,
            "dataset_path": data_path,
            "dataset_hash": modal_artifacts.sha256_json(data.hex()),
            "data_artifact_id": materialization_id,
            "parent_hashes": [materialization_id],
            "data_parent_hashes": [plan.split_artifact_id],
            "arguments": shlex.split(job.training_command)[3:],
            "environment": {},
            "checkpoints": [],
            "saved_artifacts": ["adapter", "tokenizer", "trainer_state"],
            "output_hash": output_hash,
        }
        producer_files["run-manifest.json"] = (
            canonical_json(manifest) + "\n"
        ).encode("utf-8")
    else:
        manifest = canonical_json(
            {
                "schema_version": 1,
                "kind": "phase_marker_checkpoint_selection",
                "run_kind": "pilot",
                "arm": job.arm,
                "seed": 42,
                "config_hash": plan.config_hash,
                "model_revision": plan.model_revision,
                "completed": True,
            }
        ).encode("utf-8") + b"\n"
        producer_files = {
            "manifest.json": manifest,
            "evidence.jsonl": b"{}\n",
        }
    command = job.training_command if stage == "train" else job.selection_command
    paths = tuple(sorted(producer_files))
    receipt = AttemptReceipt(
        schema_version=1,
        run_id=plan.run_id,
        bundle_id=plan.bundle_id,
        stage=stage,
        arm=job.arm,
        seed=42,
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"canonical:{stage}:{job.arm}")),
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=plan.source_hash,
        dependency_lock_hash=plan.dependency_lock_hash,
        model_cache_artifact_id="e" * 64,
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
        output_hashes=tuple(
            hashlib.sha256(producer_files[path]).hexdigest() for path in paths
        ),
        failure_reason=None,
        artifact_id="",
    )
    receipt = replace(receipt, artifact_id=receipt.recomputed_artifact_id())
    receipt_payload = asdict(receipt)
    receipt_payload["expected_outputs"] = list(receipt.expected_outputs)
    receipt_payload["output_hashes"] = list(receipt.output_hashes)
    files = {
        f"{producer}/{path}": content for path, content in producer_files.items()
    }
    files[
        f"/runs/{plan.run_id}/receipts/canonical/{stage}/{job.arm}.json"
    ] = (canonical_json(receipt_payload) + "\n").encode("utf-8")
    return files, receipt_payload


def _canonical_stage_a_publications(
    plan: modal_plan.PilotPlan, stage: str,
) -> tuple[dict[str, dict[str, bytes]], dict[str, dict[str, object]]]:
    files_by_arm: dict[str, dict[str, bytes]] = {}
    receipts: dict[str, dict[str, object]] = {}
    for job in plan.jobs:
        files, receipt = _canonical_stage_a_files(plan, job, stage)
        files_by_arm[job.arm] = files
        receipts[job.arm] = receipt
    return files_by_arm, receipts


def _publishing_stage_a_function(
    plan: modal_plan.PilotPlan,
    stage: str,
    events: list[tuple[object, ...]],
    runs: StageARunsClient,
    *,
    before_first: Callable[[], None] | None = None,
) -> tuple[StageAMapFunction, dict[str, dict[str, object]]]:
    publications, receipts = _canonical_stage_a_publications(plan, stage)
    function = StageAMapFunction(
        stage,
        receipts,
        events,
        before_first=before_first,
        after_result=lambda arm: runs.files.update(publications[arm]),
    )
    return function, receipts


def _damage_stage_a_publication(
    publications: dict[str, dict[str, bytes]],
    receipts: dict[str, dict[str, object]],
    *,
    stage: str,
    arm: str,
    fault: str,
) -> None:
    receipt_suffix = f"/receipts/canonical/{stage}/{arm}.json"
    receipt_path = next(
        path for path in publications[arm] if path.endswith(receipt_suffix)
    )
    if fault == "missing":
        publications[arm].pop(receipt_path)
        return
    if fault == "corrupt":
        manifest_name = "run-manifest.json" if stage == "train" else "manifest.json"
        manifest_path = next(
            path for path in publications[arm] if path.endswith(f"/{manifest_name}")
        )
        publications[arm][manifest_path] = b'{"corrupt":true}\n'
        return
    if fault == "mismatched":
        persisted = dict(receipts[arm])
        persisted["attempt_id"] = f"post-reload-{stage}-{arm}"
        unsigned = dict(persisted)
        unsigned.pop("artifact_id")
        persisted["artifact_id"] = modal_artifacts.sha256_json(unsigned)
        publications[arm][receipt_path] = (
            canonical_json(persisted) + "\n"
        ).encode("utf-8")
        return
    raise AssertionError(f"unsupported Stage A publication fault: {fault}")


def _stage_a_summary(
    plan: modal_plan.PilotPlan,
    training: dict[str, dict[str, object]],
    selection: dict[str, dict[str, object]],
) -> dict[str, object]:
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "stage-a",
        "run_id": plan.run_id,
        "training_receipt_ids": [training[job.arm]["artifact_id"] for job in plan.jobs],
        "selection_receipt_ids": [selection[job.arm]["artifact_id"] for job in plan.jobs],
        "behavior_gate_checked_artifact_ids": [],
        "next_command": "./.venv/bin/python -m phase_marker.behavior run",
        "stopped_before_behavior": True,
    }
    payload["artifact_id"] = modal_artifacts.sha256_json(payload)
    return payload


def test_stage_a_validates_all_training_before_selection_and_stops(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if selection overlapped training validation or behavior was invoked."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, events)
    training, training_results = _publishing_stage_a_function(
        plan, "train", events, runs
    )
    selection, selection_results = _publishing_stage_a_function(
        plan,
        "selection",
        events,
        runs,
        before_first=lambda: (
            events.count(("validated", "train")) == 6
            or pytest.fail("selection began before six training receipts were validated")
        ),
    )
    finalizer = StageAFinalizer(_stage_a_summary(plan, training_results, selection_results), events)
    real_validate = modal_artifacts.validate_job_receipt_payload

    def validate(*args: object, **kwargs: object) -> dict[str, object]:
        result = real_validate(*args, **kwargs)
        events.append(("validated", kwargs["stage"]))
        return result

    monkeypatch.setattr(imported_adapter, "validate_job_receipt_payload", validate)
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )
    real_validate_canonical = imported_adapter._validate_volume_canonical_output

    def validate_canonical(**kwargs: object) -> dict[str, object]:
        result = real_validate_canonical(**kwargs)
        job = kwargs["job"]
        events.append(("canonical-validated", kwargs["stage"], job.arm))
        return result

    monkeypatch.setattr(
        imported_adapter, "_validate_volume_canonical_output", validate_canonical
    )
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda approved: events.append(("tags", approved.run_id)),
    )

    summary = imported_adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=False,
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        runs_client=runs,
    )

    assert len(training.calls) == 6
    assert len(selection.calls) == 6
    assert len(finalizer.calls) == 1
    assert runs.reload_count == 2
    assert summary["stopped_before_behavior"] is True
    assert summary == finalizer.summary
    first_gpu = min(index for index, event in enumerate(events) if event[0] in {"train", "selection"})
    assert events.index(("tags", plan.run_id)) < first_gpu
    assert sum(event[0] in {"train", "selection"} for event in events) == 12
    assert all(event[0] != "behavior" for event in events)
    completed_validations = [
        event for event in events if event[0] == "canonical-validated"
    ]
    assert completed_validations == [
        *(("canonical-validated", "train", job.arm) for job in plan.jobs),
        *(("canonical-validated", "selection", job.arm) for job in plan.jobs),
    ]


@pytest.mark.parametrize("completed_stage", ("train", "selection"))
@pytest.mark.parametrize("fault", ("missing", "corrupt", "mismatched"))
def test_stage_a_post_reload_canonical_failure_aborts_before_next_stage(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    completed_stage: str,
    fault: str,
) -> None:
    """Would fail if a remote return could substitute for canonical volume evidence."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, events)
    training_files, training_results = _canonical_stage_a_publications(plan, "train")
    selection_files, selection_results = _canonical_stage_a_publications(plan, "selection")
    target = plan.jobs[-1].arm
    if completed_stage == "train":
        _damage_stage_a_publication(
            training_files, training_results,
            stage="train", arm=target, fault=fault,
        )
    else:
        _damage_stage_a_publication(
            selection_files, selection_results,
            stage="selection", arm=target, fault=fault,
        )
    training = StageAMapFunction(
        "train", training_results, events,
        after_result=lambda arm: runs.files.update(training_files[arm]),
    )
    selection = StageAMapFunction(
        "selection", selection_results, events,
        after_result=lambda arm: runs.files.update(selection_files[arm]),
    )
    finalizer = StageAFinalizer(
        _stage_a_summary(plan, training_results, selection_results), events
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", lambda _: None)

    with pytest.raises(ValueError, match="canonical|receipt|artifact|output"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
        )

    if completed_stage == "train":
        assert selection.calls == []
    else:
        assert len(selection.calls) == 6
    assert finalizer.calls == []


def test_training_failure_prevents_every_selection(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if any selection could launch after an incomplete training matrix."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    training_results: dict[str, object] = {
        job.arm: _stage_a_receipt(plan, job, "train") for job in plan.jobs
    }
    training_results["dot"] = RuntimeError("boom")
    selection_results = {
        job.arm: _stage_a_receipt(plan, job, "selection") for job in plan.jobs
    }
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", training_results, events)
    selection = StageAMapFunction("selection", selection_results, events)
    finalizer = StageAFinalizer(
        _stage_a_summary(
            plan,
            {job.arm: _stage_a_receipt(plan, job, "train") for job in plan.jobs},
            selection_results,
        ),
        events,
    )

    monkeypatch.setattr(
        imported_adapter, "apply_approved_app_tags", lambda approved: None
    )
    with pytest.raises(RuntimeError, match="dot"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=EmptyStageARunsClient(events),
        )
    assert selection.calls == []
    assert finalizer.calls == []


def test_selection_failure_prevents_cpu_finalization(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an incomplete selection matrix could publish a stop summary."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, events)
    training, _ = _publishing_stage_a_function(plan, "train", events, runs)
    selection_files, complete_selection = _canonical_stage_a_publications(
        plan, "selection"
    )
    selection_results: dict[str, object] = dict(complete_selection)
    selection_results["dot"] = RuntimeError("selection boom")
    selection = StageAMapFunction(
        "selection",
        selection_results,
        events,
        after_result=lambda arm: runs.files.update(selection_files[arm]),
    )
    finalizer = StageAFinalizer({}, events)
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", lambda _: None)
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )

    with pytest.raises(RuntimeError, match="dot.*selection boom"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
        )
    assert len(training.calls) == 6
    assert finalizer.calls == []


def test_stage_a_mismatched_run_id_aborts_before_preflight_tags_or_remote_calls(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a partial run identity could inspect or mutate remote state."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tags.append)

    with pytest.raises(ValueError, match="full approved run ID"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id[:-1],
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=EmptyStageARunsClient(events),
        )
    assert events == []
    assert tags == []
    assert training.calls == [] and selection.calls == [] and finalizer.calls == []


def test_explicit_resume_revalidates_existing_and_schedules_only_missing_arms(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if resume reused quarantine or overwrote already-canonical arms."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files: dict[str, bytes] = {}
    for job in plan.jobs[:2]:
        arm_files, _ = _canonical_stage_a_files(plan, job, "train")
        files.update(arm_files)
    failed_attempt = f"/runs/{plan.run_id}/attempts/failed-dot/receipt.json"
    files[failed_attempt] = b'{"validated":false,"promoted":false}\n'
    original_files = dict(files)
    events: list[tuple[object, ...]] = []
    runs = StageARunsClient(files, events)
    training_files, training_results = _canonical_stage_a_publications(plan, "train")
    selection_files, selection_results = _canonical_stage_a_publications(plan, "selection")
    training = StageAMapFunction(
        "train",
        training_results,
        events,
        after_result=lambda arm: runs.files.update(training_files[arm]),
    )
    selection = StageAMapFunction(
        "selection",
        selection_results,
        events,
        after_result=lambda arm: runs.files.update(selection_files[arm]),
    )
    finalizer = StageAFinalizer(
        _stage_a_summary(plan, training_results, selection_results), events
    )
    validated_existing: list[tuple[str, str]] = []
    real_validate = modal_artifacts.validate_canonical_job_output

    def validate_existing(*args: object, **kwargs: object) -> dict[str, object]:
        result = real_validate(*args, **kwargs)
        job = kwargs["job_payload"]
        validated_existing.append((str(kwargs["stage"]), str(job["arm"])))
        return result

    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_output", validate_existing
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda approved: events.append(("tags", approved.run_id)),
    )

    summary = imported_adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=True,
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        runs_client=runs,
    )

    assert [payload["job"]["arm"] for payload in training.calls] == [
        "dot", "random", "direct", "filler"
    ]
    assert [payload["job"]["arm"] for payload in selection.calls] == [
        "semantic", "glyph", "dot", "random", "direct", "filler"
    ]
    assert validated_existing == [
        ("train", "semantic"),
        ("train", "glyph"),
        *(("train", job.arm) for job in plan.jobs),
        *(("selection", job.arm) for job in plan.jobs),
    ]
    printed = json.loads(capsys.readouterr().out)
    assert printed["resume"] is True
    assert printed["missing_training_arms"] == ["dot", "random", "direct", "filler"]
    assert summary["stopped_before_behavior"] is True
    assert all(runs.files[path] == content for path, content in original_files.items())
    assert runs.files[failed_attempt] == original_files[failed_attempt]


@pytest.mark.parametrize("corruption", ("receipt", "manifest"))
def test_resume_corrupt_canonical_output_aborts_before_tags_or_remote_calls(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    """Would fail if resume scheduled around stale canonical evidence."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files, _ = _canonical_stage_a_files(plan, plan.jobs[0], "train")
    target = next(
        path for path in files
        if path.endswith("semantic.json") if corruption == "receipt"
    ) if corruption == "receipt" else next(
        path for path in files if path.endswith("run-manifest.json")
    )
    files[target] = b'{"corrupt":true}\n'
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tags.append)

    with pytest.raises(ValueError, match="receipt|manifest|canonical"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=True,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=StageARunsClient(files, events),
        )
    assert tags == []
    assert training.calls == []
    assert selection.calls == []
    assert finalizer.calls == []


def test_resume_semantically_invalid_self_consistent_training_aborts_before_remote_calls(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if matching receipt hashes replaced producer semantic validation."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files, receipt = _canonical_stage_a_files(plan, plan.jobs[0], "train")
    manifest_path = next(path for path in files if path.endswith("run-manifest.json"))
    receipt_path = next(path for path in files if path.endswith("semantic.json"))
    manifest = json.loads(files[manifest_path])
    manifest["saved_artifacts"] = []
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    files[manifest_path] = manifest_bytes
    output_index = receipt["expected_outputs"].index("run-manifest.json")
    receipt["output_hashes"][output_index] = hashlib.sha256(manifest_bytes).hexdigest()
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    files[receipt_path] = (canonical_json(receipt) + "\n").encode("utf-8")
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tags.append)

    with pytest.raises(ValueError, match="training|completion|semantic"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=True,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=StageARunsClient(files, events),
        )
    assert tags == []
    assert training.calls == [] and selection.calls == [] and finalizer.calls == []


def test_initial_stage_a_refuses_existing_output_before_tags_or_remote_calls(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if normal mode implicitly reused or overwrote a canonical arm."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files, _ = _canonical_stage_a_files(plan, plan.jobs[0], "train")
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tags.append)

    with pytest.raises(FileExistsError, match="use --resume"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=StageARunsClient(files, events),
        )
    assert tags == []
    assert training.calls == [] and selection.calls == [] and finalizer.calls == []


@pytest.mark.parametrize(
    "existing_path",
    (
        "stage-a-summary.json",
        "artifacts/phase-marker/checkpoints/pilot/seed-42/unapproved/run-manifest.json",
    ),
)
def test_initial_stage_a_refuses_summary_or_unexpected_canonical_namespace(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    existing_path: str,
) -> None:
    """Would fail if stale terminal or orphan state could launch a fresh GPU graph."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files = {f"/runs/{plan.run_id}/{existing_path}": b"{}\n"}
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tags.append)

    with pytest.raises((FileExistsError, ValueError), match="canonical|summary"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=StageARunsClient(files, events),
        )
    assert tags == []
    assert training.calls == [] and selection.calls == [] and finalizer.calls == []
