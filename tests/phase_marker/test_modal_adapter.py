from __future__ import annotations

import builtins
import base64
from contextlib import contextmanager
from dataclasses import asdict, replace
from datetime import datetime, timedelta
import hashlib
import io
import importlib
import importlib.util
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import shutil
import subprocess
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
EXPECTED_LOCKED_RUNTIME_IMPORTS = (
    "accelerate",
    "datasets",
    "einops",
    "huggingface_hub",
    "modal",
    "numpy",
    "peft",
    "google.protobuf",
    "safetensors",
    "sentencepiece",
    "statsmodels",
    "tokenizers",
    "torch",
    "transformers",
    "vllm",
)


def _adapter_execution_provenance(stage: str) -> dict[str, object]:
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
        "modal_function_call_id": f"fc-{stage}",
        "modal_input_id": f"in-{stage}",
        "python_version": "3.12.test",
        "torch_version": "2.7.test",
        "cuda_runtime_version": "12.8.test",
        "cuda_driver_version": (
            "not-observed-cpu" if stage in {"smoke", "finalizer"} else "570.test"
        ),
        "runtime_versions": [
            {"module": module, "version": "locked-test-version"}
            for module in EXPECTED_LOCKED_RUNTIME_IMPORTS
        ],
    }


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
        self,
        local_path: str,
        remote_path: str,
        *,
        copy: bool,
        ignore: object = None,
    ) -> FakeImage:
        self.operations.append(
            (
                "add_local_dir",
                local_path,
                remote_path,
                {"copy": copy, "ignore": ignore},
            )
        )
        return self

    def add_local_file(
        self, local_path: str, remote_path: str, *, copy: bool
    ) -> FakeImage:
        self.operations.append(("add_local_file", local_path, remote_path, {"copy": copy}))
        return self

    def run_commands(self, *commands: str) -> FakeImage:
        self.operations.append(("run_commands", *commands))
        return self

    def workdir(self, path: str) -> FakeImage:
        self.operations.append(("workdir", path))
        return self


class FakeVolumeMount:
    def __init__(self, volume: FakeVolume) -> None:
        self.volume = volume
        self.read_only = True


class FakeModalNotFoundError(Exception):
    pass


class FakeVolume:
    def __init__(
        self, modal: FakeModal, name: str, create_if_missing: bool,
        environment_name: str | None,
    ) -> None:
        self._modal = modal
        self.name = name
        self.create_if_missing = create_if_missing
        self.environment_name = environment_name
        self.read_only_calls = 0
        self.read_only_handle = FakeVolumeMount(self)

    def read_only(self) -> FakeVolumeMount:
        self.read_only_calls += 1
        return self.read_only_handle

    def hydrate(self) -> None:
        self._modal.rpc_calls.append(("volume_hydrate", self.name))

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

    def reload(self) -> None:
        self.events.append(("reload",))

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


class LocalBatchVolume(RecordingVolume):
    """Outside-App Volume reads must not use the mounted-filesystem reload API."""

    def reload(self) -> None:
        raise AssertionError("local Volume batch clients cannot reload mounts")


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
        self.function_options: dict[str, dict[str, object]] = {}
        self.local_entrypoints: list[str] = []

    def function(self, **options: object) -> Callable[[Callable[..., object]], FakeRemoteFunction]:
        self._modal.declaration_calls.append(("function", dict(options)))

        def decorate(function: Callable[..., object]) -> FakeRemoteFunction:
            remote = FakeRemoteFunction(self._modal, function)
            self.remote_functions[function.__name__] = remote
            self.function_options[function.__name__] = dict(options)
            self._modal.declaration_calls.append(("function_decorated", function.__name__))
            return remote

        return decorate

    def local_entrypoint(
        self, **options: object
    ) -> Callable[[Callable[..., object]], Callable[..., object]]:
        self._modal.declaration_calls.append(("local_entrypoint", dict(options)))

        def decorate(function: Callable[..., object]) -> Callable[..., object]:
            self.local_entrypoints.append(function.__name__)
            self._modal.declaration_calls.append(("local_entrypoint_decorated", function.__name__))
            return function

        return decorate

    def initialize(self, existing_volumes: set[str]) -> None:
        """Model Modal's selected-app resource hydration boundary."""
        for options in self.function_options.values():
            volumes = options.get("volumes", {})
            assert isinstance(volumes, dict)
            for mount in volumes.values():
                volume = mount.volume if isinstance(mount, FakeVolumeMount) else mount
                assert isinstance(volume, FakeVolume)
                if volume.name in existing_volumes:
                    continue
                if not volume.create_if_missing:
                    raise FileNotFoundError(volume.name)
                existing_volumes.add(volume.name)

    def set_tags(self, tags: dict[str, str]) -> None:
        if self._modal.importing:
            raise AssertionError("adapter attempted a client RPC during import")
        if any(
            len(value) > 63 or re.fullmatch(r"[A-Za-z0-9._-]+", value) is None
            for value in tags.values()
        ):
            raise ValueError("invalid Modal tag value")
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
        self.exception = SimpleNamespace(NotFoundError=FakeModalNotFoundError)

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

    def _from_name(
        self, name: str, *, create_if_missing: bool,
        environment_name: str | None = None,
    ) -> FakeVolume:
        volume = FakeVolume(self, name, create_if_missing, environment_name)
        self.volumes.append(volume)
        self.declaration_calls.append(
            (
                "Volume.from_name", name,
                {
                    "create_if_missing": create_if_missing,
                    "environment_name": environment_name,
                },
            )
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


def _load_inspection_adapter(
    monkeypatch: pytest.MonkeyPatch, fake_modal: FakeModal,
) -> ModuleType:
    path = REPO_ROOT / "modal_phase_marker_inspect.py"
    spec = importlib.util.spec_from_file_location(
        "modal_phase_marker_inspect_under_test", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.delitem(sys.modules, "modal_phase_marker", raising=False)
    monkeypatch.setitem(sys.modules, "modal", fake_modal)
    monkeypatch.setitem(sys.modules, spec.name, module)
    try:
        spec.loader.exec_module(module)
    finally:
        fake_modal.importing = False
    module.fake_modal = fake_modal
    return module


@pytest.fixture
def imported_adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    return _load_adapter(monkeypatch, FakeModal())


@pytest.fixture
def imported_inspection_adapter(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    path = REPO_ROOT / "modal_phase_marker_inspect.py"
    assert path.is_file(), "standalone inspection adapter is missing"
    return _load_inspection_adapter(monkeypatch, FakeModal())


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


def _operator_approval_kwargs(
    plan: modal_plan.PilotPlan,
    *,
    action: str,
    resume: bool | None = None,
    smoke_receipt_artifact_id: str | None = None,
    model_cache_artifact_id: str | None = None,
) -> dict[str, str]:
    approval = modal_plan.action_approval_payload(
        plan,
        action=action,
        resume=resume,
        smoke_receipt_artifact_id=smoke_receipt_artifact_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    return {
        "approved_plan_digest": plan.plan_digest,
        "approved_action_digest": str(approval["approval_digest"]),
    }


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
    assert changed.run_label == original.run_label
    assert changed.plan_digest != original.plan_digest
    assert changed.run_id != original.run_id


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
    assert imported_adapter.INPUT_MOUNT_ROOT == Path("/mnt/inputs")
    assert imported_adapter.MODEL_MOUNT_ROOT == Path("/mnt/model")
    assert imported_adapter.RUN_MOUNT_ROOT == Path("/mnt/runs")
    assert imported_adapter.JOB_INPUT_MOUNT_ROOT == Path("/inputs")
    assert imported_adapter.JOB_MODEL_MOUNT_ROOT == Path("/model-cache")
    assert imported_adapter.JOB_RUN_MOUNT_ROOT == Path("/runs")
    assert "glyph-reasoning-vol" not in imported_adapter.source_text
    assert "/vol/work" not in imported_adapter.source_text

    fake = imported_adapter.fake_modal
    assert fake.rpc_calls == []
    assert len(fake.apps) == 4
    assert all(candidate.include_source is False for candidate in fake.apps)
    assert all(candidate.tags == {
        "experiment": "phase-marker",
        "run-kind": "pilot",
        "seed": "42",
    } for candidate in fake.apps)
    assert [(volume.name, volume.create_if_missing) for volume in fake.volumes] == [
        *((name, False) for name in imported_adapter.VOLUME_NAMES),
        (imported_adapter.VOLUME_NAMES[1], True),
        (imported_adapter.VOLUME_NAMES[2], True),
    ]
    assert {volume.environment_name for volume in fake.volumes} == {"main"}
    assert not hasattr(imported_adapter, "inspection_runs_volume")

    assert len(fake.images) == 1
    assert imported_adapter.cpu_image is imported_adapter.gpu_image
    assert imported_adapter.gpu_image.operations == [
        ("from_registry", imported_adapter.BASE_IMAGE, {"add_python": "3.12"}),
        ("pip_install_from_requirements", "requirements-modal-phase-marker.txt"),
        (
            "add_local_dir",
            "phase_marker",
            "/opt/glyph_reasoning/phase_marker",
            {
                "copy": True,
                "ignore": imported_adapter._ignore_unhashed_phase_source,
            },
        ),
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
        ("workdir", "/opt/glyph_reasoning"),
    ]
    assert imported_adapter._ignore_unhashed_phase_source(Path("planner.py")) is False
    assert imported_adapter._ignore_unhashed_phase_source(Path("notes.txt")) is True
    assert (
        imported_adapter._ignore_unhashed_phase_source(
            Path("__pycache__/planner.cpython-312.pyc")
        )
        is True
    )
    assert ":12.8.1-cudnn-devel-ubuntu22.04" not in imported_adapter.BASE_IMAGE

    gpu_options = next(
        call[1] for call in fake.declaration_calls
        if call[0] == "function"
        and call[1].get("gpu") == "H100"
        and "/inputs" in call[1]["volumes"]
    )
    assert gpu_options["image"] is imported_adapter.gpu_image
    assert gpu_options["timeout"] == 14_400
    assert gpu_options["max_containers"] == 2
    assert gpu_options["retries"] == 0
    assert gpu_options["volumes"]["/inputs"].volume is imported_adapter.inputs_volume
    assert gpu_options["volumes"]["/model-cache"].volume is imported_adapter.model_volume
    assert gpu_options["volumes"]["/runs"] is imported_adapter.runs_volume
    assert gpu_options["volumes"]["/inputs"].read_only is True
    assert gpu_options["volumes"]["/model-cache"].read_only is True

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
        "volumes": {"/model-cache": imported_adapter.cache_model_volume},
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
    assert smoke_options["volumes"]["/mnt/runs"] is imported_adapter.smoke_runs_volume

    assert set(imported_adapter.app.remote_functions) == {
        "run_training_job",
        "run_selection_job",
        "finalize_stage_a_remote",
        "recover_stage_a_orphans_remote",
    }
    assert isinstance(imported_adapter.cache_model_remote, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.smoke_remote, imported_adapter.RemoteFunction)
    assert [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ] == [
        "stage_inputs", "cache_model", "smoke", "run_stage_a",
    ]
    assert "plan" not in [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ]


def test_bootstrap_apps_initialize_in_order_from_an_empty_volume_namespace(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if an action app hydrates a volume owned by a later action."""
    existing: set[str] = set()

    imported_adapter.stage_inputs_app.initialize(existing)
    assert existing == set()
    assert imported_adapter.stage_inputs_app.local_entrypoints == ["stage_inputs"]
    assert imported_adapter.stage_inputs_app.remote_functions == {}

    # The stage-input entrypoint creates this volume only after validating the
    # operator-bound action digest. Model that completed boundary before cache.
    existing.add(imported_adapter.VOLUME_NAMES[0])
    imported_adapter.cache_model_app.initialize(existing)
    assert existing == set(imported_adapter.VOLUME_NAMES[:2])
    assert imported_adapter.cache_model_app.local_entrypoints == ["cache_model"]
    assert set(imported_adapter.cache_model_app.remote_functions) == {
        "cache_model_remote"
    }

    imported_adapter.smoke_app.initialize(existing)
    assert existing == set(imported_adapter.VOLUME_NAMES)
    assert imported_adapter.smoke_app.local_entrypoints == ["smoke"]
    assert set(imported_adapter.smoke_app.remote_functions) == {"smoke_remote"}

    imported_adapter.app.initialize(existing)
    assert existing == set(imported_adapter.VOLUME_NAMES)
    assert imported_adapter.app.local_entrypoints == ["run_stage_a"]
    assert set(imported_adapter.app.remote_functions) == {
        "run_training_job",
        "run_selection_job",
        "finalize_stage_a_remote",
        "recover_stage_a_orphans_remote",
    }


def test_inspection_adapter_declares_no_compute_capability(
    imported_inspection_adapter: ModuleType,
) -> None:
    """Would fail if status import could declare an image, function, GPU, or new volume."""
    adapter = imported_inspection_adapter
    fake = adapter.fake_modal

    assert "modal_phase_marker" not in sys.modules
    assert len(fake.apps) == 1
    assert adapter.app.include_source is False
    assert fake.images == []
    assert fake.rpc_calls == []
    assert not any(call[0] == "Image.from_registry" for call in fake.declaration_calls)
    assert not any(call[0] == "function" for call in fake.declaration_calls)
    assert fake.apps[0].remote_functions == {}

    assert [(volume.name, volume.create_if_missing) for volume in fake.volumes] == [
        ("phase-marker-pilot-runs-v1", False),
    ]
    declared = fake.volumes[0]
    assert declared.environment_name == "main"
    assert declared.read_only_calls == 1
    assert adapter.runs_volume is declared.read_only_handle
    assert adapter.runs_volume.read_only is True
    assert [
        call[1] for call in fake.declaration_calls
        if call[0] == "local_entrypoint_decorated"
    ] == ["status", "download_evidence"]


def test_inspection_entrypoints_delegate_only_to_the_read_only_handle(
    imported_inspection_adapter: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if either inspection command regained a writable or compute handle."""
    adapter = imported_inspection_adapter
    run_id = (
        "pilot-s42-cfg-11111111-split-22222222-src-333333333333-plan-"
        + "4" * 64
    )
    destination = tmp_path / "evidence"
    seen: list[tuple[str, object]] = []

    def inspect_status(volume: object, *, run_id: str) -> dict[str, object]:
        seen.append(("status", volume))
        return {"run_id": run_id, "valid": True}

    def inspect_download(
        volume: object, *, run_id: str, destination: Path,
    ) -> tuple[Path, ...]:
        seen.append(("download", volume))
        return (destination / "stage-a-summary.json",)

    monkeypatch.setattr(adapter, "status_local", inspect_status)
    monkeypatch.setattr(adapter, "download_evidence_local", inspect_download)

    adapter.status(run_id)
    adapter.download_evidence(run_id, str(destination))

    assert seen == [
        ("status", adapter.runs_volume),
        ("download", adapter.runs_volume),
    ]
    assert adapter.fake_modal.rpc_calls == []
    assert capsys.readouterr().out.splitlines() == [
        canonical_json({"run_id": run_id, "valid": True}),
        canonical_json({
            "run_id": run_id,
            "destination": str(destination),
            "files": [str(destination / "stage-a-summary.json")],
        }),
    ]


def _declared_compute_image_python_paths(image: FakeImage) -> tuple[str, ...]:
    selected: set[str] = set()
    for operation in image.operations:
        if operation[0] == "add_local_dir":
            local_path = Path(str(operation[1]))
            options = operation[3]
            assert isinstance(options, dict)
            ignore = options["ignore"]
            local_root = REPO_ROOT / local_path
            for candidate in local_root.rglob("*"):
                if not candidate.is_file():
                    continue
                relative = candidate.relative_to(local_root)
                if ignore is None or ignore(relative) is False:
                    selected.add(candidate.relative_to(REPO_ROOT).as_posix())
        elif operation[0] == "add_local_file":
            local_path = Path(str(operation[1]))
            if local_path.suffix == ".py":
                selected.add(local_path.as_posix())
    return tuple(sorted(selected))


def _compute_image_can_import_top_level_module(
    image: FakeImage, module_name: str,
) -> bool:
    runtime_workdir = PurePosixPath("/")
    remote_python_files: set[PurePosixPath] = set()
    for operation in image.operations:
        if operation[0] == "add_local_file":
            remote_path = PurePosixPath(str(operation[2]))
            if remote_path.suffix == ".py":
                remote_python_files.add(remote_path)
        elif operation[0] == "workdir":
            runtime_workdir = PurePosixPath(str(operation[1]))
    return runtime_workdir / f"{module_name}.py" in remote_python_files


def test_compute_image_can_import_declared_remote_function_module(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if Modal cannot import the module owning remote functions."""
    assert _compute_image_can_import_top_level_module(
        imported_adapter.gpu_image, "modal_phase_marker"
    )


def test_compute_image_python_set_exactly_matches_source_hash_set(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if Modal could execute a Python byte absent from the source hash."""
    source_paths = modal_artifacts.source_tree_relative_paths(REPO_ROOT)
    assert _declared_compute_image_python_paths(imported_adapter.gpu_image) == source_paths

    records = [
        {
            "path": relative,
            "sha256": hashlib.sha256((REPO_ROOT / relative).read_bytes()).hexdigest(),
        }
        for relative in source_paths
    ]
    assert hash_source_tree(REPO_ROOT) == modal_artifacts.sha256_json(records)


def test_only_action_scoped_bootstrap_handles_can_create_volumes(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if Stage A or a read path retained volume-creation authority."""
    creating = {
        volume for volume in imported_adapter.fake_modal.volumes
        if volume.create_if_missing
    }
    assert creating == {
        imported_adapter.cache_model_volume,
        imported_adapter.smoke_runs_volume,
    }
    for options in imported_adapter.app.function_options.values():
        for mount in options.get("volumes", {}).values():
            volume = mount.volume if isinstance(mount, FakeVolumeMount) else mount
            assert volume not in creating
    assert not hasattr(imported_adapter, "gpu_resources")


@pytest.mark.parametrize(
    ("remote_name", "action"),
    [
        ("cache_model_remote", "cache-model"),
        ("smoke_remote", "smoke"),
    ],
)
def test_direct_cpu_remote_calls_require_the_exact_action_approval_envelope(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    remote_name: str,
    action: str,
) -> None:
    """Would fail if a direct function invocation bypassed local authorization."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    remote = getattr(imported_adapter, remote_name)

    with pytest.raises(ValueError, match="remote action payload"):
        remote.local(modal_plan.pilot_plan_payload(plan))
    approval = modal_plan.action_approval_payload(plan, action=action)
    tampered = dict(approval)
    tampered["approval_digest"] = "0" * 64
    with pytest.raises(ValueError, match="approval digest"):
        remote.local({"plan": modal_plan.pilot_plan_payload(plan), "approval": tampered})


def test_direct_gpu_remote_call_requires_stage_a_evidence_bound_approval(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if H100 execution accepted an unapproved or cross-mode payload."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = asdict(plan.jobs[0])
    job["expected_outputs"] = list(job["expected_outputs"])

    with pytest.raises(ValueError, match="remote job payload"):
        imported_adapter.run_training_job.local(
            {"plan": modal_plan.pilot_plan_payload(plan), "job": job}
        )
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id="3" * 64,
        model_cache_artifact_id="4" * 64,
    )
    tampered = dict(approval)
    tampered["resume"] = True
    with pytest.raises(ValueError, match="approval"):
        imported_adapter.run_training_job.local(
            {
                "plan": modal_plan.pilot_plan_payload(plan),
                "job": job,
                "approval": tampered,
            }
        )


def test_run_stage_a_entrypoint_has_no_create_tag_or_remote_on_failed_preflight(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    bundle = build_input_bundle(pilot_repo)
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id="3" * 64,
        model_cache_artifact_id="4" * 64,
    )
    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda _root: (bundle, plan)
    )
    monkeypatch.setattr(
        imported_adapter,
        "run_stage_a_local",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("Stage A smoke receipt is missing")
        ),
    )
    initial_volume_count = len(imported_adapter.fake_modal.volumes)

    with pytest.raises(ValueError, match="smoke receipt is missing"):
        imported_adapter.run_stage_a(
            repo_root=str(pilot_repo),
            approved_run_id=plan.run_id,
            acknowledge_budget_usd=1_000,
            approved_plan_digest=plan.plan_digest,
            approved_action_digest=str(approval["approval_digest"]),
            smoke_receipt_artifact_id="3" * 64,
            model_cache_artifact_id="4" * 64,
        )
    assert len(imported_adapter.fake_modal.volumes) == initial_volume_count
    assert imported_adapter.fake_modal.rpc_calls == []


@pytest.mark.parametrize("entrypoint", ("stage_inputs", "cache_model", "smoke"))
def test_operator_action_digest_is_required_before_any_external_boundary(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
) -> None:
    """Would fail if the human run label alone authorized an external action."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda _root: (bundle, plan)
    )
    initial_volume_count = len(imported_adapter.fake_modal.volumes)

    with pytest.raises(ValueError, match="plan digest|action approval"):
        getattr(imported_adapter, entrypoint)(
            repo_root=str(pilot_repo),
            approved_run_id=plan.run_id,
            acknowledge_budget_usd=1_000,
            approved_plan_digest=plan.plan_digest,
            approved_action_digest="0" * 64,
        )

    assert len(imported_adapter.fake_modal.volumes) == initial_volume_count
    assert imported_adapter.fake_modal.rpc_calls == []


def test_authorized_volume_creation_revalidates_the_exact_action_approval(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if a create-if-missing handle could be obtained cross-action."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    approval = modal_plan.action_approval_payload(plan, action="cache-model")
    initial_volume_count = len(imported_adapter.fake_modal.volumes)

    with pytest.raises(ValueError, match="approval"):
        imported_adapter._create_authorized_volume(
            imported_adapter.VOLUME_NAMES[1],
            plan_payload=modal_plan.pilot_plan_payload(plan),
            approval_payload=modal_plan.action_approval_payload(
                plan, action="smoke"
            ),
            action="cache-model",
        )
    assert len(imported_adapter.fake_modal.volumes) == initial_volume_count

    created = imported_adapter._create_authorized_volume(
        imported_adapter.VOLUME_NAMES[1],
        plan_payload=modal_plan.pilot_plan_payload(plan),
        approval_payload=approval,
        action="cache-model",
    )
    assert created.name == imported_adapter.VOLUME_NAMES[1]
    assert created.create_if_missing is True


def test_stage_a_job_resources_and_mount_permissions_are_exact(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if Stage A requested an invalid disk quota or bypassed its envelope."""
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
                "retries",
            )
        } == {
            "gpu": "H100",
            "timeout": 14_400,
            "startup_timeout": 1_200,
            "max_containers": 2,
            "retries": 0,
        }
        assert "ephemeral_disk" not in options
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
        "approval": modal_plan.action_approval_payload(
            plan,
            action="run-stage-a",
            resume=False,
            smoke_receipt_artifact_id="3" * 64,
            model_cache_artifact_id="4" * 64,
        ),
    }
    calls: list[dict[str, object]] = []

    def execute(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"stage": kwargs["stage"]}

    monkeypatch.setattr(imported_adapter, "execute_pilot_job", execute)
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(
        imported_adapter,
        "validate_stage_a_remote_dependencies",
        lambda **_: None,
    )
    monkeypatch.setattr(
        imported_adapter,
        "load_validated_canonical_stage_a_receipts",
        lambda **_: (),
    )
    provenance = _adapter_execution_provenance("train")
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda function_name: {
            **provenance,
            "modal_function_name": function_name,
        },
    )

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
        assert call["execution_provenance"]["modal_function_name"] == (
            "run_training_job" if call["stage"] == "train" else "run_selection_job"
        )


def test_cpu_finalizer_wrapper_forwards_receipts_without_loading_weights(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if finalization crossed a GPU/model or behavior execution boundary."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    plan_payload = modal_plan.pilot_plan_payload(plan)
    receipts = ({"artifact_id": "a" * 64},)
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id="3" * 64,
        model_cache_artifact_id="4" * 64,
    )
    calls: list[dict[str, object]] = []

    def finalize(**kwargs: object) -> dict[str, object]:
        calls.append(dict(kwargs))
        return {"stopped_before_behavior": True}

    monkeypatch.setattr(imported_adapter, "finalize_stage_a", finalize, raising=False)
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(
        imported_adapter,
        "load_validated_canonical_stage_a_receipts",
        lambda **_: tuple(receipts),
    )
    provenance = _adapter_execution_provenance("finalizer")
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: provenance,
    )

    result = imported_adapter.finalize_stage_a_remote.local({
        "plan": plan_payload,
        "approval": approval,
        "receipts": receipts,
    })

    assert result == {"stopped_before_behavior": True}
    assert calls == [
        {
            "plan_payload": plan_payload,
            "receipts": receipts,
            "input_root": Path("/inputs"),
            "model_root": Path("/model-cache"),
            "run_root": Path("/runs"),
                "volume": imported_adapter.runs_volume,
                "execution_provenance": provenance,
                "stage_a_approval": approval,
            }
    ]


def test_apply_approved_app_tags_validates_before_the_client_rpc(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    lock_hash = modal_plan._file_sha256(pilot_repo / LOCK_PATH.name)
    plan = _build_plan(pilot_repo, lock_hash)
    fake = imported_adapter.fake_modal
    approval = modal_plan.action_approval_payload(plan, action="stage-inputs")

    with pytest.raises(ValueError, match="run ID"):
        imported_adapter.apply_approved_app_tags(
            replace(plan, run_id="noncanonical"),
            approval_payload=approval,
            action="stage-inputs",
        )
    assert fake.rpc_calls == []

    imported_adapter.apply_approved_app_tags(
        plan, approval_payload=approval, action="stage-inputs"
    )

    encoded_plan = base64.urlsafe_b64encode(
        bytes.fromhex(plan.plan_digest)
    ).decode("ascii").rstrip("=")
    expected_tags = {
        "experiment": "phase-marker",
        "run-kind": "pilot",
        "seed": "42",
        "run-id": f"s42-{encoded_plan}",
    }
    assert len(expected_tags["run-id"]) <= 63
    assert base64.urlsafe_b64decode(encoded_plan + "=").hex() == plan.plan_digest
    assert fake.rpc_calls == [("set_tags", expected_tags)]
    assert imported_adapter.stage_inputs_app.tags == expected_tags


def test_operator_context_uses_config_independent_porcelain_status(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if local Git color/config could corrupt the clean-tree gate."""
    calls: list[tuple[list[str], dict[str, object]]] = []

    def run(args: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append((args, kwargs))
        return SimpleNamespace(stdout="?? model_cards/\n?? paper/\n")

    monkeypatch.setattr(imported_adapter.subprocess, "run", run)
    imported_adapter._build_operator_context(pilot_repo)

    assert calls == [
        (
            ["git", "status", "--porcelain=v1", "--untracked-files=normal"],
            {
                "cwd": pilot_repo.resolve(),
                "check": True,
                "capture_output": True,
                "text": True,
            },
        )
    ]


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


def test_plan_cli_cold_interpreter_blocks_any_modal_import(pilot_repo: Path) -> None:
    """Would fail if an already-cached module hid a cold Modal dependency."""
    blocker = """
import importlib.abc
import json
import sys

class BlockModal(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "modal" or fullname.startswith("modal."):
            raise RuntimeError(f"forbidden Modal import: {fullname}")
        return None

sys.meta_path.insert(0, BlockModal())
from phase_marker.modal_plan import main
main(sys.argv[1:])
"""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            blocker,
            "plan",
            "--repo-root",
            str(pilot_repo),
            "--config",
            str(CONFIG_PATH),
            "--artifact-root",
            "artifacts/phase-marker",
            "--dependency-lock",
            LOCK_PATH.name,
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"},
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(result.stdout)
    assert result.stdout == canonical_json(payload) + "\n"
    assert payload["run_id"].endswith(f"-plan-{payload['plan_digest']}")
    assert result.stderr == ""


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


def test_stage_inputs_treats_modal_missing_bundle_root_as_empty(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if Modal's empty-directory NotFound aborted first upload."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))

    class EmptyModalVolume(RecordingVolume):
        def listdir(
            self, path: str, *, recursive: bool = False,
        ) -> list[SimpleNamespace]:
            assert recursive is True
            self.events.append(("listdir", path))
            raise FakeModalNotFoundError("No such file or directory")

    volume = EmptyModalVolume()
    result = imported_adapter.stage_inputs_local(
        bundle,
        volume,
        approved_run_id=plan.run_id,
        plan=plan,
        budget_acknowledged=True,
    )

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

    for approved_run_id, approved_plan, approved_bundle, acknowledged, message in (
        (plan.run_id[:-1], plan, bundle, True, "full approved run ID"),
        (plan.run_id, plan, bundle, False, "USD 1000"),
        (
            plan.run_id,
            plan,
            replace(bundle, bundle_id="0" * 64),
            True,
            "bundle identity",
        ),
    ):
        volume = RecordingVolume()
        with pytest.raises(ValueError, match=message):
            imported_adapter.stage_inputs_local(
                approved_bundle,
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
            acknowledge_budget_usd=1_000,
            **_operator_approval_kwargs(plan, action="stage-inputs"),
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
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="stage-inputs"),
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

    def tag(approved_plan: object, **_kwargs: object) -> None:
        assert approved_plan is plan
        volume.events.append(("tags",))

    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda root: (bundle, plan)
    )
    monkeypatch.setattr(imported_adapter, "inputs_volume", volume)
    monkeypatch.setattr(imported_adapter, "apply_approved_app_tags", tag)
    monkeypatch.setattr(
        imported_adapter,
        "_create_authorized_volume",
        lambda *args, **kwargs: volume,
    )

    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="stage-inputs"),
    )

    assert [event[0] for event in volume.events] == [
        "listdir", "listdir", "tags", "batch_upload",
    ]
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


def test_stage_entrypoint_repreflights_authorized_volume_after_missing_read(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Would fail if first-run staging reads a no-create handle before bootstrap."""
    bundle = build_input_bundle(pilot_repo)
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    writable = RecordingVolume()
    events: list[str] = []

    class MissingVolume:
        def listdir(self, *_args: object, **_kwargs: object) -> object:
            raise FakeModalNotFoundError("input volume is absent")

    def create(name: str, **kwargs: object) -> RecordingVolume:
        assert name == imported_adapter.VOLUME_NAMES[0]
        assert kwargs["action"] == "stage-inputs"
        events.append("create")
        return writable

    monkeypatch.setattr(
        imported_adapter, "_build_operator_context", lambda _root: (bundle, plan)
    )
    monkeypatch.setattr(imported_adapter, "inputs_volume", MissingVolume())
    monkeypatch.setattr(imported_adapter, "_create_authorized_volume", create)
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda *_args, **_kwargs: events.append("tags"),
    )

    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="stage-inputs"),
    )

    assert events == ["create", "tags"]
    assert [event[0] for event in writable.events] == ["listdir", "batch_upload"]
    assert json.loads(capsys.readouterr().out.splitlines()[-1]) == {
        "bundle_id": bundle.bundle_id,
        "uploaded": True,
    }


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
            {
                "bos_token_id": 151643,
                "eos_token_id": [151645, 151643],
                "pad_token_id": 151643,
            },
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

    def reject_hard_link(*args: object, **kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(importlib, "import_module", import_module)
    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "smoke_runs_volume", run_volume)
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: _adapter_execution_provenance("smoke"),
    )

    result = imported_adapter.smoke_remote.local({
        "plan": modal_plan.pilot_plan_payload(plan),
        "approval": modal_plan.action_approval_payload(plan, action="smoke"),
    })

    assert imported == list(imported_adapter.LOCKED_RUNTIME_IMPORTS)
    assert run_volume.commit_count == 1
    receipt_path = Path(str(result["receipt_path"]))
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["validated"] is True
    assert receipt["run_id"] == plan.run_id
    assert receipt["source_hash"] == plan.source_hash
    assert receipt["bundle_id"] == plan.bundle_id
    assert receipt["model_revision"] == QWEN25_7B_TOKENIZER_REVISION
    assert receipt["plan_digest"] == plan.plan_digest
    assert receipt["modal_function_name"] == "smoke_remote"
    assert receipt["modal_input_id"] == "in-smoke"
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

    def reject_hard_link(*args: object, **kwargs: object) -> None:
        raise PermissionError(1, "Operation not permitted")

    monkeypatch.setattr(importlib, "import_module", fail_import)
    monkeypatch.setattr(os, "link", reject_hard_link)
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "smoke_runs_volume", run_volume)
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: _adapter_execution_provenance("smoke"),
    )

    with pytest.raises(ImportError, match="simulated locked import failure"):
        imported_adapter.smoke_remote.local({
            "plan": modal_plan.pilot_plan_payload(plan),
            "approval": modal_plan.action_approval_payload(plan, action="smoke"),
        })

    receipts = list((run_root / f"runs/{plan.run_id}/receipts/smoke").glob("*.json"))
    assert len(receipts) == 1
    receipt = json.loads(receipts[0].read_text(encoding="utf-8"))
    assert receipt["validated"] is False
    assert "ImportError" in receipt["failure_reason"]
    assert receipts[0].stem == receipt["artifact_id"]
    assert run_volume.commit_count == 1


def test_cpu_smoke_refuses_commit_if_partial_evidence_cleanup_fails(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if smoke could commit after partial evidence cleanup failed."""
    plan, input_root, model_root, run_root = _prepare_smoke_roots(pilot_repo, tmp_path)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(__version__="locked-test-version"),
    )
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "smoke_runs_volume", run_volume)
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: _adapter_execution_provenance("smoke"),
    )

    real_write = os.write
    write_calls = 0

    def fail_second_write(descriptor: int, content: bytes) -> int:
        nonlocal write_calls
        write_calls += 1
        if write_calls == 1:
            return real_write(descriptor, content[:3])
        if write_calls == 2:
            raise OSError("injected evidence write failure")
        return real_write(descriptor, content)

    real_unlink = os.unlink
    cleanup_attempted = False

    def fail_provenance_cleanup(
        path: object, *args: object, **kwargs: object,
    ) -> None:
        nonlocal cleanup_attempted
        if (
            path == "input-bundle-manifest.json"
            and kwargs.get("dir_fd") is not None
        ):
            cleanup_attempted = True
            raise OSError("injected evidence cleanup failure")
        real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(os, "write", fail_second_write)
    monkeypatch.setattr(os, "unlink", fail_provenance_cleanup)

    with pytest.raises(
        modal_artifacts._EvidencePublicationCleanupError,
        match="evidence cleanup failed",
    ) as captured:
        imported_adapter.smoke_remote.local({
            "plan": modal_plan.pilot_plan_payload(plan),
            "approval": modal_plan.action_approval_payload(plan, action="smoke"),
        })

    assert isinstance(captured.value.__cause__, OSError)
    assert "injected evidence write failure" in str(captured.value.__cause__)
    notes = getattr(captured.value, "__notes__", [])
    assert any("original failure: OSError" in note for note in notes)
    assert any("unlink failed: OSError" in note for note in notes)
    assert cleanup_attempted is True
    assert run_volume.commit_count == 0
    assert not list((run_root / f"runs/{plan.run_id}/receipts/smoke").glob("*.json"))


def test_cpu_smoke_refuses_commit_if_evidence_identity_is_unavailable(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if an unidentified evidence path could reach volume.commit()."""
    plan, input_root, model_root, run_root = _prepare_smoke_roots(pilot_repo, tmp_path)

    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda _name: SimpleNamespace(__version__="locked-test-version"),
    )
    monkeypatch.setattr(imported_adapter, "CODE_ROOT", pilot_repo)
    monkeypatch.setattr(imported_adapter, "INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "RUN_MOUNT_ROOT", run_root)
    run_volume = CommitOnlyVolume()
    monkeypatch.setattr(imported_adapter, "smoke_runs_volume", run_volume)
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: _adapter_execution_provenance("smoke"),
    )

    provenance_path = (
        run_root / f"runs/{plan.run_id}/provenance/input-bundle-manifest.json"
    )
    real_fstat = os.fstat
    identity_calls = 0

    def reject_provenance_identity(descriptor: int) -> os.stat_result:
        nonlocal identity_calls
        if os.readlink(f"/proc/self/fd/{descriptor}") == str(provenance_path):
            identity_calls += 1
            raise OSError("injected persistent fstat failure")
        return real_fstat(descriptor)

    monkeypatch.setattr(os, "fstat", reject_provenance_identity)

    with pytest.raises(
        modal_artifacts._EvidencePublicationCleanupError,
        match="evidence cleanup failed",
    ) as captured:
        imported_adapter.smoke_remote.local({
            "plan": modal_plan.pilot_plan_payload(plan),
            "approval": modal_plan.action_approval_payload(plan, action="smoke"),
        })

    assert identity_calls == 2
    assert isinstance(captured.value.__cause__, OSError)
    assert "persistent fstat failure" in str(captured.value.__cause__)
    assert run_volume.commit_count == 0
    assert not list((run_root / f"runs/{plan.run_id}/receipts/smoke").glob("*.json"))


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
        "_create_authorized_volume",
        lambda name, **kwargs: {
            imported_adapter.VOLUME_NAMES[0]: imported_adapter.inputs_volume,
            imported_adapter.VOLUME_NAMES[1]: imported_adapter.model_volume,
            imported_adapter.VOLUME_NAMES[2]: imported_adapter.runs_volume,
        }[name],
    )
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

    stage_approval = modal_plan.action_approval_payload(
        plan, action="stage-inputs"
    )
    imported_adapter.stage_inputs(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="stage-inputs"),
    )
    stage_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert stage_output[0] == {
        "operation": "stage-inputs",
        "action": "upload",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": stage_approval["approval_digest"],
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

    cache_approval = modal_plan.action_approval_payload(
        plan, action="cache-model"
    )
    imported_adapter.cache_model(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="cache-model"),
    )
    cache_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert cache_output[0] == {
        "operation": "cache-model",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": cache_approval["approval_digest"],
        "model_revision": QWEN25_7B_TOKENIZER_REVISION,
        "cpu": 4.0,
        "memory_mib": 32_768,
        "timeout_seconds": 7_200,
        "destination": f"{imported_adapter.VOLUME_NAMES[1]}:/model-cache/canonical",
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "budget_acknowledged_usd": 1_000.0,
    }

    smoke_approval = modal_plan.action_approval_payload(plan, action="smoke")
    imported_adapter.smoke(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        **_operator_approval_kwargs(plan, action="smoke"),
    )
    smoke_output = [json.loads(line) for line in capsys.readouterr().out.splitlines()]
    assert smoke_output[0] == {
        "operation": "smoke",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "approval_digest": smoke_approval["approval_digest"],
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
    ("approved_run_id", "acknowledge_budget_usd", "message"),
    (("truncated", 1_000, "full approved run ID"), ("valid", 999, "USD 1000")),
)
def test_operator_entrypoints_reject_before_tags_or_remote_boundary(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    approved_run_id: str,
    acknowledge_budget_usd: float,
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
            acknowledge_budget_usd=acknowledge_budget_usd,
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
    smoke_id = "3" * 64
    cache_id = "4" * 64
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=True,
        smoke_receipt_artifact_id=smoke_id,
        model_cache_artifact_id=cache_id,
    )

    imported_adapter.run_stage_a(
        repo_root=str(pilot_repo),
        approved_run_id=plan.run_id,
        acknowledge_budget_usd=1_000,
        resume=True,
        approved_plan_digest=plan.plan_digest,
        approved_action_digest=str(approval["approval_digest"]),
        smoke_receipt_artifact_id=smoke_id,
        model_cache_artifact_id=cache_id,
    )

    assert calls == [
        {
            "approved_run_id": plan.run_id,
            "budget_acknowledged": True,
            "resume": True,
            "training_function": imported_adapter.run_training_job,
                "selection_function": imported_adapter.run_selection_job,
                "finalizer_function": imported_adapter.finalize_stage_a_remote,
                "recovery_function": imported_adapter.recover_stage_a_orphans_remote,
                "runs_client": imported_adapter.runs_volume,
            "inputs_client": imported_adapter.inputs_volume,
            "model_client": imported_adapter.model_volume,
            "smoke_receipt_artifact_id": smoke_id,
            "model_cache_artifact_id": cache_id,
            "approval_payload": approval,
        }
    ]
    assert capsys.readouterr().out == canonical_json(summary) + "\n"


def _stage_a_test_dependency_evidence(
    plan: modal_plan.PilotPlan,
) -> tuple[dict[str, bytes], dict[str, object], dict[str, object]]:
    file_paths = tuple(sorted((
        "config.json",
        "generation_config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "model-00001-of-00001.safetensors",
    )))
    file_contents = {
        path: b"x" * (index + 1) for index, path in enumerate(file_paths)
    }
    manifest: dict[str, object] = {
        "schema_version": 1,
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "model_revision": plan.model_revision,
        "files": [
            {
                "path": path,
                "size": len(file_contents[path]),
                "sha256": hashlib.sha256(file_contents[path]).hexdigest(),
            }
            for path in file_paths
        ],
    }
    manifest["artifact_id"] = modal_artifacts.sha256_json(manifest)
    smoke: dict[str, object] = {
        "schema_version": 1,
        "stage": "smoke",
        "hardware": "CPU",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "config_hash": plan.config_hash,
        "split_artifact_id": plan.split_artifact_id,
        "materialization_artifact_ids": list(plan.materialization_artifact_ids),
        "source_hash": plan.source_hash,
        "dependency_lock_hash": plan.dependency_lock_hash,
        "canonical_dependency_lock_path": plan.canonical_dependency_lock_path,
        "bundle_id": plan.bundle_id,
        "bundle_manifest_artifact_id": plan.bundle_manifest_artifact_id,
        "bundle_files": [asdict(item) for item in plan.bundle_files],
        "modal_environment": plan.modal_environment,
        "model_revision": plan.model_revision,
        "model_cache_artifact_id": manifest["artifact_id"],
        "imports": [
            {"module": module, "version": "locked-test-version"}
            for module in EXPECTED_LOCKED_RUNTIME_IMPORTS
        ],
        "modal_app_id": "ap-smoke-test",
        "modal_app_name": "phase-marker-pilot-stage-a",
        "modal_function_name": "smoke_remote",
        "modal_function_call_id": "fc-smoke-test",
        "modal_input_id": "in-smoke-test",
        "python_version": "3.12.test",
        "torch_version": "2.7.test",
        "cuda_runtime_version": "12.8.test",
        "cuda_driver_version": "not-observed-cpu",
        "validated": True,
        "failure_reason": None,
    }
    smoke["artifact_id"] = modal_artifacts.sha256_json(smoke)
    return file_contents, manifest, smoke


def _stage_a_receipt(
    plan: modal_plan.PilotPlan, job: modal_plan.PilotJob, stage: str,
) -> dict[str, object]:
    _files, manifest, smoke = _stage_a_test_dependency_evidence(plan)
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id=str(smoke["artifact_id"]),
        model_cache_artifact_id=str(manifest["artifact_id"]),
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
        seed=job.seed,
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{stage}:{job.arm}")),
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=plan.source_hash,
        dependency_lock_hash=plan.dependency_lock_hash,
        model_cache_artifact_id=str(manifest["artifact_id"]),
        stage_a_action_digest=str(approval["approval_digest"]),
        stage_a_resume=False,
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
        output_hashes=tuple(hashlib.sha256(path.encode("utf-8")).hexdigest() for path in paths),
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


class StageAMapFunction:
    def __init__(
        self,
        stage: str,
        results: dict[str, object],
        events: list[tuple[object, ...]],
        *,
        before_first: Callable[[], None] | None = None,
        after_result: Callable[[str, object], None] | None = None,
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
                if isinstance(result, dict):
                    approval = payload.get("approval")
                    if isinstance(approval, dict) and "artifact_id" in result:
                        result = dict(result)
                        result["plan_digest"] = approval["plan_digest"]
                        result["stage_a_action_digest"] = approval["approval_digest"]
                        result["stage_a_resume"] = approval["resume"]
                        result["smoke_receipt_artifact_id"] = approval[
                            "smoke_receipt_artifact_id"
                        ]
                        result["model_cache_artifact_id"] = approval[
                            "model_cache_artifact_id"
                        ]
                        result["bundle_manifest_artifact_id"] = approval[
                            "bundle_manifest_artifact_id"
                        ]
                        result["modal_environment"] = approval[
                            "modal_environment"
                        ]
                        unsigned = dict(result)
                        unsigned.pop("artifact_id")
                        result["artifact_id"] = modal_artifacts.sha256_json(unsigned)
                        self.results[arm] = result
                if self.after_result is not None:
                    self.after_result(arm, result)
                yield result

        return results()


class StageAFinalizer:
    def __init__(self, summary: dict[str, object], events: list[tuple[object, ...]]) -> None:
        self.summary = summary
        self.events = events
        self.calls: list[object] = []

    def remote(self, payload: object) -> dict[str, object]:
        self.calls.append(payload)
        self.events.append(("finalizer",))
        if not isinstance(payload, dict):
            return self.summary
        approval = payload.get("approval")
        receipts = payload.get("receipts")
        if not isinstance(approval, dict) or not isinstance(receipts, list):
            return self.summary
        training = [receipt for receipt in receipts if receipt.get("stage") == "train"]
        selection = [
            receipt for receipt in receipts if receipt.get("stage") == "selection"
        ]
        summary = dict(self.summary)
        summary.update(
            {
                "plan_digest": approval["plan_digest"],
                "stage_a_action_digest": approval["approval_digest"],
                "stage_a_resume": approval["resume"],
                "modal_environment": approval["modal_environment"],
                "smoke_receipt_artifact_id": approval[
                    "smoke_receipt_artifact_id"
                ],
                "model_cache_artifact_id": approval[
                    "model_cache_artifact_id"
                ],
                "bundle_manifest_artifact_id": approval[
                    "bundle_manifest_artifact_id"
                ],
                "receipt_approval_history": modal_artifacts._receipt_approval_history(
                    receipts
                ),
                "training_receipt_ids": [
                    receipt["artifact_id"] for receipt in training
                ],
                "selection_receipt_ids": [
                    receipt["artifact_id"] for receipt in selection
                ],
                "elapsed_gpu_seconds": {
                    "training": sum(
                        float(receipt["elapsed_seconds"]) for receipt in training
                    ),
                    "selection": sum(
                        float(receipt["elapsed_seconds"]) for receipt in selection
                    ),
                    "total": sum(
                        float(receipt["elapsed_seconds"]) for receipt in receipts
                    ),
                },
            }
        )
        summary.pop("artifact_id", None)
        summary["artifact_id"] = modal_artifacts.sha256_json(summary)
        return summary


def _publish_stage_a_result(
    files: dict[str, bytes], receipt: object, runs: StageARunsClient,
) -> None:
    published = dict(files)
    if isinstance(receipt, dict):
        receipt_paths = [
            path for path in published if "/receipts/canonical/" in path
        ]
        if receipt_paths:
            receipt_path = receipt_paths[0]
            persisted = json.loads(published[receipt_path])
            if not str(persisted.get("attempt_id", "")).startswith("post-reload-"):
                published[receipt_path] = (
                    canonical_json(receipt) + "\n"
                ).encode("utf-8")
    runs.files.update(published)


class EmptyStageARunsClient:
    def __init__(self, events: list[tuple[object, ...]]) -> None:
        self.events = events

    def read_file(self, path: str) -> list[bytes]:
        self.events.append(("read_file", path))
        raise FileNotFoundError(path)

    def listdir(self, path: str, *, recursive: bool = False) -> list[object]:
        assert recursive is True
        self.events.append(("listdir", path))
        raise FileNotFoundError(path)

    def reload(self) -> None:
        raise AssertionError("local Volume batch clients cannot reload mounts")


class StageARunsClient(RecordingVolume):
    def __init__(
        self, files: dict[str, bytes], events: list[tuple[object, ...]],
    ) -> None:
        super().__init__(files)
        self.events = events

    def reload(self) -> None:
        raise AssertionError("local Volume batch clients cannot reload mounts")


def _stage_a_dependency_kwargs(
    plan: modal_plan.PilotPlan,
    pilot_repo: Path,
    runs: RecordingVolume,
    *,
    resume: bool,
) -> dict[str, object]:
    bundle = build_input_bundle(pilot_repo)
    inputs = LocalBatchVolume(_bundle_volume_files(bundle, pilot_repo))
    file_contents, manifest, smoke = _stage_a_test_dependency_evidence(plan)
    model_cache_artifact_id = str(manifest["artifact_id"])
    snapshot_root = (
        "/canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        f"{plan.model_revision}"
    )
    model_files = {
        f"{snapshot_root}/{path}": content
        for path, content in file_contents.items()
    }
    model_files[
        "/canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots/"
        f"{plan.model_revision}.manifest.json"
    ] = (canonical_json(manifest) + "\n").encode("utf-8")
    model = LocalBatchVolume(model_files)
    smoke_id = str(smoke["artifact_id"])
    bundle_manifest = (canonical_json(asdict(bundle)) + "\n").encode("utf-8")
    runs.files[
        f"/runs/{plan.run_id}/provenance/input-bundle-manifest.json"
    ] = bundle_manifest
    runs.files[
        f"/runs/{plan.run_id}/receipts/smoke/{smoke_id}.json"
    ] = (canonical_json(smoke) + "\n").encode("utf-8")
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=resume,
        smoke_receipt_artifact_id=smoke_id,
        model_cache_artifact_id=model_cache_artifact_id,
    )
    return {
        "inputs_client": inputs,
        "model_client": model,
        "smoke_receipt_artifact_id": smoke_id,
        "model_cache_artifact_id": model_cache_artifact_id,
        "approval_payload": approval,
    }


def test_stage_a_dependency_preflight_binds_exact_bundle_cache_and_smoke(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if H100 approval could name unreviewed staged evidence."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    runs = LocalBatchVolume()
    kwargs = _stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False)

    evidence = imported_adapter.preflight_stage_a_dependencies(
        plan,
        inputs_client=kwargs["inputs_client"],
        model_client=kwargs["model_client"],
        runs_client=runs,
        smoke_receipt_artifact_id=kwargs["smoke_receipt_artifact_id"],
        model_cache_artifact_id=kwargs["model_cache_artifact_id"],
    )

    assert evidence.bundle_id == plan.bundle_id
    assert evidence.model_cache_artifact_id == kwargs["model_cache_artifact_id"]
    assert evidence.smoke_receipt_artifact_id == kwargs["smoke_receipt_artifact_id"]
    with pytest.raises(ValueError, match="model-cache artifact"):
        imported_adapter.preflight_stage_a_dependencies(
            plan,
            inputs_client=kwargs["inputs_client"],
            model_client=kwargs["model_client"],
            runs_client=runs,
            smoke_receipt_artifact_id=kwargs["smoke_receipt_artifact_id"],
            model_cache_artifact_id="0" * 64,
        )

    model = kwargs["model_client"]
    assert isinstance(model, RecordingVolume)
    snapshot_file = next(
        path for path in model.files if path.endswith("/config.json")
    )
    model.files[snapshot_file] = b"tampered-but-still-listed"
    with pytest.raises(ValueError, match="model-cache file"):
        imported_adapter.preflight_stage_a_dependencies(
            plan,
            inputs_client=kwargs["inputs_client"],
            model_client=model,
            runs_client=runs,
            smoke_receipt_artifact_id=kwargs["smoke_receipt_artifact_id"],
            model_cache_artifact_id=kwargs["model_cache_artifact_id"],
        )


def test_stage_a_dependency_preflight_rejects_self_hashed_extra_smoke_schema(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if a self-hashed receipt could invent its own smoke schema."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    runs = RecordingVolume()
    kwargs = _stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False)
    original_id = str(kwargs["smoke_receipt_artifact_id"])
    original_path = f"/runs/{plan.run_id}/receipts/smoke/{original_id}.json"
    smoke = json.loads(runs.files.pop(original_path))
    smoke["unapproved_extra"] = "self-hashed"
    smoke.pop("artifact_id")
    smoke["artifact_id"] = modal_artifacts.sha256_json(smoke)
    changed_id = str(smoke["artifact_id"])
    runs.files[f"/runs/{plan.run_id}/receipts/smoke/{changed_id}.json"] = (
        canonical_json(smoke) + "\n"
    ).encode("utf-8")

    with pytest.raises(ValueError, match="smoke receipt"):
        imported_adapter.preflight_stage_a_dependencies(
            plan,
            inputs_client=kwargs["inputs_client"],
            model_client=kwargs["model_client"],
            runs_client=runs,
            smoke_receipt_artifact_id=changed_id,
            model_cache_artifact_id=kwargs["model_cache_artifact_id"],
        )


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
        training_files, _training_receipt = _canonical_stage_a_files(
            plan, job, "train"
        )
        training_manifest = next(
            content
            for path, content in training_files.items()
            if path.endswith("run-manifest.json")
        )
        evidence = b"{}\n"
        materialization_id = dict(
            zip(
                (candidate.arm for candidate in plan.jobs),
                plan.materialization_artifact_ids,
                strict=True,
            )
        )[job.arm]
        training_manifest_hash = hashlib.sha256(training_manifest).hexdigest()
        checkpoint_hash = hashlib.sha256(
            f"checkpoint:{job.arm}".encode("utf-8")
        ).hexdigest()
        selection_root = (
            f"artifacts/phase-marker/checkpoint-selections/pilot/seed-42/{job.arm}"
        )
        training_root = (
            f"artifacts/phase-marker/checkpoints/pilot/seed-42/{job.arm}"
        )
        split_path = Path(plan.local_repo_root) / "artifacts/phase-marker/splits/manifest.json"
        validation_path = (
            Path(plan.local_repo_root) / "artifacts/phase-marker/splits/validation.jsonl"
        )
        manifest_payload: dict[str, object] = {
            "schema_version": 1,
            "kind": "phase_marker_checkpoint_selection",
            "config_hash": plan.config_hash,
            "run_kind": "pilot",
            "arm": job.arm,
            "seed": 42,
            "selected_on": "validation",
            "evidence_scope": "experiment",
            "origin_verification": "execution_receipt_or_rerun_required",
            "backend": "vllm",
            "model_id": "Qwen/Qwen2.5-7B-Instruct",
            "model_revision": plan.model_revision,
            "criterion": {
                "primary": "maximize_strict_validation_exact_answer_accuracy",
                "tie_break_1": "higher_mean_gold_answer_logprob",
                "tie_break_2": "earliest_checkpoint_step",
            },
            "split_artifact_id": plan.split_artifact_id,
            "split_manifest_hash": hashlib.sha256(split_path.read_bytes()).hexdigest(),
            "validation_examples_file": "artifacts/phase-marker/splits/validation.jsonl",
            "validation_examples_hash": hashlib.sha256(
                validation_path.read_bytes()
            ).hexdigest(),
            "training_manifest_file": f"{training_root}/run-manifest.json",
            "training_manifest_hash": training_manifest_hash,
            "materialization_artifact_id": materialization_id,
            "candidates": [{
                "path": f"{training_root}/checkpoint-1",
                "checkpoint_hash": checkpoint_hash,
                "step": 1,
                "strict_accuracy": 1.0,
                "mean_gold_answer_logprob": -0.1,
                "row_count": 1,
            }],
            "evidence_file": f"{selection_root}/evidence.jsonl",
            "evidence_hash": hashlib.sha256(evidence).hexdigest(),
            "selected_path": f"{training_root}/checkpoint-1",
            "selected_checkpoint_hash": checkpoint_hash,
            "selected_step": 1,
            "parent_hashes": [
                plan.split_artifact_id,
                materialization_id,
                training_manifest_hash,
            ],
            "completed": True,
        }
        manifest_payload["artifact_id"] = modal_artifacts.sha256_json(
            manifest_payload
        )
        manifest = (canonical_json(manifest_payload) + "\n").encode("utf-8")
        producer_files = {
            "manifest.json": manifest,
            "evidence.jsonl": evidence,
        }
    command = job.training_command if stage == "train" else job.selection_command
    _files, cache_manifest, smoke_receipt = _stage_a_test_dependency_evidence(plan)
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id=str(smoke_receipt["artifact_id"]),
        model_cache_artifact_id=str(cache_manifest["artifact_id"]),
    )
    paths = tuple(sorted(producer_files))
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
        attempt_id=str(uuid.uuid5(uuid.NAMESPACE_OID, f"canonical:{stage}:{job.arm}")),
        command=command,
        command_hash=hashlib.sha256(command.encode("utf-8")).hexdigest(),
        source_hash=plan.source_hash,
        dependency_lock_hash=plan.dependency_lock_hash,
        model_cache_artifact_id=str(cache_manifest["artifact_id"]),
        stage_a_action_digest=str(approval["approval_digest"]),
        stage_a_resume=False,
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
        output_hashes=tuple(
            hashlib.sha256(producer_files[path]).hexdigest() for path in paths
        ),
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
    receipt_payload = asdict(receipt)
    receipt_payload["expected_outputs"] = list(receipt.expected_outputs)
    receipt_payload["output_hashes"] = list(receipt.output_hashes)
    receipt_payload["materialization_artifact_ids"] = list(
        receipt.materialization_artifact_ids
    )
    receipt_payload["runtime_versions"] = [
        {"module": module, "version": version}
        for module, version in receipt.runtime_versions
    ]
    receipt_payload["bundle_files"] = [
        asdict(item) for item in receipt.bundle_files
    ]
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


def _write_direct_remote_mounts(
    plan: modal_plan.PilotPlan,
    pilot_repo: Path,
    root: Path,
    *,
    include_training: bool,
    include_selection: bool = False,
) -> tuple[Path, Path, Path, dict[str, object], list[dict[str, object]]]:
    """Materialize one complete, self-derived direct-call dependency envelope."""
    input_root = root / "inputs"
    model_root = root / "model-cache"
    run_root = root / "runs"
    bundle = build_input_bundle(pilot_repo)
    bundle_root = input_root / "bundles" / bundle.bundle_id
    for item in bundle.files:
        destination = bundle_root / item.path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes((pilot_repo / item.path).read_bytes())
    bundle_manifest = (canonical_json(asdict(bundle)) + "\n").encode("utf-8")
    (bundle_root / "bundle-manifest.json").write_bytes(bundle_manifest)

    snapshot = (
        model_root / "canonical/models--Qwen--Qwen2.5-7B-Instruct/snapshots"
        / plan.model_revision
    )
    snapshot.mkdir(parents=True)
    model_payloads = {
        "config.json": {
            "architectures": ["Qwen2ForCausalLM"], "hidden_size": 3584,
            "intermediate_size": 18944, "model_type": "qwen2",
            "num_attention_heads": 28, "num_hidden_layers": 28,
            "num_key_value_heads": 4, "vocab_size": 152064,
        },
        "generation_config.json": {
            "bos_token_id": 151643, "eos_token_id": [151645, 151643],
            "pad_token_id": 151643,
        },
        "tokenizer.json": {
            "version": "1.0",
            "added_tokens": [{"id": 0, "content": "<|endoftext|>"}],
            "normalizer": {"type": "NFC"},
            "pre_tokenizer": {"type": "Sequence", "pretokenizers": []},
            "post_processor": {"type": "ByteLevel"},
            "decoder": {"type": "ByteLevel"},
            "model": {
                "type": "BPE", "vocab": {"<|endoftext|>": 0, "t": 1, "o": 2, "to": 3},
                "merges": ["t o"],
            },
        },
        "tokenizer_config.json": {
            "tokenizer_class": "Qwen2Tokenizer",
            "chat_template": "{% for message in messages %}{{ message['content'] }}{% endfor %}",
            "model_max_length": 131072,
        },
        "model.safetensors.index.json": {
            "metadata": {"total_size": 8},
            "weight_map": {
                "model.layers.0.weight": "model-00001-of-00002.safetensors",
                "model.layers.1.weight": "model-00002-of-00002.safetensors",
            },
        },
    }
    for name, payload in model_payloads.items():
        (snapshot / name).write_text(json.dumps(payload), encoding="utf-8")
    (snapshot / "model-00001-of-00002.safetensors").write_bytes(b"first\n")
    (snapshot / "model-00002-of-00002.safetensors").write_bytes(b"second\n")
    cache_manifest = build_model_cache_manifest(snapshot)
    (snapshot.parent / f"{plan.model_revision}.manifest.json").write_text(
        canonical_json(asdict(cache_manifest)) + "\n", encoding="utf-8"
    )

    run = run_root / "runs" / plan.run_id
    provenance = run / "provenance/input-bundle-manifest.json"
    provenance.parent.mkdir(parents=True)
    provenance.write_bytes(bundle_manifest)
    _files, _old_manifest, smoke = _stage_a_test_dependency_evidence(plan)
    smoke = dict(smoke)
    smoke["model_cache_artifact_id"] = cache_manifest.artifact_id
    smoke.pop("artifact_id")
    smoke["artifact_id"] = modal_artifacts.sha256_json(smoke)
    smoke_path = run / f"receipts/smoke/{smoke['artifact_id']}.json"
    smoke_path.parent.mkdir(parents=True)
    smoke_path.write_text(canonical_json(smoke) + "\n", encoding="utf-8")
    approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=False,
        smoke_receipt_artifact_id=str(smoke["artifact_id"]),
        model_cache_artifact_id=cache_manifest.artifact_id,
    )
    canonical_receipts: list[dict[str, object]] = []
    for stage, enabled in (("train", include_training), ("selection", include_selection)):
        if not enabled:
            continue
        for job in plan.jobs:
            files, receipt = _canonical_stage_a_files(plan, job, stage)
            receipt = dict(receipt)
            receipt["model_cache_artifact_id"] = cache_manifest.artifact_id
            receipt["smoke_receipt_artifact_id"] = smoke["artifact_id"]
            receipt["stage_a_action_digest"] = approval["approval_digest"]
            receipt["bundle_manifest_artifact_id"] = plan.bundle_manifest_artifact_id
            receipt["modal_environment"] = plan.modal_environment
            receipt.pop("artifact_id")
            receipt["artifact_id"] = modal_artifacts.sha256_json(receipt)
            for remote_path, content in files.items():
                destination = run_root / remote_path.lstrip("/")
                destination.parent.mkdir(parents=True, exist_ok=True)
                if "/receipts/canonical/" in remote_path:
                    destination.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
                else:
                    destination.write_bytes(content)
            canonical_receipts.append(receipt)
    return input_root, model_root, run_root, approval, canonical_receipts


@pytest.mark.parametrize("damage", ("missing", "corrupt"))
def test_direct_training_remote_aborts_before_fake_gpu_body_when_smoke_invalid(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    damage: str,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    input_root, model_root, run_root, approval, _ = _write_direct_remote_mounts(
        plan, pilot_repo, tmp_path / "direct", include_training=False
    )
    smoke_path = next((run_root / "runs" / plan.run_id / "receipts/smoke").iterdir())
    if damage == "missing":
        smoke_path.unlink()
    else:
        smoke_path.write_bytes(b"{}\n")
    calls: list[object] = []
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(imported_adapter, "execute_pilot_job", lambda **kw: calls.append(kw))
    monkeypatch.setattr(imported_adapter, "JOB_INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "JOB_MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "JOB_RUN_MOUNT_ROOT", run_root)
    job = asdict(plan.jobs[0])
    job["expected_outputs"] = list(job["expected_outputs"])
    with pytest.raises(ValueError, match="smoke"):
        imported_adapter.run_training_job.local(
            {"plan": modal_plan.pilot_plan_payload(plan), "job": job, "approval": approval}
        )
    assert calls == []


@pytest.mark.parametrize("arm", ("semantic", "glyph", "dot", "random", "direct", "filler"))
@pytest.mark.parametrize("damage", ("missing", "corrupt", "approval"))
def test_direct_selection_remote_aborts_if_any_training_is_invalid(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    arm: str,
    damage: str,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    input_root, model_root, run_root, approval, _ = _write_direct_remote_mounts(
        plan, pilot_repo, tmp_path / "direct", include_training=True
    )
    run = run_root / "runs" / plan.run_id
    receipt_path = run / f"receipts/canonical/train/{arm}.json"
    if damage == "missing":
        receipt_path.unlink()
    elif damage == "corrupt":
        producer = run / f"artifacts/phase-marker/checkpoints/pilot/seed-42/{arm}"
        next(path for path in producer.rglob("*") if path.is_file()).write_bytes(b"corrupt")
    else:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        receipt["smoke_receipt_artifact_id"] = "d" * 64
        receipt["stage_a_action_digest"] = modal_artifacts._stage_a_action_digest(
            plan_digest=plan.plan_digest,
            resume=False,
            smoke_receipt_artifact_id="d" * 64,
            model_cache_artifact_id=str(approval["model_cache_artifact_id"]),
            bundle_manifest_artifact_id=plan.bundle_manifest_artifact_id,
            modal_environment=plan.modal_environment,
        )
        receipt.pop("artifact_id")
        receipt["artifact_id"] = modal_artifacts.sha256_json(receipt)
        receipt_path.write_text(canonical_json(receipt) + "\n", encoding="utf-8")
    calls: list[object] = []
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(imported_adapter, "execute_pilot_job", lambda **kw: calls.append(kw))
    monkeypatch.setattr(imported_adapter, "JOB_INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "JOB_MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "JOB_RUN_MOUNT_ROOT", run_root)
    monkeypatch.setattr(modal_artifacts, "validate_canonical_job_semantics", lambda **_: None)
    job = asdict(plan.jobs[0])
    job["expected_outputs"] = list(job["expected_outputs"])
    with pytest.raises(ValueError):
        imported_adapter.run_selection_job.local(
            {"plan": modal_plan.pilot_plan_payload(plan), "job": job, "approval": approval}
        )
    assert calls == []


def test_direct_selection_remote_validates_all_six_training_parents_before_body(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    input_root, model_root, run_root, approval, _ = _write_direct_remote_mounts(
        plan, pilot_repo, tmp_path / "direct", include_training=True
    )
    semantic: list[str] = []
    bodies: list[dict[str, object]] = []
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(imported_adapter, "JOB_INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "JOB_MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "JOB_RUN_MOUNT_ROOT", run_root)
    monkeypatch.setattr(
        modal_artifacts,
        "validate_canonical_job_semantics",
        lambda **kwargs: semantic.append(str(kwargs["job_payload"]["arm"])),
    )
    monkeypatch.setattr(
        imported_adapter,
        "_collect_modal_execution_provenance",
        lambda _name: _adapter_execution_provenance("selection"),
    )
    monkeypatch.setattr(
        imported_adapter,
        "execute_pilot_job",
        lambda **kwargs: bodies.append(dict(kwargs)) or {"ok": True},
    )
    job = asdict(plan.jobs[0])
    job["expected_outputs"] = list(job["expected_outputs"])

    assert imported_adapter.run_selection_job.local(
        {"plan": modal_plan.pilot_plan_payload(plan), "job": job, "approval": approval}
    ) == {"ok": True}
    assert semantic == [job.arm for job in plan.jobs]
    assert len(bodies) == 1


def test_direct_finalizer_rejects_forged_results_without_canonical_receipt(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    input_root, model_root, run_root, approval, receipts = _write_direct_remote_mounts(
        plan,
        pilot_repo,
        tmp_path / "direct",
        include_training=True,
        include_selection=True,
    )
    missing = (
        run_root / "runs" / plan.run_id
        / "receipts/canonical/selection/filler.json"
    )
    missing.unlink()
    finalizer_calls: list[object] = []
    monkeypatch.setattr(imported_adapter, "runs_volume", RecordingVolume())
    monkeypatch.setattr(imported_adapter, "JOB_INPUT_MOUNT_ROOT", input_root)
    monkeypatch.setattr(imported_adapter, "JOB_MODEL_MOUNT_ROOT", model_root)
    monkeypatch.setattr(imported_adapter, "JOB_RUN_MOUNT_ROOT", run_root)
    monkeypatch.setattr(modal_artifacts, "validate_canonical_job_semantics", lambda **_: None)
    monkeypatch.setattr(
        imported_adapter,
        "finalize_stage_a",
        lambda **kwargs: finalizer_calls.append(kwargs) or {},
    )

    with pytest.raises(ValueError, match="canonical Stage A receipt"):
        imported_adapter.finalize_stage_a_remote.local(
            {
                "plan": modal_plan.pilot_plan_payload(plan),
                "approval": approval,
                "receipts": receipts,
            }
        )
    assert finalizer_calls == []
    assert not (
        run_root / "runs" / plan.run_id / "stage-a-summary.json"
    ).exists()


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
        after_result=lambda arm, receipt: _publish_stage_a_result(
            publications[arm], receipt, runs
        ),
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
    receipt_matrix = [
        *(training[job.arm] for job in plan.jobs),
        *(selection[job.arm] for job in plan.jobs),
    ]
    first = receipt_matrix[0]
    payload: dict[str, object] = {
        "schema_version": 1,
        "stage": "stage-a",
        "run_id": plan.run_id,
        "plan_digest": plan.plan_digest,
        "stage_a_action_digest": first["stage_a_action_digest"],
        "stage_a_resume": first["stage_a_resume"],
        "modal_environment": first["modal_environment"],
        "smoke_receipt_artifact_id": first["smoke_receipt_artifact_id"],
        "model_cache_artifact_id": first["model_cache_artifact_id"],
        "bundle_manifest_artifact_id": first[
            "bundle_manifest_artifact_id"
        ],
        "receipt_approval_history": modal_artifacts._receipt_approval_history(
            receipt_matrix
        ),
        "training_receipt_ids": [training[job.arm]["artifact_id"] for job in plan.jobs],
        "selection_receipt_ids": [selection[job.arm]["artifact_id"] for job in plan.jobs],
        "behavior_gate_checked_artifact_ids": [],
        "elapsed_gpu_seconds": {
            "training": sum(
                float(training[job.arm]["elapsed_seconds"]) for job in plan.jobs
            ),
            "selection": sum(
                float(selection[job.arm]["elapsed_seconds"]) for job in plan.jobs
            ),
            "total": sum(
                float(receipts[job.arm]["elapsed_seconds"])
                for receipts in (training, selection)
                for job in plan.jobs
            ),
        },
        "finalizer_provenance": _adapter_execution_provenance("finalizer"),
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
        lambda approved, **_: events.append(("tags", approved.run_id)),
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
        **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
    )

    assert len(training.calls) == 6
    assert len(selection.calls) == 6
    assert len(finalizer.calls) == 1
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


def test_stage_a_flushes_complete_plan_before_first_tag_or_remote_call(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume_events: list[tuple[object, ...]] = []
    side_effects: list[tuple[object, ...]] = []
    runs = StageARunsClient({}, volume_events)
    training, training_receipts = _publishing_stage_a_function(
        plan, "train", side_effects, runs
    )
    selection, selection_receipts = _publishing_stage_a_function(
        plan, "selection", side_effects, runs
    )
    finalizer = StageAFinalizer(
        _stage_a_summary(plan, training_receipts, selection_receipts), side_effects
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda *_args, **_kwargs: side_effects.append(("tags",)),
    )

    def record_print(*args: object, **kwargs: object) -> None:
        payload = json.loads(str(args[0]))
        side_effects.append(("plan-print", kwargs.get("flush"), payload["operation"]))

    monkeypatch.setattr(builtins, "print", record_print)
    imported_adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=False,
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        runs_client=runs,
        **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
    )
    external = [
        event for event in side_effects
        if event[0] in {"plan-print", "tags", "train", "selection", "finalizer"}
    ]
    assert external[0] == ("plan-print", True, "run-stage-a")
    assert external[1] == ("tags",)


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
        after_result=lambda arm, receipt: _publish_stage_a_result(
            training_files[arm], receipt, runs
        ),
    )
    selection = StageAMapFunction(
        "selection", selection_results, events,
        after_result=lambda arm, receipt: _publish_stage_a_result(
            selection_files[arm], receipt, runs
        ),
    )
    finalizer = StageAFinalizer(
        _stage_a_summary(plan, training_results, selection_results), events
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )
    monkeypatch.setattr(
        imported_adapter, "apply_approved_app_tags", lambda _plan, **_: None
    )

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
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
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
    runs = StageARunsClient({}, events)
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
        imported_adapter, "apply_approved_app_tags", lambda approved, **_: None
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
            runs_client=runs,
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
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
        after_result=lambda arm, receipt: _publish_stage_a_result(
            selection_files[arm], receipt, runs
        ),
    )
    finalizer = StageAFinalizer({}, events)
    monkeypatch.setattr(
        imported_adapter, "apply_approved_app_tags", lambda _plan, **_: None
    )
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
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
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
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )

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
        after_result=lambda arm, receipt: _publish_stage_a_result(
            training_files[arm], receipt, runs
        ),
    )
    selection = StageAMapFunction(
        "selection",
        selection_results,
        events,
        after_result=lambda arm, receipt: _publish_stage_a_result(
            selection_files[arm], receipt, runs
        ),
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
        lambda approved, **_: events.append(("tags", approved.run_id)),
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
        **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=True),
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
    assert [item["arm"] for item in printed["missing_training"]] == [
        "dot", "random", "direct", "filler"
    ]
    assert summary["stopped_before_behavior"] is True
    assert all(runs.files[path] == content for path, content in original_files.items())
    assert runs.files[failed_attempt] == original_files[failed_attempt]


def test_explicit_resume_returns_existing_complete_summary_without_remote_calls(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a post-finalizer client crash forced a conflicting re-finalization."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    runs, _training_receipts, _selection_receipts = _complete_status_volume(plan)
    summary = json.loads(
        runs.files[f"/runs/{plan.run_id}/stage-a-summary.json"]
    )
    events: list[tuple[object, ...]] = []
    training = StageAMapFunction("train", {}, events)
    selection = StageAMapFunction("selection", {}, events)
    finalizer = StageAFinalizer({}, events)
    tags: list[object] = []
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda approved, **_: tags.append(approved),
    )
    monkeypatch.setattr(
        imported_adapter, "validate_canonical_job_semantics", lambda **_: None
    )

    result = imported_adapter.run_stage_a_local(
        plan,
        approved_run_id=plan.run_id,
        budget_acknowledged=True,
        resume=True,
        training_function=training,
        selection_function=selection,
        finalizer_function=finalizer,
        runs_client=runs,
        **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=True),
    )

    assert result == summary
    assert training.calls == []
    assert selection.calls == []
    assert finalizer.calls == []
    assert tags == []


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
    runs = StageARunsClient(files, events)
    tags: list[object] = []
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )

    with pytest.raises(ValueError, match="receipt|manifest|canonical"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=True,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=True),
        )
    assert tags == []
    assert training.calls == []
    assert selection.calls == []
    assert finalizer.calls == []


def _stage_a_promotion_lease(
    receipt_payload: dict[str, object],
) -> tuple[bytes, dict[str, object], datetime, dict[str, object]]:
    receipt = modal_artifacts.load_attempt_receipt_payload(receipt_payload)
    owner_started = datetime.now().astimezone() - timedelta(hours=6)
    receipt = replace(
        receipt,
        attempt_id=str(uuid.uuid4()),
        started_at=owner_started.isoformat(timespec="microseconds"),
        finished_at=(owner_started + timedelta(minutes=1)).isoformat(
            timespec="microseconds"
        ),
        artifact_id="",
    )
    receipt = replace(receipt, artifact_id=receipt.recomputed_artifact_id())
    normalized_receipt = asdict(receipt)
    normalized_receipt["expected_outputs"] = list(receipt.expected_outputs)
    normalized_receipt["output_hashes"] = list(receipt.output_hashes)
    normalized_receipt["materialization_artifact_ids"] = list(
        receipt.materialization_artifact_ids
    )
    normalized_receipt["runtime_versions"] = [
        {"module": module, "version": version}
        for module, version in receipt.runtime_versions
    ]
    created_at = owner_started + timedelta(seconds=1)
    lease = modal_artifacts._promotion_lease_payload(
        receipt, created_at=created_at
    )
    expired_at = datetime.fromisoformat(str(lease["recover_after"])) + timedelta(
        microseconds=1
    )
    return (
        (canonical_json(lease) + "\n").encode("utf-8"),
        lease,
        expired_at,
        normalized_receipt,
    )


@pytest.mark.parametrize(
    ("state", "expected_error", "move_producer", "canonical_complete"),
    (
        ("live-lease", "live promotion lease", None, False),
        ("malformed-lease", "promotion lease", None, False),
        ("producer-without-lease", "authenticated expired lease", None, False),
        ("expired-producer", None, True, False),
        ("expired-lease-only", None, False, False),
        ("expired-complete", None, False, True),
    ),
)
def test_resume_promotion_lease_state_matrix(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    state: str,
    expected_error: str | None,
    move_producer: bool | None,
    canonical_complete: bool,
) -> None:
    """Would fail if resume guessed ownership or recovered a live promotion."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = plan.jobs[0]
    files, receipt = _canonical_stage_a_files(plan, job, "train")
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    producer = imported_adapter._volume_producer_path(plan.run_id, "train", job.arm)
    lock = imported_adapter._volume_promotion_lock_path(plan.run_id, "train", job.arm)
    lease_bytes, lease, expired_at, receipt = _stage_a_promotion_lease(receipt)
    files[receipt_path] = (canonical_json(receipt) + "\n").encode("utf-8")
    observed_at = expired_at
    if not canonical_complete:
        files.pop(receipt_path)
    if state == "expired-lease-only":
        files = {
            path: content
            for path, content in files.items()
            if not path.startswith(producer.rstrip("/") + "/")
        }
    if state != "producer-without-lease":
        files[lock] = b"{}\n" if state == "malformed-lease" else lease_bytes
    if state == "live-lease":
        observed_at = datetime.fromisoformat(str(lease["lease_created_at"]))
    runs = StageARunsClient(files, [])

    if expected_error is not None:
        with pytest.raises((FileExistsError, ValueError), match=expected_error):
            imported_adapter._preflight_stage_a_outputs(
                plan, resume=True, runs_client=runs, now=observed_at
            )
        return

    preflight = imported_adapter._preflight_stage_a_outputs(
        plan, resume=True, runs_client=runs, now=observed_at
    )

    assert len(preflight.recoveries) == 1
    recovery = preflight.recoveries[0]
    assert recovery["stage"] == "train"
    assert recovery["arm"] == job.arm
    assert recovery["producer_path"] == producer
    assert recovery["lock_path"] == lock
    assert recovery["quarantine_root"].startswith(
        f"/runs/{plan.run_id}/attempts/orphan-recovery-"
    )
    assert recovery["move_producer"] is move_producer
    assert recovery["lock_artifact_id"] == lease["artifact_id"]
    assert (job.arm in preflight.training) is canonical_complete


def test_direct_orphan_recovery_rejects_live_authenticated_lease(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
) -> None:
    """Would fail if the remote trust boundary relied on local expiry preflight."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = plan.jobs[0]
    receipt_payload = _stage_a_receipt(plan, job, "train")
    started = datetime.now().astimezone()
    receipt_payload.update(
        attempt_id=str(uuid.uuid4()),
        started_at=started.isoformat(timespec="microseconds"),
        finished_at=started.isoformat(timespec="microseconds"),
    )
    receipt_payload.pop("artifact_id")
    receipt_payload["artifact_id"] = modal_artifacts.sha256_json(receipt_payload)
    receipt = modal_artifacts.load_attempt_receipt_payload(receipt_payload)
    lease = modal_artifacts._promotion_lease_payload(
        receipt, created_at=started + timedelta(microseconds=1)
    )
    spec = imported_adapter._stage_a_orphan_recovery_spec(
        plan,
        stage="train",
        arm=job.arm,
        producer_present=False,
        receipt_present=False,
        lock_present=True,
        move_producer=False,
        lease=lease,
    )
    mount = tmp_path / "runs-mount"
    local_lock = mount.joinpath(
        *PurePosixPath(str(spec["lock_path"])).parts[1:]
    )
    local_lock.parent.mkdir(parents=True)
    local_lock.write_text(canonical_json(lease) + "\n", encoding="utf-8")

    with pytest.raises(FileExistsError, match="live promotion lease"):
        imported_adapter._recover_stage_a_orphans(
            plan_payload=modal_plan.pilot_plan_payload(plan),
            recoveries=[spec],
            input_root=pilot_repo,
            run_mount_root=mount,
        )

    quarantine = mount.joinpath(
        *PurePosixPath(str(spec["quarantine_root"])).parts[1:]
    )
    assert local_lock.is_file()
    assert not quarantine.exists()


def test_direct_orphan_recovery_rejects_inconsistent_move_state(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
) -> None:
    """Would fail if a direct call could remove the lease but retain its orphan."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = plan.jobs[0]
    files, receipt_payload = _canonical_stage_a_files(plan, job, "train")
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    files.pop(receipt_path)
    lease_bytes, lease, _expired_at, _receipt = _stage_a_promotion_lease(
        receipt_payload
    )
    lock = imported_adapter._volume_promotion_lock_path(
        plan.run_id, "train", job.arm
    )
    files[lock] = lease_bytes
    spec = imported_adapter._stage_a_orphan_recovery_spec(
        plan,
        stage="train",
        arm=job.arm,
        producer_present=True,
        receipt_present=False,
        lock_present=True,
        move_producer=True,
        lease=lease,
    )
    spec["move_producer"] = False
    mount = tmp_path / "runs-mount"
    for remote_path, content in files.items():
        local = mount.joinpath(*PurePosixPath(remote_path).parts[1:])
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
    producer = mount.joinpath(
        *PurePosixPath(str(spec["producer_path"])).parts[1:]
    )
    local_lock = mount.joinpath(*PurePosixPath(lock).parts[1:])

    with pytest.raises(ValueError, match="recovery record identity"):
        imported_adapter._recover_stage_a_orphans(
            plan_payload=modal_plan.pilot_plan_payload(plan),
            recoveries=[spec],
            input_root=pilot_repo,
            run_mount_root=mount,
        )

    assert producer.is_dir()
    assert local_lock.is_file()


def test_orphan_quarantine_hard_crash_preserves_first_move_and_retries_uniquely(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a hard crash rolled back, reused, deleted, or adopted residue."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = plan.jobs[0]
    files, _receipt = _canonical_stage_a_files(plan, job, "train")
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    receipt = json.loads(files[receipt_path])
    files.pop(receipt_path)
    producer = imported_adapter._volume_producer_path(plan.run_id, "train", job.arm)
    lock = imported_adapter._volume_promotion_lock_path(plan.run_id, "train", job.arm)
    lease_bytes, _lease, expired_at, _receipt = _stage_a_promotion_lease(receipt)
    files[lock] = lease_bytes
    preflight = imported_adapter._preflight_stage_a_outputs(
        plan, resume=True, runs_client=StageARunsClient(files, []), now=expired_at
    )
    first = preflight.recoveries[0]
    mount = tmp_path / "runs-mount"
    for remote_path, content in files.items():
        local = mount.joinpath(*PurePosixPath(remote_path).parts[1:])
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)

    real_rename = imported_adapter.os.rename
    rename_calls = 0

    def crash_after_producer(
        source: object, destination: object, *args: object, **kwargs: object,
    ) -> None:
        nonlocal rename_calls
        rename_calls += 1
        if rename_calls == 2:
            raise KeyboardInterrupt("simulated hard crash")
        real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(imported_adapter.os, "rename", crash_after_producer)
    with pytest.raises(KeyboardInterrupt, match="hard crash"):
        imported_adapter._recover_stage_a_orphans(
            plan_payload=modal_plan.pilot_plan_payload(plan),
            recoveries=[first],
            input_root=pilot_repo,
            run_mount_root=mount,
        )

    first_quarantine = mount.joinpath(
        *PurePosixPath(str(first["quarantine_root"])).parts[1:]
    )
    local_producer = mount.joinpath(*PurePosixPath(producer).parts[1:])
    local_lock = mount.joinpath(*PurePosixPath(lock).parts[1:])
    assert (first_quarantine / "producer").is_dir()
    assert not local_producer.exists()
    assert local_lock.is_file()

    monkeypatch.setattr(imported_adapter.os, "rename", real_rename)
    mounted_files = {
        "/" + path.relative_to(mount).as_posix(): path.read_bytes()
        for path in mount.rglob("*")
        if path.is_file()
    }
    retry_preflight = imported_adapter._preflight_stage_a_outputs(
        plan,
        resume=True,
        runs_client=StageARunsClient(mounted_files, []),
        now=expired_at,
    )
    assert len(retry_preflight.recoveries) == 1
    second = retry_preflight.recoveries[0]
    assert second["quarantine_root"] != first["quarantine_root"]
    result = imported_adapter._recover_stage_a_orphans(
        plan_payload=modal_plan.pilot_plan_payload(plan),
        recoveries=[second],
        input_root=pilot_repo,
        run_mount_root=mount,
    )
    second_quarantine = mount.joinpath(
        *PurePosixPath(str(second["quarantine_root"])).parts[1:]
    )
    assert result["quarantined"] == [second]
    assert (first_quarantine / "producer").is_dir()
    assert (second_quarantine / "promotion.lock").is_file()
    assert not local_lock.exists()


def test_orphan_quarantine_detects_source_inode_swap_without_deleting_residue(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Would fail if a checked path could be swapped before its quarantine rename."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    job = plan.jobs[0]
    files, receipt = _canonical_stage_a_files(plan, job, "train")
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    files.pop(receipt_path)
    producer = imported_adapter._volume_producer_path(plan.run_id, "train", job.arm)
    lock = imported_adapter._volume_promotion_lock_path(plan.run_id, "train", job.arm)
    lease_bytes, _lease, expired_at, _receipt = _stage_a_promotion_lease(receipt)
    files[lock] = lease_bytes
    recovery = imported_adapter._preflight_stage_a_outputs(
        plan,
        resume=True,
        runs_client=StageARunsClient(files, []),
        now=expired_at,
    ).recoveries[0]
    mount = tmp_path / "runs-mount"
    for remote_path, content in files.items():
        local = mount.joinpath(*PurePosixPath(remote_path).parts[1:])
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(content)
    local_producer = mount.joinpath(*PurePosixPath(producer).parts[1:])
    attacker = local_producer.parent / ".attacker-swap"
    attacker.mkdir()
    (attacker / "unvalidated.txt").write_text("attacker", encoding="utf-8")
    parked = local_producer.parent / ".validated-original"
    real_rename = imported_adapter.os.rename
    swapped = False

    def swap_before_rename(
        source: object, destination: object, *args: object, **kwargs: object,
    ) -> None:
        nonlocal swapped
        if not swapped and source == job.arm and destination == "producer":
            swapped = True
            source_fd = int(kwargs["src_dir_fd"])
            real_rename(job.arm, parked.name, src_dir_fd=source_fd, dst_dir_fd=source_fd)
            real_rename(attacker.name, job.arm, src_dir_fd=source_fd, dst_dir_fd=source_fd)
        real_rename(source, destination, *args, **kwargs)

    monkeypatch.setattr(imported_adapter.os, "rename", swap_before_rename)
    with pytest.raises(OSError, match="identity changed"):
        imported_adapter._recover_stage_a_orphans(
            plan_payload=modal_plan.pilot_plan_payload(plan),
            recoveries=[recovery],
            input_root=pilot_repo,
            run_mount_root=mount,
        )

    quarantine = mount.joinpath(
        *PurePosixPath(str(recovery["quarantine_root"])).parts[1:]
    )
    assert swapped is True
    assert parked.is_dir()
    assert (quarantine / "producer" / "unvalidated.txt").read_text() == "attacker"
    assert mount.joinpath(*PurePosixPath(lock).parts[1:]).is_file()


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
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )
    runs = StageARunsClient(files, events)

    with pytest.raises(ValueError, match="training|completion|semantic"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=True,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=True),
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
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )
    runs = StageARunsClient(files, events)

    with pytest.raises(FileExistsError, match="use --resume"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
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
    monkeypatch.setattr(
        imported_adapter,
        "apply_approved_app_tags",
        lambda plan, **_: tags.append(plan),
    )
    runs = StageARunsClient(files, events)

    with pytest.raises((FileExistsError, ValueError), match="canonical|summary"):
        imported_adapter.run_stage_a_local(
            plan,
            approved_run_id=plan.run_id,
            budget_acknowledged=True,
            resume=False,
            training_function=training,
            selection_function=selection,
            finalizer_function=finalizer,
            runs_client=runs,
            **_stage_a_dependency_kwargs(plan, pilot_repo, runs, resume=False),
        )
    assert tags == []
    assert training.calls == [] and selection.calls == [] and finalizer.calls == []


def _status_dependency_files(plan: modal_plan.PilotPlan) -> dict[str, bytes]:
    _cache_files, _cache_manifest, smoke = _stage_a_test_dependency_evidence(plan)
    smoke_id = str(smoke["artifact_id"])
    bundle_manifest = {
        "schema_version": 1,
        "bundle_id": plan.bundle_id,
        "files": [asdict(item) for item in plan.bundle_files],
        "artifact_ids": [
            plan.split_artifact_id,
            *plan.materialization_artifact_ids,
        ],
    }
    return {
        f"/runs/{plan.run_id}/provenance/input-bundle-manifest.json": (
            canonical_json(bundle_manifest) + "\n"
        ).encode("utf-8"),
        f"/runs/{plan.run_id}/receipts/smoke/{smoke_id}.json": (
            canonical_json(smoke) + "\n"
        ).encode("utf-8"),
    }


def _complete_status_volume(
    plan: modal_plan.PilotPlan,
) -> tuple[StageARunsClient, dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    events: list[tuple[object, ...]] = []
    files: dict[str, bytes] = {}
    training: dict[str, dict[str, object]] = {}
    selection: dict[str, dict[str, object]] = {}
    for stage, receipts in (("train", training), ("selection", selection)):
        publications, stage_receipts = _canonical_stage_a_publications(plan, stage)
        for job in plan.jobs:
            files.update(publications[job.arm])
            receipts[job.arm] = stage_receipts[job.arm]
    summary = _stage_a_summary(plan, training, selection)
    files[f"/runs/{plan.run_id}/stage-a-summary.json"] = (
        canonical_json(summary) + "\n"
    ).encode("utf-8")
    files.update(_status_dependency_files(plan))
    return StageARunsClient(files, events), training, selection


def test_status_reads_validated_receipts_and_producer_manifests_without_mutation(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if status inferred completion from names or wrote remote state."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    before = dict(volume.files)

    result = imported_adapter.status_local(volume, run_id=plan.run_id)

    assert result["training"] == {job.arm: "complete" for job in plan.jobs}
    assert result["selection"] == {job.arm: "complete" for job in plan.jobs}
    assert result["summary"] == "complete"
    assert result["stopped_before_behavior"] is True
    assert result["valid"] is True
    assert volume.files == before
    assert not any(event[0] in {"batch_upload", "commit", "reload"} for event in volume.events)


@pytest.mark.parametrize("tamper", ("path", "size", "hash", "order"))
def test_status_rejects_ordered_bundle_file_record_tampering(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tamper: str,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    path = f"/runs/{plan.run_id}/receipts/canonical/train/semantic.json"
    receipt = json.loads(volume.files[path])
    records = list(receipt["bundle_files"])
    if tamper == "path":
        records[0] = {**records[0], "path": "configs/other.toml"}
    elif tamper == "size":
        records[0] = {**records[0], "size": records[0]["size"] + 1}
    elif tamper == "hash":
        records[0] = {**records[0], "sha256": "d" * 64}
    else:
        records[0], records[1] = records[1], records[0]
    receipt["bundle_files"] = records
    receipt.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(receipt)
    volume.files[path] = (canonical_json(receipt) + "\n").encode("utf-8")

    result = imported_adapter.status_local(volume, run_id=plan.run_id)
    assert result["training"]["semantic"] == "invalid"
    assert result["valid"] is False


def test_status_and_evidence_require_exact_bundle_manifest_provenance(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
) -> None:
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    provenance = f"/runs/{plan.run_id}/provenance/input-bundle-manifest.json"
    volume.files.pop(provenance)

    result = imported_adapter.status_local(volume, run_id=plan.run_id)
    assert result["summary"] == "invalid"
    assert result["valid"] is False
    with pytest.raises(ValueError, match="validated complete"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=tmp_path / "evidence"
        )


@pytest.mark.parametrize(
    "damage", ("missing-smoke", "missing-provenance", "missing-both", "corrupt-provenance")
)
def test_partial_status_requires_receipt_advertised_smoke_and_provenance(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    damage: str,
) -> None:
    """Would fail if summary absence hid missing shared dependency evidence."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    volume.files.pop(f"/runs/{plan.run_id}/stage-a-summary.json")
    provenance = f"/runs/{plan.run_id}/provenance/input-bundle-manifest.json"
    smoke = next(
        path for path in volume.files
        if path.startswith(f"/runs/{plan.run_id}/receipts/smoke/")
    )
    if damage in {"missing-smoke", "missing-both"}:
        volume.files.pop(smoke)
    if damage in {"missing-provenance", "missing-both"}:
        volume.files.pop(provenance)
    elif damage == "corrupt-provenance":
        volume.files[provenance] = b"{}\n"

    result = imported_adapter.status_local(volume, run_id=plan.run_id)

    assert result["summary"] == "pending"
    assert result["valid"] is False
    assert result["errors"]


def test_status_validates_mixed_fresh_and_resume_receipt_approval_history(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if status flattened a legitimate resumed Stage A lineage."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, training, selection = _complete_status_volume(plan)
    first = training[plan.jobs[0].arm]
    resumed_approval = modal_plan.action_approval_payload(
        plan,
        action="run-stage-a",
        resume=True,
        smoke_receipt_artifact_id=str(first["smoke_receipt_artifact_id"]),
        model_cache_artifact_id=str(first["model_cache_artifact_id"]),
    )
    for job in plan.jobs[3:]:
        receipt = dict(selection[job.arm])
        receipt["stage_a_action_digest"] = resumed_approval["approval_digest"]
        receipt["stage_a_resume"] = True
        unsigned = dict(receipt)
        unsigned.pop("artifact_id")
        receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
        selection[job.arm] = receipt
        volume.files[
            f"/runs/{plan.run_id}/receipts/canonical/selection/{job.arm}.json"
        ] = (canonical_json(receipt) + "\n").encode("utf-8")
    summary = _stage_a_summary(plan, training, selection)
    summary["stage_a_action_digest"] = resumed_approval["approval_digest"]
    summary["stage_a_resume"] = True
    summary.pop("artifact_id")
    summary["artifact_id"] = modal_artifacts.sha256_json(summary)
    volume.files[f"/runs/{plan.run_id}/stage-a-summary.json"] = (
        canonical_json(summary) + "\n"
    ).encode("utf-8")

    result = imported_adapter.status_local(volume, run_id=plan.run_id)

    assert result["valid"] is True
    assert result["summary"] == "complete"
    assert len(summary["receipt_approval_history"]) == 2
    assert {
        item["stage_a_resume"] for item in summary["receipt_approval_history"]
    } == {False, True}


def test_status_rejects_self_hashed_incomplete_receipt_approval_history(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if summary approval history could omit a canonical receipt."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    summary_path = f"/runs/{plan.run_id}/stage-a-summary.json"
    summary = json.loads(volume.files[summary_path])
    summary["receipt_approval_history"][0]["receipt_artifact_ids"].pop()
    summary.pop("artifact_id")
    summary["artifact_id"] = modal_artifacts.sha256_json(summary)
    volume.files[summary_path] = (canonical_json(summary) + "\n").encode("utf-8")

    result = imported_adapter.status_local(volume, run_id=plan.run_id)

    assert result["summary"] == "invalid"
    assert result["valid"] is False
    assert any("summary" in error for error in result["errors"])


def test_status_reports_partial_failed_and_invalid_evidence_without_completion(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if partial, failed, or hash-mismatched bytes looked complete."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    first = plan.jobs[0]
    train_files, _receipt = _canonical_stage_a_files(plan, first, "train")
    volume = StageARunsClient(train_files, [])
    partial = imported_adapter.status_local(volume, run_id=plan.run_id)
    assert partial["training"][first.arm] == "complete"
    assert set(partial["training"].values()) == {"complete", "pending"}
    assert set(partial["selection"].values()) == {"pending"}
    assert partial["stopped_before_behavior"] is False

    failed = _stage_a_receipt(plan, plan.jobs[1], "train")
    failed.update(
        exit_status=1,
        validated=False,
        promoted=False,
        failure_reason="RuntimeError: producer crashed",
        failure_stage="command",
    )
    unsigned = dict(failed)
    unsigned.pop("artifact_id")
    failed["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    volume.files[
        f"/runs/{plan.run_id}/receipts/attempts/{failed['attempt_id']}.json"
    ] = (canonical_json(failed) + "\n").encode("utf-8")
    failed_status = imported_adapter.status_local(volume, run_id=plan.run_id)
    assert failed_status["training"][plan.jobs[1].arm] == "failed"

    adapter_path = next(
        path for path in volume.files if path.endswith("/semantic/adapter_config.json")
    )
    volume.files[adapter_path] += b"corrupt"
    invalid = imported_adapter.status_local(volume, run_id=plan.run_id)
    assert invalid["training"][first.arm] == "invalid"
    assert invalid["valid"] is False
    assert invalid["stopped_before_behavior"] is False


def test_status_rejects_unknown_or_unsafe_run_identity(
    imported_adapter: ModuleType,
) -> None:
    """Would fail if status scanned an unbound or traversal-derived namespace."""
    with pytest.raises(ValueError, match="canonical run ID"):
        imported_adapter.status_local(RecordingVolume(), run_id="../secrets")
    with pytest.raises(ValueError, match="unknown run"):
        imported_adapter.status_local(
            RecordingVolume(),
            run_id=(
                "pilot-s42-cfg-11111111-split-22222222-src-333333333333-"
                f"plan-{'4' * 64}"
            ),
        )


def test_status_does_not_accept_a_producer_directory_without_its_receipt(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if a canonical directory name alone implied completion."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files, _receipt = _canonical_stage_a_files(plan, plan.jobs[0], "train")
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    files.pop(receipt_path)

    result = imported_adapter.status_local(StageARunsClient(files, []), run_id=plan.run_id)

    assert result["training"][plan.jobs[0].arm] == "invalid"
    assert result["valid"] is False


def test_status_rejects_self_consistent_wrong_model_revision(
    imported_adapter: ModuleType, pilot_repo: Path,
) -> None:
    """Would fail if internal consistency replaced the frozen model identity."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    files, receipt = _canonical_stage_a_files(plan, plan.jobs[0], "train")
    manifest_path = next(path for path in files if path.endswith("run-manifest.json"))
    receipt_path = next(path for path in files if "/receipts/canonical/" in path)
    manifest = json.loads(files[manifest_path])
    manifest["model_revision"] = "f" * 40
    manifest["tokenizer_revision"] = "f" * 40
    manifest_bytes = (canonical_json(manifest) + "\n").encode("utf-8")
    files[manifest_path] = manifest_bytes
    output_index = receipt["expected_outputs"].index("run-manifest.json")
    receipt["output_hashes"][output_index] = hashlib.sha256(manifest_bytes).hexdigest()
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    files[receipt_path] = (canonical_json(receipt) + "\n").encode("utf-8")

    result = imported_adapter.status_local(StageARunsClient(files, []), run_id=plan.run_id)

    assert result["training"][plan.jobs[0].arm] == "invalid"
    assert result["valid"] is False


def _replace_canonical_output_and_rebind_receipt(
    files: dict[str, bytes],
    receipt: dict[str, object],
    *,
    output_path: str,
    content: bytes,
) -> None:
    files[output_path] = content
    output_name = output_path.rsplit("/", maxsplit=1)[-1]
    output_index = receipt["expected_outputs"].index(output_name)
    receipt["output_hashes"][output_index] = hashlib.sha256(content).hexdigest()
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    receipt_path = next(
        path
        for path in files
        if path.endswith(f"/receipts/canonical/{receipt['stage']}/{receipt['arm']}.json")
    )
    files[receipt_path] = (canonical_json(receipt) + "\n").encode("utf-8")


@pytest.mark.parametrize(
    "fault",
    ("model-cache", "full-split", "selection-parent", "selection-schema"),
)
def test_status_rejects_self_consistent_wrong_complete_lineage(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    fault: str,
) -> None:
    """Would fail if self-hashed evidence could sever full Stage A lineage."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, training, selection = _complete_status_volume(plan)
    arm = plan.jobs[0].arm
    if fault == "model-cache":
        receipt = training[arm]
        receipt["model_cache_artifact_id"] = "d" * 64
        unsigned = dict(receipt)
        unsigned.pop("artifact_id")
        receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
        volume.files[
            f"/runs/{plan.run_id}/receipts/canonical/train/{arm}.json"
        ] = (canonical_json(receipt) + "\n").encode("utf-8")
    elif fault == "full-split":
        receipt = training[arm]
        manifest_path = next(
            path
            for path in volume.files
            if path.endswith(f"/checkpoints/pilot/seed-42/{arm}/run-manifest.json")
        )
        manifest = json.loads(volume.files[manifest_path])
        manifest["data_parent_hashes"] = [plan.split_artifact_id[:8] + "d" * 56]
        _replace_canonical_output_and_rebind_receipt(
            volume.files,
            receipt,
            output_path=manifest_path,
            content=(canonical_json(manifest) + "\n").encode("utf-8"),
        )
    else:
        receipt = selection[arm]
        manifest_path = next(
            path
            for path in volume.files
            if path.endswith(
                f"/checkpoint-selections/pilot/seed-42/{arm}/manifest.json"
            )
        )
        manifest = json.loads(volume.files[manifest_path])
        if fault == "selection-parent":
            wrong_parent = "d" * 64
            manifest["training_manifest_hash"] = wrong_parent
            manifest["parent_hashes"][2] = wrong_parent
        else:
            manifest["unapproved_extra"] = "self-hashed"
        manifest.pop("artifact_id")
        manifest["artifact_id"] = modal_artifacts.sha256_json(manifest)
        _replace_canonical_output_and_rebind_receipt(
            volume.files,
            receipt,
            output_path=manifest_path,
            content=(canonical_json(manifest) + "\n").encode("utf-8"),
        )

    result = imported_adapter.status_local(volume, run_id=plan.run_id)

    expected_stage = "selection" if fault.startswith("selection") else "training"
    assert result[expected_stage][arm] == "invalid"
    assert result["valid"] is False


def _failed_attempt_payload(
    plan: modal_plan.PilotPlan, job: modal_plan.PilotJob,
) -> dict[str, object]:
    receipt = _stage_a_receipt(plan, job, "train")
    receipt.update(
        exit_status=1,
        validated=False,
        promoted=False,
        failure_reason="RuntimeError: failed attempt",
        failure_stage="command",
    )
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    return receipt


@pytest.mark.parametrize(
    "fault", ("malformed", "hash", "filename", "command", "shared-identity")
)
def test_status_reports_every_invalid_attempt_receipt(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    fault: str,
) -> None:
    """Would fail if malformed or unapproved attempts disappeared as pending."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    receipt = _failed_attempt_payload(plan, plan.jobs[0])
    filename = f"{receipt['attempt_id']}.json"
    content: bytes
    if fault == "malformed":
        content = b"not json"
    else:
        if fault == "hash":
            receipt["failure_reason"] = "changed without rehash"
        elif fault == "filename":
            filename = "different-attempt.json"
        elif fault == "command":
            receipt["command"] = "./.venv/bin/python -m phase_marker.training train --arm random"
            receipt["command_hash"] = hashlib.sha256(
                str(receipt["command"]).encode("utf-8")
            ).hexdigest()
        elif fault == "shared-identity":
            receipt["model_cache_artifact_id"] = "d" * 64
        if fault not in {"hash", "filename"}:
            unsigned = dict(receipt)
            unsigned.pop("artifact_id")
            receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
        content = (canonical_json(receipt) + "\n").encode("utf-8")
    files, _canonical = _canonical_stage_a_files(plan, plan.jobs[1], "train")
    files[f"/runs/{plan.run_id}/receipts/attempts/{filename}"] = content

    result = imported_adapter.status_local(StageARunsClient(files, []), run_id=plan.run_id)

    assert result["valid"] is False
    assert any("attempt" in error for error in result["errors"])


@pytest.mark.parametrize(
    "fault", ("a100", "missing-gpu", "missing-output", "arbitrary-output")
)
def test_status_rejects_invalid_successful_attempt_execution_evidence(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    fault: str,
) -> None:
    """Would fail if an unapproved successful execution looked trustworthy."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    receipt = _stage_a_receipt(plan, plan.jobs[0], "train")
    if fault == "a100":
        receipt["observed_gpu"] = "NVIDIA A100-SXM4-80GB"
    elif fault == "missing-gpu":
        receipt["observed_gpu"] = None
    elif fault == "missing-output":
        index = receipt["expected_outputs"].index("run-manifest.json")
        receipt["expected_outputs"].pop(index)
        receipt["output_hashes"].pop(index)
    else:
        receipt["expected_outputs"] = ["arbitrary-output.bin"]
        receipt["output_hashes"] = ["d" * 64]
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    filename = f"{receipt['attempt_id']}.json"
    files = {
        f"/runs/{plan.run_id}/receipts/attempts/{filename}": (
            canonical_json(receipt) + "\n"
        ).encode("utf-8")
    }
    files.update(_status_dependency_files(plan))

    result = imported_adapter.status_local(
        StageARunsClient(files, []), run_id=plan.run_id
    )

    assert result["valid"] is False
    assert any("attempt" in error for error in result["errors"])


def test_status_accepts_successful_h200_attempt_execution_evidence(
    imported_adapter: ModuleType,
    pilot_repo: Path,
) -> None:
    """Would fail if the approved H200-compatible boundary were narrowed to H100."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    receipt = _stage_a_receipt(plan, plan.jobs[0], "train")
    receipt["observed_gpu"] = "NVIDIA H200 NVL"
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    filename = f"{receipt['attempt_id']}.json"
    files = {
        f"/runs/{plan.run_id}/receipts/attempts/{filename}": (
            canonical_json(receipt) + "\n"
        ).encode("utf-8")
    }
    files.update(_status_dependency_files(plan))

    result = imported_adapter.status_local(
        StageARunsClient(files, []), run_id=plan.run_id
    )

    assert result["valid"] is True


@pytest.mark.parametrize("fault", ("wrong-source", "a100"))
def test_status_rejects_invalid_lone_failed_attempt_identity(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    fault: str,
) -> None:
    """Would fail if a lone failed attempt escaped run and GPU binding."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    receipt = _failed_attempt_payload(plan, plan.jobs[0])
    if fault == "wrong-source":
        receipt["source_hash"] = "d" * 64
    else:
        receipt["observed_gpu"] = "NVIDIA A100-SXM4-80GB"
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    filename = f"{receipt['attempt_id']}.json"
    files = {
        f"/runs/{plan.run_id}/receipts/attempts/{filename}": (
            canonical_json(receipt) + "\n"
        ).encode("utf-8")
    }

    result = imported_adapter.status_local(
        StageARunsClient(files, []), run_id=plan.run_id
    )

    assert result["valid"] is False
    assert any("attempt" in error for error in result["errors"])


@pytest.mark.parametrize(
    "observed_gpu", (None, "NVIDIA H100 80GB HBM3", "NVIDIA H200 NVL")
)
def test_status_accepts_approved_failed_attempt_gpu_evidence(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    observed_gpu: str | None,
) -> None:
    """Would fail if failed attempts required hardware they may not have observed."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    receipt = _failed_attempt_payload(plan, plan.jobs[0])
    receipt["observed_gpu"] = observed_gpu
    unsigned = dict(receipt)
    unsigned.pop("artifact_id")
    receipt["artifact_id"] = modal_artifacts.sha256_json(unsigned)
    filename = f"{receipt['attempt_id']}.json"
    files = {
        f"/runs/{plan.run_id}/receipts/attempts/{filename}": (
            canonical_json(receipt) + "\n"
        ).encode("utf-8")
    }
    files.update(_status_dependency_files(plan))

    result = imported_adapter.status_local(
        StageARunsClient(files, []), run_id=plan.run_id
    )

    assert result["valid"] is True
    assert result["training"][plan.jobs[0].arm] == "failed"


class MutatingDownloadVolume(StageARunsClient):
    def __init__(self, files: dict[str, bytes], target: str) -> None:
        super().__init__(files, [])
        self.target = target
        self.target_reads = 0

    def read_file(self, path: str) -> list[bytes]:
        content = super().read_file(path)
        if path == self.target:
            self.target_reads += 1
            if self.target_reads > 1:
                return [content[0] + b"mutated-after-status"]
        return content


class RelistMutatingDownloadVolume(StageARunsClient):
    def __init__(self, files: dict[str, bytes], run_root: str) -> None:
        super().__init__(files, [])
        self.run_root = run_root
        self.run_root_lists = 0

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        entries = super().listdir(path, recursive=recursive)
        if path == self.run_root:
            self.run_root_lists += 1
            if self.run_root_lists >= 3:
                entries.append(
                    SimpleNamespace(
                        path=f"{self.run_root}/receipts/attempts/{'f' * 64}.json",
                        type="file",
                    )
                )
        return entries


class LateDestinationDownloadVolume(StageARunsClient):
    def __init__(
        self, files: dict[str, bytes], run_root: str, destination: Path,
    ) -> None:
        super().__init__(files, [])
        self.run_root = run_root
        self.destination = destination
        self.run_root_lists = 0

    def listdir(self, path: str, *, recursive: bool = False) -> list[SimpleNamespace]:
        entries = super().listdir(path, recursive=recursive)
        if path == self.run_root:
            self.run_root_lists += 1
            if self.run_root_lists == 3:
                self.destination.mkdir()
        return entries


def test_download_rereads_every_advertised_producer_byte_after_status(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if excluded weights could mutate between status and download."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    target = next(
        path for path in volume.files if path.endswith("adapter_model.safetensors")
    )
    mutating = MutatingDownloadVolume(volume.files, target)
    destination = tmp_path / "evidence"

    with pytest.raises(ValueError, match="producer bytes|changed during download"):
        imported_adapter.download_evidence_local(
            mutating, run_id=plan.run_id, destination=destination,
        )

    assert mutating.target_reads >= 2
    assert not destination.exists()


def test_download_rejects_dangling_symlink_destination_without_replacing_it(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if exists() treated a dangling destination symlink as absent."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    destination = tmp_path / "evidence"
    missing = tmp_path / "missing-target"
    destination.symlink_to(missing, target_is_directory=True)

    with pytest.raises(FileExistsError, match="destination already exists"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=destination,
        )

    assert destination.is_symlink()
    assert destination.readlink() == missing
    assert not missing.exists()


def test_download_atomic_publish_refuses_late_created_empty_destination(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if final publication replaced a concurrently created directory."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    destination = tmp_path / "evidence"
    run_root = f"/runs/{plan.run_id}"
    late = LateDestinationDownloadVolume(volume.files, run_root, destination)

    with pytest.raises(FileExistsError, match="destination already exists"):
        imported_adapter.download_evidence_local(
            late, run_id=plan.run_id, destination=destination,
        )

    assert late.run_root_lists == 3
    assert destination.is_dir()
    assert list(destination.iterdir()) == []


def test_download_relists_complete_allowlist_immediately_before_publication(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if a late allowlisted file escaped the validated snapshot."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    run_root = f"/runs/{plan.run_id}"
    mutating = RelistMutatingDownloadVolume(volume.files, run_root)
    destination = tmp_path / "evidence"

    with pytest.raises(ValueError, match="allowlist changed during download"):
        imported_adapter.download_evidence_local(
            mutating, run_id=plan.run_id, destination=destination,
        )

    assert mutating.run_root_lists >= 3
    assert not destination.exists()


def test_download_rejects_self_hashed_smoke_receipt_with_extra_schema(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if arbitrary self-hashed JSON were accepted as a smoke receipt."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, training, _selection = _complete_status_volume(plan)
    _files, _manifest, smoke = _stage_a_test_dependency_evidence(plan)
    referenced_id = str(
        training[plan.jobs[0].arm]["smoke_receipt_artifact_id"]
    )
    smoke["arbitrary"] = "must not export"
    smoke.pop("artifact_id")
    smoke["artifact_id"] = modal_artifacts.sha256_json(smoke)
    volume.files[
        f"/runs/{plan.run_id}/receipts/smoke/{referenced_id}.json"
    ] = (canonical_json(smoke) + "\n").encode("utf-8")

    with pytest.raises(ValueError, match="validated complete Stage A evidence"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=tmp_path / "evidence",
        )


@pytest.mark.parametrize(
    "fault", ("empty", "missing", "duplicate", "wrong")
)
def test_download_rejects_smoke_receipt_without_exact_locked_imports(
    imported_adapter: ModuleType,
    pilot_repo: Path,
    tmp_path: Path,
    fault: str,
) -> None:
    """Would fail if smoke import evidence did not match the locked runtime exactly."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, training, _selection = _complete_status_volume(plan)
    imports = [
        {"module": module, "version": "locked-test-version"}
        for module in EXPECTED_LOCKED_RUNTIME_IMPORTS
    ]
    if fault == "empty":
        imports = []
    elif fault == "missing":
        imports.pop()
    elif fault == "duplicate":
        imports.append(dict(imports[-1]))
    else:
        imports[-1] = {"module": "unapproved_runtime", "version": "1.0"}
    _files, _manifest, smoke = _stage_a_test_dependency_evidence(plan)
    referenced_id = str(
        training[plan.jobs[0].arm]["smoke_receipt_artifact_id"]
    )
    smoke["imports"] = imports
    smoke.pop("artifact_id")
    smoke["artifact_id"] = modal_artifacts.sha256_json(smoke)
    volume.files[
        f"/runs/{plan.run_id}/receipts/smoke/{referenced_id}.json"
    ] = (canonical_json(smoke) + "\n").encode("utf-8")

    with pytest.raises(ValueError, match="validated complete Stage A evidence"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=tmp_path / "evidence",
        )


def _add_downloadable_logs(
    volume: StageARunsClient, plan: modal_plan.PilotPlan,
) -> None:
    for stage in ("train", "selection"):
        for job in plan.jobs:
            canonical_path = (
                f"/runs/{plan.run_id}/receipts/canonical/{stage}/{job.arm}.json"
            )
            receipt = json.loads(volume.files[canonical_path])
            attempt_id = str(receipt["attempt_id"])
            volume.files[
                f"/runs/{plan.run_id}/receipts/attempts/{attempt_id}.json"
            ] = volume.files[canonical_path]
            volume.files[
                f"/runs/{plan.run_id}/attempts/{attempt_id}/logs/{stage}.log"
            ] = f"{stage}:{job.arm}\n".encode("utf-8")


def test_download_evidence_is_atomic_and_strictly_allowlisted(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if weights, credentials, cache bytes, or arbitrary files escaped."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    _add_downloadable_logs(volume, plan)
    run_root = f"/runs/{plan.run_id}"
    volume.files[f"{run_root}/.modal.toml"] = b"token = 'secret'\n"
    volume.files[f"{run_root}/arbitrary.bin"] = b"not evidence"
    volume.files[f"{run_root}/model-cache/raw.bin"] = b"model"
    destination = tmp_path / "stage-a-evidence"

    downloaded = imported_adapter.download_evidence_local(
        volume, run_id=plan.run_id, destination=destination,
    )

    assert destination.is_dir()
    assert len(downloaded) == 63
    assert (destination / "provenance/input-bundle-manifest.json").is_file()
    relative = {path.relative_to(destination).as_posix() for path in downloaded}
    assert "stage-a-summary.json" in relative
    assert sum(path.endswith("run-manifest.json") for path in relative) == 6
    assert sum(path.endswith("adapter_config.json") for path in relative) == 6
    assert sum(
        path.endswith("manifest.json") and "checkpoint-selections" in path
        for path in relative
    ) == 6
    assert sum(path.endswith("evidence.jsonl") for path in relative) == 6
    assert sum(path.endswith(".log") for path in relative) == 12
    assert sum(path.startswith("receipts/canonical/") for path in relative) == 12
    assert sum(path.startswith("receipts/attempts/") for path in relative) == 12
    assert sum(path.startswith("receipts/smoke/") for path in relative) == 1
    assert not any("adapter_model" in path or "model-cache" in path for path in relative)
    assert ".modal.toml" not in relative
    assert "arbitrary.bin" not in relative
    assert all(path.is_file() for path in downloaded)


def test_download_exports_only_summary_referenced_successful_smoke_receipt(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if unrelated smoke history polluted the evidence export."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, training, _selection = _complete_status_volume(plan)
    _files, _manifest, unrelated = _stage_a_test_dependency_evidence(plan)
    unrelated["validated"] = False
    unrelated["failure_reason"] = "RuntimeError: unrelated failed smoke"
    unrelated.pop("artifact_id")
    unrelated["artifact_id"] = modal_artifacts.sha256_json(unrelated)
    unrelated_id = str(unrelated["artifact_id"])
    volume.files[
        f"/runs/{plan.run_id}/receipts/smoke/{unrelated_id}.json"
    ] = (canonical_json(unrelated) + "\n").encode("utf-8")

    destination = tmp_path / "evidence"
    downloaded = imported_adapter.download_evidence_local(
        volume, run_id=plan.run_id, destination=destination,
    )

    referenced_id = str(
        training[plan.jobs[0].arm]["smoke_receipt_artifact_id"]
    )
    exported_smoke = {
        path.relative_to(destination).as_posix()
        for path in downloaded
        if path.relative_to(destination).as_posix().startswith("receipts/smoke/")
    }
    assert exported_smoke == {f"receipts/smoke/{referenced_id}.json"}
    assert not (destination / f"receipts/smoke/{unrelated_id}.json").exists()


def test_download_evidence_rejects_an_unreceipted_log_escape(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if an arbitrary file under a log-shaped path were exportable."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    volume.files[
        f"/runs/{plan.run_id}/attempts/not-a-receipted-attempt/logs/train.log"
    ] = b"arbitrary bytes"

    destination = tmp_path / "evidence"
    with pytest.raises(ValueError, match="log lacks its bound attempt receipt"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=destination,
        )
    assert not destination.exists()


def test_download_evidence_refuses_existing_destination_and_invalid_status(
    imported_adapter: ModuleType, pilot_repo: Path, tmp_path: Path,
) -> None:
    """Would fail if download overwrote local data or exported corrupt evidence."""
    plan = _build_plan(pilot_repo, modal_plan._file_sha256(pilot_repo / LOCK_PATH.name))
    volume, _training, _selection = _complete_status_volume(plan)
    destination = tmp_path / "existing"
    destination.mkdir()
    sentinel = destination / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="destination already exists"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=destination,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"

    destination = tmp_path / "corrupt"
    receipt_path = next(
        path for path in volume.files
        if path.endswith("/receipts/canonical/train/semantic.json")
    )
    volume.files[receipt_path] += b"corrupt"
    with pytest.raises(ValueError, match="validated complete Stage A evidence"):
        imported_adapter.download_evidence_local(
            volume, run_id=plan.run_id, destination=destination,
        )
    assert not destination.exists()
