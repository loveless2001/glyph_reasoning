from __future__ import annotations

import builtins
from dataclasses import replace
import hashlib
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
from phase_marker.modal_artifacts import build_input_bundle
import phase_marker.modal_plan as modal_plan
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
    assert gpu_options["volumes"]["/mnt/inputs"].volume is imported_adapter.inputs_volume
    assert gpu_options["volumes"]["/mnt/model"].volume is imported_adapter.model_volume
    assert gpu_options["volumes"]["/mnt/runs"] is imported_adapter.runs_volume
    assert gpu_options["volumes"]["/mnt/inputs"].read_only is True
    assert gpu_options["volumes"]["/mnt/model"].read_only is True

    assert set(imported_adapter.app.remote_functions) == {"gpu_resources", "status_resources"}
    assert isinstance(imported_adapter.gpu_resources, imported_adapter.RemoteFunction)
    assert isinstance(imported_adapter.status_resources, imported_adapter.RemoteFunction)
    assert [
        call[1] for call in fake.declaration_calls if call[0] == "local_entrypoint_decorated"
    ] == ["status"]


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
