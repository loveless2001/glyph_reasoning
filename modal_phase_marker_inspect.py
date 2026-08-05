"""Read-only Modal entrypoints for inspecting existing phase-marker runs."""

from __future__ import annotations

from pathlib import Path

import modal

from phase_marker.io import canonical_json
from phase_marker.modal_inspection import download_evidence_local, status_local
from phase_marker.modal_plan import MODAL_ENVIRONMENT


APP_NAME = "phase-marker-pilot-stage-a-inspect"
RUNS_VOLUME_NAME = "phase-marker-pilot-runs-v1"

app = modal.App(
    APP_NAME,
    tags={"experiment": "phase-marker", "run-kind": "inspection"},
    include_source=False,
)
runs_volume = modal.Volume.from_name(
    RUNS_VOLUME_NAME,
    environment_name=MODAL_ENVIRONMENT,
    create_if_missing=False,
).read_only()


@app.local_entrypoint()
def status(run_id: str) -> None:
    """Print validated read-only status for one canonical run."""
    print(canonical_json(status_local(runs_volume, run_id=run_id)))


@app.local_entrypoint(name="download-evidence")
def download_evidence(run_id: str, destination: str) -> None:
    """Explicitly write one compact validated evidence bundle locally."""
    paths = download_evidence_local(
        runs_volume, run_id=run_id, destination=Path(destination),
    )
    print(canonical_json({
        "run_id": run_id,
        "destination": str(Path(destination)),
        "files": [str(path) for path in paths],
    }))
