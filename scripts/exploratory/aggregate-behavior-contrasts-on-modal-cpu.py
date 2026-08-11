"""Aggregate held-out behavior results on a Modal CPU next to the volume.

Streams records.jsonl inside the datacenter (no WAN transfer), verifies the
byte stream against the behavior receipt hash while reading, and returns only
compact per-cell accuracy aggregates.
"""
import modal

RUN = (
    "pilot-s42-cfg-f112fda5-split-3b15bcc3-src-173973c847e6-"
    "plan-a0412ddaa183dbab512417322a32ca534435535d8c0327e347370bee25f3c2a6"
)
app = modal.App("phase-marker-preliminary-analysis")
runs_volume = modal.Volume.from_name("phase-marker-pilot-runs-v1")
image = modal.Image.debian_slim(python_version="3.12")


@app.function(image=image, volumes={"/runs": runs_volume}, timeout=1800, cpu=4.0, memory=8192)
def aggregate() -> dict:
    import hashlib
    import json
    from collections import defaultdict

    base = f"/runs/runs/{RUN}/artifacts/phase-marker/raw-generations/pilot"
    receipt = json.load(open(f"/runs/runs/{RUN}/receipts/canonical/behavior.json"))
    expected_hash = receipt["output_hashes"]["records.jsonl"]

    digest = hashlib.sha256()
    cells = defaultdict(lambda: [0, 0])
    by_source = defaultdict(lambda: [0, 0])
    with open(f"{base}/records.jsonl", "rb") as handle:
        for raw in handle:
            digest.update(raw)
            row = json.loads(raw)
            decoding = row.get("decoding") if isinstance(row.get("decoding"), dict) else {}
            kind = row.get("evaluation_kind") or decoding.get("evaluation_kind")
            perturbation = row.get("perturbation") or decoding.get("perturbation")
            correct = 1 if row["score"]["correct"] else 0
            key = f"{kind}|{row['training_arm']}|{row['prompt_condition']}|{perturbation or '-'}"
            cells[key][0] += 1
            cells[key][1] += correct
            if kind == "primary":
                source_key = f"{row['training_arm']}|{row['prompt_condition']}|{row['source']}"
                by_source[source_key][0] += 1
                by_source[source_key][1] += correct
    if digest.hexdigest() != expected_hash:
        raise ValueError("records byte stream does not match the behavior receipt")
    return {
        "records_hash": digest.hexdigest(),
        "cells": dict(sorted(cells.items())),
        "primary_by_source": dict(sorted(by_source.items())),
    }


@app.local_entrypoint()
def main() -> None:
    import json

    print(json.dumps(aggregate.remote(), sort_keys=True))
