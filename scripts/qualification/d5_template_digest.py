#!/usr/bin/env python
"""Canonical template-hit and template-feature digests of D5 on the CPU (contract-6, protocol E5).

Usage:
    python scripts/qualification/d5_template_digest.py --out OUT.json [--row D5] [--seed 101]

Run from the repository root on the CPU, with ``OPENDDE_ROOT_DIR``,
``OPENDDE_QUAL_DATA`` and ``OPENDDE_QUAL_KALIGN`` set as for a qualification
run (the private asset copy). It builds the row's inference configuration the
way ``pred`` does (``run_rows.build_command`` and the row's overrides, with
the template cache in a fresh directory), featurizes the row's job with
``InferenceDataset.process_one`` after seeding as inference does, and runs
no model. It records:

- (a) the template-hit stage: for every ``TemplateHitFeaturizer.get_templates``
  call, the query and the selected realigned hits (name, aligned columns,
  sum of probabilities, aligned query and hit strings, query and hit
  indices), canonical JSON hashed with SHA-256;
- (b) the model-consumed template features: every ``template*`` entry of the
  input feature dict, by key in sorted order, with dtype, shape and the
  SHA-256 of its contiguous bytes, and one digest over all of them.

Run it twice on one host to check both digests are stable there, then on
another host to compare them. A same-host repeat cannot detect differences
between CPUs; equal digests show equality of these measured outputs for the
frozen input in the tested environments, not general determinism. If (b)
differs, ``per_key`` localizes it.
"""

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

import provenance
import run_rows

MANIFEST = Path("scripts/qualification/inputs/manifest.json")


def _canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def _apply(configs, path: list[str], value) -> None:
    node = configs
    for part in path[:-1]:
        node = node[part] if isinstance(node, dict) else getattr(node, part)
    if isinstance(node, dict):
        node[path[-1]] = value
    else:
        setattr(node, path[-1], value)


def _last(args, name):
    found = [i for i, a in enumerate(args[:-1]) if a == name]
    return args[found[-1] + 1] if found else None


def build_configs(row: dict, manifest: dict, out: Path):
    from opendde.config.inference import build_inference_config
    from opendde.config.model_registry import DEFAULT_MODEL_NAME
    from opendde.config.schema import OpenDDEConfig
    from opendde.data.tools import kalign

    overrides, cli, _ = run_rows.build_command(row, manifest, out, "101")
    configs = build_inference_config(
        model_name=DEFAULT_MODEL_NAME, fill_required_with_null=True
    )
    for text in overrides:
        key, _, raw = text.partition("=")
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        _apply(configs, key.split("."), value)
    for flag in ("use_msa", "use_template", "use_rna_msa"):
        value = _last(cli, f"--{flag}")
        setattr(configs, flag, value is not None and value.lower() == "true")
    configs = OpenDDEConfig.model_validate(configs.model_dump())
    binary = _last(cli, "--kalign_binary_path")
    if configs.use_template:
        if binary is None or not (
            Path(binary).is_file() and os.access(binary, os.X_OK)
        ):
            raise SystemExit(
                f"error: --kalign_binary_path must be executable: {binary}"
            )
        configs.data.template.kalign_binary_path = kalign.resolve_kalign_binary(binary)
    configs.input_json_path = row["input"]
    configs.dump_dir = str(out)
    return configs, cli


def tensor_digest(value) -> dict:
    array = (
        value.detach().cpu().contiguous().numpy()
        if torch.is_tensor(value)
        else np.ascontiguousarray(value)
    )
    return {
        "dtype": str(array.dtype),
        "shape": list(array.shape),
        "sha256": hashlib.sha256(array.tobytes()).hexdigest(),
    }


def measure(row_id: str, seed: int) -> dict:
    from opendde.data.inference.infer_dataloader import InferenceDataset
    from opendde.data.template import template_utils
    from opendde.utils.seed import seed_everything

    manifest = json.loads(MANIFEST.read_text())
    row = {r["id"]: r for r in manifest["rows"]}[row_id]
    out = Path(tempfile.mkdtemp(prefix="d5-template-")) / "out"
    configs, cli = build_configs(row, manifest, out)
    source = Path(row["input"]).read_bytes()
    _, jobs = provenance.effective_input(source, configs.use_msa)
    job = [j for j in jobs if j.get("name") == row["job"]][0]

    calls = []
    original = template_utils.TemplateHitFeaturizer.get_templates

    def get_templates(self, sequence_uid, query_sequence, hits, max_template_date=None):
        result = original(self, sequence_uid, query_sequence, hits, max_template_date)
        search = result[0]
        calls.append(
            {
                "query": query_sequence,
                "hits": [
                    {k: v for k, v in dataclasses.asdict(h).items() if k != "index"}
                    for h in search.hits
                ],
                "n_errors": len(search.errors),
                "n_warnings": len(search.warnings),
            }
        )
        return result

    template_utils.TemplateHitFeaturizer.get_templates = get_templates
    try:
        dataset = InferenceDataset(configs=configs, inputs=[job])
        seed_everything(seed=seed, deterministic=True)
        data, _, _ = dataset.process_one(job)
    finally:
        template_utils.TemplateHitFeaturizer.get_templates = original
    feats = data["input_feature_dict"]
    per_key = {
        k: tensor_digest(feats[k]) for k in sorted(feats) if k.startswith("template")
    }
    hit_stage = [{k: v for k, v in c.items() if not k.startswith("n_")} for c in calls]
    template = configs.data.template
    return {
        "row": row_id,
        "seed": seed,
        "input": row["input"],
        "input_sha256": provenance.sha256_bytes(source),
        "hits_sha256": _canonical(hit_stage),
        "features_sha256": _canonical(per_key),
        "per_key": per_key,
        "hits": [
            {
                "query_length": len(c["query"]),
                "names": [h["name"] for h in c["hits"]],
                "errors": c["n_errors"],
                "warnings": c["n_warnings"],
            }
            for c in calls
        ],
        "environment": {
            "host": platform.node(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "kalign": {
                "path": template.kalign_binary_path,
                **provenance.file_state(template.kalign_binary_path),
            },
            "release_dates": provenance.file_state(template.release_dates_path),
            "obsolete_pdbs": provenance.file_state(template.obsolete_pdbs_path),
            "mmcif_dir": template.prot_template_mmcif_dir,
            "cli_args": cli,
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--row", default="D5")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = measure(args.row, args.seed)
    args.out.write_text(json.dumps(result, indent=1) + "\n")
    print(
        json.dumps(
            {k: result[k] for k in ("row", "hits_sha256", "features_sha256", "hits")},
            indent=1,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
