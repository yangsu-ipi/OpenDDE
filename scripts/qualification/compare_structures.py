#!/usr/bin/env python
"""Paired structure comparison of two qualification runs.

Usage:
    python scripts/qualification/compare_structures.py RUN_A RUN_B
        --rows D1,D2,... [--seed-pairs 101:101,102:102] [--manifest PATH]
        --out OUT.json

Run from the repository root with ``OPENDDE_ROOT_DIR`` set. Each run
directory holds ``<row>/<job>/seed_<s>/predictions`` written with
``sorted_by_ranking_score=false`` (``run_rows.py`` does this), so
``sample_k`` is original sample index k. Sample k of seed ``a`` in RUN_A is
compared with sample k of seed ``b`` in RUN_B for each ``a:b`` pair; the
default pairs every development seed with itself. RUN_A and RUN_B may be the
same directory for a seed-to-seed contrast.

Metrics follow ``qual_structures.py`` and the protocol's comparison rules.
Rows with ``comparison.exclude_constraint_window`` in the manifest exclude the
constraint residues and their neighbours from structure metrics.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from biotite.structure.io import pdbx
from qual_structures import build_reference, compare_pair
from scipy.stats import kendalltau

from opendde.data.inference.json_to_feature import SampleDictToFeatures

SCALAR_CONFIDENCE = ("ranking_score", "ptm", "iptm", "plddt", "gpde", "has_clash")


def load_coords(cif: Path, atom_array) -> np.ndarray:
    block = pdbx.CIFFile.read(str(cif)).block
    pred = pdbx.get_structure(block, model=1, use_author_fields=False)
    same = (
        len(pred) == len(atom_array)
        and np.array_equal(pred.chain_id, atom_array.chain_id)
        and np.array_equal(pred.res_id, atom_array.res_id)
        and np.array_equal(pred.atom_name, atom_array.atom_name)
        and np.array_equal(
            np.char.upper(pred.element.astype(str)),
            np.char.upper(atom_array.element.astype(str)),
        )
    )
    if not same:
        raise ValueError(f"{cif} does not match the featurized atom order")
    return pred.coord.astype(np.float64)


def _samples(pred_dir: Path, job: str) -> dict[int, Path]:
    return {
        int(p.stem.rsplit("_", 1)[1]): p for p in pred_dir.glob(f"{job}_sample_*.cif")
    }


def _confidence(pred_dir: Path, job: str, k: int) -> dict[str, float]:
    path = pred_dir / f"{job}_summary_confidence_sample_{k}.json"
    data = json.loads(path.read_text())
    return {
        key: float(data[key])
        for key in SCALAR_CONFIDENCE
        if isinstance(data.get(key), (int, float))
    }


def _is_guided(row: dict[str, Any]) -> bool:
    flags = row["flags"]
    return any(
        a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
    )


def compare_row(row, run_a: Path, run_b: Path, seed_pairs) -> list[dict[str, Any]]:
    jobs = json.loads(Path(row["input"]).read_text())
    window = (row.get("comparison") or {}).get("exclude_constraint_window")
    results = []
    for job in jobs:
        atom_array = SampleDictToFeatures(job, _is_guided(row)).get_atom_array()
        ref = build_reference(atom_array, job, exclude_window=window)
        for seed_a, seed_b in seed_pairs:
            dir_a = run_a / row["id"] / job["name"] / f"seed_{seed_a}" / "predictions"
            dir_b = run_b / row["id"] / job["name"] / f"seed_{seed_b}" / "predictions"
            sa, sb = _samples(dir_a, job["name"]), _samples(dir_b, job["name"])
            entry: dict[str, Any] = {
                "id": row["id"],
                "job": job["name"],
                "seeds": [seed_a, seed_b],
            }
            if not sa or sorted(sa) != sorted(sb):
                entry["status"] = (
                    f"fail: sample sets differ ({sorted(sa)} vs {sorted(sb)})"
                )
                results.append(entry)
                continue
            samples = []
            conf_a, conf_b = [], []
            for k in sorted(sa):
                pair = compare_pair(
                    ref, load_coords(sa[k], atom_array), load_coords(sb[k], atom_array)
                )
                ca = _confidence(dir_a, job["name"], k)
                cb = _confidence(dir_b, job["name"], k)
                pair["confidence_diff"] = {
                    key: abs(ca[key] - cb[key]) for key in ca.keys() & cb.keys()
                }
                pair["sample"] = k
                conf_a.append(ca.get("ranking_score"))
                conf_b.append(cb.get("ranking_score"))
                samples.append(pair)
            ranking = {}
            if None not in conf_a and None not in conf_b and len(conf_a) > 1:
                tau = kendalltau(conf_a, conf_b).statistic
                ranking = {
                    "top1_same": int(np.argmax(conf_a)) == int(np.argmax(conf_b)),
                    "kendall_tau": None if np.isnan(tau) else float(tau),
                }
            entry.update(
                {
                    "assignments": len(ref.assignments),
                    "anchor": ref.anchor_type,
                    "excluded_atoms": int(ref.exclude_mask.sum()),
                    "samples": samples,
                    "ranking": ranking,
                    "status": _worst(samples),
                }
            )
            results.append(entry)
    return results


def _worst(samples) -> str:
    statuses = [s["status"] for s in samples]
    for prefix in ("fail", "inconclusive"):
        hits = [s for s in statuses if s.startswith(prefix)]
        if hits:
            return hits[0]
    return "ok"


def summarize(entry: dict[str, Any]) -> dict[str, Any]:
    samples = entry.get("samples") or []
    out = {k: entry[k] for k in ("id", "job", "seeds", "status")}
    if not samples:
        return out

    def stat(values, fn):
        values = [v for v in values if v is not None]
        return fn(values) if values else None

    out["anchor_rmsd_max"] = stat([s.get("anchor_rmsd") for s in samples], max)
    out["polymer_rmsd_max"] = stat([s.get("polymer_rmsd") for s in samples], max)
    for name in samples[0].get("lddt", {}):
        out[f"lddt_{name}_min"] = stat(
            [s["lddt"][name]["score"] for s in samples if "lddt" in s], min
        )
    ligs = [lig for s in samples for lig in s.get("ligands", [])]
    if ligs:
        out["ligand_rmsd_max"] = max(lig["rmsd"] for lig in ligs)
        out["ligand_centroid_max"] = max(lig["centroid_shift"] for lig in ligs)
        out["pocket_jaccard_min"] = min(lig["pocket_jaccard"] for lig in ligs)
    contacts = [c for s in samples for c in s.get("contacts", [])]
    if contacts:
        out["contact_diff_max"] = max(c["diff"] for c in contacts)
    conf = [s["confidence_diff"].get("ranking_score") for s in samples]
    out["ranking_score_diff_max"] = stat(conf, max)
    out["ranking"] = entry.get("ranking")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_a", type=Path)
    parser.add_argument("run_b", type=Path)
    parser.add_argument("--rows", required=True)
    parser.add_argument("--seed-pairs")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    rows = {r["id"]: r for r in manifest["rows"]}
    if args.seed_pairs:
        seed_pairs = [
            tuple(int(x) for x in p.split(":")) for p in args.seed_pairs.split(",")
        ]
    else:
        seed_pairs = [(s, s) for s in manifest["development_seeds"]]
    entries = []
    for row_id in args.rows.split(","):
        entries.extend(compare_row(rows[row_id], args.run_a, args.run_b, seed_pairs))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {"run_a": str(args.run_a), "run_b": str(args.run_b), "entries": entries},
            indent=1,
            default=lambda v: v.item() if isinstance(v, np.generic) else str(v),
        )
    )
    summary = [summarize(e) for e in entries]
    print(json.dumps(summary, indent=1, default=str))
    return 0 if all(e["status"] == "ok" for e in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
