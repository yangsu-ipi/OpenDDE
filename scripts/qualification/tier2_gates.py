#!/usr/bin/env python
"""Measure and evaluate tier 2 gates for a candidate run against ``off``.

Usage:
    python scripts/qualification/tier2_gates.py measure OFF_RUN CAND_RUN
        --rows D1,... [--seed-pairs 101:101,102:102] --out MEASURE.json
    python scripts/qualification/tier2_gates.py evaluate MEASURE.json
        --limits LIMITS.json [--out RESULT.json]

Run ``measure`` from the repository root with ``OPENDDE_ROOT_DIR`` and (for
reference rows) ``OPENDDE_QUAL_DATA`` set. Per row and job it records:

- paired samples (``compare_structures.compare_row``): outcome classification,
  same-outcome lDDT overall and per polymer type, p95 per-atom error,
  pairwise interface contacts, gated ligand poses, confidence differences and
  guidance status;
- ensembles, ``off`` samples of the first seed of each pair against candidate
  samples of the second: reference lDDT and Fnat medians and minima where a
  deposited structure exists, median mean pLDDT, within-ensemble spread
  (median pairwise anchor RMSD), the closest pair of samples from one seed
  (a collapsed ensemble repeats samples), and the normalized L1 distance between the
  two runs' interchain contact-frequency maps, keyed by entity and residue so
  that equivalent copies are interchangeable.

``evaluate`` applies a limits file (class defaults plus per-row overrides) and
reports every gate's value, limit and verdict. Quality gates are one-sided:
a candidate may be better than ``off``.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from compare_structures import _confidence, _samples, compare_row, load_coords
from qual_structures import anchor_distance, build_reference, interchain_contacts
from reference_metrics import reference_map, row_reference, score_sample

from opendde.data.inference.json_to_feature import SampleDictToFeatures

SAME_OUTCOME = 2.0
GATED_LIGANDS = ("6OI", "ATR", "NAG", "PCG")


def _guided(row) -> bool:
    flags = row["flags"]
    return any(
        a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
    )


def _load_set(run: Path, row_id: str, job: str, seeds, atoms):
    out = []
    for seed in seeds:
        pred = run / row_id / job / f"seed_{seed}" / "predictions"
        for k, path in sorted(_samples(pred, job).items()):
            out.append((seed, k, load_coords(path, atoms), _confidence(pred, job, k)))
    return out


def _contact_frequency(samples, atoms, mask) -> dict[tuple[str, str], float]:
    chain_id = np.asarray(atoms.chain_id)
    entity_key = np.array(
        [f"{e}:{r}" for e, r in zip(atoms.label_entity_id, atoms.res_id)]
    )
    residue_key = np.array([f"{c}:{r}" for c, r in zip(atoms.chain_id, atoms.res_id)])
    lookup = dict(zip(residue_key, entity_key))
    counts: dict[tuple[str, str], float] = {}
    for _, _, xyz, _ in samples:
        keys = set()
        for a, b in interchain_contacts(xyz, mask, chain_id, residue_key):
            ea, eb = lookup[a], lookup[b]
            keys.add((ea, eb) if ea < eb else (eb, ea))
        for key in keys:
            counts[key] = counts.get(key, 0.0) + 1.0
    return {k: v / len(samples) for k, v in counts.items()}


def _spread(ref, samples) -> tuple[float | None, float | None]:
    """Median pairwise anchor RMSD over all samples, and the smallest
    pairwise anchor RMSD between two samples of the same seed."""
    if len(samples) < 2:
        return None, None
    d, same_seed = [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            value = anchor_distance(ref, samples[i][2], samples[j][2])
            d.append(value)
            if samples[i][0] == samples[j][0]:
                same_seed.append(value)
    return float(np.median(d)), (float(min(same_seed)) if same_seed else None)


def measure_row(row, off_run: Path, cand_run: Path, seed_pairs) -> list[dict[str, Any]]:
    paired = compare_row(row, off_run, cand_run, seed_pairs)
    window = (row.get("comparison") or {}).get("exclude_constraint_window")
    seeds_a = sorted({a for a, _ in seed_pairs})
    seeds_b = sorted({b for _, b in seed_pairs})
    out = []
    for job in json.loads(Path(row["input"]).read_text()):
        atoms = SampleDictToFeatures(job, _guided(row)).get_atom_array()
        ref = build_reference(atoms, job, exclude_window=window)
        a_set = _load_set(off_run, row["id"], job["name"], seeds_a, atoms)
        b_set = _load_set(cand_run, row["id"], job["name"], seeds_b, atoms)
        samples = [
            s for e in paired if e["job"] == job["name"] for s in e.get("samples", [])
        ]
        statuses = [e["status"] for e in paired if e["job"] == job["name"]]
        entry: dict[str, Any] = {
            "id": row["id"],
            "job": job["name"],
            "paired_status": statuses,
            "pairs": samples,
            "n_off": len(a_set),
            "n_cand": len(b_set),
        }
        mask = ref.polymer_mask & ~ref.exclude_mask
        if len(set(atoms.chain_id[mask].tolist())) > 1 and a_set and b_set:
            fa = _contact_frequency(a_set, atoms, mask)
            fb = _contact_frequency(b_set, atoms, mask)
            total = sum(fa.values())
            l1 = sum(abs(fa.get(k, 0.0) - fb.get(k, 0.0)) for k in set(fa) | set(fb))
            entry["contact_freq_l1"] = l1 / total if total else None
        entry["spread_off"], entry["min_pair_off"] = _spread(ref, a_set)
        entry["spread_cand"], entry["min_pair_cand"] = _spread(ref, b_set)
        entry["plddt_median_off"] = (
            float(np.median([c["plddt"] for *_, c in a_set])) if a_set else None
        )
        entry["plddt_median_cand"] = (
            float(np.median([c["plddt"] for *_, c in b_set])) if b_set else None
        )
        reference = row_reference(row, job["name"])
        if reference and a_set and b_set:
            rmap = reference_map(
                atoms, reference, ref.exclude_mask if window is not None else None
            )
            for tag, data in (("off", a_set), ("cand", b_set)):
                scores = [score_sample(rmap, xyz) for _, _, xyz, _ in data]
                ok = [s for s in scores if s["status"] == "ok"]
                entry[f"ref_{tag}"] = {
                    "n_ok": len(ok),
                    "lddt": [s["lddt"]["polymer"] for s in ok],
                    "fnat": [s.get("fnat") for s in ok],
                }
        out.append(entry)
    return out


def _quantile(values, fn):
    values = [v for v in values if v is not None]
    return fn(values) if values else None


def _limit(limits: dict[str, Any], row_id: str, job: str, key: str):
    rows = limits.get("rows", {})
    for name in (f"{row_id}/{job}", row_id):
        if name in rows and key in rows[name]:
            return rows[name][key]
    cls = None
    for name in (f"{row_id}/{job}", row_id):
        if name in rows and "class" in rows[name]:
            cls = rows[name]["class"]
            break
    return limits.get("classes", {}).get(cls, {}).get(key)


def evaluate_entry(entry: dict[str, Any], limits: dict[str, Any]) -> dict[str, Any]:
    rid, job = entry["id"], entry["job"]
    lim = lambda key: _limit(limits, rid, job, key)  # noqa: E731
    gates: dict[str, Any] = {}

    def gate(name, value, limit, ok):
        gates[name] = {"value": value, "limit": limit, "pass": bool(ok)}

    pairs = entry["pairs"]
    gate(
        "paired_conclusive",
        entry["paired_status"],
        "ok",
        all(s == "ok" for s in entry["paired_status"]) and pairs,
    )
    if not pairs:
        return gates
    same = [p for p in pairs if p.get("anchor_rmsd", 1e9) <= SAME_OUTCOME]
    n_min = lim("n_min")
    if n_min is not None:
        gate("outcome_count", len(same), n_min, len(same) >= n_min)
    lddt_min = lim("lddt_min")
    if lddt_min is not None and same:
        v = min(p["lddt"]["polymer"]["score"] for p in same)
        gate("lddt_polymer", v, lddt_min, v >= lddt_min)
        for kind in ("protein", "dna", "rna"):
            vals = [p["lddt"][kind]["score"] for p in same if kind in p["lddt"]]
            if vals:
                gate(f"lddt_{kind}", min(vals), lddt_min, min(vals) >= lddt_min)
    e_max = lim("p95_max")
    if e_max is not None and same:
        v = max(p["atom_error"]["p95"] for p in same)
        gate("p95_error", v, e_max, v <= e_max)
    fnat_min = lim("pair_fnat_min")
    vals = [
        p["interface"]["fnat"]
        for p in same
        if p.get("interface", {}).get("fnat") is not None
    ]
    if fnat_min is not None and vals:
        gate("pair_interface_fnat", min(vals), fnat_min, min(vals) >= fnat_min)
    ligs = [g for p in same for g in p.get("ligands", []) if g.get("gated")]
    if ligs:
        v = max(g["rmsd"] for g in ligs)
        gate(
            "ligand_rmsd", v, limits["ligand_rmsd_max"], v <= limits["ligand_rmsd_max"]
        )
        v = min(g["pocket_jaccard"] for g in ligs)
        gate(
            "ligand_pocket",
            v,
            limits["ligand_pocket_min"],
            v >= limits["ligand_pocket_min"],
        )
    rs_max = lim("ranking_score_max")
    if rs_max is not None:
        scope = pairs if lim("ranking_all_pairs") else same
        vals = [p["confidence_diff"].get("ranking_score", 0.0) for p in scope]
        if vals:
            gate("ranking_score", max(vals), rs_max, max(vals) <= rs_max)
    pl_max = lim("plddt_max")
    if pl_max is not None and same:
        v = max(p["confidence_diff"].get("plddt", 0.0) for p in same)
        gate("mean_plddt", v, pl_max, v <= pl_max)
    guidance = [p["guidance"] for p in pairs if "guidance" in p]
    rule = lim("guidance")
    if rule and guidance:
        a = [g["a"] for g in guidance]
        b = [g["b"] for g in guidance]
        if rule["kind"] == "contacts":
            v = sum(x["contacts_ok"] for x in b)
            gate("guidance_contacts", v, rule["min"], v >= rule["min"])
        elif rule["kind"] == "epitope":
            v, off = sum(x["epitope_ok"] for x in b), sum(x["epitope_ok"] for x in a)
            gate("guidance_epitope", v, off - rule["slack"], v >= off - rule["slack"])
        elif rule["kind"] == "contact_mean":
            ma = float(np.mean([x["contact_mean"] for x in a]))
            mb = float(np.mean([x["contact_mean"] for x in b]))
            ok = abs(mb - ma) <= rule["delta"] and mb <= rule.get("max", float("inf"))
            gate(
                "guidance_contact_mean", [ma, mb], [rule["delta"], rule.get("max")], ok
            )
    if entry.get("ref_off") and entry.get("ref_cand"):
        ra, rb = entry["ref_off"], entry["ref_cand"]
        d_med = float(np.median(ra["lddt"]) - np.median(rb["lddt"]))
        gate(
            "ensemble_ref_lddt_median",
            d_med,
            lim("ref_lddt_median_drop"),
            d_med <= lim("ref_lddt_median_drop"),
        )
        d_min = float(min(ra["lddt"]) - min(rb["lddt"]))
        gate(
            "ensemble_ref_lddt_lowest",
            d_min,
            lim("ref_lddt_lowest_drop"),
            d_min <= lim("ref_lddt_lowest_drop"),
        )
        fa = [x for x in ra["fnat"] if x is not None]
        fb = [x for x in rb["fnat"] if x is not None]
        if fa and fb and lim("ref_fnat_median_drop") is not None:
            d = float(np.median(fa) - np.median(fb))
            gate(
                "ensemble_ref_fnat_median",
                d,
                lim("ref_fnat_median_drop"),
                d <= lim("ref_fnat_median_drop"),
            )
    if entry.get("spread_off") and entry.get("spread_cand") is not None:
        ratio = entry["spread_cand"] / entry["spread_off"]
        lo, hi = lim("spread_ratio") or (0.0, float("inf"))
        gate("ensemble_spread_ratio", ratio, [lo, hi], lo <= ratio <= hi)
    dup_min = limits.get("duplicate_min_anchor_rmsd")
    if dup_min is not None and entry.get("min_pair_cand") is not None:
        v = entry["min_pair_cand"]
        gate("no_duplicate_samples", v, dup_min, v >= dup_min)
    l1_max = lim("contact_freq_l1_max")
    if l1_max is not None and entry.get("contact_freq_l1") is not None:
        v = entry["contact_freq_l1"]
        gate("ensemble_contact_frequency", v, l1_max, v <= l1_max)
    pl_ens = lim("plddt_median_delta")
    if pl_ens is not None and entry.get("plddt_median_off") is not None:
        v = abs(entry["plddt_median_cand"] - entry["plddt_median_off"])
        gate("ensemble_plddt_median", v, pl_ens, v <= pl_ens)
    return gates


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("off_run", type=Path)
    m.add_argument("cand_run", type=Path)
    m.add_argument("--rows", required=True)
    m.add_argument("--seed-pairs", default="101:101,102:102")
    m.add_argument(
        "--manifest",
        type=Path,
        default=Path("scripts/qualification/inputs/manifest.json"),
    )
    m.add_argument("--out", type=Path, required=True)
    e = sub.add_parser("evaluate")
    e.add_argument("measure", type=Path)
    e.add_argument("--limits", type=Path, required=True)
    e.add_argument("--out", type=Path)
    args = parser.parse_args()
    if args.command == "measure":
        rows = {r["id"]: r for r in json.loads(args.manifest.read_text())["rows"]}
        seed_pairs = [
            tuple(int(x) for x in p.split(":")) for p in args.seed_pairs.split(",")
        ]
        entries = []
        for row_id in args.rows.split(","):
            entries.extend(
                measure_row(rows[row_id], args.off_run, args.cand_run, seed_pairs)
            )
            print(f"measured {row_id}", flush=True)
        for entry in entries:
            for pair in entry["pairs"]:
                for lig in pair.get("ligands", []):
                    lig["gated"] = lig.get("res_name") in GATED_LIGANDS
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {
                    "off": str(args.off_run),
                    "cand": str(args.cand_run),
                    "seed_pairs": seed_pairs,
                    "entries": entries,
                },
                indent=1,
                default=float,
            )
        )
        return 0
    data = json.loads(args.measure.read_text())
    limits = json.loads(args.limits.read_text())
    results = []
    for entry in data["entries"]:
        gates = evaluate_entry(entry, limits)
        failed = [k for k, g in gates.items() if not g["pass"]]
        results.append(
            {
                "id": entry["id"],
                "job": entry["job"],
                "pass": not failed,
                "failed": failed,
                "gates": gates,
            }
        )
        print(
            f"{entry['id']:4s} {entry['job'][:20]:20s} {'PASS' if not failed else 'FAIL ' + ','.join(failed)}"
        )
    if args.out:
        args.out.write_text(
            json.dumps(
                {"measure": str(args.measure), "results": results},
                indent=1,
                default=float,
            )
        )
    return 0 if all(r["pass"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
