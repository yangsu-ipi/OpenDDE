#!/usr/bin/env python
"""Measure and evaluate tier 2 gates for a candidate run against ``off``.

Usage:
    python scripts/qualification/tier2_gates.py measure OFF_RUN CAND_RUN
        --rows D1,... [--seed-pairs 101:101,102:102] --out MEASURE.json
    python scripts/qualification/tier2_gates.py evaluate MEASURE.json
        --limits LIMITS.json [--rows D1,...] [--out RESULT.json]

Run from the repository root with ``OPENDDE_ROOT_DIR`` and (for reference
rows) ``OPENDDE_QUAL_DATA`` set. Per row and job, ``measure`` records:

- the original sample indices found for each seed in both runs;
- paired samples (``compare_structures.compare_row``): outcome classification,
  same-outcome lDDT overall and per polymer type, p95 per-atom error,
  pairwise interface contacts in both directions, gated ligand poses,
  confidence differences and guidance status;
- ensembles, ``off`` samples of the first seed of each pair against candidate
  samples of the second: every sample's reference score (status, lDDT overall
  and per polymer type, Fnat, fnonnat and the lowest per-chain-pair Fnat)
  where a deposited structure exists, each sample's mean pLDDT,
  within-ensemble spread (median pairwise anchor RMSD, over all samples and
  within each seed), the closest pair of samples from one seed, and the
  normalized L1 distance between the two runs' interchain contact-frequency
  maps, keyed by entity and residue so that equivalent copies are
  interchangeable;
- for inputs with Zn, the paired Zn same-site count from ``zn_sites.py``.

``evaluate`` fails closed. The gates a job needs follow from its input and
manifest row (``job_facts``): polymer types, chain count, gated ligands, Zn,
guidance and reference. Every needed gate reports ``pass``, ``fail`` or
``n/a``; ``n/a`` is used only where a declared rule allows it (no
same-outcome pairs, too few matched Zn ions, a reference without
interchain contacts). An absent value, an undeclared limit, a missing class
assignment, a missing job or a sample count other than the one declared is a
failure. Quality gates are one-sided: a candidate may be better than
``off``. ipTM differences are reported, not gated.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from compare_structures import _confidence, _samples, compare_row, load_coords
from qual_structures import anchor_distance, build_reference, interchain_contacts
from reference_metrics import reference_map, row_reference, score_sample

from opendde.data.inference.json_to_feature import SampleDictToFeatures

SAME_OUTCOME = 2.0
GATED_LIGANDS = ("6OI", "ATR", "NAG", "PCG")
POLYMER_KEYS = {"proteinChain": "protein", "dnaSequence": "dna", "rnaSequence": "rna"}
REFERENCE_SCORES = ("fnat", "fnonnat", "fnat_min_pair")
NO_SAME = "no same-outcome pairs (count floor, C1)"


def _guided(row) -> bool:
    flags = row["flags"]
    return any(
        a == "--use_tfg_guidance" and b == "true" for a, b in zip(flags, flags[1:])
    )


def _finite(value) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def job_facts(row: dict[str, Any], job: dict[str, Any]) -> dict[str, Any]:
    """What a job's input and manifest row declare, independent of any run."""
    types, chains, ligands = set(), 0, {}
    for entity in job["sequences"]:
        ((kind, value),) = entity.items()
        count = int(value.get("count", 1))
        if kind in POLYMER_KEYS:
            types.add(POLYMER_KEYS[kind])
            chains += count
        elif kind in ("ligand", "ion"):
            code = str(value.get(kind, ""))
            code = code[4:] if code.startswith("CCD_") else code
            ligands[code] = ligands.get(code, 0) + count
    reference = row.get("reference")
    if isinstance(reference, dict):
        reference = reference.get(job["name"])
    return {
        "types": sorted(types),
        "multichain": chains > 1,
        "gated_ligands": sum(n for c, n in ligands.items() if c in GATED_LIGANDS),
        "zn": ligands.get("ZN", 0) > 0,
        "guided": _guided(row),
        "reference": bool(reference),
    }


def _load_set(run: Path, row_id: str, job: str, seeds, atoms):
    out, found = [], {}
    for seed in seeds:
        pred = run / row_id / job / f"seed_{seed}" / "predictions"
        files = _samples(pred, job)
        found[str(seed)] = sorted(files)
        for k, path in sorted(files.items()):
            out.append((seed, k, load_coords(path, atoms), _confidence(pred, job, k)))
    return out, found


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


def _spread(ref, samples) -> dict[str, float | None]:
    """Median pairwise anchor RMSD over all samples and over same-seed pairs,
    and the smallest same-seed pairwise anchor RMSD."""
    if len(samples) < 2:
        return {"all": None, "same_seed": None, "min_same_seed": None}
    d, same_seed = [], []
    for i in range(len(samples)):
        for j in range(i + 1, len(samples)):
            value = anchor_distance(ref, samples[i][2], samples[j][2])
            d.append(value)
            if samples[i][0] == samples[j][0]:
                same_seed.append(value)
    return {
        "all": float(np.median(d)),
        "same_seed": float(np.median(same_seed)) if same_seed else None,
        "min_same_seed": float(min(same_seed)) if same_seed else None,
    }


def _reference_scores(rmap, data) -> list[dict[str, Any]]:
    out = []
    for _, _, xyz, _ in data:
        s = score_sample(rmap, xyz)
        rec: dict[str, Any] = {"status": s["status"]}
        if s["status"] == "ok":
            rec["lddt"] = s["lddt"]
            for key in REFERENCE_SCORES:
                rec[key] = s.get(key)
        out.append(rec)
    return out


def _zn_summary(row, off_run: Path, cand_run: Path, seed_pairs, job_name: str):
    import zn_sites

    for entry in zn_sites.measure_row(row, off_run, cand_run, seed_pairs):
        if entry["job"] == job_name:
            s = zn_sites.summarize(entry)
            keys = (
                "same_outcome_samples",
                "same_site",
                "same_site_of",
                "same_site_fraction",
                "bound_off",
                "bound_cand",
            )
            return {k: s[k] for k in keys}
    return None


def measure_row(row, off_run: Path, cand_run: Path, seed_pairs) -> list[dict[str, Any]]:
    paired = compare_row(row, off_run, cand_run, seed_pairs)
    window = (row.get("comparison") or {}).get("exclude_constraint_window")
    seeds_a = sorted({a for a, _ in seed_pairs})
    seeds_b = sorted({b for _, b in seed_pairs})
    out = []
    for job in json.loads(Path(row["input"]).read_text()):
        atoms = SampleDictToFeatures(job, _guided(row)).get_atom_array()
        ref = build_reference(atoms, job, exclude_window=window)
        a_set, found_a = _load_set(off_run, row["id"], job["name"], seeds_a, atoms)
        b_set, found_b = _load_set(cand_run, row["id"], job["name"], seeds_b, atoms)
        samples = [
            s for e in paired if e["job"] == job["name"] for s in e.get("samples", [])
        ]
        statuses = [e["status"] for e in paired if e["job"] == job["name"]]
        entry: dict[str, Any] = {
            "id": row["id"],
            "job": job["name"],
            "paired_status": statuses,
            "pairs": samples,
            "samples_off": found_a,
            "samples_cand": found_b,
        }
        mask = ref.polymer_mask & ~ref.exclude_mask
        if len(set(atoms.chain_id[mask].tolist())) > 1 and a_set and b_set:
            fa = _contact_frequency(a_set, atoms, mask)
            fb = _contact_frequency(b_set, atoms, mask)
            total = sum(fa.values())
            l1 = sum(abs(fa.get(k, 0.0) - fb.get(k, 0.0)) for k in set(fa) | set(fb))
            entry["contact_freq_l1"] = l1 / total if total else None
        for tag, data in (("off", a_set), ("cand", b_set)):
            spread = _spread(ref, data)
            entry[f"spread_{tag}"] = spread["all"]
            entry[f"spread_same_seed_{tag}"] = spread["same_seed"]
            entry[f"min_pair_{tag}"] = spread["min_same_seed"]
            plddt = [c.get("plddt") for *_, c in data]
            entry[f"plddt_median_{tag}"] = (
                float(np.median(plddt))
                if plddt and all(_finite(v) for v in plddt)
                else None
            )
        reference = row_reference(row, job["name"])
        if reference:
            rmap = reference_map(
                atoms, reference, ref.exclude_mask if window is not None else None
            )
            for tag, data in (("off", a_set), ("cand", b_set)):
                entry[f"ref_{tag}"] = _reference_scores(rmap, data)
        if (atoms.res_name == "ZN").any():
            entry["zn"] = _zn_summary(row, off_run, cand_run, seed_pairs, job["name"])
        out.append(entry)
    return out


def _class_of(limits: dict[str, Any], row_id: str, job: str) -> str | None:
    rows = limits.get("rows", {})
    for name in (f"{row_id}/{job}", row_id):
        if name in rows and "class" in rows[name]:
            return rows[name]["class"]
    return None


def _limit(limits: dict[str, Any], row_id: str, job: str, key: str):
    """Row/job override, then row override, then class value, then a
    top-level value shared by every class."""
    rows = limits.get("rows", {})
    for name in (f"{row_id}/{job}", row_id):
        if name in rows and key in rows[name]:
            return rows[name][key]
    cls = _class_of(limits, row_id, job)
    value = limits.get("classes", {}).get(cls, {}).get(key)
    return limits.get(key) if value is None else value


class Gates:
    def __init__(self):
        self.gates: dict[str, dict[str, Any]] = {}

    def check(self, name, value, limit, ok):
        status = "pass" if ok else "fail"
        self.gates[name] = {"status": status, "value": value, "limit": limit}

    def fail(self, name, reason, value=None):
        self.gates[name] = {"status": "fail", "value": value, "reason": reason}

    def not_applicable(self, name, reason):
        self.gates[name] = {"status": "n/a", "reason": reason}

    def report(self, name, value):
        self.gates[name] = {"status": "report", "value": value}


def evaluate_entry(
    entry: dict[str, Any], limits: dict[str, Any], facts: dict[str, Any]
) -> dict[str, Any]:
    rid, job = entry["id"], entry["job"]
    g = Gates()
    cls = _class_of(limits, rid, job)
    if cls is None or cls not in limits.get("classes", {}):
        g.fail("limits_declared", f"no declared class for {rid}/{job}", value=cls)
        return g.gates
    g.check("limits_declared", cls, "declared", True)

    def lim(key):
        return _limit(limits, rid, job, key)

    def need(name, key):
        value = lim(key)
        if value is None:
            g.fail(name, f"limit {key} not declared")
        return value

    n_seeds, per_seed = lim("seeds"), lim("samples_per_seed")
    if n_seeds is None or per_seed is None:
        g.fail("sample_count", "seeds or samples_per_seed not declared")
        return g.gates
    for tag in ("off", "cand"):
        found = entry.get(f"samples_{tag}") or {}
        ok = len(found) == n_seeds and all(
            list(v) == list(range(per_seed)) for v in found.values()
        )
        g.check(f"sample_count_{tag}", found, [n_seeds, per_seed], ok)

    pairs = entry.get("pairs") or []
    statuses = entry.get("paired_status") or []
    g.check(
        "paired_conclusive",
        statuses,
        "ok",
        bool(statuses)
        and all(s == "ok" for s in statuses)
        and len(pairs) == n_seeds * per_seed,
    )
    if not all(_finite(p.get("anchor_rmsd")) for p in pairs):
        g.fail("anchor_rmsd", "missing or non-finite anchor RMSD")
    same = [
        p
        for p in pairs
        if _finite(p.get("anchor_rmsd")) and p["anchor_rmsd"] <= SAME_OUTCOME
    ]

    n_min = need("outcome_count", "n_min")
    if n_min is not None:
        g.check("outcome_count", len(same), n_min, len(same) >= n_min)
    _paired_structure_gates(g, lim, need, facts, same)
    _confidence_gates(g, lim, need, pairs, same)
    rule = lim("guidance")
    if facts["guided"]:
        if rule is None:
            g.fail("guidance", "guided row without a declared guidance rule")
        elif not pairs or any("guidance" not in p for p in pairs):
            g.fail("guidance", "guidance status absent")
        else:
            _guidance_gate(g, rule, [p["guidance"] for p in pairs])
    elif rule is not None:
        g.fail("guidance", "guidance rule declared for an unguided row")
    if facts["zn"]:
        _zn_gate(g, need, entry.get("zn"))
    if facts["reference"]:
        _reference_gates(g, entry, need, facts, n_seeds * per_seed)
    _ensemble_gates(g, need, entry, facts)
    return g.gates


def _paired_structure_gates(g, lim, need, facts, same) -> None:
    lddt_min = lim("lddt_min")
    for kind in ["polymer"] + facts["types"]:
        name = f"lddt_{kind}"
        if lddt_min is None:
            g.fail(name, "limit lddt_min not declared")
        elif not same:
            g.not_applicable(name, NO_SAME)
        else:
            vals = [p.get("lddt", {}).get(kind, {}).get("score") for p in same]
            if not all(_finite(v) for v in vals):
                g.fail(name, "lDDT absent or non-finite in a same-outcome pair")
            else:
                g.check(name, min(vals), lddt_min, min(vals) >= lddt_min)

    e_max = lim("p95_max")
    p95 = [p.get("atom_error", {}).get("p95") for p in same]
    if e_max is not None:
        if not same:
            g.not_applicable("p95_error", NO_SAME)
        elif not all(_finite(v) for v in p95):
            g.fail("p95_error", "p95 error absent in a same-outcome pair")
        else:
            g.check("p95_error", max(p95), e_max, max(p95) <= e_max)
    elif same and all(_finite(v) for v in p95):
        g.report("p95_error", max(p95))

    if facts["multichain"]:
        for name, key, field, lower in (
            ("pair_interface_fnat", "pair_fnat_min", "fnat", True),
            ("pair_interface_fnonnat", "pair_fnonnat_max", "fnonnat", False),
        ):
            limit = need(name, key)
            if limit is None:
                continue
            if not same:
                g.not_applicable(name, NO_SAME)
                continue
            vals = [(p.get("interface") or {}).get(field) for p in same]
            if not all(_finite(v) for v in vals):
                g.fail(name, f"interface {field} absent in a same-outcome pair")
            elif lower:
                g.check(name, min(vals), limit, min(vals) >= limit)
            else:
                g.check(name, max(vals), limit, max(vals) <= limit)

    if facts["gated_ligands"]:
        rmsd_max = need("ligand_rmsd", "ligand_rmsd_max")
        pocket_min = need("ligand_pocket", "ligand_pocket_min")
        if rmsd_max is None or pocket_min is None:
            return
        if not same:
            g.not_applicable("ligand_rmsd", NO_SAME)
            g.not_applicable("ligand_pocket", NO_SAME)
            return
        ligs = [[x for x in p.get("ligands", []) if x.get("gated")] for p in same]
        complete = all(
            len(x) == facts["gated_ligands"]
            and all(
                _finite(y.get("rmsd"))
                and _finite(y.get("pocket_jaccard"))
                and not y.get("truncated")
                for y in x
            )
            for x in ligs
        )
        if not complete:
            g.fail("ligand_rmsd", "gated ligand absent or inconclusive")
            g.fail("ligand_pocket", "gated ligand absent or inconclusive")
            return
        flat = [y for x in ligs for y in x]
        v = max(y["rmsd"] for y in flat)
        g.check("ligand_rmsd", v, rmsd_max, v <= rmsd_max)
        v = min(y["pocket_jaccard"] for y in flat)
        g.check("ligand_pocket", v, pocket_min, v >= pocket_min)


def _confidence_gates(g, lim, need, pairs, same) -> None:
    rs_max = need("ranking_score", "ranking_score_max")
    if rs_max is not None:
        scope = pairs if lim("ranking_all_pairs") else same
        vals = [(p.get("confidence_diff") or {}).get("ranking_score") for p in scope]
        if not scope:
            g.not_applicable("ranking_score", NO_SAME)
        elif not all(_finite(v) for v in vals):
            g.fail("ranking_score", "ranking-score difference absent")
        else:
            g.check("ranking_score", max(vals), rs_max, max(vals) <= rs_max)
    pl_max = need("mean_plddt", "plddt_max")
    if pl_max is not None:
        vals = [(p.get("confidence_diff") or {}).get("plddt") for p in same]
        if not same:
            g.not_applicable("mean_plddt", NO_SAME)
        elif not all(_finite(v) for v in vals):
            g.fail("mean_plddt", "mean pLDDT difference absent")
        else:
            g.check("mean_plddt", max(vals), pl_max, max(vals) <= pl_max)
    iptm = [(p.get("confidence_diff") or {}).get("iptm") for p in pairs]
    if iptm and all(_finite(v) for v in iptm):
        g.report("iptm", max(iptm))


def _guidance_gate(g: Gates, rule: dict[str, Any], guidance) -> None:
    a = [x["a"] for x in guidance]
    b = [x["b"] for x in guidance]
    kind = rule.get("kind")
    if kind == "contacts":
        if not all("contacts_ok" in x for x in a + b):
            g.fail("guidance_contacts", "contact status absent")
            return
        v = sum(x["contacts_ok"] for x in b)
        off = sum(x["contacts_ok"] for x in a)
        limit = rule["min"] if "min" in rule else off - rule["slack"]
        g.check("guidance_contacts", v, limit, v >= limit)
    elif kind == "epitope":
        if not all("epitope_ok" in x for x in a + b):
            g.fail("guidance_epitope", "epitope status absent")
            return
        v, off = sum(x["epitope_ok"] for x in b), sum(x["epitope_ok"] for x in a)
        g.check("guidance_epitope", v, off - rule["slack"], v >= off - rule["slack"])
    elif kind == "contact_mean":
        vals_a = [x.get("contact_mean") for x in a]
        vals_b = [x.get("contact_mean") for x in b]
        if not all(_finite(v) for v in vals_a + vals_b):
            g.fail("guidance_contact_mean", "constrained distances absent")
            return
        ma, mb = float(np.mean(vals_a)), float(np.mean(vals_b))
        ok = abs(mb - ma) <= rule["delta"] and mb <= rule.get("max", float("inf"))
        g.check("guidance_contact_mean", [ma, mb], [rule["delta"], rule.get("max")], ok)
    else:
        g.fail("guidance", f"unknown guidance rule {kind!r}")


def _zn_gate(g: Gates, need, zn) -> None:
    z_min = need("zn_same_site", "zn_same_site_min")
    z_matched = need("zn_same_site", "zn_min_matched")
    if z_min is None or z_matched is None:
        return
    if (
        not zn
        or not _finite(zn.get("same_site_of"))
        or not _finite(zn.get("same_site"))
    ):
        g.fail("zn_same_site", "Zn measurement absent")
    elif zn["same_site_of"] < z_matched:
        g.not_applicable(
            "zn_same_site",
            f"{zn['same_site_of']} matched Zn ions in same-outcome pairs, fewer "
            f"than {z_matched}: Zn placement not claimed",
        )
    else:
        v = zn["same_site"] / zn["same_site_of"]
        g.check("zn_same_site", v, z_min, v >= z_min)


def _drop(g, name, off_values, cand_values, limit, fn) -> None:
    """One-sided gate on ``fn(off) - fn(cand)``."""
    d = float(fn(off_values) - fn(cand_values))
    g.check(name, d, limit, d <= limit)


def _reference_gates(g: Gates, entry, need, facts, expected: int) -> None:
    runs = {}
    for tag in ("off", "cand"):
        scores = entry.get(f"ref_{tag}")
        ok = (
            isinstance(scores, list)
            and len(scores) == expected
            and all(
                s.get("status") == "ok"
                and all(
                    _finite((s.get("lddt") or {}).get(k))
                    for k in ["polymer"] + facts["types"]
                )
                for s in scores
            )
        )
        count = len(scores) if isinstance(scores, list) else None
        g.check(f"reference_complete_{tag}", count, expected, ok)
        runs[tag] = scores if ok else None
    if runs["off"] is None or runs["cand"] is None:
        return
    ra, rb = runs["off"], runs["cand"]
    for kind in ["polymer"] + facts["types"]:
        prefix = "ref" if kind == "polymer" else "ref_type"
        for stat, fn in (("median", np.median), ("lowest", min)):
            name = f"ensemble_ref_lddt_{kind}_{stat}"
            limit = need(name, f"{prefix}_lddt_{stat}_drop")
            if limit is not None:
                va = [s["lddt"][kind] for s in ra]
                vb = [s["lddt"][kind] for s in rb]
                _drop(g, name, va, vb, limit, fn)
    if not facts["multichain"]:
        return
    for field, key, lower_is_worse in (
        ("fnat", "ref_fnat_median_drop", True),
        ("fnonnat", "ref_fnonnat_median_rise", False),
        ("fnat_min_pair", "ref_fnat_min_pair_median_drop", True),
    ):
        name = f"ensemble_ref_{field}_median"
        va = [s.get(field) for s in ra]
        vb = [s.get(field) for s in rb]
        if all(v is None for v in va + vb):
            g.not_applicable(name, f"reference has no interchain contacts for {field}")
            continue
        limit = need(name, key)
        if limit is None:
            continue
        if not all(_finite(v) for v in va + vb):
            g.fail(name, f"reference {field} absent for some samples")
        elif lower_is_worse:
            _drop(g, name, va, vb, limit, np.median)
        else:
            _drop(g, name, vb, va, limit, np.median)


def _ensemble_gates(g: Gates, need, entry, facts) -> None:
    ratio = need("ensemble_spread_ratio", "spread_ratio")
    so, sc = entry.get("spread_off"), entry.get("spread_cand")
    if ratio is not None:
        if not (_finite(so) and _finite(sc)) or so <= 0:
            g.fail("ensemble_spread_ratio", "spread absent or zero", value=[so, sc])
        else:
            lo, hi = ratio
            g.check("ensemble_spread_ratio", sc / so, [lo, hi], lo <= sc / so <= hi)
    dup_min = need("no_duplicate_samples", "duplicate_min_anchor_rmsd")
    if dup_min is not None:
        v = entry.get("min_pair_cand")
        if not _finite(v):
            g.fail("no_duplicate_samples", "closest same-seed pair absent")
        else:
            g.check("no_duplicate_samples", v, dup_min, v >= dup_min)
    if facts["multichain"]:
        l1_max = need("ensemble_contact_frequency", "contact_freq_l1_max")
        v = entry.get("contact_freq_l1")
        if l1_max is not None:
            if not _finite(v):
                g.fail("ensemble_contact_frequency", "contact frequency absent")
            else:
                g.check("ensemble_contact_frequency", v, l1_max, v <= l1_max)
    pl_ens = need("ensemble_plddt_median", "plddt_median_delta")
    if pl_ens is not None:
        a, b = entry.get("plddt_median_off"), entry.get("plddt_median_cand")
        if not (_finite(a) and _finite(b)):
            g.fail("ensemble_plddt_median", "median pLDDT absent")
        else:
            d = abs(b - a)
            g.check("ensemble_plddt_median", d, pl_ens, d <= pl_ens)
    for tag in ("off", "cand"):
        v = entry.get(f"spread_same_seed_{tag}")
        if _finite(v):
            g.report(f"spread_same_seed_{tag}", v)


def _row_order(row_id: str):
    return (row_id[0], int(row_id[1:])) if row_id[1:].isdigit() else (row_id, 0)


def _result(row_id, job, gates):
    failed = [k for k, v in gates.items() if v["status"] == "fail"]
    return {
        "id": row_id,
        "job": job,
        "pass": not failed,
        "failed": failed,
        "gates": gates,
    }


def evaluate(data, limits, manifest, rows=None) -> list[dict[str, Any]]:
    """Evaluate every job of every requested row; a row or job without a
    measurement fails."""
    table = {r["id"]: r for r in manifest["rows"]}
    by_key = {(e["id"], e["job"]): e for e in data["entries"]}
    wanted = rows or sorted({e["id"] for e in data["entries"]}, key=_row_order)
    results = []
    for row_id in wanted:
        if row_id not in table:
            missing = {"status": "fail", "reason": "row not in manifest"}
            results.append(_result(row_id, "-", {"row": missing}))
            continue
        row = table[row_id]
        for job in json.loads(Path(row["input"]).read_text()):
            entry = by_key.get((row_id, job["name"]))
            if entry is None:
                gates = {"measured": {"status": "fail", "reason": "job not measured"}}
            else:
                gates = evaluate_entry(entry, limits, job_facts(row, job))
            results.append(_result(row_id, job["name"], gates))
    return results


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    m = sub.add_parser("measure")
    m.add_argument("off_run", type=Path)
    m.add_argument("cand_run", type=Path)
    m.add_argument("--rows", required=True)
    m.add_argument("--seed-pairs", default="101:101,102:102")
    e = sub.add_parser("evaluate")
    e.add_argument("measure", type=Path)
    e.add_argument("--limits", type=Path, required=True)
    e.add_argument("--rows", help="rows that must be present (default: all measured)")
    for p in (m, e):
        p.add_argument(
            "--manifest",
            type=Path,
            default=Path("scripts/qualification/inputs/manifest.json"),
        )
        p.add_argument("--out", type=Path, required=p is m)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    if args.command == "measure":
        rows = {r["id"]: r for r in manifest["rows"]}
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
    rows = args.rows.split(",") if args.rows else None
    results = evaluate(data, limits, manifest, rows)
    for r in results:
        status = "PASS" if r["pass"] else "FAIL " + ",".join(r["failed"])
        print(f"{r['id']:4s} {r['job'][:20]:20s} {status}")
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
