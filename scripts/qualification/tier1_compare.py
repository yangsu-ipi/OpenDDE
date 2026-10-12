#!/usr/bin/env python
"""Compare tier 1 captures of a reference run and another run.

Usage:
    python scripts/qualification/tier1_compare.py REF_DIR OTHER_DIR --out OUT.json
        [--rows D1,...]

Both directories come from ``run_rows.py --tier1-capture`` (``<row>/<job>/
seed_<s>/tier1.pt``). REF_DIR is the stock ``off`` capture. OTHER_DIR is a
capture of another configuration made with ``--tier1-replay REF_DIR``, so its
``head_fixed`` record is the confidence head run on the reference's recorded
inputs; REF_DIR itself may be passed as OTHER_DIR to check self-consistency.
With ``--rows`` only those rows' captures are compared, so a capture root that
also holds other rows gives entries for exactly the rows of one evidence set.
Comparing all fifteen development rows needs about 30 GB of host memory (on
gnode2, a CPU-only job with 16 CPUs and no ``--mem``); 8 CPUs ran out of
memory in the stand-in rehearsal.

Per row, job and seed it reports:
- input embedding, single and pair representations: relative Frobenius error,
  and the 99.9th percentile and maximum of the absolute error divided by the
  reference tensor's RMS (absolute values when that RMS is zero);
- distogram probabilities: the largest absolute difference, and the fraction
  of token pairs whose most likely bin changes, overall and among pairs whose
  reference top-two probability margin is at least each of ``MARGINS``;
- confidence head on fixed inputs: the largest absolute probability
  difference for pLDDT, PAE, PDE and resolved, the largest difference in
  per-atom expected pLDDT, and the largest pTM, ipTM and
  ``0.8 ipTM + 0.2 pTM`` differences over samples.

Each entry records the SHA-256 of the two captures it read (``ref_sha256``,
``other_sha256``), and the output records the checkout's commit and
uncommitted paths (``source``), so ``accept.py`` can match them with the
capturing runs' records.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import torch

import provenance

MARGINS = (0.01, 0.05, 0.1, 0.2)
TAIL_QUANTILE = 0.999
TENSORS = ("s_inputs", "s_trunk", "z_trunk")
HEADS = ("plddt", "pae", "pde", "resolved")


def tensor_error(ref: torch.Tensor, other: torch.Tensor) -> dict[str, float]:
    if ref.shape != other.shape:
        raise ValueError(f"shape {tuple(ref.shape)} vs {tuple(other.shape)}")
    a = ref.float().flatten()
    b = other.float().flatten()
    diff = (b - a).abs()
    norm = a.norm().item()
    rms = (a.square().mean().sqrt()).item()
    k = max(1, int(round(TAIL_QUANTILE * diff.numel())))
    tail = diff.kthvalue(k).values.item()
    scale = rms if rms > 0 else 1.0
    return {
        "rel_fro": (diff.norm().item() / norm) if norm > 0 else diff.norm().item(),
        "tail_over_rms": tail / scale,
        "max_over_rms": diff.max().item() / scale,
        "ref_rms": rms,
        "finite": bool(torch.isfinite(b).all()),
    }


def distogram_error(ref: torch.Tensor, other: torch.Tensor) -> dict[str, Any]:
    ref = ref.float()
    other = other.float()
    top2 = ref.topk(2, dim=-1).values
    margin = top2[..., 0] - top2[..., 1]
    changed = ref.argmax(dim=-1) != other.argmax(dim=-1)
    out: dict[str, Any] = {
        "max_abs_prob": (other - ref).abs().max().item(),
        "argmax_changed": changed.float().mean().item(),
    }
    for m in MARGINS:
        sel = margin >= m
        out[f"argmax_changed_margin_{m}"] = (changed & sel).float().sum().item() / max(
            1, sel.sum().item()
        )
    out["finite"] = bool(torch.isfinite(other).all())
    return out


def head_error(ref: dict[str, Any], other: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name in HEADS:
        pa = torch.softmax(ref[f"{name}_logits"].float(), dim=-1)
        pb = torch.softmax(other[f"{name}_logits"].float(), dim=-1)
        out[f"{name}_max_abs_prob"] = (pb - pa).abs().max().item()
        out[f"{name}_finite"] = bool(torch.isfinite(pb).all())
    pa = torch.softmax(ref["plddt_logits"].float(), dim=-1)
    pb = torch.softmax(other["plddt_logits"].float(), dim=-1)
    n_bins = pa.shape[-1]
    centers = (torch.arange(n_bins, dtype=torch.float32) + 0.5) / n_bins
    out["expected_plddt_max_abs"] = ((pb - pa) @ centers).abs().max().item()
    dptm = (other["ptm"] - ref["ptm"]).float()
    diptm = (other["iptm"] - ref["iptm"]).float()
    out["ptm_max_abs"] = dptm.abs().max().item()
    out["iptm_max_abs"] = diptm.abs().max().item()
    out["ranking_core_max_abs"] = (0.8 * diptm + 0.2 * dptm).abs().max().item()
    return out


def compare_capture(ref_path: Path, other_path: Path) -> dict[str, Any]:
    ref = torch.load(ref_path, map_location="cpu")
    other = torch.load(other_path, map_location="cpu")
    out: dict[str, Any] = {
        name: tensor_error(ref[name], other[name]) for name in TENSORS
    }
    out["distogram"] = distogram_error(ref["distogram_probs"], other["distogram_probs"])
    head = other.get("head_fixed", other.get("head"))
    out["head_source"] = "head_fixed" if "head_fixed" in other else "head"
    out["head"] = head_error(ref["head"], head)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("ref", type=Path)
    parser.add_argument("other", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rows", help="compare only these comma-separated rows")
    args = parser.parse_args()
    rows = args.rows.split(",") if args.rows else None
    if rows is not None and len(set(rows)) != len(rows):
        parser.error("--rows names a row more than once")
    entries = []
    for ref_path in sorted(args.ref.glob("*/*/seed_*/tier1.pt")):
        rel = ref_path.relative_to(args.ref)
        row, job, seed = rel.parts[0], rel.parts[1], rel.parts[2]
        if rows is not None and row not in rows:
            continue
        other_path = args.other / rel
        entry = {"id": row, "job": job, "seed": int(seed.split("_")[1])}
        entry["ref_sha256"] = provenance.sha256_file(ref_path)
        if not other_path.exists():
            entry["status"] = "fail: missing capture"
        else:
            entry["other_sha256"] = provenance.sha256_file(other_path)
            entry.update(compare_capture(ref_path, other_path))
            entry["status"] = "ok"
        entries.append(entry)
        print(
            json.dumps({k: entry[k] for k in ("id", "job", "seed", "status")}),
            flush=True,
        )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(
            {
                "ref": str(args.ref),
                "other": str(args.other),
                "source": provenance.git_state(),
                "entries": entries,
            },
            indent=1,
        )
    )
    return 0 if all(e["status"] == "ok" for e in entries) else 1


if __name__ == "__main__":
    raise SystemExit(main())
