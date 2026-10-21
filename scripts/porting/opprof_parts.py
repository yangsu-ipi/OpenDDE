"""Sampler kernel time by part and kernel class from opprof profiler records (kit-item disposition, plan revision 117).

Usage: python scripts/porting/opprof_parts.py DIR   (DIR holds <row>.opprof.json written in the profiler mode)

For each part of the diffusion module (diffusion_transformer, atom_attention_encoder, atom_attention_decoder,
diffusion_conditioning, ...) prints its kernel time as a share of the row's total kernel time, split into GEMM
kernels, attention kernels and all other kernels (elementwise, LayerNorm, reductions, copies). A fused stack that
keeps fp32 GEMMs and the stock attention (dit_fused or atom_fused without 16-bit operands) can remove at most the
"other" time of its part, plus whatever merging GEMMs gains; the GEMM share bounds the latter.
"""

import glob
import json
import sys

GEMM_WORDS = ("gemm", "gemv", "cutlass", "xmma", "cublas", "splitk")
ATTN_WORDS = ("fmha", "attention", "flash", "attn")


def kernel_class(name: str) -> str:
    low = name.lower()
    if any(w in low for w in ATTN_WORDS):
        return "attention"
    if any(w in low for w in GEMM_WORDS):
        return "gemm"
    return "other"


def main() -> int:
    rows = {}
    for f in glob.glob(f"{sys.argv[1]}/*.opprof.json"):
        d = json.load(open(f))
        rows[d["row"]] = d
    order = sorted(rows, key=lambda r: int(r[1:]))
    parts = sorted(
        {p for d in rows.values() for p in d.get("sampler_part_kernels", {})}
    )
    classes = ("gemm", "attention", "other")
    print(
        f"{'row':5s} {'part':26s} {'total%':>7s} "
        + " ".join(f"{c + '%':>10s}" for c in classes)
    )
    sums = {p: dict.fromkeys(classes, 0.0) for p in parts}
    grand = 0.0
    for r in order:
        d = rows[r]
        total = d["kernel_us_total"]
        grand += total
        for p in parts:
            ks = d.get("sampler_part_kernels", {}).get(p, [])
            per = dict.fromkeys(classes, 0.0)
            for k in ks:
                per[kernel_class(k["name"])] += k["us"]
            for c in classes:
                sums[p][c] += per[c]
            print(
                f"{r:5s} {p:26s} {100 * sum(per.values()) / total:7.1f} "
                + " ".join(f"{100 * per[c] / total:10.1f}" for c in classes)
            )
    print("summed over the rows (share of all kernel time):")
    for p in parts:
        print(
            f"      {p:26s} {100 * sum(sums[p].values()) / grand:7.1f} "
            + " ".join(f"{100 * sums[p][c] / grand:10.1f}" for c in classes)
        )
    print("largest kernels per part, summed over the rows:")
    for p in parts:
        agg = {}
        for d in rows.values():
            for k in d.get("sampler_part_kernels", {}).get(p, []):
                agg[k["name"]] = agg.get(k["name"], 0.0) + k["us"]
        for name, us in sorted(agg.items(), key=lambda kv: -kv[1])[:8]:
            print(
                f"      {p:26s} {100 * us / grand:6.2f}% [{kernel_class(name)}] {name[:110]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
