"""Upper bounds per kit item from opprof events records (kit-item disposition, plan revision 117).

Usage: python scripts/porting/opprof_items.py DIR   (DIR holds <row>.opevents.json for D1-D15)

Each kit item can save at most the exclusive GPU time of the module families it would replace. Prints, per
item group, that time as a share of the warm forward's GPU span summed over the rows, and the largest share
on any one row, to compare with the disposition thresholds (3% summed, 5% on every row).
"""

import json, glob, sys

d_dir = sys.argv[1]
rows = {}
for f in glob.glob(f"{d_dir}/*.opevents.json"):
    d = json.load(open(f))
    rows[d["row"]] = d
order = sorted(rows, key=lambda r: int(r[1:]))
PAIR_TOPS = {
    "pairformer_stack",
    "msa_module",
    "template_embedder",
    "confidence_head",
    "structural_token_refiner",
    "distogram_head",
    "input_embedder",
}


def top(k):
    return k.split("/")[0]


def cls(k):
    return k.split("/")[1]


ITEMS = {
    "triangle attention, all stacks (triattn_exact on sm_120; triattn_core)": lambda k: (
        cls(k) == "TriangleAttention"
    ),
    "triangle multiplication, all stacks (trimul_core over trimul_triton)": lambda k: (
        cls(k).startswith("TriangleMultiplication")
    ),
    "pair transitions (transition_core)": lambda k: (
        cls(k) == "Transition" and top(k) != "diffusion_module"
    ),
    "pair-stack LayerNorms (ln_core)": lambda k: (
        cls(k) in ("OpenFoldLayerNorm", "FusedLayerNorm")
        and top(k) != "diffusion_module"
    ),
    "pair-stack glue: stack/module self time (arm_z transpose-free part)": lambda k: (
        cls(k)
        in (
            "PairformerStack",
            "MSAModule",
            "TemplateEmbedder",
            "MSAPairWeightedAveraging",
            "OuterProductMean",
        )
    ),
    "whole pair stack (arm_u bf16 pair stack, upper bound)": lambda k: (
        top(k) in PAIR_TOPS
    ),
    "sampler attention (dit_attn_apb, atom_attn_apb)": lambda k: (
        top(k) == "diffusion_module" and cls(k) == "AttentionPairBias"
    ),
    "sampler transformer, all (dit_fused + dit_lowp, atom_fused)": lambda k: (
        top(k) == "diffusion_module"
        and cls(k)
        in (
            "AttentionPairBias",
            "ConditionedTransitionBlock",
            "AdaptiveLayerNorm",
            "DiffusionTransformer",
            "OpenFoldLayerNorm",
            "Transition",
            "AtomTransformer",
            "AtomAttentionEncoder",
            "AtomAttentionDecoder",
        )
    ),
    "atom encoder/decoder self (atom_fused part)": lambda k: (
        cls(k) in ("AtomAttentionEncoder", "AtomAttentionDecoder", "AtomTransformer")
    ),
    "diffusion conditioning (cond_dedupe)": lambda k: (
        cls(k) in ("DiffusionConditioning", "FourierEmbedding")
    ),
    "outside every module (sampler loop, guidance)": lambda k: k == "(root)/OpenDDE",
    "diffusion module self (sampler loop glue)": lambda k: (
        k == "diffusion_module/DiffusionModule"
    ),
}
span = {r: rows[r]["span_ms"] for r in order}
S = sum(span.values())
print(f"rows {len(order)}; GPU span of the warm forward summed: {S / 1e3:.1f} s")
print(f"{'item group':72s} {'sum %':>6s} {'max row %':>9s}")
for name, sel in ITEMS.items():
    per = {
        r: sum(v for k, v in rows[r]["families_ms"].items() if sel(k)) for r in order
    }
    tot = sum(per.values())
    mx = max((100 * per[r] / span[r], r) for r in order)
    print(f"{name:72s} {100 * tot / S:6.1f} {mx[0]:7.1f} {mx[1]}")
