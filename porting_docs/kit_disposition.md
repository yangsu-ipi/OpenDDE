# Kit item dispositions

Plan revision 117 requires every kit item to reach a disposition on sm_120 and on sm_80: ported, or shown (a) not worth porting, (b) numerically worse, or (c) not applicable to OpenDDE 1.2. The thresholds, set by the user on 2026-10-19:

- **(a)** the item saves under 3% of model forward time summed over D1-D15 and under 5% on every row, by the controlled timing protocol. An upper bound is enough: an item can save at most the time of the work it replaces.
- **(b)** alone on top of `exact`, it fails a frozen tier 1 or tier 2 gate on D1-D15 against `off`, or its kernel-level error against an fp32 reference is above 1.25× stock's.
- **(c)** the code path it changes does not run under 1.2's defaults or supported options, or 1.2 or this port already covers it.

The item list and what each kit item does are in [kit_fast_comparison.md](kit_fast_comparison.md). This file holds the evidence and the status per GPU.

## sm_120 evidence (gnode2, 2026-10-20)

All runs: RTX PRO 6000 Blackwell Max-Q, clean checkout of `aea6a3e` (`~/opendde-prof`), `--optimization_mode exact --deterministic true`, development rows D1-D15 at development seeds 101 and 102, the shared weights. Outputs are under `/mnt/compbio-data/yang.su/opendde-porting/dev-sm120/disp-*`; job scripts in `~/opendde-dev-checks/disposition/`.

- **Phase profile** (job 54775, `run_rows.py --phase-timing`, as sm_80 job 347). Summed over D1-D15 and both seeds, model forward time is 2,122 s: diffusion 46.7%, trunk 44.8%, confidence 6.5%, structural expansion 2.0%. Featurization is a further 5.3% and output writing 1.8% of the total time. Diffusion is 76% of D12 (guided) and over 60% of D1, D2 and D6; the trunk is 56% of D4.
- **Module-family GPU time** (jobs 54822 and, for D12, 54836; `scripts/porting/opprof/sitecustomize.py` in its events mode, summarized by `scripts/porting/opprof_items.py`). The warm seed's model forward is timed with CUDA events around every module of the listed families; each family is charged its span minus the spans of the families inside it, so the shares are disjoint and include idle time inside a module. The timed call took within about 1% of the unprofiled run's time (D5: 67.17 s against 67.3 s). In D12, 6,400 sampler-block frames (6,000 conditioned transitions, 200 atom encoders, 200 atom decoders) never got their closing hook and were closed with the enclosing module, so D12's split between the conditioned transition and the sampler's glue is approximate; the sampler total is exact. The records were produced by the hook before its docstring was rewritten and the file formatted; the logic is unchanged.
- **Kernel time** (job 54797, the hook's profiler mode, D1-D3, D5-D7, D13-D15; the other six rows exceeded the job's 29 GB host memory). On these rows the profiled kernel time is close to or above the unprofiled wall time of the warm forward, except D7 (60%) and D6 (93%), so the GPU is close to fully busy on all but the smallest rows. Kernel durations are inflated by the profiler, so this is a rough reading; a direct idle-time measurement is still to do.

Upper bounds on sm_120 (share of the warm forward's GPU span, 1,043.6 s summed over D1-D15):

| Work a kit item would replace | Kit items | Summed | Largest row |
| --- | --- | --- | --- |
| Triangle attention, all pair stacks (stock cuEquivariance on sm_120) | `triattn_exact` (sm_120), `triattn_core` | 24.0% | 37.2% (D4) |
| Triangle multiplication, all stacks (already `trimul_triton`) | `trimul_core` | 13.6% | 17.6% (D4) |
| Pair transitions (already `transition_triton`) | `transition_core` | 8.4% | 10.4% (D5) |
| Pair-stack LayerNorms | `ln_core` | 2.7% | 3.4% (D5) |
| Stack self time (residual adds, transposes; PairformerStack, MSAModule, TemplateEmbedder) | `arm_z` (part) | 3.2% | 4.0% (D5) |
| Whole pair stack | `arm_u` + `arm_u23` | 52.7% | 72.1% (D4) |
| Sampler attention, token and atom | `dit_attn_apb`, `atom_attn_apb`, `dit_attn_exact` | 21.4% | 31.6% (D1) |
| Whole sampler transformer, token and atom | `dit_fused` + `dit_lowp`, `atom_fused` | 45.6% | 79.6% (D6) |
| Atom encoder and decoder self time | `atom_fused` (part) | 2.7% | 7.7% (D6) |
| Diffusion conditioning | `cond_dedupe` | 0.1% | 0.6% (D7) |
| Outside every module (sampler loop, guidance) | `stepgraph` (part), host items | 1.3% | 9.3% (D11) |

## Status

| Kit item | sm_120 | sm_80 |
| --- | --- | --- |
| Host items (`drop_bond_mask`, `tmpl_dedup`, `json_oneshot`, `structok_sync`, `sched_host`, `dit_hoist`, `zprep_hoist`) | ported | ported |
| `fpf_trimul_exact` + `trimul_exact`, `transition_exact` | ported (`trimul_triton`, `transition_triton`) | ported |
| `triattn_exact` | to measure: bound 24.0% / 37.2%; needs an sm_120 kernel bitwise to cuEquivariance's | ported |
| `lnstream` | ported as a bug fix outside the modes | ported as a bug fix |
| `ln_core` | **(a)**: bound 2.7% summed, 3.4% at most on a row; also (c), it wraps only `FusedLayerNorm`, which 1.2's default `LAYERNORM_TYPE=torch` never runs | pending the A100 session |
| `cond_dedupe` | **(a)**: bound 0.1% summed, 0.6% at most on a row; `dit_hoist` already computes the step-invariant conditioning once | pending |
| `served_levers_hook`, `cueq_tuned_cache` | **(c)**: the kit's overlay infrastructure; this port dispatches inline, and stock already reads cuEquivariance's packaged per-card tuning cache | **(c)** |
| `sampler_amp`, `writer_overlap`, `xl_tri_ln`, `dit_attn_bf16` | the kit's own measurements: in no shipped mode, neutral or negative | the same |
| `arm_z` | to measure: bound 3.2% / 4.0% is above the 3% summed threshold | pending |
| `stepgraph`, `keep_pool`, `alloc_auto`, `prefetch`, `dit_align` | to measure: not bounded by module time; timed directly (`prefetch` also on a multi-input run) | pending |
| `triattn_core` | ported as `triattn_k2b`; rejected at the held-out draw | ported as `triattn_native`; rejected at the held-out draw |
| `trimul_core`, `transition_core` | to measure: bounds 13.6% and 8.4% over the bit-identical kernels | pending |
| `arm_u` + `arm_u23` | to measure: bound 52.7% | pending |
| `chunk_lift` | to measure: needs a decision fixed by the input and settings | pending |
| `dit_attn_apb`, `atom_attn_apb`, `dit_fused` + `dit_lowp`, `atom_fused` | to measure: bounds 21.4% (attention) and 45.6% (whole sampler transformer) | pending |
| `dit_attn_exact` | to measure: bound 21.4%; the kit has only an sm_90a build | pending |
| `big` items, `rowpair_tp` | decided by capacity (plan steps 5 and 6) | the same |

## Next on sm_120

1. Direct idle-time measurement of the warm forward, for `stepgraph` and the remaining host-overhead items.
2. Switch-type items timed directly with the controlled protocol: `alloc_auto` (the expandable-segments allocator), `keep_pool` (in-forward `empty_cache` calls made no-ops) and `prefetch` on a multi-input run.
3. Prototypes in order of bound: the sampler fast tier, triangle attention (`triattn_exact` on sm_120), the bf16 pair stack, then `trimul_core`, `transition_core`, `arm_z`, `dit_align` and `chunk_lift`.

The sm_80 column waits for one A100 session, which starts with the replacement-host test; the same profile and measurements are then repeated there, because bounds measured on sm_120 do not carry over.
