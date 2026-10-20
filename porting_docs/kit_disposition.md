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

- **Switch timing** (job 54841, `run_rows.py --phase-timing`, D1-D15 under `exact`, passes alternating on one GPU: base, alloc, keep, base, alloc, keep; compared with `scripts/porting/phase_compare.py`, which sums each row's `predict` phase time over both seeds). The two base passes agree within 0.25% on every row but D6 (1.1%). `alloc_auto` (`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`) saves -0.08% summed and at most 2.1% on a row (D7); `keep_pool` (in-forward `torch.cuda.empty_cache` made a no-op by the hook's trial switch; 6 calls per process) saves 0.04% summed and at most 1.1% on a row. Both leave all 480 D1-D15 artifacts byte-identical to `baseline_off.json`.
- **Prefetch** (job 54842: one query of four development inputs, D3, D8, D14 and D15, which share a recipe, at seeds 101 and 102 under `exact`; `run_opendde.py --set num_workers=0|1`, alternating, two passes each). Whole-process wall time 519.2 s with the 1.2 default (0 workers) and 513.3 s with one worker, a 1.1% saving. With one worker all 120 prediction artifacts differ from the default run, structures included, for every input (none has a SMILES ligand), while two default runs are byte-identical: the worker changes how the run consumes random numbers, so a prefetching loader is not an `exact` change in 1.2 without separating featurization's random state from the model's.
- **`atomattn_triton` alone on top of `exact`** (job 55004; scratch clone of `779fd09` plus the port; `fast` with `optimization.triattn_k2b=false`; passes alternating exact, candidate, exact, candidate with phase timing; tier 1 capture under `exact` and replay under the candidate). The kernel served every row (one call per process falls back where autograd is on, in D12's guidance). The candidate's two passes are byte-identical (480 artifacts). Against `exact`, its development gates pass: single-cycle tier 1, 32 of 32 entries `ok` (`tier1_compare.py`); tier 2 under the frozen sm_120 limits, all 16 D1-D15 jobs pass with the Zn claim qualified (post job 55060), so it is not (b). Model forward time: 2,101.4 s under `exact` and 2,072.5 s under the candidate, a 1.38% saving; the largest row saving is 2.5% (D1, where the two `exact` passes differ by 2.6%), D12 saves 0.17%. A first run (job 54854) never used the kernel, because the port refused the input embedder's 3-D calls and the model's permuted bias layout; it was discarded and the port fixed. The port is not committed (user decision, 2026-10-20); it is archived as a patch against `779fd09` in `/mnt/compbio-data/yang.su/opendde-porting/archive/atomattn_triton-2026-10-20/` (SHA-256 `80784c8b…`, with a README), for measuring on sm_80.
- **GPU idle time** (job 54841's last pass, the hook's idle mode: NVML utilization sampled every 20 ms during the warm forward). The GPU is busy at least 94% of the time on every row but D6 (87.9%) and D7 (62.7%), the two smallest inputs. Idle time summed over D1-D15 is about 27 s of 1,045 s (2.6%); it bounds what host-overhead items can save.
- **Sampler attention kernels** (kernel benchmarks, jobs 54844 and 54848, `kernel-bench-sm120/bench_dit_apb.py` and `bench_atom_apb.py` with the kit's Triton files copied unchanged into `kit/apb` with their SHA-256 sums; random inputs at the sampler's shapes, 5 samples; error against an fp64 reference beside stock's). Stock 1.2 runs both in fp32 (error about 4e-7 and 3e-7 relative RMS).
  - Token attention (`fpf_apb`, 16 heads of 48, 86-1,218 tokens): the 16-bit and TF32 operand variants are up to 1.84x faster but have 750-7,000x stock's error; the fp32-faithful variants (`tf32x3`, `ieee`) are within 0.8-1.5x of stock's error but 1.8-3.3x slower than stock.
  - Atom attention (`fpf_atom`, 4 heads of 32, windows 32x128, 1,836-9,142 atoms): with fp32 (`ieee`) operands the error is 1.0x stock's and the call is 3.5-10.1x faster than stock's `_local_attention`; `tf32x3` is 2.3-5.6x faster at 1.5x stock's error; the kit's default `tf32rn` and the 16-bit variants have 1,400-11,000x stock's error.

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
| Outside every module (sampler loop, guidance) | host items | 1.3% | 9.3% (D11) |
| GPU idle time in the warm forward | `stepgraph`, host-overhead items | 2.6% | 37.3% (D7) |

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
| `alloc_auto` | **(a)**: -0.08% summed, at most 2.1% on a row; outputs byte-identical | pending |
| `keep_pool` | **(a)**: 0.04% summed, at most 1.1% on a row; outputs byte-identical | pending |
| `stepgraph` | to measure: idle time bounds it at 2.6% summed but 37% on D7 and 12% on D6, so the per-row bound is not met; it refuses guidance (D9-D12) and the kit has no sm_120 bit-exact window | pending |
| `prefetch` | **(a)**: 1.1% of wall time on a four-input query, and featurization is 5.3% of D1-D15's total time; it also changes every output in 1.2 (random-state consumption), so it would not be `exact` as it stands | pending |
| `dit_align` | to measure: timed directly | pending |
| `triattn_core` | ported as `triattn_k2b`; rejected at the held-out draw | ported as `triattn_native`; rejected at the held-out draw |
| `trimul_core`, `transition_core` | to measure: bounds 13.6% and 8.4% over the bit-identical kernels | pending |
| `arm_u` + `arm_u23` | to measure: bound 52.7% | pending |
| `chunk_lift` | to measure: needs a decision fixed by the input and settings | pending |
| `dit_attn_apb` | **(b)**: every variant faster than stock has 750-7,000x stock's kernel error; the variants within 1.25x of it are slower than stock | pending |
| `atom_attn_apb` | **(a)**: ported for measurement as `atomattn_triton` (the kit kernel with fp32 operands: 1.0x stock's kernel error, 3.5-10x faster per call), it saves 1.38% of model forward time summed over D1-D15 and at most 2.5% on a row, because atom attention is a small share of the run | pending (the kit has sm_80 cells) |
| `dit_fused` + `dit_lowp`, `atom_fused` | to measure: bound 45.6% (whole sampler transformer); `dit_lowp`'s fp16 operands are expected to fail (b) as the 16-bit attention variants do; the kit measured them against TF32-on stock, while 1.2 runs the sampler in fp32 | pending |
| `dit_attn_exact` | **(c)**: its kernel uses sm_90a-only instructions (wgmma, setmaxnreg) and the kit ships it only for compute capability 9.0, so it cannot run on sm_120 | **(c)**, as on sm_120 |
| `big` items, `rowpair_tp` | decided by capacity (plan steps 5 and 6) | the same |

## Next on sm_120

1. The development diagnostic of `atomattn_triton` alone on top of `exact` (job 54854: alternating timed passes, tier 1 and tier 2 gates against `exact`, which is byte-identical to `off`).
2. Prototypes in order of bound: the rest of the sampler fast tier at fp32 (`dit_fused`, `atom_fused`), triangle attention (`triattn_exact` on sm_120), the bf16 pair stack, then `trimul_core`, `transition_core`, `arm_z`, `stepgraph` (small rows), `dit_align` and `chunk_lift`.

The sm_80 column waits for one A100 session, which starts with the replacement-host test; the same profile and measurements are then repeated there, because bounds measured on sm_120 do not carry over.
