# The kit's `fast` mode compared with this port

Date: 2026-10-19. Sources: the optimization kit at commit `f4f62fa` (`opendde/CHANGES.md`, `opendde/README.md`, `opendde/configs/a100.env`, and the shared core's `common/opt_core/opt_core/kernels/triattn/TRIATTN_CELLS.json`); this port at `porting/inline-optimizations` (`opendde/config/optimization.py`, `opendde/model/optimized/dispatch.py`); the [line ledger](qualification_line_ledger.md) for the held-out results.

## Summary

- On each GPU, this port's `fast` adds exactly one item to `exact`: a triangle-attention kernel. On sm_80 it is `triattn_native`, the same kernel the kit's `fast` serves there (`triattn_native@v11`); on sm_120 it is `k2b`, chosen by this port's own tile sweep, since the kit has no sm_120 measurements.
- The kit's `fast` has seven groups of changes that alter numerics. This port carries one of them, triangle attention, and none of the other six: the bf16 pair stack, fast triangle multiplication, fast pair transition, Triton pair-row LayerNorms, the attention chunk lift and the sampler's fast tier.
- The kit's `fast` also runs its `exact`-tier scheduling items. This port's `exact` has the host and data-path ones, but not the CUDA graph replay of the denoiser step, input prefetch, skipped cache flushes, the expandable-segments allocator or the aligned pair-bias allocation.
- Both of this port's `fast` candidates, each differing from `off` only in triangle-attention rounding, were rejected at their first held-out draws on tier 2 gates. Following the protocol, the failures have not been diagnosed on held-out outputs.

## Every kit item: mode, GPUs and port status

The kit targets A100 (sm_80) and H100/H200 (sm_90). Its host items run on any GPU; its kernel items run where the shared core's cell tables vouch for a row: CUDA-native kernels for sm_90a or sm_80, Triton elsewhere. sm_120 appears nowhere in the kit. Most "no" entries below are planned but not started; the reason says which. The plan's "Pairformer fast path" group (the transpose-free pair stack and the fast Triton triangle, transition and LayerNorm kernels) is to be ported after exact parity, which `exact` has met on both GPUs since 2026-10-17; of that group only triangle attention has been taken up so far, and the rest has not been attempted or judged unsuitable.

### `exact` (`fast` and `big` also run its host items)

| Kit item | Kit GPUs | sm_120 | sm_80 | Reason |
| --- | --- | --- | --- | --- |
| `drop_bond_mask`, `tmpl_dedup`, `json_oneshot`, `structok_sync`, `sched_host` | any | ported | ported | Host and data-path changes with identical bytes (`c584e6d`; as `template_dedup`, `structural_token_sync`, `host_noise_schedule`) |
| `dit_hoist`, `zprep_hoist` | any | ported | ported | The same (`7fddcb4`) |
| `dit_align` | any | no | no | Omitted from the plan's scope table and never evaluated; added in plan revision 116 as an optional `exact` item, to be checked for byte identity with `off` like `dit_hoist` |
| `stepgraph` | sm_90; sm_80 up to 1,024 tokens | no | no | Optional `exact` item: needs graph-capture tests and a measured bit-exact window per GPU, and must stay off under guidance |
| `prefetch` | any | no | no | Optional: adds a worker process and must step aside to keep RDKit's random state as in stock |
| `keep_pool` | any | no | no | Optional: less fragmentation control for fewer synchronizations |
| `alloc_auto` | any | no | no | Optional: a process-wide allocator setting |
| `lnstream` | prebuilt sm_80 and sm_90 | as a bug fix | as a bug fix | Ported outside the modes (`0950ab5`): a correctness bug, used only by `fast_layernorm` |
| `served_levers_hook`, `cueq_tuned_cache` | any | no | no | Overlay infrastructure, excluded by the plan; no tile tables are shipped anyway |
| `arm_z` (transpose-free pair update, stock kernels) | any | no | no | Not started: in the plan's "Pairformer fast path", planned after exact parity, which has been met |
| `fpf_trimul_exact` + `trimul_exact` | CUDA sm_90a and sm_80, only where a bit-exact record exists (none on A100) | ported as `trimul_triton` | ported as `trimul_triton` | This port's Triton gated projections around cuEquivariance's LayerNorms and contraction give the stock bits; about 2x per call (`33e09d2`, `f79dcb4`) |
| `transition_exact` | per the core's cells | ported as `transition_triton` | ported as `transition_triton` | Bit-identical, about 1.15x per call |
| `triattn_exact` + `triattn_conf` | H100 stacks the core vouches for; the sm_80 cells list the CUDA v3 kernel | no | ported as `triattn_exact` | Bitwise cuEquivariance on A100, 1.04-1.71x from 192 tokens. Its tiling is sized for sm_80's 163 KB of shared memory (sm_120 allows about 99 KB per block), and the kit has no sm_120 exact records |
| `dit_attn_exact` | prebuilt sm_90a only | no | no | No sm_80 or sm_120 build exists; the kit itself runs stock on A100 |

This port's `exact` also has bit-identical kernels that the kit's `exact` does not serve on these GPUs: `trimul_triton` and `transition_triton` on both GPUs, and `triattn_exact` on A100. On A100 the kit's `exact` runs the stock operation for triangle multiplication and sampler attention, because it has no bit-exact record there.

### `fast` (rounding changes; replaces `arm_z` and the exact-tier kernel items)

The kit describes these as bf16 re-association and fused-kernel rounding.

| Kit item | Kit GPUs | sm_120 | sm_80 | Reason |
| --- | --- | --- | --- | --- |
| `triattn_core` + `triattn_conf` | CUDA sm_90a and sm_80, Triton elsewhere; the sm_80 cells serve `triattn_native@v11` | ported as `triattn_k2b` | ported as `triattn_native` | The only `fast` item ported; rejected at the first held-out draw on both GPUs |
| `arm_u` + `arm_u23` (bf16 pair stack on the trunk, structure refiner and confidence pairformer) | any, from 300 tokens | no | no | Not started: "Pairformer fast path", planned after exact parity, which has been met |
| `trimul_core` | Triton | no | no | sm_80: no fast candidate beat the bit-identical kernel by more than 5-10%, and the core's native triangle-multiplication cubins are CUDA 13 builds. sm_120: not attempted; triangle attention came first |
| `transition_core` (`sep16`) | Triton | no | no | Not started: "Pairformer fast path", planned after exact parity, which has been met. The bit-identical version already runs |
| `ln_core` | Triton, C = 384 | no | no | Wraps `FusedLayerNorm` only, which never runs under 1.2's default `LAYERNORM_TYPE=torch`; it needs a new dispatch point, worth it only if the profile shows LayerNorm time matters |
| `chunk_lift` | any, up to 1,024 tokens | no | no | Omitted from the plan's scope table and never evaluated; added in plan revision 116 to be evaluated with the fast path. Its choice depends on free GPU memory at run time, so it needs a decision fixed by the input and settings, or identical choices across repeats |
| `dit_attn_apb`, `atom_attn_apb`, `cond_dedupe`, `dit_fused` + `dit_lowp`, `atom_fused` | Triton and CUDA through the core's apb provider | no | no | "Diffusion and atom fast path", after the Pairformer path. Tuned against TF32-on stock while 1.2's sampler runs full FP32, so their error will be larger than the kit reported; they must also fit the 1.2 guidance loop |

### `big`

| Kit item | Kit GPUs | sm_120 | sm_80 | Reason |
| --- | --- | --- | --- | --- |
| `pair_offload_trunk`, `pair_offload_struct`, `pair_offload_conf`, `diffz`, `bigln_guard`, `free_templ`, `sample_chunk`, `no_dit_hoist` | any; gates sized per card (64 GiB boundary) | no | no | Conditional on measured capacity: ported only if target inputs do not fit under `fast`. So far D4 (1,218 tokens) peaks at 43 GiB on the 80 GB A100 |
| `rowpair_tp`, `tp_triatt`, `struct_pair_bf16` (`--n_gpu P`) | multiple GPUs of one host | no | no | Deferred: about 23k lines, overlaps native Fold-CP, refuses guidance, and suits NVLink better than gnode2's PCIe |

Registered in the kit but in no shipped mode, and excluded by the plan as neutral or negative: `sampler_amp`, `writer_overlap`, `xl_tri_ln`, `dit_attn_bf16`.

In short: every exact host item is ported to both GPUs; bit-identical kernels are ported for triangle multiplication and the pair transition on both GPUs and for triangle attention on sm_80; from `fast` only triangle attention is ported, and both kernels were rejected at their draws. The optional exact scheduling items, the rest of the Pairformer and sampler fast paths, and all of `big` are planned but not ported.

## Triangle attention on sm_80

- **Kernel.** The kit's cell table serves `triattn_native@v11` for bf16 at head dimension 32 in every cell for 4, 8 and 12 heads, and for 2 heads in every cell except 2 heads at up to 256 tokens (`k2b`). This port's `triattn_native` is that kernel's sm_80 member, built from source (`csrc/triattn_sm80`).
- **Call sites.** Both serve every pair-stack triangle attention: in this port, every `TriangleAttention` module (trunk, template stack, MSA module, structure refiner, confidence head) under bf16 autocast.
- **Size window.** This port serves 192 to 2,047 tokens and runs `triattn_exact` below 192, where the native kernel was slower in its benchmark (0.85× `triattn_exact` at 128 tokens, 1.17× at 192, 2.1-2.4× from 300; 1.7-3.7× stock; error against an fp32 reference 1.03-1.06× stock's; kernel benchmark job 341, 2026-10-18). The kit composes its trunk items out when every item of a call has fewer than 300 tokens (`MODEL_OPT_SMALL_INPUT_FLOOR_TOKENS`), and its cells run to 6,144 tokens. Between 192 and 300 tokens, this port's `fast` therefore changes rounding where the kit's would not.
- **`k2b` on A100.** In this port's benchmark `k2b` was slower than `triattn_exact` (error against fp32 about 2×10⁻⁴), so it has no sm_80 row.
- **End-to-end effect** (job 347, D4 and D12 with phase timing, `exact` and `fast` alternating on one otherwise idle GPU): D4 total 736 → 661 s (0.90×), trunk 367 → 292 s (0.79×); D12 total 646 → 638 s (0.99×), trunk 72 → 64 s (0.89×), diffusion 529 s in both. Peak memory was unchanged (D4 43.1 GiB). On a guided row like D12 the sampler dominates the run time, and that is where the kit's sampler fast tier, which this port does not have, is aimed.

## Triangle attention on sm_120

The kit ships configurations for A100, H100 and H200 only, and its cell table has no compute-capability 12.0 cells. Its `k2b` is the measured first choice mainly on B200 and B300 (24 of 28 bf16 head-dimension-32 cells on B200, 23 of 25 on B300); on the RTX PRO 6000 it would serve `k2b` only by inheritance. This port chose `k2b` for sm_120 from its own tile sweep. Testing `k2b` on B200 or B300 is an open item in [TODO.md](TODO.md).

## What the held-out results do and do not say

- Both candidates differed from `off` only in triangle attention: every other `fast` item is in `exact`, which is byte-identical to `off` on D1-D15.
- `sm120-fast-1` (k2b, contract-6) failed 4 of 2,538 checks: H3 Zn same-site, H6 and H7 under guidance, H10 p95 error. `sm80-fast-1` (`triattn_native`, contract-7) failed 3 of 2,498: H5 lowest reference lDDT, H6 and H7 outcome counts. Both passed every development check and every held-out tier 1, trace, repeat, provenance and seed check.
- The kit describes its `fast` as "within stock's seed-to-seed variation". This project's tier 2 gates are not a seed-to-seed comparison: they pair each candidate sample with the stock sample of the same seed, under limits calibrated on harmless contrasts (kernel swap, precision, LayerNorm build).
- These results do not show whether the kit's full `fast` would pass or fail the same gates. They do show that a single triangle-attention rounding change, on two GPUs and with two different kernels, was enough to fail held-out tier 2 gates on rows with restraints. Each further fast-tier item would add its own rounding change for those gates to absorb.
- Each `fast` claim has used one of its three draws. A later candidate needs a fresh code commit descending from `df5779c`, its own rehearsal and the user's go-ahead.
