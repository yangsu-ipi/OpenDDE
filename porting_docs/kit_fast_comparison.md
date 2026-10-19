# The kit's `fast` mode compared with this port

Date: 2026-10-19. Sources: the optimization kit at commit `f4f62fa` (`opendde/CHANGES.md`, `opendde/README.md`, `opendde/configs/a100.env`, and the shared core's `common/opt_core/opt_core/kernels/triattn/TRIATTN_CELLS.json`); this port at `porting/inline-optimizations` (`opendde/config/optimization.py`, `opendde/model/optimized/dispatch.py`); the [line ledger](qualification_line_ledger.md) for the held-out results.

## Summary

- On each GPU, this port's `fast` adds exactly one item to `exact`: a triangle-attention kernel. On sm_80 it is `triattn_native`, the same kernel the kit's `fast` serves there (`triattn_native@v11`); on sm_120 it is `k2b`, chosen by this port's own tile sweep, since the kit has no sm_120 measurements.
- The kit's `fast` has seven groups of changes that alter numerics. This port carries one of them, triangle attention, and none of the other six: the bf16 pair stack, fast triangle multiplication, fast pair transition, Triton pair-row LayerNorms, the attention chunk lift and the sampler's fast tier.
- The kit's `fast` also runs its `exact`-tier scheduling items. This port's `exact` has the host and data-path ones, but not the CUDA graph replay of the denoiser step, input prefetch, skipped cache flushes, the expandable-segments allocator or the aligned pair-bias allocation.
- Both of this port's `fast` candidates, each differing from `off` only in triangle-attention rounding, were rejected at their first held-out draws on tier 2 gates. Following the protocol, the failures have not been diagnosed on held-out outputs.

## Numerics-changing items (the kit's `fast` tier)

The kit's `fast` replaces its exact-tier kernel items (`arm_z`, `fpf_trimul_exact`/`trimul_exact`, `triattn_exact`, `transition_exact`, `dit_attn_exact`) with these fast-tier bindings. The kit describes their numerics as bf16 re-association and fused-kernel rounding.

| Kit item | What it does | This port, sm_80 | This port, sm_120 |
| --- | --- | --- | --- |
| `arm_u` + `arm_u23` | bf16 pair stack on the trunk, structure refiner and confidence pairformer: cast-once prologue, transpose-free block, the arm's triangle-multiplication route | not ported | not ported |
| `triattn_core` + `triattn_conf` | fast triangle attention at every pair-stack site; the row per card, dtype, head count and size comes from the core's cell table | `triattn_native` (see below) | `triattn_k2b` |
| `trimul_core` | bf16 Triton triangle multiplication | not ported: no fast candidate measured more than 5-10% faster than the bit-identical one (`trimul_triton`) | not ported |
| `transition_core` (`sep16`) | fused pair transition, fast variant | not ported; the bit-identical `transition_triton` runs | not ported; `transition_triton` runs |
| `ln_core` | Triton pair-row LayerNorms at C = 384 | not ported | not ported |
| `chunk_lift` | un-chunked fp32 attention scores where free memory allows; only when every item has at most 1,024 tokens | not ported | not ported |
| Sampler fast tier | `dit_attn_apb` (token attention), `atom_attn_apb` (atom transformers), `cond_dedupe`, `dit_fused` + `dit_lowp` (the token DiffusionTransformer as one fused schedule, fp16 operands), `atom_fused` | not ported | not ported |

The plan's scope table already places the bf16 pair stack and the Triton kernels in "Pairformer fast path" and the sampler items in "Diffusion and atom fast path", both after exact parity.

## Exact-tier items the kit's `fast` also runs

| Kit item | What it does | This port's `exact` |
| --- | --- | --- |
| `drop_bond_mask`, `tmpl_dedup`, `json_oneshot`, `structok_sync`, `sched_host`, `dit_hoist`, `zprep_hoist` | host and data-path changes | ported (`drop_bond_mask`, `template_dedup`, `json_oneshot`, `structural_token_sync`, `host_noise_schedule`, `dit_hoist`, `zprep_hoist`) |
| `stepgraph` | the denoiser step captured into a CUDA graph and replayed; on sm_80 only up to 1,024 tokens, since replay is not bit-exact above that | not ported (an optional `exact` item in the plan) |
| `prefetch`, `keep_pool`, `alloc_auto` | featurise the next input during this one; make in-forward `empty_cache()` calls no-ops; the expandable-segments allocator | not ported (optional `exact` items in the plan) |
| `dit_align` | the denoiser's pair biases in an aligned allocation | not ported |
| `lnstream` | upstream's fused LayerNorm launched on the current CUDA stream | not in a mode; the plan treats it as a bug fix outside the modes |
| `cueq_tuned_cache` | exports cuEquivariance's tuning-cache location; no tiles are shipped | not needed |

This port's `exact` also has bit-identical kernels that the kit's `exact` does not serve on these GPUs: `trimul_triton` and `transition_triton` on both GPUs, and `triattn_exact` on A100. On A100 the kit's `exact` runs the stock operation for triangle multiplication and sampler attention, because it has no bit-exact record there.

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
