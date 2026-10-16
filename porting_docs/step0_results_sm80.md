# Step 0 results (sm_80, A100)

Stock step 0 on an A100 cloud VM (user go-ahead 2026-10-14). Stock code only: no candidate exists, and only development rows D1-D15, repository examples and development seeds 101 and 102 ran. No held-out input ran. Jobs ran through the gnode2-compat Slurm mock on the VM. The VM was unreachable (SSH refused) on 2026-10-15, after all outputs had been relayed; protocol E6 records the retention gaps and the replacement-host acceptance test. The VM was ephemeral, so the setup recipe and job scripts are on gnode2 in `~/opendde-dev-checks/a100/` (`README.md` has the rebuild steps), and all outputs, logs and tier 1 captures are on `/mnt/compbio-data/yang.su/opendde-a100/2026-10-14/`, each directory verified against a sha256 manifest computed on the VM. Prediction artifacts are not committed.

## Environment and source

- Host: 2x NVIDIA A100 80GB PCIe (sm_80), driver 595.99.02 / CUDA 13.2, Ubuntu 22.04, 60 CPUs, 236 GB RAM.
- Stack (`opendde-dev` on the VM): created from gnode2's `opendde-dev` conda explicit list and pip freeze; the installed pip freeze is identical to gnode2's (Python 3.11.17, torch 2.7.1+cu128, Triton 3.3.1, cuEquivariance 0.10.0, NumPy 2.4.1, SciPy 1.17.1, nvcc 12.8, gcc 14). The GPU architecture is the main variable, as the plan requires.
- Source: `cedfa29` (`qual-sm120-contract-6` plus the revision 96 docs commit), clean checkout, `PYTHONPATH` set to it.
- Assets: the VM's shared OpenDDE store and the colabfold env's Kalign 2.04 were byte-identical to the six frozen sm_120 digests in `runtime_assets.json`. Runs used a private copy (`/ephemeral/porting/opendde-qual-assets`, plain `cp`, read-only), which passed `assets.py verify`. The qualification data (13 MB) was copied from gnode2; `freeze_inputs.py check`, `check --external` and `check --effective` passed on the VM. No sm_80 entry was frozen in `runtime_assets.json`.

## Item 2: environment checks

| Check | Result |
|---|---|
| CPU test suite (`pytest tests -m "not network"`, job 42) | 1,041 passed, 4 skipped, 4 deselected (network), 69 subtests passed, 206 s |
| `tests/test_triton_compatibility.py`, `tests/smoke/test_cueq_runtime.py` (job 43) | 3 passed |
| Stock `opendde pred` smoke, `protein_200` | succeeded, 71 s |
| Triangle kernels under `auto` | resolved to cuEquivariance for both multiplication and attention |
| `fast_layernorm` extension | builds and loads for sm_80 (first build about 300 s on the VM) |
| Representative Triton kernel | compiles and runs (environment check, job 40) |
| cuEquivariance vs torch triangle kernels, N=128 (job 40) | trimul max diff 9.7e-8 fp32 / 9.8e-4 bf16; triatt 1.9e-8 / 1.2e-4; all finite |
| D1 and D5 through `run_rows.py` with contract-6 provenance (job 41) | both exit 0; D5 parser read the frozen bytes, referenced files stable, private Kalign used |

## D5 template preprocessing (protocol: run D5 on the provisioned A100)

`d5_template_digest.py` on the VM's CPU, twice (job 42): the template-hit digest (`c4d3b1ed…`) and template-feature digest (`0c951289…`) are equal across the two runs and equal to the sm_120 reference (`porting_docs/qualification_records/d5_template_digests_sm120_2026-10-14.json`); the same four hits (5ocm_A, 3zgy_A, 5a9s_B, 5a9r_A) and every per-key template tensor match. Kalign was copied unchanged (same sha256 as the frozen sm_120 binary).

## Item 3: stock reproducibility

Deterministic recipe, default `auto` kernels (cuEquivariance), bf16, TF32 off, torch LayerNorm.

| Input | Result |
|---|---|
| `protein_200`, seed 101, two runs (job 43) | byte-identical, 15 of 15 artifacts |
| `7wux` (1,218 tokens, MSA, ligands), seed 101, two runs (job 43) | byte-identical, 15 of 15 artifacts |
| D1-D15 through `run_rows.py`, seeds 101 and 102: `off` traced (job 44) vs `off` untraced (job 46) | byte-identical, 480 of 480 artifacts |

A scan of the `protein_200` artifacts found no run-specific values. The reproducible configuration on sm_80 is `auto`, as on sm_120.

**Cross-GPU drift.** None of the 480 sm_80 `off` artifacts is byte-identical to `baseline_off.json` (sm_120). As expected, an A100 `off` run would fail the sm_120 drift check; the sm_80 contract needs its own baseline (plan, A100 continuity).

## Item 5: phase profile

Same inputs and settings as gnode2 job 48613 (unmodified defaults, torch LayerNorm, `auto` kernels, trunk cache off; seed 101 cold, 102 warm), job 50. Warm, synchronized wall time; memory is peak allocated device memory within the phase. In parentheses: the sm_120 value.

| Phase | `7r6r` 245 tok | `1a14` 612 tok | `7wux` 1,218 tok |
|---|---|---|---|
| featurization | 0.6 s (0.2) | 2.4 s (1.7) | 3.0 s (4.0) |
| trunk | 6.7 s (4.2) / 3.8 GiB | 44.7 s (37.7) / 7.5 GiB | 246.8 s (175.8) / 18.8 GiB |
| structural expansion | 0.2 s (0.2) / 4.9 GiB | 1.7 s (1.2) / 13.7 GiB | 9.4 s (10.3) / 43.6 GiB |
| diffusion | 20.8 s (14.7) / 4.2 GiB | 58.6 s (29.4) / 10.7 GiB | 146.1 s (87.7) / 32.2 GiB |
| confidence | 1.1 s (0.6) / 4.6 GiB | 7.2 s (4.5) / 11.4 GiB | 37.6 s (21.0) / 35.0 GiB |
| output writing | 0.9 s (0.4) | 4.6 s (2.1) | 16.8 s (7.3) |
| total | 30.4 s (20.3) | 119.2 s (76.6) | 460.1 s (306.2) |

- The A100 is 1.5x (total) slower than gnode2's power-limited RTX PRO 6000 here: the trunk 1.2-1.6x, diffusion 1.4-2.0x. Diffusion is a larger share on sm_80 (49% for 1a14 vs 38%).
- Device-memory peaks per phase are identical to sm_120 (same stack, same shapes); the peak is again structural expansion (43.6 GiB at 1,218 tokens).
- Host memory peak 11.1 GiB (7wux), as on sm_120. The job ran while the other GPU was idle; one run per configuration, so no variance is reported.

## Item 6: pre-fix baselines and stream diagnostics

| Baseline | Result |
|---|---|
| Single GPU, `LAYERNORM_TYPE=fast_layernorm`, two runs (`protein_200`) | byte-identical, 15 of 15; no fallback warning |
| Single GPU, trunk cache miss (key `2fc3174e…`, same as sm_120) then hit | byte-identical, 15 of 15 |
| Fold-CP, two A100s, torch LayerNorm, `--sample 1`, torch triangle kernels (job 51) | byte-identical, 3 of 3 |
| Fold-CP, `fast_layernorm`, same settings | byte-identical, 3 of 3 |

**Fold-CP needs no NCCL change on sm_80.** The env's torch-pinned NCCL 2.26.2 passed the minimal all-reduce and send/recv check and both Fold-CP baselines ran (123-145 s). On sm_120 the same NCCL crashed and the baselines needed 2.28.9 in a temporary env.

Targeted `fast_layernorm` checks (`layernorm_stream_diag.py --out`, job 53), pre-fix — the same failures as sm_120:
- **Side-stream correctness: fails** (5 of 5 trials for each shape and dtype, e.g. [65536, 128] bf16); the default-stream repeat is equal.
- **Graph capture: fails silently.** Capture and replay report ok for every shape, dtype and direction, but the replayed results do not match eager (`pass` false in all 12 cases).

## Item 8: random-input trace

`protein_200`: manifest check and fault-injection self-test pass; two traced runs match; traced and untraced artifacts are byte-identical. D1-D15 `off` traces: manifest and self-test pass on all 15 rows. Draw identity, comparing with `off`: torch triangle kernels (`raw` hashes) and `fast_layernorm` (`raw`) match on all 15 rows; fp32 (`cmp` hashes, after casting to the comparison dtype) matches on all 14 rows that ran. The sm_80 stock contrasts therefore share random inputs with `off`, as on sm_120.

## Tested D4 fp32 recipes do not fit on an 80 GB A100

Stock D4 (7wux, 1,218 tokens) with `--dtype fp32` ran out of device memory, in the full run (job 47) and in the tier 1 replay (job 49): in the trunk, cuEquivariance's fused triangle-multiplication kernel asked for 7.6 GiB with 74.5 GiB allocated. That kernel processes the whole pair tensor and ignores OpenDDE's chunk settings; reserved-but-unallocated memory was 0.4 GiB, so allocator settings would not help. A rerun with torch triangle kernels (user's option 2, job 55, to be compared with the bf16 torch-kernel run) passed the trunk and then ran out of memory in structural token expansion (triangle-attention chunk loop, 15.1 GiB requested), the stage that is also the bf16 peak (43.6 GiB). **D4 therefore has no fp32 contrast on sm_80** (option 1, the fallback agreed when the user chose option 2 on 2026-10-15). The other 14 rows have one. Stock bf16 fits every row, including D4. These failures establish a memory limit for the tested cuEquivariance and torch fp32 recipes at D4/H10's size (1,218 tokens); they do not establish that every fp32 implementation would fail. A forced smaller triangle-attention chunk was not tried and would change more than precision. Keep the missing D4 fp32 contrast explicit in the sm_80 audit. These failures alone do not trigger `big`: stock bf16 fits, and that decision still depends on intended `fast` peaks and the agreed memory margin.

## Calibration against the frozen sm_120 limits

All D1-D15 runs used `run_rows.py` with contract-6 provenance at seeds 101 and 102 (five samples each): `off` traced (44), `off` untraced (46), torch triangle kernels traced (45), fp32 traced (47; D4 failed, above), `LAYERNORM_TYPE=fast_layernorm` traced (48), and single-cycle tier 1 captures of `off` with torch-kernel and fp32 replays (49; D4 fp32 failed). Analysis job 54 (CPU) evaluated them with the frozen sm_120 limit files (sha256 `b4b8fdd3…` tier 1, `d0294133…` tier 2). This shows how the sm_120 contract would treat sm_80 stock contrasts; it does not set sm_80 limits.

**Tier 1 (single cycle): passes with margin.** Torch kernels 32 of 32 checks, fp32 30 of 30 (D4 absent). Worst values as a fraction of the limit: pair `rel_fro` 0.30 (kernel) and 0.42 (fp32), pair tail 0.37 and 0.47, single 0.37 and 0.45, input embedding 0 and 0.37, distogram argmax changes 0 and 0.01, confidence head on fixed inputs at most 0.20 (PAE/PDE, kernel) and exactly 0 in fp32 (the head runs in fp32). These ratios are close to sm_120's.

**Tier 2 harmless contrasts: not all pass under the sm_120 limits.**

| Contrast | Jobs passing | Failing gates (value vs limit) |
|---|---|---|
| Torch triangle kernels | 12 of 16 | D3 and D13 `1a14_unconstrained` paired lDDT polymer/protein 0.9473 vs ≥0.95; D9 whole-ensemble spread ratio (`ensemble_spread_ratio`) 1.301 vs ≤1.25 (archived key and value confirmed 2026-10-15); D10 ensemble reference fnonnat median rise 0.0626 vs ≤0.06 |
| bf16 vs fp32 (D4 absent) | 12 of 15 | D3 and D13 `1a14_unconstrained` lDDT 0.9474 vs 0.95; D10 lDDT 0.9486 vs 0.95 and pair interface Fnat 0.673 vs ≥0.70 |
| `fast_layernorm` | 14 of 16 | D3 and D13 `1a14_unconstrained` lDDT 0.9496 vs 0.95 |

The D9 label above follows `tier2_gates.py`: `ensemble_spread_ratio` gates the ratio of median pairwise anchor RMSD across the whole ensemble. `spread_same_seed_*` is reported only; the separate same-seed duplicate gates still apply. The archived evaluation confirms key `ensemble_spread_ratio` and value 1.30097 (`spread_cand` 0.6483 / `spread_off` 0.4983; same-seed spreads 0.637 and 0.492, report-only).

D3 and D13's `1a14_unconstrained` job are the same job (one result, counted twice, as on sm_120). Every harmless contrast passed all 16 jobs on sm_120 under these limits. The failures are close to the limits; the `1a14_unconstrained` lDDT failure appears in all three contrasts, so the sm_80 stock contrast on that input is wider than sm_120's. The sm_80 limit audit (`limit_audit.py`, precision vs kernel) marks two groups where the kernel swap binds and fails the precision-only rederivation: flexible/epitope reference fnonnat median (kernel 0.0626 at D10, limit 0.06) and the low/stable spread ratio (kernel 0.263 at D9 against |log| limit 0.223). D4's precision values are missing from the audit.

Zn: the Zn claim is "qualified" for D14 and D15 under all three harmless contrasts.

**Negative controls (built from sm_80 `off`, sm_120 limits): every applicable job fails**, except D1 under 0.5 Å jitter, the same declared limitation as on sm_120.

| Control | Jobs failing |
|---|---|
| displaced | 12 of 12 |
| hinge | 14 of 14 |
| jitter | 13 of 14 (D1 passes) |
| collapsed | 14 of 14 |
| ligand out of pocket (D4, D5, D8) | 3 of 3 |
| guidance removed (D9, D10, D12) | 3 of 3 |
| Zn moved / Zn wrong site (D14, D15) | 2 of 2 each |
| partial collapse 0.1 Å / half size | 15 of 15 each |

The half-size partial-collapse sizes came from sm_80 `off` (half the closest same-seed pair from the kernel measurement): 0.06-0.83 Å.

## What this means for the sm_80 contract

- The sm_80 checks support stock reproducibility, trace-manifest coverage, matching D5 template digests and tier 1 margins under the tested stack. They do not establish byte-identical predictions across GPUs.
- The sm_120 tier 2 limits cannot be reused as they stand: three harmless sm_80 contrasts fail them, mainly paired lDDT on `1a14_unconstrained`, plus near-limit D9/D10 gates. sm_80 needs its own calibrated limits (the plan already requires an sm_80 freeze), and its own `baseline_off.json`.
- D4 has no fp32 contrast under the tested recipes on 80 GB. Its kernel and `fast_layernorm` results provide narrower coverage; any limit supported only by those contrasts must declare the precision gap and its reduced assurance (protocol E6).
- These are calibration measurements only; choosing sm_80 limits, the contract structure (shared family or GPU-scoped) and the sm_80 freeze are user decisions. Protocol E6 requires a per-gate derivation record and negative-control reevaluation under the proposed sm_80 limits, before tuning.

## Outputs

On gnode2 at `/mnt/compbio-data/yang.su/opendde-a100/2026-10-14/`:
- `runs/step0/{cpu-42,ab-43,profile-50,foldcp-51,lndiag-53}`, `runs/smoke-41`;
- `runs/cal/{off,repeat,k,fp32,ln,t1_off,t1_k,t1_fp32,cap_off,cap_k,cap_fp32,fp32_torch,controls,analysis}` (tier 1 captures kept as the reference copy; do not regenerate);
- `logs/`, `jobs/` (as run on the VM).
