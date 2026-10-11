# Step 0 results (sm_120)

Results for step 0 of `inline_optimizations_plan.md` (revision 19) on gnode2. Runs are identified by Slurm job ID; prediction artifacts and logs stay on gnode2 and are not committed.

## Environment and source

- Host: gnode2, NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (sm_120), driver 595.84. Jobs ran in the `interactive` partition.
- Stack (`opendde-dev`): Python 3.11.17, torch 2.7.1+cu128, Triton 3.3.1, cuEquivariance 0.10.0, NumPy 2.4.1, SciPy 1.17.1, RDKit 2025.09.3, biotite 1.4.0.
- Weights: the shared read-only store; `opendde.pt` matches the 1.2 manifest sha256.
- Source: commit `0238914` plus the uncommitted step 0 instrumentation (items 4 and 8 below). Every GPU job saved `git diff HEAD` (sha256 `420674bf…`) and a tarball of untracked files with its results. The instrumentation is off by default; job A showed it leaves stock artifacts byte-identical when the draw recorder is on.

## Item 2: environment checks

| Check | Result |
|---|---|
| CPU test suite (`pytest tests -m "not network"`, CPU only, job 48606) | 783 passed, 4 skipped (Apple MPS only), 4 deselected (network), 26 s |
| `tests/test_triton_compatibility.py`, `tests/smoke/test_cueq_runtime.py` (job 48612) | 3 passed |
| Stock `opendde pred` smoke run, `protein_200` (job 48612) | succeeded, 29 s, 5 CIFs and 10 confidence JSONs |
| Triangle kernels under `auto` | resolved to cuEquivariance for both multiplication and attention |
| `fast_layernorm` extension | builds and loads for sm_120 (first build about 140 s) |
| Representative Triton kernel | compiles and runs on Triton 3.3.1 (environment check, 2026-10-09) |

## Item 3: stock reproducibility

Deterministic recipe (`--deterministic true`), default `auto` triangle kernels (cuEquivariance), bf16, TF32 off, torch LayerNorm, seed 101.

| Input | Runs | Result |
|---|---|---|
| `protein_200` (200 tokens, no MSA), job 48612 | 2 | byte-identical, 15 of 15 artifacts |
| `7wux` (1,218 tokens, 9,142 atoms, MSA, ligands), job 48614 | 2 | byte-identical, 15 of 15 artifacts |

The reproducible configuration is therefore `auto`; step 1 can use it for `exact` byte-identity comparisons. The fallback to torch triangle kernels was not needed.

Prediction artifacts are the files under `<job>/seed_<seed>/predictions/`: `<name>_sample_<rank>.cif`, `<name>_summary_confidence_sample_<rank>.json` and `<name>_full_data_sample_<rank>.json`, five of each with the default `--sample 5`. A scan of all 15 found no output, home or data paths, hostname, user name or timestamp-like strings.

## Item 4: synchronized phase timing

Added `opendde/utils/instrumentation.py`, a private in-process hook module with no CLI option, config key or environment variable. On a single GPU, `_foldcp_stage_context` returns `instrumentation.phase(stage_name)` instead of `nullcontext()`; with no recorder installed this is still a null context. `scripts/qualification/run_opendde.py --phase-timing` installs a recorder that synchronizes the device at each boundary and records wall time and peak device memory per phase, and also times featurization, batch preparation and output writing in the runner.

## Item 5: phase profile

Unmodified 1.2 defaults (bf16, TF32 off, torch LayerNorm, `auto` kernels, trunk cache off, `--sample 5 --step 200 --cycle 10`), precomputed MSAs, no templates. One whole GPU (job 48613). Each input ran in its own process with seeds 101 (cold) and 102 (warm). Times are synchronized wall time; memory is peak allocated device memory within the phase.

The inputs are provisional profile inputs taken from the repository examples, not the frozen qualification set.

| Phase | `7r6r` 245 tok, warm | `1a14` 612 tok, warm | `7wux` 1,218 tok, warm |
|---|---|---|---|
| featurization | 0.2 s | 1.7 s | 4.0 s |
| trunk | 4.2 s (21%) / 3.8 GiB | 37.7 s (49%) / 7.5 GiB | 175.8 s (57%) / 18.8 GiB |
| structural expansion | 0.2 s / 4.9 GiB | 1.2 s / 13.7 GiB | 10.3 s / 43.6 GiB |
| diffusion | 14.7 s (72%) / 4.2 GiB | 29.4 s (38%) / 10.7 GiB | 87.7 s (29%) / 32.2 GiB |
| confidence | 0.6 s / 4.6 GiB | 4.5 s / 11.4 GiB | 21.0 s / 35.0 GiB |
| output writing | 0.4 s | 2.1 s | 7.3 s |
| total | 20.3 s | 76.6 s | 306.2 s |

- Cold runs took 3.9 s, 3.1 s and 6.2 s longer than warm. Featurization accounts for 3.3 s, 2.8 s and 4.8 s of that; every model phase was within about 2% of its warm time.
- `1a14` has 10,448 MSA rows against 703 for `7r6r` and 1,302 for `7wux`, which accounts for its relatively heavy trunk.
- The device-memory peak is in structural token expansion, not the trunk: 43.6 GiB at 1,218 tokens. If it scales roughly with the square of the token count (an extrapolation from one point, not measured), a 96 GB card runs out near 1,750 tokens. This bears on the `big` decision.
- Host memory peaked at 11.1 GiB (`7wux`).

## Item 6: pre-fix baselines and stream diagnostics

Stock CLI, default stream, deterministic recipe, `auto` kernels, `protein_200`, seed 101 (job 48614).

| Baseline | Result |
|---|---|
| Single GPU, `LAYERNORM_TYPE=fast_layernorm`, two runs | byte-identical, 15 of 15 artifacts |
| Single GPU, trunk cache miss (writes key `2fc3174e…`) then hit | hit and miss byte-identical, 15 of 15 artifacts |
| Fold-CP, default LayerNorm, two GPUs, `--sample 1`, torch triangle kernels (job 48707) | byte-identical, 3 of 3 artifacts; needs NCCL 2.28.9 (below) |
| Fold-CP, `fast_layernorm`, same settings (job 48729) | byte-identical, 3 of 3 artifacts; needs NCCL 2.28.9 (below) |

Targeted `fast_layernorm` checks (`scripts/qualification/layernorm_stream_diag.py`), pre-fix:

- **Side-stream correctness: fails.** On a non-default stream, with a controlled delay before the input and gradient writes, every trial (5 of 5, for each shape and dtype) produced non-finite outputs and gradients. The forward kernel read the input before the delayed write on its stream completed. The same kernel on the default stream is repeatable and matches torch LayerNorm (fp32 difference 0; bf16 within 0.031).
- **Graph capture: fails silently.** Capture raised no error, but PyTorch warned that the captured graph was empty. Replay with a new input left the LayerNorm output and the input and weight gradients stale; only the bias gradient, computed outside the extension, matched. Graph capture of `fast_layernorm` is therefore wrong without any exception, which is worse than a capture failure.

### Fold-CP blocker: NCCL 2.26.2 on sm_120

Every Fold-CP run (jobs 48618 and 48623) aborted with `CUDA error: an illegal memory access` in the NCCL watchdog, right after environment initialization and before the model loaded. A minimal two-process `all_reduce` reproduces it outside OpenDDE, so the cause is the communication stack, not OpenDDE (job 48626):

| NCCL configuration (two RTX PRO 6000, PCIe, no NVLink, IOMMU passthrough) | Minimal all-reduce |
|---|---|
| 2.26.2+cuda12.2 (the `nvidia-nccl-cu12` that torch 2.7.1+cu128 pins), default P2P transport | illegal memory access |
| 2.26.2, `NCCL_P2P_DISABLE=1` (shared-memory transport) | illegal memory access |
| 2.26.2, `NCCL_PROTO=Simple` | illegal memory access |
| 2.26.2, `NCCL_CUMEM_ENABLE=0` | illegal memory access |
| 2.26.2, `NCCL_P2P_DISABLE=1 NCCL_PROTO=Simple` | illegal memory access |
| 2.28.9+cuda12.9, preloaded from another shared environment | passes (all-reduce and send/recv) |
| 2.30.7+cuda12.9, preloaded from another shared environment | passes |

With NCCL 2.28.9 preloaded, a two-GPU Fold-CP run of `protein_200` completed in 28 s. The production `opendde` environment on gnode2 pins the same NCCL 2.26.2, so its Fold-CP mode is presumably affected as well (not tested).

**Decision (user, 2026-10-10):** produce the Fold-CP baselines in a temporary conda environment and delete it afterwards. The temporary environment was a clone of `opendde-dev` whose only package difference was `nvidia-nccl-cu12` 2.28.9 instead of 2.26.2 (pip notes that torch 2.7.1 declares `==2.26.2`). Its `fast_layernorm` extension was built once, single-process, in a separate extension cache before `torchrun` started. Both Fold-CP baselines above came from it at commit `45bb734`; the environment and its extension cache were then deleted. Later Fold-CP comparisons against these baselines must use NCCL 2.28.9 as well.

**Fold-CP rank divergence (stock behavior).** The first `fast_layernorm` Fold-CP attempt in the temporary environment, before its compiler paths were set, hung until the job's time limit instead of failing. Both ranks tried to JIT-build the extension at once. One rank's build failed and it fell back to torch LayerNorm with a warning, while the other loaded the existing build, so the two ranks ran different LayerNorm implementations and deadlocked. The soft per-rank fallback that `LAYERNORM_TYPE` relies on is therefore not rank-consistent under Fold-CP when the extension build differs between ranks. This is outside the port's scope; it is recorded for the user to decide whether to act on it.

## Item 7: cross-precision contrast

### Draw match (pilot)

Traced stock runs of `protein_200`, deterministic, bf16 and `--dtype fp32` (job 48615): all 3,005 draws matched the expected manifest in both runs, and every draw matched byte for byte under both the raw and comparison-dtype hashes. All draws are fp32 in both precisions, because the sampler runs with autocast disabled.

The same holds with guidance. Rigid-contact guidance on `1a14_contact` (`--use_tfg_guidance true --enable_tf32 false`, MSA, deterministic, job 48620) drew 4,005 identical draws in bf16 and fp32 under both hashes.

The native bf16-versus-fp32 contrast therefore shares its random inputs for unguided and rigid-contact runs, and the replay diagnostic is not needed. The contrast distances themselves need the frozen qualification inputs and the comparison tooling, which are not built yet.

### Contrast distances

Jobs 48765 (bf16) and 48766 (fp32) ran every development row through `run_rows.py`: stock `off`, seeds 101 and 102, 5 samples, deterministic, TF32 off, torch LayerNorm, `sorted_by_ranking_score=false`, draw trace on, at commit `d5619dc` with the comparison tooling uncommitted (committed unchanged as `3c021ce`). The frozen-input digest check passed first. Every row ran except D5, which needed `kalign` and template mmCIF files that gnode2 lacked; it ran later with frozen template data (job 48810, below).

**Random inputs.** For all 13 rows, every bf16 draw matched its manifest and every fp32 draw, under the comparison-dtype hash (6,010 draws per unguided row including D5, 8,010 per guided row, 12,020 for D13). All draws are fp32 in both precisions. So the native contrast shares its random inputs on every row and isolates numerical precision; no replay is needed (plan, item 7). The D5 traces were empty; the first analysis job (48780) passed them vacuously, and `trace_tools.py` now fails a trace from a failed or empty run (job 48788 confirmed D5 now fails).

**Distances.** `compare_structures.py` paired sample k of bf16 with sample k of fp32 for each seed (10 pairs per row; job 48788), and sample k of seed 101 with sample k of seed 102 within the bf16 run (5 pairs; job 48780). All comparisons were conclusive: every lDDT had contacts and no ligand map search was truncated. Anchor is protein Cα, or C4′ for D6 and D7. Ligand RMSDs are symmetry-aware; D13's two jobs gave exactly D2's and D3's numbers.

| Row | bf16 vs fp32: anchor RMSD median / max (pairs > 2 Å) | bf16 vs fp32: polymer lDDT median / min | seed 101 vs 102: anchor RMSD median / max | seed 101 vs 102: polymer lDDT median / min |
|---|---|---|---|---|
| D1 protein, no MSA | 0.60 / 1.94 Å (0/10) | 0.976 / 0.810 | 7.21 / 13.90 Å | 0.527 / 0.463 |
| D2 protein–DNA | 0.34 / 1.17 Å (0/10) | 0.999 / 0.994 | 2.06 / 4.50 Å | 0.946 / 0.896 |
| D3 three chains | 2.17 / 15.92 Å (5/10) | 0.975 / 0.856 | 2.79 / 14.51 Å | 0.939 / 0.859 |
| D4 large, ligands | 0.11 / 0.14 Å (0/10) | 1.000 / 0.999 | 3.77 / 4.40 Å | 0.981 / 0.978 |
| D5 templates, ATR | 0.03 / 0.04 Å (0/10) | 1.000 / 0.997 | 0.75 / 1.00 Å | 0.976 / 0.971 |
| D6 RNA | 0.06 / 0.34 Å (0/10) | 1.000 / 0.982 | 2.31 / 2.42 Å | 0.927 / 0.914 |
| D7 DNA duplex | 0.01 / 0.11 Å (0/10) | 1.000 / 0.998 | 0.20 / 0.48 Å | 0.998 / 0.996 |
| D8 covalent glycan | 2.47 / 20.57 Å (5/10) | 0.917 / 0.682 | 6.06 / 27.33 Å | 0.825 / 0.673 |
| D9 rigid contact | 0.26 / 0.94 Å (0/10) | 0.998 / 0.973 | 0.60 / 0.73 Å | 0.982 / 0.969 |
| D10 epitope | 8.42 / 16.66 Å (9/10) | 0.918 / 0.882 | 10.58 / 19.06 Å | 0.916 / 0.866 |
| D11 generic, `rho=0` | 2.57 / 19.95 Å (5/10) | 0.990 / 0.899 | 12.27 / 22.69 Å | 0.907 / 0.867 |
| D12 generic, `rho=1.0` | 2.55 / 19.98 Å (5/10) | 0.992 / 0.900 | 12.33 / 22.64 Å | 0.907 / 0.867 |

D11 and D12 exclude the constraint residues and neighbours within 3 positions.

Ligands, bf16 vs fp32 (max RMSD over 10 pairs): D4's specific ligand 6OI 0.93 Å (seed contrast 2.11 Å); D5's ATR 0.45 Å with centroid shifts under 0.06 Å (seed contrast 1.13 Å); D4's PEG fragment P4G 31.1 Å and Mg²⁺ ions up to 29.1 Å, which also move 30–66 Å between seeds; D8's covalent NAG 25.5 Å, moving with the chain rearrangement in the same samples, and ≤ 1.3 Å in the samples whose anchor RMSD is ≤ 2.5 Å.

Constraint distances, largest bf16-vs-fp32 difference of a constrained Cα–Cα distance: D9 0.44 Å, D11 0.06 Å, D12 3.69 Å (seed contrast 0.76, 0.12 and 4.20 Å).

Confidence: the largest bf16-vs-fp32 `ranking_score` difference per sample is ≤ 0.002 on D1, D2, D4, D6 and D7, 0.02–0.03 on D9 and D10, and 0.09–0.17 on D3, D8, D11 and D12. The top-ranked sample agreed between precisions for both seeds on D1, D2, D4, D6, D7, D9 and D11, and for one of the two seeds on D3, D8, D10 and D12.

What this shows:
- On the stable inputs (D1, D2, D4 polymer, D5, D6, D7, D9) the precision contrast is small, at most 1.94 Å anchor RMSD, and well below seed-to-seed variation.
- On flexible multi-chain inputs (D3, D8, D10, D11, D12) precision alone, with identical random inputs, switches about half the samples to a different chain arrangement, 9–20 Å away. That is as large as seed-to-seed variation. The distribution is bimodal: medians of 2–3 Å (8 Å on D10) with tails at 16–20 Å. This is the heavy tail the plan anticipated: per-sample maximum limits on these inputs would reject a harmless change.
- Nonspecific ligands and ions (D4's P4G and Mg²⁺) flip sites even under a precision change; the specific ligand 6OI stays under 1 Å.
- D12's constrained distances differ by up to 3.7 Å between precisions, against 0.06 Å for D11. Under `rho=1.0` the D12 per-sample gradient check cannot use a tight per-pair limit: a zeroed gradient (as at `rho=0`) moves the distances by a similar amount.

**Tooling cost.** The bf16-vs-fp32 comparison of all 12 rows took 17 min with a peak of 17.0 GB resident; the first attempt at 8 CPUs (15 GB default) was OOM-killed. Run it with at least 12 CPUs on gnode2, since CPU jobs set no `--mem`.

**D5 (template input).** The first attempt failed in 3 s: `opendde-dev` has no `kalign`, and gnode2's read-only store has no `search_database/mmcif`, where OpenDDE's default remote PDBe fetch would also write. With the user's approval (option 1):
- D5 uses the kalign 2.04 binary from gnode2's shared colabfold env (the version OpenFold's wrapper targets) through `--kalign_binary_path`; no environment changed.
- One stock run with fetching on (job 48809, bf16, seed 101) fetched the four templates OpenDDE selected under its 2021-09-30 template date cutoff, 5ocm, 3zgy, 5a9s and 5a9r, into a private directory, now read-only; the log reports "Found 4 templates".
- D5's manifest row now sets that directory, `fetch_remote=false` and a per-run template cache, through the host variables `OPENDDE_QUAL_DATA` and `OPENDDE_QUAL_KALIGN`; `external_digests.json` holds the four files' SHA-256 (`freeze_inputs.py check --external`).
- Job 48810 ran D5 at bf16 and fp32 from the frozen data: the bf16 seed-101 artifacts are byte-identical to the fetch run's (15 files), and the results are in the table above.

### Reference feasibility check (ensemble gate)

Job 48887 (CPU), stock only, after the revision 22 decision to base the tier 2 ensemble gate on deposited structures. The nine reference entries (development 7r6r, 1a14, 7wux, 9fm7, 9gmw, 9lh2; held-out 7pzb, 9sat, 9xqn) were fetched once from PDBe into the read-only `${OPENDDE_QUAL_DATA}/references/` and added to `external_digests.json`; no held-out input was run. `reference_metrics.py` scored all 140 bf16 and 140 fp32 item 7 samples (13 rows, two seeds of five) against them: reference-directional lDDT of polymer heavy atoms, and Fnat-style interface contacts (deposited interchain residue contacts within 5 Å that the prediction reproduces). The `9fm7` asymmetric unit holds one chain of the predicted dimer, so it is scored against assembly 1 (lDDT 0.90, Fnat 0.91). The 9gmw reference resolves 56 of D6's 86 nucleotides.

| Row | Reference lDDT median, bf16 / fp32 (seed 101 / 102) | Fnat median, bf16 / fp32 | Displaced-chain control: lDDT / Fnat median | Ensemble spread, bf16 / fp32 |
|---|---|---|---|---|
| D2 | 0.789 / 0.788 (0.787 / 0.791) | 0.684 / 0.684 | 0.650 / 0.297 | 3.88 / 3.84 Å |
| D3 | 0.876 / 0.879 (0.876 / 0.879) | 0.448 / 0.453 | 0.845 / 0.026 | 10.47 / 2.10 Å |
| D4 | 0.923 / 0.923 (0.923 / 0.923) | 0.897 / 0.901 | 0.855 / 0.426 | 3.54 / 3.53 Å |
| D5 | 0.908 / 0.908 (0.910 / 0.907) | 0.904 / 0.898 | 0.744 / 0.188 | 0.84 / 0.82 Å |
| D6 | 0.824 / 0.823 (0.823 / 0.825) | single chain | not applicable | 1.94 / 1.98 Å |
| D8 | 0.783 / 0.780 (0.784 / 0.783) | 0.069 / 0.062 | 0.782 / 0.069 | 5.94 / 16.21 Å |
| D9 | 0.909 / 0.910 (0.909 / 0.910) | 0.812 / 0.802 | 0.860 / 0.208 | 0.59 / 0.59 Å |
| D10 | 0.884 / 0.882 (0.883 / 0.884) | 0.469 / 0.479 | 0.852 / 0.062 | 10.58 / 12.86 Å |
| D11 | 0.904 / 0.905 (0.905 / 0.903) | 0.816 / 0.816 | 0.878 / 0.026 | 12.06 / 16.16 Å |
| D12 | 0.903 / 0.904 (0.903 / 0.903) | 0.816 / 0.816 | 0.877 / 0.026 | 12.01 / 15.67 Å |

The control moves the smallest mapped polymer chain 8 Å outward and turns it 20° in every bf16 sample, then rescores. Ensemble spread is the median pairwise anchor RMSD among a run's ten samples; a collapsed ensemble has spread 0. D13's jobs match D2 and D3. Rows without a reference: D1 mean pLDDT median 48.0 / 48.8, spread 4.53 / 4.58 Å; D7 96.2 / 96.1, spread 0.25 / 0.25 Å.

What this shows:
- With ten samples, the median reference lDDT of stock bf16 and fp32 agrees within 0.004 on every row, as do the two seeds, and the median Fnat within 0.010. The displaced-chain control lowers the median lDDT by 0.026-0.164 and the median Fnat by 0.39-0.79 on every reference row except D8. A distribution gate on reference lDDT and Fnat is therefore feasible at ten samples for D2, D3, D4, D5 and D9-D12 (and D13).
- D8 is the exception: stock reproduces almost none of 9lh2's deposited interfaces (Fnat about 0.07, worst chain pair 0), so moving a chain changes neither metric. The reference gate checks only D8's chains individually.
- The interface results match the inputs' design: unconstrained 1a14 (D3) misses the antibody-antigen interface (worst-pair Fnat 0), and rigid-contact guidance (D9) recovers it (0.70-0.72).
- Ensemble spread separates a collapsed ensemble from stock on every row, but varies widely between precisions on flexible rows (D3 10.47 versus 2.10 Å, D8 5.94 versus 16.21 Å), so any spread limit must be a loose ratio.
- D6 is a single chain, so the displaced-chain control does not apply; it needs another negative control.

### Tier 1 calibration and tier 2 positive control

Jobs 48900-48903, stock only, commit `44e1fd1`, every development row, seeds 101 and 102, deterministic. Three configurations with the wrapper's tier 1 capture: `off` (bf16, default cuEquivariance triangle kernels), a same-precision kernel swap (bf16, `--triatt_kernel torch --trimul_kernel torch`) and fp32. The swap and fp32 runs also ran the confidence head on `off`'s recorded inputs. Hook checks: with capture on, all 13 rows' prediction artifacts were byte-identical to the item 7 bf16 run, and with capture and replay on, byte-identical to the item 7 fp32 run; on D2, replaying `off`'s own inputs gave zero difference and the hook's pTM and ipTM matched the summary JSON. Captures take 23 GB (`off`, swap) and 30 GB (fp32).

Tier 1 against `off`, largest value over the two seeds. "1a14 rows" are D3, D9-D12 and D13's `1a14` job, which share the same trunk inputs; D2 and D13's `7r6r` job are identical.

| Quantity | Kernel swap: other rows | Kernel swap: 1a14 rows / D8 | fp32: other rows | fp32: 1a14 rows / D8 |
|---|---|---|---|---|
| Input embedding, relative error | 0 | 0 / 0 | 0.0012-0.0018 | 0.0014 / 0.0015 |
| Single representation, relative error | 0.0050-0.0063 | 0.0115 / 0.0115 | 0.0076-0.0132 | 0.0122 / 0.0143 |
| Pair representation, relative error | 0.0178-0.0294 | 0.155 / 0.181 | 0.0370-0.184 | 0.168 / 0.236 |
| Pair representation, p99.9 error / RMS | 0.11-0.28 | 1.15 / 1.35 | 0.34-1.20 | 1.25 / 1.51 |
| Distogram, largest probability difference | 0.015-0.085 | 0.225 / 0.330 | 0.045-0.117 | 0.300 / 0.543 |
| Distogram, most likely bin changed | 0.3-2.1% | 1.6% / 1.4% | 0.9-8.1% | 1.8% / 2.0% |
| Same, reference top-two margin ≥ 0.1 | 0 | 0.058% / 0.009% | 0 | 0.080% / 0.047% |
| Fixed-input head, PAE / PDE probability | ≤ 0.0021 | ≤ 0.0021 | 0 | 0 |
| Fixed-input head, pLDDT / resolved probability, pTM, ipTM | < 1e-4 | < 1e-4 | 0 | 0 |

D1 (no MSA, low confidence) is the outlier among the other rows in fp32: pair error 0.184 and 8.1% changed bins.

What this shows:
- The input embedding has no triangle kernels, so the swap leaves it bit-identical; fp32 moves it by about 0.15%.
- On most rows the pair representation moves 2-3% under the swap and about 4% in fp32. On the multichain 1a14 and 9lh2 inputs it moves 15-24%: ten recycling cycles amplify small differences there, as the sampled structures already showed. A pair limit that is tight on the stable rows is weak on these.
- Changed distogram bins are almost all near-ties: with a top-two margin of at least 0.1 the changed fraction is at most 0.08%.
- The confidence head runs in fp32 with autocast disabled in every configuration, so on identical inputs fp32 reproduces `off`'s head exactly; the kernel swap, which reaches the head's own pairformer, moves PAE and PDE probabilities by at most 0.002 and pTM and ipTM by less than 1e-4. The fp32 contrast cannot set a head limit (it is zero).

Tier 2 positive control, `off` versus the kernel swap, against the revision 22 draft gates:
- Outcome counts: stable rows 10 of 10 (floor 9), D1 10 of 10 (floor 7), D3 8, D8 7, D11 4, D12 5 (floor 2), D10 0 (floor 0).
- Same-outcome polymer lDDT at least 0.927 (D8, draft 0.84), 0.969 on the other flexible rows and 0.976 or more on stable rows; per-type lDDT at least 0.998 on D2, D6 and D7. p95 error on D4-D7 at most 0.26 Å (draft 1.0 Å).
- Gated ligands: 6OI at most 0.86 Å with pocket Jaccard at least 0.92, ATR at most 0.43 Å, NAG at most 0.61 Å. (D4's 3.3 Å maximum is a report-only Mg²⁺ ion.)
- Confidence: |Δ `ranking_score`| at most 0.0003 on D2 and D4-D7, 0.010 on D9 and 0.053 elsewhere in same-outcome pairs; |Δ mean pLDDT| at most 1.01.
- Guidance: D9 contacts 10 of 10; D10 epitope 10 of 10, as `off`; D11 mean constrained distance 7.99 versus 8.00 Å; D12 6.17 versus 5.76 Å (draft: within 1.0 Å and at most 7.0 Å).
- Ensemble: median reference lDDT within 0.004, lowest-sample lDDT within 0.010 and median Fnat within 0.016 of `off`; spread ratios 0.84-1.51.

The harmless kernel swap passes every draft tier 2 gate, so none of them is too strict for it.

### Single-cycle tier 1 calibration

Jobs 49092-49095, stock only, commit `84f3d8e`: the same three configurations with tier 1 capture, run with `--cycle 1 --sample 1` on the eight distinct trunk inputs D1-D8 (D9-D13 reuse D2's and D3's trunk inputs; the full calibration gave them identical trunk metrics). One cycle still runs every trunk module once; it only drops the repeated recycling. Largest value over seeds 101 and 102:

| Quantity | Kernel swap, D1-D8 | fp32, D1-D8 | Full-cycle range, for comparison (swap / fp32) |
|---|---|---|---|
| Input embedding, relative error | 0 | 0.0012-0.0018 | same |
| Single representation, relative error | 0.0045-0.0072 | 0.0074-0.0091 | 0.005-0.012 / 0.008-0.014 |
| Pair representation, relative error | 0.0183-0.0301 | 0.0292-0.0408 | 0.018-0.181 / 0.037-0.236 |
| Pair representation, p99.9 error / RMS | 0.12-0.30 | 0.30-0.37 | 0.11-1.35 / 0.34-1.51 |
| Distogram, largest probability difference | 0.013-0.072 | 0.031-0.118 | 0.015-0.330 / 0.045-0.543 |
| Distogram, changed bins with margin ≥ 0.1 | 0 | 0 | up to 0.06% / 0.08% |
| Fixed-input head, PAE / PDE probability | ≤ 0.0020 | 0 | same |
| Fixed-input head, pLDDT, pTM, ipTM | < 1e-4 | 0 | same |

With one cycle, the 1a14 input (D3) and D8 behave like every other row: pair error 2.2% under the kernel swap against 15.5% and 18.1% with ten cycles, and D1's fp32 pair error falls from 18.4% to 4.0%. The spread across rows shrinks from a factor of ten to under two, so a single set of tier 1 limits can serve every row. A single-cycle `off` run of all eight inputs took about 8 min (sum of per-row times), against about 40 min for the ten-cycle, five-sample run.

### Tier 2 negative controls and limit proposal

Jobs 49121 (control build) and 49134-49142 (measurements), stock only, CPU. `make_controls.py` built six negative-control candidate runs from the stock `off` run of the tier 1 calibration (job 48900), copying confidence files unchanged so only structures differ:
- **displaced:** the smallest polymer chain moved 8 Å and turned 20° in every sample (11 multichain jobs);
- **hinge:** the last 30% of the largest chain turned 20° about the hinge residue (all 14 jobs, including single-chain D1 and D6);
- **jitter:** Gaussian noise of 0.5 Å per coordinate on every heavy atom (14 jobs);
- **collapsed:** each seed's samples replaced by its sample 0 (14 jobs);
- **ligand out:** each gated ligand moved 6 Å outward (D4, D5, D8);
- **guidance removed:** stock D11 (`rho=0`, generic restraint) as the D9 and D12 candidate, and stock unconstrained D3 as the D10 candidate.

`tier2_gates.py measure` scored these and three stock contrasts (bf16 versus fp32 from item 7, the kernel swap, and seed 101 versus 102) on paired, ensemble and guidance quantities. It adds a pairwise interface metric (the fraction of `off`'s interchain residue contacts within 5 Å that the candidate keeps), an ensemble contact-frequency distance (normalized L1 between the two runs' interchain contact frequencies, keyed by entity and residue), and the closest same-seed sample pair. `tier2_gates.py evaluate` applied the proposed limits (`scripts/qualification/inputs/tier2_limits.json`; protocol C9):

| Contrast or control | Jobs passing every gate |
|---|---|
| bf16 vs fp32 | 14 of 14 |
| Kernel swap | 14 of 14 |
| Seed 101 vs 102 (different random inputs; context) | 1 of 14 |
| Displaced chain | 0 of 11 |
| Hinge | 0 of 14 |
| Jitter | 1 of 14 (D1) |
| Collapsed ensemble | 0 of 14 |
| Ligand out | 0 of 3 |
| Guidance removed | 0 of 3 |

Notable results:
- A displaced DNA strand in D2 leaves the protein anchor unchanged, so its pairs stay same-outcome; the per-type DNA lDDT (0.546), the pairwise interface Fnat (0.43), the reference ensemble and the contact frequency (0.60) catch it.
- On D8 the displaced chain is not in contact in stock predictions, so contact frequency barely moves (0.05); every pair switches outcome, so the count gate catches it.
- The D12 gradient check rejects the guidance-removed control: stock `rho=0` samples give a mean constrained distance of 8.00 Å against 5.76 Å.
- Before the duplicate-sample gate, the collapsed control passed on D3, D8, D9, D10 and D13's `1a14` job; with it, every collapsed job fails (repeated samples have anchor RMSD 0, against at least 0.10 Å between any two stock samples of one seed).
- D1 jitter passes because D1's own precision contrast exceeds the noise; this is the limit of what a low-confidence input can show.

### Zn stock calibration (D14, D15)

Jobs 49181-49183 (stock `off` bf16, fp32 and `torch` triangle kernels, commit `0e10ff3`; seeds 101 and 102, 5 samples, deterministic, with the random-input trace and a single-cycle tier 1 capture) and 49186 (analysis). D14 is 6E4Y with four Zn, D15 is 9MUY with two (protocol, freeze record). `scripts/qualification/zn_sites.py` measures Zn placement: coordinating atoms are polymer N, O and S within 2.8 Å; paired ions are matched one-to-one by pocket-relative distance (the candidate's polymer atoms within 8 Å of the `off` ion superposed onto `off`'s), and deposited sites are scored after the reference chain assignment with ions assigned to sites one-to-one.

- **Paired draws and tier 1.** Every trace matches its manifest (6,010 draws per row); bf16 and fp32 draws agree under the comparable hash and the kernel swap's raw draws are identical. Single-cycle tier 1 is inside the approved limits on both rows: kernel swap / fp32 maxima are embedding 0 / 0.0015, single 0.0066 / 0.0085, pair 0.0195 / 0.0308, pair p99.9/RMS 0.24 / 0.33, no distogram change with margin at least 0.1, head probabilities 0.0019 / 0, pTM, ipTM and ranking combination 0.
- **Structures.** All ten pairs per row and contrast are same-outcome (anchor RMSD at most 1.75 Å). Lowest same-outcome lDDT: D14 0.989 under both; D15 0.94995 under fp32 and 0.964 under the kernel swap, so D15 would fail the proposed stable-class lDDT limit of 0.95 under the precision contrast and needs a row override if it joins the stable class. Ranking-score differences are at most 0.002; reference lDDT and Fnat medians move by at most 0.005.
- **Deposited Zn sites are not reproduced.** In D14, the constant-domain site (heavy His167, light Asn142/Asn143) holds a Zn coordinated by His167 and Asn143 (2 of its 3 deposited ligands) in 9 of 10 `off` samples; D14's two other sites with polymer ligands are never occupied. D15's two sites are each partly occupied in 1 of 10 samples. Stock instead places Zn at its own preferred sites: D14 heavy His61 (10 of 10 samples), light Asp156/His194 (9), heavy His167 with light Asn143 (9), light His96 with peptide Glu18 (4); D15 light His52 (8), light Asp150/His188 (6). Deposited-site occupancy is therefore reported, not gated.
- **Stock placement is reproducible under harmless changes.** Of the ions `off` binds (39 in D14, 20 in D15), the paired candidate ion sits within 2.0 Å in the pocket frame for 37 of 39 and 19 of 20 under fp32, and 38 of 39 and 19 of 20 under the kernel swap; the coordinating-residue sets are identical for 36 of 39 and 20 of 20 in both. The exceptions: one D14 ion in seed 102 sample 0 moves 40 Å under both contrasts, one D14 ion in seed 101 sample 0 moves 24 Å under fp32, and a single-Asp D15 ion moves 4.6-4.8 Å. A seed change (101 against 102, different random inputs) keeps 14 of 20 and 7 of 10.
- **Zn negative controls** (job 49203, `make_controls.py` kinds `zn_moved` and `zn_wrong_site`, built from stock `off`). In every sample one bound Zn, chosen with the sample's own random generator, either moves 6 Å out of its pocket away from its coordinating atoms (further in 2 Å steps up to 12 Å if that clashes) or moves 2.1 Å from a free His, Asp, Glu or Cys donor at least 8 Å from every Zn and 2.5 Å from every other heavy atom. Two D14 samples had no clash-free exit up to 12 Å and were copied unchanged; every other sample was changed. Polymer coordinates are untouched, so every sample pair stays same-outcome.

| Contrast | D14: same site / bound `off` ions | D15 |
|---|---|---|
| fp32 (harmless) | 37 / 39 (0.95) | 19 / 20 (0.95) |
| Kernel swap (harmless) | 38 / 39 (0.97) | 19 / 20 (0.95) |
| Seed 101 against 102 (same-outcome pairs only) | 14 / 20 (0.70) | 3 / 6 (0.50) |
| Moved Zn | 31 / 39 (0.79) | 10 / 20 (0.50) |
| Wrong site | 29 / 39 (0.74) | 10 / 20 (0.50) |

  "Same site" means the paired candidate ion lies within 2.0 Å of the `off` ion in the `off` pocket frame, counted over same-outcome pairs. A limit of 0.85 passes both harmless contrasts on both rows and fails both controls on both rows. The margin to the moved-Zn control on D14 is small because only one of four ions moves per sample and two samples were unchanged; a fault that displaces Zn in only a few samples would pass.

### Fail-closed tier 2 evaluation

Measurement jobs 49232-49242 and evaluation job 49252 (CPU), from the revised `tier2_gates.py` (protocol C9). Run directories combine the earlier stock runs: item 7 bf16/fp32 and tier 1 `off`/`torch`-kernel runs for D1-D13, the Zn calibration runs for D14 and D15. Every reference score succeeded on every sample of every row.

| Contrast or control | Jobs passing | Notes |
|---|---|---|
| bf16 vs fp32 | 16 of 16 | |
| Kernel swap | 16 of 16 | D10's same-outcome gates not applicable (no same-outcome pair; count floor 0) |
| Seed 101 vs 102 | 0 of 16 | different random inputs; also fails the sample-count gates by construction |
| Displaced chain | 0 of 12 | |
| Hinge | 0 of 14 | |
| Jitter | 1 of 14 | D1 passes (documented limitation) |
| Collapsed | 0 of 14 | |
| Ligand out of pocket | 0 of 3 | |
| Guidance removed | 0 of 3 | |
| Moved Zn, wrong-site Zn | 0 of 2 each | |

New-gate stock maxima (harmless contrasts) against the limits: same-outcome pairwise new-contact fraction 0.071 on stable rows (0.15), 0.278 on D9 (0.60), 0.198 on D10 (0.40) and 0.395 on flexible rows (0.80); per-type reference lDDT drop 0.004 median and 0.011 lowest (0.01 / 0.02); median reference fnonnat rise 0.005 on stable rows (0.02) and 0.028 on flexible rows (0.06); median lowest per-chain-pair Fnat drop 0.006 (0.05).

### Partial-collapse control

Control definition and applicable rows recorded before measurement (protocol C8; copy on gnode2 at `tier2v2/partial_collapse_declaration.md`, 20:12 UTC). Build job 49254, measurement jobs 49255-49259 (stock contrasts re-measured to record every same-seed pair distance), evaluation 49271. Samples 1-3 of each seed are copies of sample 0 with Gaussian noise; size 1 is 0.1 Å, size 2 half the row's smallest stock closest-same-seed pair (0.12-0.71 Å). D7 is exempt.

| Same-seed pairs below 0.75 × `off`'s closest same-seed pair (of 20) | Largest count | Jobs over 3 |
|---|---|---|
| bf16 vs fp32 | 2 (D3) | 0 of 16 |
| Kernel swap | 1 | 0 of 16 |
| Seed 101 vs 102 (10 pairs, one seed of `off`) | 5 (D3) | D3, D8, D13 `1a14` |
| Partial collapse, 0.1 Å | 12 | 15 of 15 |
| Partial collapse, half size | 12 | 15 of 15 |

With the gate added, both harmless contrasts still pass every gate on all 16 jobs, and both control sizes fail on every applicable job. Without it, the 0.1 Å control passes on D3, D8-D12, D13 `1a14` and D14, and the half-size control on D3, D11, D12, D13 `1a14` and D14; on the stable rows the copies also fail the outcome count, because a copy of sample 0 is paired with `off`'s sample 1-3. The median same-seed spread ratio could not serve: bf16 versus fp32 alone gives 0.18 on D3.

### Limit provenance audit

`scripts/qualification/limit_audit.py` (protocol C8) reads the current evaluation outputs of both harmless contrasts (`tier2v2/eval_precision.json` and `eval_kernel.json`, evaluation 49271; same `tier2_limits.json` and `tier2_gates.py` as the committed files) and writes `tier2v2/limit_audit.json`. Each gated value is put on a scale where the limit is an upper bound (deficit from 1 for lDDT, Fnat, pocket Jaccard and Zn same-site; |log ratio| for the spread ratio). Limits are grouped by gate and value, with per-type lDDT pooled into the all-polymer limit it shares (C9). The precision-only limit is twice the worst bf16-versus-fp32 value, rounded up to 1, 1.5, 2, 2.5, 3, 4, 5, 6 or 8 times a power of ten (C9's doubling rule). A group needs action only when the kernel swap is binding and the current limit is looser than the precision-only limit.

| Gate (limit) | Classes | Limit, as badness | bf16 vs fp32 worst | Kernel swap worst | Binding | Precision-only | Action |
|---|---|---|---|---|---|---|---|
| `ensemble_contact_frequency` (0.2) | stable | 0.2 | 0.0919 (D9) | 0.0758 (D9) | precision | 0.2 | none |
| `ensemble_contact_frequency` (1.0) | epitope, flexible | 1 | 0.745 (D8) | 0.755 (D10) | kernel | 1.5 | none |
| `ensemble_plddt_median` (1.5) | all | 1.5 | 0.763 (D1) | 0.213 (D8) | precision | 2 | none |
| `ensemble_ref_fnat_median` (0.05) | reference rows | 0.05 | 0.0104 (D9) | 0.0052 (D9) | precision | 0.025 | none |
| `ensemble_ref_fnat_min_pair_median` (0.05) | reference rows | 0.05 | 0.0062 (D5) | 0 | precision | 0.015 | none |
| `ensemble_ref_fnonnat_median` (0.02) | stable | 0.02 | 0.0047 (D9) | 0.0043 (D2) | precision | 0.01 | none |
| `ensemble_ref_fnonnat_median` (0.06) | epitope, flexible | 0.06 | 0.0074 (D8) | 0.0283 (D12) | kernel | 0.015 | **rederive: kernel fails** |
| `ensemble_ref_lddt_*_lowest` (0.02) | reference rows | 0.02 | 0.0111 (D8) | 0.0104 (D8) | precision | 0.025 | none |
| `ensemble_ref_lddt_*_median` (0.01) | reference rows | 0.01 | 0.0031 (D8) | 0.0037 (D10) | kernel | 0.008 | **rederive: kernel passes** |
| `ensemble_spread_ratio` (0.1-10) | epitope, flexible | 2.30 | 1.61 (D3) | 0.413 (D12) | precision | 4 | none |
| `ensemble_spread_ratio` (0.8-1.25) | low, stable | 0.223 | 0.0655 (D14) | 0.0527 (D9) | precision | 0.15 | none |
| `lddt_*` (0.78) | low | 0.22 | 0.190 (D1) | 0.0052 (D1) | precision | 0.4 | none |
| `lddt_*` (0.85) | D8 | 0.15 | 0.110 (D8) | 0.0733 (D8) | precision | 0.25 | none |
| `lddt_*` (0.90) | D15 | 0.1 | 0.0500 (D15) | 0.0358 (D15) | precision | 0.15 | none |
| `lddt_*` (0.95) | epitope, flexible, stable | 0.05 | 0.0269 (D9) | 0.0307 (D3) | kernel | 0.06 | none |
| `ligand_pocket` (0.6) | D4, D5, D8 | 0.4 | 0.0909 (D4) | 0.0769 (D4) | precision | 0.2 | none |
| `ligand_rmsd` (2.0 Å) | D4, D5, D8 | 2 | 0.934 (D4) | 0.862 (D4) | precision | 2 | none |
| `mean_plddt` (2.0) | all | 2 | 1.03 (D1) | 1.01 (D8) | precision | 2.5 | none |
| `p95_error` (1.0 Å) | D4-D7 | 1 | 0.287 (D6) | 0.258 (D6) | precision | 0.6 | none |
| `pair_interface_fnat` (0.55) | D8 | 0.45 | 0.339 (D8) | 0.245 (D8) | precision | 0.8 | none |
| `pair_interface_fnat` (0.60) | D3, D11, D12, D13 `1a14` | 0.4 | 0.146 (D11) | 0.280 (D12) | kernel | 0.3 | **rederive: kernel passes** |
| `pair_interface_fnat` (0.70) | D9, D10 | 0.3 | 0.219 (D9) | 0.170 (D9) | precision | 0.5 | none |
| `pair_interface_fnat` (0.90) | stable | 0.1 | 0.0677 (D15) | 0.0519 (D15) | precision | 0.15 | none |
| `pair_interface_fnonnat` (0.15) | stable | 0.15 | 0.0709 (D15) | 0.0699 (D15) | precision | 0.15 | none |
| `pair_interface_fnonnat` (0.40) | D10 | 0.4 | 0.198 (D10) | no same-outcome pair | precision | 0.4 | none |
| `pair_interface_fnonnat` (0.60) | D9 | 0.6 | 0.278 (D9) | 0.226 (D9) | precision | 0.6 | none |
| `pair_interface_fnonnat` (0.80) | flexible | 0.8 | 0.395 (D8) | 0.394 (D12) | precision | 0.8 | none |
| `ranking_score` (0.01) | low, stable | 0.01 | 0.0041 (D5) | 0.0004 (D15) | precision | 0.01 | none |
| `ranking_score` (0.05) | D9 | 0.05 | 0.0241 (D9) | 0.0103 (D9) | precision | 0.05 | none |
| `ranking_score` (0.10) | epitope, flexible | 0.1 | 0.0783 (D8) | 0.0531 (D8) | precision | 0.2 | none |
| `zn_same_site` (0.85) | D14, D15 | 0.15 | 0.0513 (D14) | 0.0500 (D15) | precision | 0.15 | none |

Rules, not stock-variation limits: outcome-count floors, guidance counts and tolerances (C7), the 0.01 Å exact-duplicate floor and the near-duplicate count (declared from bf16 versus fp32 only, C8). The confidence limits, ligand limits, the p95 limit, the same-outcome lDDT rule (stock minimum minus 0.05, capped at 0.95), the outcome-count rule and the guidance tolerances were proposed in commit `983367e` (2026-10-11 02:25 UTC), before the kernel-swap run (job 48901, finished 05:54 UTC), so the kernel swap cannot have set them.

Findings:
- **One limit was set by the kernel swap and cannot be rederived from precision alone:** the flexible and epitope median reference fnonnat rise, 0.06. Its value is the doubled kernel-swap maximum on D12 (0.028); bf16 versus fp32 gives at most 0.0074, and the precision-only limit 0.015 would fail the harmless kernel swap. Precision alone therefore underestimates harmless variation for this statistic. The limit also applies to H1-H5 and H7. No development negative control depends on it: every control job that fails it also fails at least one other gate.
- **Two limits are looser than precision alone supports, and the kernel swap passes the tighter value:** the median reference lDDT drop, 0.01 against a precision-only 0.008 (kernel swap 0.0037), and the flexible pairwise interface Fnat, 0.60 against a precision-only 0.70 (kernel swap 0.720, margin 0.020). The Fnat limit applies only to D3, D11, D12 and D13 `1a14`; held-out flexible rows take D8's 0.55. Tightening a limit cannot make a failing control pass, and bf16 versus fp32 passes the precision-only value by construction.
- **Kernel binding without action:** the flexible and epitope contact-frequency L1 (1.0 against a precision-only 1.5) and the same-outcome lDDT 0.95 (precision-only 0.94; the limit also follows the earlier precision rule).
- **Confirmed clean:** the Zn threshold (precision binding), D10's paired limits (no kernel-swap pair) and the near-duplicate gate.

**Tightenings applied (2026-10-11, user decision).** `tier2_limits.json` now has median reference lDDT drop 0.008 (all-polymer and per-type) and flexible pairwise interface Fnat 0.70. Re-evaluation of every contrast and control with the tightened file (job 49299, CPU; `tier2v3/eval_*.json` and `tier2v3/limit_audit.json`):

| Contrast or control | Jobs passing | Change from the previous limits |
|---|---|---|
| bf16 vs fp32 | 16 of 16 | none |
| Kernel swap | 16 of 16 | none |
| Seed 101 vs 102 | 0 of 16 | interface Fnat now also fails on D3 and D13 `1a14` |
| Collapsed | 0 of 14 | median reference lDDT now also fails on D8 |
| Guidance removed | 0 of 3 | median reference lDDT now also fails on D10 |
| Displaced, hinge, jitter, ligand out, Zn moved, Zn wrong site, partial collapse (both sizes) | unchanged (D1 jitter, see below) | none from the tightening |

Controls measured before the near-duplicate gate (job 49252's measurements) now fail it as well, because they lack same-seed pair distances; that comes from the fail-closed evaluator, not the tightening. D1's jitter control now reports a failure only for that reason; it passes every gate its measurement covers, so D1's documented jitter limitation stands. The audit rerun leaves one flagged group, the flexible and epitope median reference fnonnat limit (0.06), which awaits the third harmless contrast.

## Item 8: random-input trace

Six RNG call sites carry stable symbolic names: `sampler.initial_noise`, `sampler.step_noise`, `augmentation.rotation`, `augmentation.translation`, `tfg.eps` and `tfg.outer_noise`. Each draw is captured before scaling and hashed per original sample index (`run_opendde.py --draw-trace`). `scripts/qualification/trace_tools.py` derives the expected manifest from the recorded sampling settings.

| Check (`protein_200`, unguided, job 48612) | Result |
|---|---|
| Manifest | 3,005 expected, 3,005 observed, no missing, extra or duplicate keys |
| Self-test: omit each site, duplicate a record, omit one and duplicate another | every fault detected |
| Two traced stock runs | all 3,005 raw hashes equal |
| Traced versus untraced artifacts | byte-identical |

Guided runs (job 48620, `1a14`, MSA, deterministic, rigid-body guidance as in `examples/tfg`):

| Check | Result |
|---|---|
| Manifest, contact constraint and pocket (epitope) constraint | 4,005 expected and observed each: the unguided sites plus one `tfg.outer_noise` per step; no `tfg.eps` because the default Monte Carlo standard deviation is 0 |
| Self-test, both inputs | every fault detected |
| Two traced contact runs | all 4,005 raw hashes equal; artifacts byte-identical |
| Traced versus untraced contact artifacts | byte-identical |

So guided stock runs with rigid contact are also byte-reproducible under the deterministic recipe.

CPU unit tests (`tests/test_instrumentation.py`) cover unguided runs and guidance with zero and nonzero Monte Carlo noise, including per-site counts, scopes and unchanged outputs.

## D12 rho sweep

Job 48751, `1a14_contact` (D11 input) with `OPENDDE_RIGID_CONTACT=off`, MSA, seed 101, 5 samples, deterministic, bf16, TF32 off, stock `off`. `rho` was set with `run_opendde.py --set sample_diffusion.guidance.rho=…`, with `sorted_by_ranking_score=false` so samples pair by original index. A CPU check in the job confirmed the override reaches the guidance config. Comparisons are against the `rho=0` sample with the same index.

| `rho` | Cα RMSD vs `rho=0` | Max atom shift | Restrained Cα–Cα distances (3.5–8 Å) | Ranking score | Diffusion / total |
|---|---|---|---|---|---|
| 0 | — | — | all ≈ 8.0 Å | 0.51–0.61 | 36 s / 90 s |
| 0.01 | 0.01–0.07 Å | 0.08–0.84 Å | ≈ 8.0 Å | unchanged | 141 s / 194 s |
| 0.1 | 0.06–0.34 Å | 0.7–5.2 Å | mostly ≈ 8.0 Å, some 6.6–7.8 Å | unchanged | 140 s / 194 s |
| 1.0 | 0.67–1.30 Å | 8–14 Å | 3.5–6.5 Å | unchanged | 141 s / 194 s |

All coordinates were finite and peak device memory stayed at 10.7–10.8 GiB. Generic guidance breaks the backbone on this input already at `rho=0`: consecutive Cα gaps of up to 17 Å at and next to the restrained residues (L29–31, L91–94, N247–248, H54–56), widening to up to 35 Å at `rho=1.0`. Sample 0 of the unguided D3 output (job 48613) and of the rigid-contact D9 output (job 48620) has no such gaps; its only short Cα steps are cis-prolines. The user chose `rho=1.0` and scoped D11 and D12 to path agreement (qualification protocol).

## Nucleic-acid-only runnability

Jobs 48754 and 48756, stock `off`, seeds 101 and 102, 5 samples, deterministic, bf16, TF32 off, each case run twice. D7 is the synthetic `CGCGAATTCGCG` duplex (`dnaSequence`, `count: 2`). The first D6-with-RNA-MSA attempt passed `--use_msa false` and was refused by 1.2 (`--use_rna_msa true requires --use_msa true`); job 48756 reran it with both switches on.

| Case | Flags | Tokens / atoms | MSA rows | Time per run | Repeat | Finite | Ranking score | LDDT sample 0 vs 1 |
|---|---|---|---|---|---|---|---|---|
| D6, RNA MSA | `--use_msa true --use_rna_msa true` | 86 / 1,836 | 1,727 | 27 s | byte-identical (30 files) | yes | 0.154 | 0.935, 0.950 |
| D6, no MSA (as H9) | `--use_msa false --use_rna_msa false` | 86 / 1,836 | 1 | 28 s | byte-identical (30 files) | yes | 0.154–0.155 | 0.926, 0.943 |
| D7, DNA duplex | `--use_msa false` | 24 / 494 | — | 22 s | byte-identical (30 files) | yes | 0.569–0.571 | 0.998, 0.999 |

Ranking score is shown per sample range over both seeds; LDDT is heavy-atom `biotite.structure.lddt` with default settings for seeds 101 and 102, only to confirm the metric returns finite scores on these outputs. The RNA monomer has `iptm` 0, which keeps its ranking score low. D7's samples are nearly identical to each other, so its seed-to-seed spread is small.

## Not yet done

- The qualification protocol: inputs, seeds, pairing, held-out pool and symmetry screen are frozen (`qualification_protocol.md`, freeze record; job 48760). Comparison rules are fixed (protocol). Acceptance and guidance-success criteria and the numerical contract follow, informed by the item 7 contrast.
