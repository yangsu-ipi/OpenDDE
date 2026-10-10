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

## Item 7: cross-precision draw match (pilot)

Traced stock runs of `protein_200`, deterministic, bf16 and `--dtype fp32` (job 48615): all 3,005 draws matched the expected manifest in both runs, and every draw matched byte for byte under both the raw and comparison-dtype hashes. All draws are fp32 in both precisions, because the sampler runs with autocast disabled.

The same holds with guidance. Rigid-contact guidance on `1a14_contact` (`--use_tfg_guidance true --enable_tf32 false`, MSA, deterministic, job 48620) drew 4,005 identical draws in bf16 and fp32 under both hashes.

The native bf16-versus-fp32 contrast therefore shares its random inputs for unguided and rigid-contact runs, and the replay diagnostic is not needed. The contrast distances themselves need the frozen qualification inputs and the comparison tooling, which are not built yet.

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

- Item 7 contrast distances: need the frozen qualification inputs and paired seeds, and the comparison tooling (LDDT, ligand RMSD, chain mapping).
- The qualification protocol: `qualification_protocol.md` holds the approved inputs, seeds and held-out pool; freezing, criteria and comparison rules follow.
