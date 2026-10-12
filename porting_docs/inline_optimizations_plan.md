# Inline optimization porting plan

Revision 50. Changes and reasons are recorded in `inline_optimizations_plan_changelog.md`.

Status: step 0 is under way. The qualification inputs and comparison rules are recorded in `qualification_protocol.md`. The single-cycle tier 1 limits and the full tier 2 numeric contract are approved for sm_120. Tier 2 stock calibration is complete; held-out classes are fixed without inspecting their predictions. The contract includes the reduced-assurance items in protocol E3 and the acceptance command's implementation choices and trust boundary in E5. The sm_120 contract is frozen (2026-10-12) as a pushed annotated tag on the freeze commit, with the limit hashes in `acceptance.json` as a readable record (protocol E5). The current tag is `qual-sm120-contract-2`, which amends `qual-sm120-contract-1` to add the development-only `dev-rehearsal` route (exit 4); the limit files are byte-identical, and the D1 smoke test passed at the new tag (job 49458). `accept.py` requires the draw's `code_commit` to descend from the tag with no change under `scripts/qualification` except the held-out log. `qual-sm120-contract-1` is superseded. The proposed rehearsal-to-draw binding in E5 (a pre-draw receipt, one draw per candidate and a newest-contract check) is not implemented or frozen. Implement it, amend the contract as `qual-sm120-contract-3` and rerun the D1 smoke test at that tag before porting starts, so the port branches from contract-3; a candidate branched from contract-2 would carry contract-2's `accept.py`, without the binding. A later run-wrapper change likewise needs a new tag and D1 smoke test. The current freeze permits sm_120 tuning but does not make the held-out workflow ready to run. The port itself still waits for the user's go-ahead. A100 needs its own stock checks and limit freeze before sm_80 tuning.

The acceptance command (`scripts/qualification/accept.py`) checks the declared rows, runs, recipes, random draws, baseline artifacts and held-out Git history. It also checks per-row same-seed pair mappings, run-to-measurement input digests, fresh successful runs, exactly one measurement per expected key, and the contract tag; it offers the development-only rehearsal route. All 156 qualification tests pass on gnode2 at `qual-sm120-contract-2`. The documented development-only D1 GPU smoke test passed at `qual-sm120-contract-2` (job 49458; first run job 49415), including byte-identical stock `off` outputs against the step 0 baseline. Candidate-side end-to-end behavior cannot be tested until a candidate exists; synthetic tests cover it for now. Before any held-out draw, the candidate must pass the `dev-rehearsal` route in the acceptance clone. Full acceptance does not yet verify that a rehearsal preceded the draw or used the same candidate arguments; E5 specifies the required repair. Acceptance runs will use the dedicated gnode2 clone `~/opendde-accept` once a candidate exists. This document remains a plan, not authorization to begin the port.

## Recorded decisions

| Date | Decision | Where it applies |
| --- | --- | --- |
| 2026-10-09 | Approved: reject `--optimization_mode` other than `off` under Fold-CP for this port. | Step 1 (Fold-CP policy), validation matrix |
| 2026-10-09 | Do not offer the LayerNorm stream fix or the trunk-cache identity fix to upstream `aurekaresearch/OpenDDE` for now. Both stay in the fork. Revisit only at the user's request. | Step 2, keeping up with upstream |
| 2026-10-10 | Require a held-out draw only for dispatch rows chosen or changed in this port. Rows kept unchanged from a kit record for that card qualify on the development set. | Step 4 gate, validation matrix, qualification protocol |
| 2026-10-11 | Accept the proposed sm_120 tier 2 numeric values for planning, without freezing the acceptance contract. The Zn rule remains a proposed stock-relative check; an inconclusive Zn result cannot support a Zn-qualified claim. | Step 0 gate, validation matrix, qualification protocol C5/C8/C9/E |
| 2026-10-11 | After the limit provenance audit: tighten the median reference lDDT drop to 0.008; keep the flexible/epitope median reference fnonnat rise at 0.06 with the kernel swap as its calibration and the third harmless contrast as its validation; set the flexible pairwise interface Fnat to 0.55 after the third contrast failed the audit's 0.70. | Qualification protocol C8/C9, `tier2_limits.json` |
| 2026-10-11 | Zn scope: H3's paired Zn check may add up to four Zn-only seeds until 10 eligible ions; H5's Zn is reported only. | Qualification protocol C5/C9/D |
| 2026-10-11 | Accept the held-out class assignments as predeclared sm_120 tests, with H1/H8/H9 marked as extrapolations and H2 PCG pose preservation limited to same-outcome pairs. Withhold approval of the full tier 2 contract until the acceptance command also binds its evidence to the declared runs and recipe. | Qualification protocol D/E, validation matrix |
| 2026-10-11 | Acceptance runs use a dedicated clone on gnode2, `~/opendde-accept`, checked out at the candidate's code commit, updated only by held-out log commits during a draw and used for nothing else. | Qualification protocol E5, held-out qualification |
| 2026-10-11 | Withhold full sm_120 tier 2 approval until paired-seed mapping, measurement-input bytes and successful fresh runs are enforced, tested and exercised once through the real run-record path. Keep the proposed numeric limits for planning. | Qualification protocol E5, validation matrix |
| 2026-10-12 | Approve the full sm_120 tier 2 contract, including E3's reduced-assurance limits and E5's implementation choices and trust boundary. The duplicate/extra-entry repair is implemented and tested; the D1 stock-side GPU smoke passed. Freeze the limit hashes before sm_120 tuning. Approval is of the contract, not a candidate or sm_80 limits. | Step 0 gate, qualification protocol C9/E, validation matrix |

## Recommendation

Port the optimization kit in functional layers: an environment and baseline check, exact scheduling and data-path changes, the portable fast kernels, then large-input memory changes if measurements on the target cards show they are needed. Keep OpenDDE 1.2.0 behavior as the default and make every optimized path opt-in during this port. Do not copy the overlay launcher, import hooks, monkeypatch framework, stock-package verifier, or the `opt_core` tree. Move only the OpenDDE-specific code and the kernels that can run on the target GPUs into the OpenDDE package. Call them from the normal model and runner code, and keep explicit fallbacks to the current PyTorch and cuEquivariance implementations.

The targets are A100 (sm_80) and RTX PRO 6000 Blackwell (sm_120). H100 and other sm_90a-only work is out of scope.

This work is isolated on `porting/inline-optimizations`, created from OpenDDE commit `9d0d9c9` (1.2.0). The source optimization kit was inspected at commit `f4f62fa`. This document is a plan only; no optimization code has been ported yet.

## What the comparison showed

### OpenDDE 1.1.1 to 1.2.0

- The optimization kit is pinned to OpenDDE 1.1.1 commit `ddfa1df`. The fork's 1.2.0 commit descends directly from that pin.
- Among the model and inference paths relevant to this port, the changes include `model/generator.py`, `model/opendde.py`, the two config default files, `data/inference/json_to_feature.py`, `tfg/*`, `utils/environment.py`, CLI defaults in `runner/batch_inference.py`, and the version files. The full commit diff also includes `runner/template_search.py`, docs, tests and examples. Nothing under `model/modules`, `model/triangular`, `model/layer_norm`, `data/core`, `data/template`, `runner/inference.py`, `runner/dumper.py` or `utils/file_io.py` changed. The `json_to_feature.py` changes only add `user_*` guidance features.
- The risky changes are in `generator.py` and `opendde.py`:
  - 1.2 adds rigid-contact and epitope interventions inside the diffusion loop.
  - 1.2 adds an optional on-disk trunk cache, `_trunk_with_cache`.
- No kit lever replaces `sample_diffusion` outright. The levers that touch it wrap it and call the original, so the 1.2 interventions would still run. Several levers do contain copies of stock function bodies, listed below. Those copies are unchanged in 1.2 but would drift silently under an overlay. Porting inline removes that risk.

### The kit measured against a different baseline

The kit's "stock" was 1.1.1 run with:
- `--dtype bf16`;
- `LAYERNORM_TYPE=fast_layernorm`;
- TF32 on, the 1.1.1 default.

Its "exact" claim is bitwise only under the deterministic recipe (`--det 1`).

OpenDDE 1.2 defaults to:
- bf16;
- TF32 off;
- `LAYERNORM_TYPE=torch`, which is unchanged from 1.1.1 and set in `opendde/model/triangular/layers.py`.

This has three consequences:
- The kit's two LayerNorm levers (`lnstream`, `ln_core`) have no effect under the 1.2 default LayerNorm.
- An `exact` mode cannot switch to `fast_layernorm` and still match `off`.
- Some fast sampler kernels (`dit_fused`, `atom_fused`, the atom-attention variant) were tuned against TF32-on stock. In 1.2 the stock sampler runs full FP32, so their distance from stock will be larger than the kit reported.

All of the kit's numerical claims have to be re-established against the 1.2 defaults.

### Kernel coverage on the target GPUs

- No kernel in the kit has an sm_120 binary or an sm_120 measurement, with one exception: the Triton `k2b` triangle-attention kernel, measured for bf16 at head dimension 32 on Triton 3.6 and torch 2.11. Other sm_120 behavior is inferred from sm_100 tables or falls back to the core's "safe settings" Triton configurations. The core's own architecture check labels sm_120 "uncertified".
- Bit-exact kernel records are per GPU and per software stack, and were taken on torch 2.7.1+cu126:
  - The bit-exact sampler-attention kernel (`dit_attn_exact`) is a prebuilt sm_90a binary only.
  - On A100, bit-exact triangle multiplication and triangle attention exist only for some shapes; elsewhere the stock operation runs.
  - On sm_120 there are no exact records at all.
- The kit's prebuilt `.so` files exist for torch 2.7.1 (cu126 and cu128), 2.10 and later. None exists for torch 2.8 or 2.9.
- The kit's triangle-multiplication cubins for sm_80 were built with CUDA 13.0, so they need NVIDIA driver R580 or newer.
- The A100 triangle-attention CUDA kernel is sized for sm_80's 163 KB of shared memory. sm_120 allows about 99 KB per block, so it would need a different tile configuration.
- Source is present for every prebuilt binary, so everything needed can be rebuilt.

### Size of the shared core

The transitive closure of what the OpenDDE kit imports from `opt_core` is large:
- about 82k lines of Python;
- 16k lines of CUDA C++;
- 31 MB of JSON tuning tables;
- 148 MB of prebuilt binaries;
- 72 MB of test vectors.

A selection limited to A100 and sm_120 is about 35-40k lines of Python, 3.6k lines of CUDA, and 3-5 MB of tables after pruning to compute capabilities 8.0 and 12.0. It drops all sm_90a units, the ESM/ef2 kernels, the JAX code and the prebuilt `.so` files. Even that keeps the generic provider and cell-selection machinery, which this plan proposes to replace with a smaller per-site dispatch (step 3).

### Software stack

- `pyproject.toml` pins `torch==2.7.1`, `triton==3.3.1` and `cuequivariance==0.10.0`.
- torch 2.7.1 ships a cu128 build that supports sm_120, and the kit's A100 tables and prebuilts are keyed to torch 2.7.1 and Triton 3.3.
- Staying on the pinned versions with the cu128 build keeps the GPU architecture as the main variable and avoids changing a public dependency pin.
- PyTorch 2.7 with cu128 includes Blackwell support and Triton 3.3. The open questions are whether the kit's Triton kernels compile, run correctly and win on sm_120 with Triton 3.3.1, and whether cuEquivariance 0.10 serves OpenDDE's triangle calls there. Step 0 checks those paths.

## Scope by optimization group

| Group | Disposition | Notes |
| --- | --- | --- |
| Environment and baseline | Do first | Verify the pinned stack on sm_120. Record a per-phase profile of unmodified 1.2. Use it to rank later groups, then use A100 and `fast` measurements for the capacity decision. |
| LayerNorm stream fix | Port as a bug fix, not behind a mode | `layer_norm_cuda_kernel.cu` launches on the legacy default stream (`<<<grid, block>>>`), not PyTorch's current stream. This is wrong under any non-default stream or CUDA graph capture, whatever the optimization mode. |
| Exact host and data-path changes | Port first | Drop the unused `bond_mask` feature, deduplicate padded template slots, remove the sampler's per-step host round-trips, reduce structural-token host reads, hoist step-invariant diffusion tensors and the pair permute, encode JSON in one pass. Optionally: prefetch the next input, skip in-forward cache flushes, set the expandable-segments allocator, guard CUDA graph replay. |
| Exact-tier kernels | Defer | On the target GPUs these mostly run the stock operation: sampler attention has no sm_80 or sm_120 build, and sm_120 has no exact records. Records from cu126 do not carry over to cu128. Revisit A100 rows only if the profile shows the trunk dominates and a cu128 requalification passes. |
| Pairformer fast path (Triton) | Port after exact parity | A cast-once Pairformer path that avoids transposes, plus Triton triangle attention, triangle multiplication, transition and pair-row LayerNorm. These are the only kernels that can run on sm_120 without new CUDA work. Keep the stock torch and cuEquivariance implementations as named fallbacks. |
| A100 CUDA kernels | Port after the Triton path, A100 only | The sm_80 triangle-attention and triangle-multiplication CUDA sources, built from source for the target stack. They do not run on sm_120. Use the sm_80 tile size there and do not port the sm_80 shared-memory layout to sm_120. |
| Diffusion and atom fast path | Port after Pairformer | Conditioning deduplication, pair-bias attention, fused token and atom schedules, low-precision operands. Fit them into the 1.2 guidance loop, and use the stock path for any call that needs gradients. |
| Single-GPU `big` mode | Conditional on measured capacity | Port if intended inputs fail to fit, or lack the agreed memory margin, on any actual target card under `fast`, including a 40 GB A100 if that is the provisioned model. The kit's offload threshold was 1,400 tokens for 80 GB cards. |
| Custom `rowpair_tp` | Defer | About 23k lines. It overlaps native Fold-CP, refuses guidance and Fold-CP, and suits NVLink better than gnode2's PCIe-only setup. Benchmark native Fold-CP first. |
| Unshipped or negative-value levers | Exclude | `writer_overlap`, `sampler_amp`, `xl_tri_ln` and the older `dit_attn_bf16` are not part of a shipped mode, or were measured as neutral or negative. |
| sm_90a and other-model kernels | Exclude | sm_90a CUDA units, the ESM/ef2/JAX kernels and every prebuilt binary. Dropping the sm_90a units also removes the FlashAttention-3 and CUTLASS license obligations. |
| Overlay infrastructure | Do not port | No `.pth` files, `sitecustomize`, runtime monkeypatches, separate `opendde-opt` executable, pinned-wheel verifier, clean-subprocess stock runner, `ACTIVE`/`LEVER` banner protocol or exit-code-3 refusal semantics. |

## Implementation plan

### 0. Verify the environment and record the stock baseline (sm_120)

The step needs the user's approval before an environment is created.

On gnode2:
1. Create a conda environment named `opendde-dev` from `~/miniforge3`. The existing `opendde` environment is reserved for production use and must not be modified. Use Python 3.11, `torch==2.7.1` from the cu128 index, `triton==3.3.1`, and the cuEquivariance 0.10.0 packages, with OpenDDE installed editable from a clone under the home directory.
2. Inside a Slurm allocation, check:
   - the CPU test suite;
   - a stock `opendde pred` smoke run;
   - that cuEquivariance 0.10 triangle kernels load and run on sm_120, and if not, what `--triatt_kernel auto` falls back to;
   - that the `fast_layernorm` extension builds for sm_120;
   - that a representative Triton kernel compiles and runs on Triton 3.3.1.
   The Triton check is a compatibility smoke test, not qualification. Its result must not be used to set numerical limits, which are fixed from stock calibration before candidate qualification outputs are inspected.
3. Check that stock output is reproducible, which the `exact` contract depends on. Run `off` twice with the deterministic recipe, the same seeds and the default `auto` triangle kernels, and compare the prediction artifacts byte for byte.
   - `--deterministic true` turns on `torch.use_deterministic_algorithms` and sets `CUBLAS_WORKSPACE_CONFIG`. Neither governs cuEquivariance's custom kernels, which `auto` selects for triangle attention and multiplication.
   - If the two runs differ, repeat with `--triatt_kernel torch --trimul_kernel torch`. Record which configuration is reproducible; step 1 uses it for `exact` comparisons.
   - List the files that count as prediction artifacts (the structure CIFs and confidence JSONs), and confirm they embed no timestamps, absolute paths or other run-specific values.
4. Add opt-in, synchronized phase timing before profiling. `_main_inference_loop` records phase times with `time.time()` and never synchronizes the GPU (`opendde/model/opendde.py`, for example the `pairformer` entry). Queued GPU work is charged to whichever later phase first forces a host wait. Several exact changes remove host waits, so these timers would move time between phases even when nothing got faster. Use CUDA events, or a `torch.cuda.synchronize()` at each phase boundary behind a profiling switch, so the default path is unchanged.
5. Profile unmodified 1.2 with the released checkpoint from the shared weights store (see the validation matrix), fixed seeds and the trunk cache disabled. Use the small, medium and large inputs from the validation matrix and the synchronized timing from item 4. Record:
   - cold and warm time per phase: featurization, trunk, structural expansion, diffusion, confidence, output writing. Report one-time compilation separately, such as the first `fast_layernorm` build, which took about 140 s on gnode2;
   - peak allocated and reserved GPU memory;
   - host memory peak;
   - output digests.
6. Before either unconditional bug fix, save stock `off` baselines for the step 2 regressions. Record the commit they were produced from. Each baseline is run twice under the deterministic recipe; record whether the runs agree byte for byte, and keep the prediction artifacts. If they differ, repeat enough runs to set an agreed comparison tolerance before changing the code. The baselines:
   - **Fold-CP, default LayerNorm:** a short, fixed input on two GPUs with `LAYERNORM_TYPE=torch`. This is a general check that the port leaves Fold-CP unchanged.
   - **Single GPU, `LAYERNORM_TYPE=fast_layernorm`:** a short, fixed input. This is the only single-GPU configuration that runs the kernel the LayerNorm stream fix changes. The step 0 profile uses the default torch LayerNorm and never reaches it.
   - **Fold-CP, `LAYERNORM_TYPE=fast_layernorm`:** the same input on two GPUs, for the same reason under Fold-CP. Fold-CP already accepts `LAYERNORM_TYPE` as a rank-consistent setting.
   - **Single GPU, trunk cache enabled:** one run that writes the cache and one that hits it, both before the fix. Record that the hit and the miss produce byte-identical artifacts. `_trunk_with_cache` skips the cache under Fold-CP, so the cache fix can only be exercised on a single GPU.

   Current OpenDDE code does not create or switch CUDA streams internally, including under Fold-CP. Produce these baselines through the stock CLI on the default stream, and record that condition. In that specified context, reproducible pre-fix `fast_layernorm` outputs are identity references. Do not extend the claim to every prediction: a direct Python caller can invoke the runner under an external CUDA stream, and later graph-capture work can also expose the launch-stream bug.

   Also record the pre-fix result of two targeted checks:
   - **Graph capture:** attempt to capture `fast_layernorm` forward and backward in a CUDA graph, and check both capture and replay where supported.
   - **Side-stream correctness:** on a non-default stream, enqueue a controlled delay before writing the LayerNorm input, then run forward and backward on that stream and compare with the torch LayerNorm reference.

   Record failures and passes alike. Capture behavior and the visibility of a stream race depend on the target stack and scheduling, so a pre-fix failure is not a prerequisite for the fix. These targeted results are diagnostics, not prediction artifacts to match.
7. Measure a cross-precision stock contrast for `fast` qualification.
   - **Native contrast.** On the qualification inputs and paired seeds, pair the original sample indices and run stock `off` with the default bf16 and with `--dtype fp32`. The sampler draws noise in the dtype of `s_inputs`. Record each raw initial Gaussian's dtype, shape and hash before schedule scaling, then compare values after casting the fp32 draw to bf16; raw hashes of different dtypes cannot establish equality. Compare the full item 8 trace before treating any pair as having identical random inputs. If the full trace matches under the comparison-dtype hashes, the native contrast already shares random inputs and no replay is needed. If a draw differs, label the observed distance a combined precision-and-stochastic contrast: it is descriptive context, not a precision estimate and not a basis for numerical limits.
   - **Replay diagnostic (conditional).** Only if a draw differs, and only if an isolated precision estimate is needed to choose those limits, run a separate diagnostic that captures one complete set of raw draws from the fp32 run and replays it into both stock precision paths, casting each draw to the receiving path's dtype; the bf16 path then receives a rounded copy and the fp32 path the exact values. The replay injector follows item 8's rule: a private in-process callback the harness installs, never public surface. Require the expected draw manifest in both replay runs and verify each supplied draw against the captured source after casting; keep capture and replay off the timing path. Reason: run the native traced contrast first, so a second hook that injects draws into the sampler is built only when it adds information.
   - **Reporting.** Record the native stock distance and, if run, the replay stock distance. Report each candidate's distance from the native bf16 and fp32 stock runs. Neither stock contrast is a hard pass/fail ceiling or evidence that fp32 is ground truth: `--dtype fp32` disables autocast and may change the executed path. Reason: a shared seed need not produce shared noise across dtypes, so the native contrast can mix stochastic and arithmetic effects; fixed-seed `off` repeatability alone also cannot set a useful tolerance for changed arithmetic.
   - **Later candidates.** Each candidate needs a fresh bf16 `off` run from its commit on its target GPU and software stack (item 8), and that run must pass the `off` drift check in the validation matrix. Reuse the Step 0 fp32 run for context while the fresh bf16 `off` remains byte-identical to the Step 0 baseline; rerun the fp32 contrast only when the `off` baseline legitimately changes. The original Step 0 contrast remains a historical baseline. Reason: the fp32 run is context only, and an unchanged bf16 `off` shows that the stock path it accompanies has not moved.
8. Add an opt-in sampler random-input trace for qualification. Record the shape, dtype and hash at each stochastic call site: the initial and per-step Gaussians in `generator.py`; the rotation matrix returned by `uniform_random_rotation` and the translation Gaussian in `utils.py`; the guidance Monte Carlo Gaussian when `eps_std` is nonzero and the guidance outer-step Gaussian in `tfg/engine.py`. Capture each Gaussian immediately after `torch.randn`, before multiplying it by the noise schedule, `s_trans`, `eps_std` or `sigma`. Capture the rotation matrix before applying it to coordinates; it is the applied random transform, not a raw NumPy variate. Use these same boundaries for replay, so each receiving path performs its own subsequent arithmetic. Reason: hashing or injecting a scaled perturbation as though it were the unscaled draw would confound the comparison or scale noise twice. Key each record by original sample index, diffusion step, chunk, RNG call site, draw kind and draw ordinal. Name each call site with a stable symbolic name declared beside the draw, not a file and line number, and record the source commit, GPU identity and software-stack fingerprint with each trace. The fingerprint names the driver, CUDA, torch, Triton and cuEquivariance versions, and also the NumPy and SciPy versions, because rotations come from SciPy's `Rotation.random` driven by the NumPy generator. Reason: line-based keys would change with unrelated edits and break comparison with older traces. Store two hashes per draw: one of the raw bytes, for exact candidate-versus-`off` checks, and one after casting to a single declared comparison dtype, for the bf16-versus-fp32 contrast in item 7. Compare translation hashes rather than assuming equality: its draw is fp32 in both paths, but the preceding dtype-dependent Gaussian uses the same torch generator and may leave it in a different state.
   - Implement the trace as a private in-process callback that the harness installs, not an environment variable, CLI option or config key. Reason: it must not become public surface, which step 7 governs.
   - Derive an expected manifest of keys and per-site, per-kind counts from `N_sample`, the step count, chunking and the enabled guidance settings. Include branches that skip a draw, such as zero guidance perturbation standard deviation, and branches that repeat one, such as guidance outer steps. Require the observed key multiset to match the manifest exactly, with no duplicate key. In a harness self-test, omit a hook at each RNG call site and inject a duplicate record; each fault must fail. Reason: a total count can still pass when one draw is missing and another is logged twice.
   - In the 1.2 code every draw count is fixed by configuration: guidance draws depend on `outer_steps`, `eps_batch` and `eps_std`, the per-step guidance noise sits outside the `mu == 0` early exit, and the rigid-contact and epitope modules draw no random numbers. The manifest is still a second encoding of the sampler's control flow. If a stock `off` trace does not match it, stop qualification and inspect the source, hooks and manifest. Rebuild the manifest from the actual draw sites and control flow, then rerun its fault-injection self-tests and fresh stock traces. Matching the observed count alone is insufficient. Reason: a mismatch can mean a stale manifest, a missing hook or changed sampler behavior. For numerical qualification, run `off` again from the candidate's source commit on the same target GPU and software stack and compare candidate draws and outputs with that baseline. An older trace, even one whose manifest still passes, is diagnostic only: key coverage cannot establish that the rest of the model stayed unchanged.
   - Check that two traced stock `off` runs agree and that enabling the trace does not change stock prediction artifacts. Run the reproducibility and artifact checks with the trace off, and use traced runs only to establish random-input identity. Reason: hashing copies each draw to the host and synchronizes every step, which can hide the stream races that item 6 looks for.
   - This trace is diagnostic and stays off for timing. Candidate-versus-`off` checks happen during qualification, after candidates exist. Reason: the first draw alone does not establish that later stochastic inputs match.

Gate and decision:
- If stock output is not reproducible under any triangle-kernel configuration, stop and ask the user: the `exact` contract cannot be tested as written.
- A failing stock smoke run or unusable stock triangle fallback blocks the baseline until its cause is diagnosed. If an optional cuEquivariance, `fast_layernorm` or kit Triton path fails, record the failure and continue the baseline and exact work through the available stock path. Qualify that optimized path only after it works. Propose a dependency-pin change to the user only if a required target path has no suitable fallback and the alternative stack has been tested.
- Use the phase profile to rank the groups by expected gain. Drop or postpone a group whose phase is too small to matter. Decide on `big` after measuring `fast` memory on the provisioned A100 as well as sm_120.
- Before tuning a `fast` kernel on either target GPU, freeze the qualification inputs, paired seeds and sample-index pairing, the symmetry-completeness screen of the inputs (see the validation matrix), that GPU's acceptance and guidance-success criteria, and a pool of reserved held-out inputs and seeds. Stock-only measurement and private qualification hooks may precede that freeze; candidate outputs and held-out predictions may not. The sm_120 freeze may precede A100 provisioning, but A100 needs its own stock check and freeze before A100 tuning. Reason: stock calibration is needed to choose limits, but thresholds chosen after seeing optimized outputs would make a marginal kernel look qualified.
- The `fast` and `big` acceptance criteria have two pass/fail tiers, both declared before tuning:
  1. **Non-sampling outputs.** Compare the trunk's single and pair representations and the distogram with `off` on identical inputs, with tight per-tensor limits. Test the confidence head separately with identical supplied `x_pred_coords` and other inputs. Reason: the head normally consumes sampled coordinates, so comparing its end-to-end outputs cannot isolate arithmetic changes in the head.
  2. **End-to-end sampled structures.** Classify paired samples as same-outcome or switched with a 2.0 Å anchor-RMSD cutoff, frozen together with the other limits. In same-outcome pairs, gate polymer LDDT for all polymer atoms, for each polymer type present and at interfaces; gate a high percentile of per-atom coordinate error only on rows whose stock same-outcome pairs sit far below the cutoff, because near the cutoff that percentile only restates it. Apply a distribution-based ensemble gate to every row, including rows with no same-outcome pairs; otherwise a zero minimum count makes the paired structure tests vacuous. Base it on lDDT or DockQ against frozen deposited structures where they exist, and on intrinsic ensemble statistics elsewhere; nearest-sample distances cannot discriminate at ten samples. Measure ensemble diversity within each seed as well as over the whole set: with two seeds of five, 25 of the 45 sample pairs cross seeds, so a whole-set median cannot see a collapse within each seed. Require the seeds and sample count declared for the row's class (ten samples unless calibration raised it), every applicable reference score and every declared row-specific gate to be present and conclusive; an absent metric cannot silently pass. Each gate reports pass, fail or not-applicable, and not-applicable is allowed only where a declared rule says so (for example, no same-outcome pairs on an epitope row). Validate the limits before freezing: a stock same-precision kernel swap must pass and constructed negative controls, including a near-duplicate partial collapse with a declared replacement count, perturbation size and applicable rows, must fail. Define the alignment, equivalent-chain assignment and ligand atom-symmetry mapping before calculating these metrics. Check ligand pose and interchain/interface contacts separately where applicable; internal ligand geometry and an all-polymer average can hide a wrong pocket or minority-chain interface. Do not use the maximum atom error as the sole gate. Reason: even with identical injected noise, a 200-step sampler can send a flexible loop, a ligand pose or a chain-symmetry choice to a different outcome; missing gates or samples would turn the test into a false pass.

  D1 remains a low-confidence, no-MSA path check. Its stock precision contrast exceeds the 0.5 Å-per-coordinate jitter control, so D1 is not a sensitive test of atom-level noise; the other development rows reject that control. Do not claim that qualification detects such noise on D1. If that claim is needed, calibrate a local-geometry check against stock before adding it as a gate. The approved sm_120 Zn metric tests pocket-relative placement against `off`, not correctness against deposited sites or coordination chemistry. It reliably rejects faults that move about half the ions (D15's controls) but not one ion in four (D14's moved-Zn control at 0.79 would pass a rerun about a quarter of the time); state the detectable fault size in those terms. Require a conclusive Zn result before claiming Zn placement is qualified. The paired check needs same-outcome pairs, which an epitope row such as H5 may never produce, and an unpaired coordination check calibrated on D14/D15 failed its validation, so H3 may add up to four Zn-only seeds to reach a conclusive result and H5's Zn is reported only. An inconclusive Zn result fails D14 and D15, which exist only to test Zn. P4G and D4 Mg stay report-only. Reason: H3 and H5 contain three Zn ions, stock D14/D15 fail to occupy most deposited Zn sites, and mobile Mg in D4 says nothing about Zn.

  Apply domain criteria on top of these: LDDT or DockQ against a reference where appropriate, ligand-pose quality where a ligand reference exists, end-to-end confidence and ranking agreement, and guidance success rates. Set absolute per-case limits after the step 0 calibration measurements and before candidate qualification outputs are inspected. A multiple of the largest stock contrast is a diagnostic proposal, not an automatic pass limit: a large contrast can make a gate too weak, while a zero contrast can make it impossible to pass. Stock seed-to-seed variation and any common-random precision diagnostic can inform limits but grant no automatic error allowance. Report the native cross-precision contrast with its draw-match result; if draws differ, it is context only.

Reason: the kit's numbers come from H100 on another software stack and OpenDDE version. Without a local profile there is no basis for the order or the scope.

### 1. Define the numerical contract and the configuration surface

The modes, all measured against the 1.2 defaults (bf16, TF32 off, `LAYERNORM_TYPE=torch`, triangle kernels `auto`):

- `off`: current OpenDDE 1.2 behavior.
- `exact`: byte-identical prediction artifacts to `off` under the deterministic recipe (`--deterministic true` plus `CUBLAS_WORKSPACE_CONFIG`). It may not change the LayerNorm implementation, precision, TF32 or kernel choice. Any optimization without proof of identity on the running GPU and stack runs the stock operation.
  - "Prediction artifacts" are the files listed in step 0, item 3.
  - Run `exact` comparisons with the triangle-kernel configuration that step 0 found reproducible. If only `torch` triangle kernels are reproducible, the byte-identity check uses them. Runs with `auto` are then compared within the agreed tolerance and documented as such.
  - In default, non-deterministic runs, `exact` matches `off` in distribution, not bit for bit. The user docs must say so.
- `fast`: deterministic across repeated runs in the same environment when the underlying kernels are deterministic, finite throughout, and within the predeclared two-tier acceptance criteria against `off` (step 0 gate): tight limits on non-sampling outputs and fixed-input confidence-head tests, plus paired-sample and domain metrics. Report the native bf16-versus-fp32 contrast and stock seed-to-seed variation separately as context, and distinguish any common-random precision diagnostic. It may select qualified Triton kernels. It does not change the global LayerNorm implementation merely because the mode changed.
- `big`: the same numerical class as `fast`, with lower peak device memory. It exists only if measurements on the target cards justify it.

Add one native configuration object for mode, GPU capability, per-optimization decisions and fallbacks. Expose it through the existing config system and CLI as `--optimization_mode {off,exact,fast,big}`, defaulting to `off`, consistent with flags such as `--triatt_kernel` and `--enable_tf32`.

Record which overlapping options were supplied before applying defaults at both CLI and direct Python entry points. Today the CLI passes `triatt_kernel="auto"` even when the option is absent. An explicit `--triatt_kernel auto` must retain the stock `auto` selection; an omitted flag may allow qualified optimization dispatch. Apply this rule to `--trimul_kernel`, `--dtype`, `--enable_tf32` and other overlapping flags. Explicit `LAYERNORM_TYPE`, cache, guidance and Fold-CP choices also take precedence. The resolved value alone does not record the user's intent.
- **CLI:** use Click's `ctx.get_parameter_source(name)`. It distinguishes `COMMANDLINE` and `ENVIRONMENT` from `DEFAULT`, so the existing option defaults and help text stay unchanged. Reason: changing the Click defaults to sentinels would change the help output and every caller that inspects them, for no gain.
- **Direct Python entry points** (`get_default_runner`, `inference_jsons` and similar): use an internal unset sentinel for omitted arguments, then resolve it to the current effective default so existing callers keep their behavior.
- **Dotted config overrides** also count as explicit choices. These are parsed by `opendde/config/inference.py` for the `runner.inference.run()` entry point, for example a supplied `--triangle_attention` or `--dtype` key. The `pred` command does not accept them. Environment variables that select a kernel or precision count as explicit too.

Keep the existing import-time `LAYERNORM_TYPE` selection for the first port, and use local dispatch for qualified pair-row LayerNorm kernels in step 3. `opendde/model/triangular/layers.py` reads the environment variable when it is imported and constructs a concrete LayerNorm module, so changing the mode later cannot switch every existing layer. If whole-model `fast_layernorm` shows a worthwhile gain, make it a separate follow-on design: resolve the choice before model construction, pass it explicitly to the constructors, preserve the environment override, and test mode changes through direct Python use. Do not implement it by changing `LAYERNORM_TYPE` after import.

Allow individual optimizations to be switched off through config keys, for ablation and debugging. Use the existing logger, not the kit's banner protocol.

Fold-CP policy for this port, approved by the user on 2026-10-09: reject `--optimization_mode` other than `off` when Fold-CP is active, with an error that names the reason. Check this at startup, before model construction. Reasons:
- The kit's levers step aside or refuse under Fold-CP. None was qualified with Fold-CP's sharded pair state and rank-synchronized sampler.
- A mode that silently runs stock would mislead users and benchmarks.

Lifting the restriction for specific exact changes is a later decision, made once each passes the Fold-CP regression in the validation matrix.

### 2. Port the exact, low-risk changes into their call sites

Port these first. Each is ordinary OpenDDE code, independent of the others, with an operator-level test against the stock path on CPU where possible:
- `bond_mask` removal;
- template slot deduplication (`TemplateEmbedder.forward`);
- one-pass JSON encoding;
- structural-token host-read reduction (`StructuralTokenExpander` role-pair projections).

Then integrate:
- the noise schedule copied to the host (`InferenceNoiseScheduler`, `centre_random_augmentation`);
- the diffusion step-invariant hoists (`dit_hoist`, `zprep_hoist`).

Optional exact items, gated because each has a real trade-off:
- **Prefetch** adds a worker process, and must step aside for one-item queries and for multi-seed runs with SMILES ligands, so the RDKit conformer random state is consumed as in stock.
- **Skipping in-forward `torch.cuda.empty_cache()`** trades lower fragmentation control for less synchronization.
- **The expandable-segments allocator** is a process-wide setting.
- **CUDA graph replay of the denoiser step** (`stepgraph`). Add graph-capture tests before enabling it. Its replay must be compared against the eager step. The kit found that on A100 replay was not bit-exact above 1,024 tokens, so measure the safe token window on each GPU rather than reusing the A100 limit.

Fix these two OpenDDE issues in the same step, outside any mode:

1. **LayerNorm stream.** Change the kernel launches in `opendde/model/layer_norm/kernel/layer_norm_cuda_kernel.cu` to use PyTorch's current CUDA stream. Build through the existing extension loader for the active architecture. Do not carry the kit's cu126 prebuilt `.so`. Reason: it is a correctness bug on any non-default stream and a prerequisite for graph capture, not an optimization.
   - The file has 28 kernel launch sites. Most use `<<<grid, block>>>` with no stream argument. The backward is already half-converted: `HostLayerNormGradient` fetches the current stream and passes it to `LayerNormParamGradStep2`, while `LayerNormParamGradStep1` and other kernels in the same function still launch on the default stream. On a side stream that is a race inside a single operation.
   - Fix every launch site, and add a check, such as a test that scans the source, that no launch is left without a stream argument.
2. **Trunk cache identity.** Audit the inputs and settings read by the trunk. Then version the cache format and key it by:
   - the feature fingerprint and `N_cycle`;
   - the loaded checkpoint's content identity;
   - relevant trunk configuration, precision and TF32 settings, and LayerNorm type;
   - the effective optimized kernel or row choices;
   - the GPU compute capability and device name;
   - the torch, CUDA runtime, Triton and cuEquivariance versions;
   - the OpenDDE version.

   Bypass the disk cache for a call if its numerical path or loaded model state cannot be identified before the lookup. Keep the disk cache off during numerical qualification until cache hits, misses, custom-checkpoint changes, same-metadata checkpoint replacement, post-load weight changes, cross-mode isolation and cross-GPU isolation are tested.

   Compute the checkpoint digest only when `OPENDDE_TRUNK_CACHE` is set, never on the default path. Bind the digest to a runner-managed model instance after a successful strict load, and reuse it for that instance while its weights remain unchanged. Path, size and modification time are not safe memo keys: replacing a file can preserve all three. Bypass the disk cache after a non-strict load or for directly constructed models unless their loaded state has a verified identity. A new process must hash again unless it has a trusted content-addressed manifest.

   Two mechanisms make this concrete:
   - **Hash the bytes that are loaded.** When the cache is enabled, read the checkpoint file once into memory, hash those bytes, and pass the same buffer to `torch.load`. This replaces the current direct `torch.load(checkpoint_path, ...)` in `runner/inference.py` on that path only. The default path is unchanged. Reason: hashing the file and then reading it again with `torch.load` leaves a window in which the file can change between the two reads. The cost is about 2.6 GB of extra host memory while loading.
   - **Track the loaded tensor set as well as its versions.** After strict loading, snapshot each named parameter's and buffer's object identity, shape, dtype, device and `_version` counter. Record identity by holding the tensor objects themselves, or weak references to them, and compare with `is`. Do not store `id()` values: Python reuses an `id()` once its object is freed, so a replacement tensor could match. Before each cache lookup, require the same named set and snapshot values; otherwise bypass the cache. Invalidate the identity explicitly when runner code calls `load_state_dict` or replaces a tensor. Version counters catch ordinary in-place changes but do not establish identity when a tensor is replaced with one at the same version, and they cannot certify arbitrary writes through raw storage. Limit cache use to the runner-managed inference lifecycle; bypass it if external mutation of the loaded model cannot be ruled out. Test parameter and buffer replacement with matching version counters as well as in-place mutation. Freeze the guard's design at this level and implement it plainly. The cache is opt-in and stays off during qualification, so further elaboration has little value.

   Strict loading is already the default (`load_strict: True` in `opendde/config/model_base.py`), so the non-strict bypass affects only unusual setups.

   Reasons:
   - Today `_trunk_with_cache` hashes only tensor features and `N_cycle`. The CLI also accepts an explicit checkpoint path, so two models or two numerical paths could share one key.
   - Stock trunk results already differ between GPU models and library versions. A cache directory shared between the A100 VM and gnode2, or across an environment upgrade, would otherwise serve results from the wrong platform.
   - Hashing the 2.6 GB checkpoint takes several seconds. Doing it at every model load would slow every run, including those that never use the cache.
   - File metadata cannot establish which weights the model actually holds. A cache hit after a same-metadata file replacement or a post-load weight change could return another model's trunk result.
   - Dropping `bond_mask` changes the feature set and hence the key, which is harmless but should be expected.

The 1.2 guidance path is a hard boundary:
- CUDA graph replay stays disabled whenever training-free guidance is enabled. Rigid-contact and epitope guidance require it.
- The other exact changes may stay enabled if their operator-level and end-to-end tests pass with guidance on.

Gate: all existing CPU tests pass; the exact operator tests pass; deterministic `exact` and `off` outputs match on sm_120 before work moves to fast kernels. After each unconditional fix, and before adding the exact changes, rerun the step 0 item 6 baselines and compare:
- after the LayerNorm stream fix: the `fast_layernorm` baselines, single-GPU and Fold-CP, and the default-LayerNorm Fold-CP baseline. Also require the targeted graph-capture and side-stream checks from step 0 item 6 to pass where supported, along with the launch-site check. A pre-fix pass does not waive the post-fix correctness checks;
- after the trunk-cache fix: the default-LayerNorm Fold-CP baseline, and a single-GPU cache miss followed by a cache hit. Both must match the pre-fix artifacts. Also confirm that the cache is still skipped under Fold-CP.

Where the stock runs were reproducible, require byte identity; otherwise use the tolerance fixed from those runs. The specified stock CLI baselines run on the default stream, so a mismatch after the fix is a regression to investigate. This identity claim does not cover direct Python calls made under an external stream. If two GPUs are unavailable, hold the Fold-CP parts of this gate open and do not claim Fold-CP `off` is preserved.

Keep exact changes behind the mode gate throughout this port so `off` remains the unmodified 1.2 path for A/B comparison. The `off` drift check in the validation matrix enforces this on every candidate commit. After each change passes the A100 and sm_120 gates, record whether it is a candidate for default promotion in a later release. Candidates are the host schedule, structural-token sync, JSON encoding, template deduplication and the hoists. A separate promotion decision would update the `off` contract, docs and tests. Reason: keeping two paths for an identical result costs maintenance, but changing the baseline during qualification would invalidate comparisons.

### 3. Bring in only the portable kernels, with a small dispatch

Create a normal package area such as `opendde/model/optimized/` for:
- GPU capability detection;
- per-call-site dispatch;
- the selected Triton kernels;
- the A100 CUDA sources, kept separate.

Do not keep the core's generic provider interfaces, `*_CELLS.json` tables or "tier word" selection. Instead give each call site (triangle attention, triangle multiplication, transition, pair-row LayerNorm, pair-bias attention) a small explicit table keyed by GPU capability, dtype and shape range. It holds only the rows measured in this port. Anything else uses the stock implementation. Reason: the generic machinery serves many models and GPUs, most of it unreachable from OpenDDE on these two GPUs. The tables alone are 31 MB, one of them 17.8 MB.

Pair-row LayerNorm needs a new dispatch point. Under the default `LAYERNORM_TYPE=torch`, `LayerNorm()` in `opendde/model/triangular/layers.py` builds `OpenFoldLayerNorm`, not `FusedLayerNorm`. The kit's `ln_core` wraps only `FusedLayerNorm.forward`, so it would never run.
- Put the dispatch in `OpenFoldLayerNorm.forward`, on the bf16 path at width 384 (`c_z` and `c_s` are both 384), with weight and bias present.
- Its reference is PyTorch's `layer_norm` in bf16 with bf16-cast weights, the current stock behavior. The kit measured speed and accuracy against the fused CUDA extension, so neither result carries over.
- Qualify this site last among the Pairformer sites, and skip it unless the step 0 profile shows LayerNorm time is significant.

Carry the license notices:
- add Anthropic's Apache-2.0 copyright line to OpenDDE's NOTICE;
- keep the Protenix Apache-2.0 notices for the FlashPairformer-derived Triton kernels;
- add PyTorch's BSD-3 notice if the bit-exact LayerNorm kernel is ever brought in;
- mark modified files as Apache-2.0 requires.

Replace monkeypatched `forward` methods with explicit dispatch inside the existing Pairformer, triangle, transition, LayerNorm, diffusion and atom modules. These situations must all select the existing implementation and log a structured fallback record rather than silently changing behavior:
- the dispatch refuses the call;
- the kernel fails to compile;
- the shape is unsupported;
- the user stated an explicit stock kernel flag;
- autograd is needed (`torch.is_grad_enabled()` and any differentiable input or module parameter requires gradients);
- the device is CPU or MPS;
- the numerical class is unknown.

Order:
1. Triton kernels on sm_120 at gnode2. Start with the stock fallback for every site. Run per-shape correctness and timing sweeps under Triton 3.3.1, and promote only measured winners. The kit's sm_120 entries are hypotheses until then. If a kernel fails with an out-of-resources error at sm_120's ~99 KB shared-memory limit, retune it or leave it on the stock path.
2. The same Triton kernels on sm_80 in the A100 VM, on the same stack.
3. The A100 CUDA triangle-attention and triangle-multiplication kernels, built from source with the cu128 toolkit, as A100-only rows. Building from source avoids depending on the kit's CUDA 13.0 cubins and the R580 driver requirement. Port them only if they beat the Triton rows on A100 at OpenDDE's shapes.

Gate: every promoted row has a correctness record, a run-to-run check, a graph-capture result where applicable, and a measured win over the fallback at the exact call shape. Include a parameter-only backward test for each dispatch site that can be called with gradients enabled. Reason: an input may not require gradients while the module's weights do.

### 4. Reconcile and port the optimized sampler

Integrate conditioning deduplication, pair-bias attention, the fused token and atom schedules, and low-precision operands into the model classes. Keep the diffusion loop in `generator.py` authoritative, so the 1.2 contact and epitope interventions still run at their current steps and Fold-CP synchronization is unchanged.

Guidance rules:
- When guidance `rho` is non-zero, `tfg/engine.py` calls the denoiser under `torch.enable_grad()` and backpropagates to `x_t`. The kit's fused sampler kernels have no gradient check, and none was confirmed to support autograd. Every fused or low-precision sampler path must route a call that needs gradients to the stock module. Reason: otherwise the guidance gradient may be silently zero or partial; the `allow_unused=True` path in the engine replaces a missing gradient with zeros. The default `rho=0` hides this, so it needs an explicit test.
- Rigid and epitope interventions rewrite `x_l` between steps. Any cached or captured state must be rebuilt from the current `x_l` after an intervention.

Test four cases separately: unguided, generic guidance (with `rho=0` and `rho>0`), rigid contact, and epitope guidance. Exact scheduling must not change intervention timing or random-number consumption. Fast mode must keep guidance success rates.

Gate: end-to-end unguided and guided runs pass the predeclared two-tier numerical and domain criteria, determinism and constraint-satisfaction checks on the qualification inputs on sm_120, then on sm_80. On each GPU where the candidate enables a port-chosen row, which includes every sm_120 candidate, it must also pass a fresh held-out set (validation matrix). Use paired seeds against `off` and report the failures as well as the pass rate.

### 5. Port single-GPU large-input memory handling (conditional)

Do this step if the step 0 profile and the `fast` measurements after step 4 show that intended inputs fail to fit, or lack an agreed memory margin, on any actual target card. Measure the provisioned A100 before the decision: it may have 40 GB or 80 GB, and the sm_120 profile cannot settle its capacity. Define the margin from measured peaks and run-to-run variation before making the decision. The kit set the offload threshold at 1,400 tokens for 80 GB cards and 1,160 below 64 GB, and the conditioning-hoist exclusion at 2,565 and 1,856 tokens. Those numbers come from H100 and A100, not from the 1.2 defaults on these GPUs.

If the step goes ahead:
- Convert the overlay's `HostPair` and row-block code into native stage implementations for the trunk, structural expansion and refinement, diffusion conditioning, confidence and distogram. Keep the existing PyTorch path at every stage.
- Several of these overlay modules copy stock function bodies, so port from the current 1.2 code, not from the copies.
- Resolve the 1.2 trunk cache deliberately. At first, disable the disk trunk cache while pair offload is active and log the reason. Re-enable it only after defining a stable serialization format for host-resident pair state and testing cache hits across guided and unguided inputs. Benchmarks must disable the disk cache so it cannot hide model work.
- Tune the offload and hoist thresholds separately for the 96 GB RTX PRO 6000 and 40 or 80 GB A100. Include host RAM and transfer time in the decision. On gnode2 a whole-GPU job gets about 29 GB of host RAM by default, which limits how much pinned pair state can be held.

Gate: `big` has a lower measured CUDA peak than `fast`, handles the target large inputs without corrupting outputs, and does not slow medium inputs down enough to make it a bad choice below its threshold.

### 6. Decide whether custom multi-GPU work remains

Benchmark native Fold-CP on the target workloads after the single-GPU port. Open a follow-on phase for the overlay's `rowpair_tp` only if native Fold-CP cannot meet a concrete capacity or throughput target. That phase needs a separate design, because:
- it overlaps Fold-CP internals;
- it currently refuses guidance;
- its trunk cache interaction is undefined (ranks would share one key while holding different row shards);
- it will be sensitive to gnode2's PCIe-only topology.

### 7. Finish the public surface and release hygiene

Update these together:
- CLI help;
- config defaults and schema;
- docs (`docs/kernels.md`, `docs/inference_instructions.md`);
- examples;
- `opendde doctor`, which should report GPU capability, the Triton and cuEquivariance status, and which optimized rows are available;
- changelog;
- package data;
- notices.

Add a machine-readable run report with the selected rows, fallbacks, software stack, GPU, warm or cold state, phase times and peak memory.

Port the kit's behavior tests (template deduplication, structural-token sync, host schedule, step-graph window, kernel binding fallbacks), not its overlay-mechanics tests. Several of the latter check stock line numbers that 1.2 already moved. Keep generated binaries, caches, benchmark outputs, checkpoints and private paths out of git.

## Keeping up with upstream OpenDDE

The fork will need to absorb future upstream releases, and inline changes spread across model files make each merge harder. To keep merges small:
- Limit edits to upstream files to thin dispatch calls, configuration plumbing and the two bug fixes. Put the implementations, dispatch tables, kernels and tests in new files, mainly `opendde/model/optimized/` and new test modules.
- Keep each optimization's hook in one recognizable form, for example one call into the optimized package at the top of the stock method, so conflicts are easy to spot and resolve.
- Before each upstream merge, list the upstream functions that the optimized code restates or depends on, and re-run the exact and fast gates afterwards.
- After each upstream merge, reconcile the random-input manifest with the sampler's draw sites, re-run its self-tests and validate it against fresh stock `off` traces (step 0 item 8). Re-establish the `off` numerical baseline on the merged commit before qualifying candidates. Reason: an upstream change may alter draw sites or model calculations without changing the manifest's keys.
- The LayerNorm stream fix and the trunk-cache identity fix stay in the fork. On 2026-10-09 the user decided not to offer them to the upstream project (`aurekaresearch/OpenDDE`) for now. Expect them to need re-applying when upstream changes those files, and revisit upstreaming only at the user's request.

## Validation matrix

Use named, fixed small, medium and large inputs for both tuning and qualification. Before the first `fast` row is tuned, set the two-tier acceptance and domain criteria (step 0 gate). Record the step 0 native cross-precision contrast and seed-to-seed variation as context; record any common-random precision diagnostic separately. Reserve a pool of inputs and seeds that will not be used for tuning. Cover:
- protein-only;
- DNA-only and RNA-only, to exercise the non-protein anchor and polymer LDDT; if either stock path cannot run in step 0, record the failure and ask the user whether to exclude that input class from this port's qualification;
- ligand and covalent-bond;
- template and MSA;
- multi-job and multi-seed;
- unguided, generic guidance with `rho=0` and with `rho>0`, rigid contact, and epitope guidance;
- Fold-CP regression: `--optimization_mode` other than `off` is rejected under Fold-CP. Compare Fold-CP `off` outputs with the step 0 baselines under the deterministic recipe, with both the default and `fast_layernorm` LayerNorm. Require byte identity where stock was reproducible; otherwise use the agreed tolerance from step 0. On gnode2 this needs two whole GPUs and `--mem=58G`. If Fold-CP support is later added for an exact change, `exact` and `off` must also match under Fold-CP.

For `fast` and `big`, record the approved two-tier limits, including a distribution-based ensemble gate that applies when no pairs are same-outcome (against frozen deposited reference structures where available), plus per-row confidence and constraint-satisfaction criteria, before tuning on each target GPU. The sm_120 single-cycle tier 1 calibration and full tier 2 contract are approved and frozen as `qual-sm120-contract-2` (amended from `qual-sm120-contract-1`; limits unchanged); A100 still needs its own stock check and freeze. All three harmless stock contrasts pass all 16 development jobs, and the constructed negative controls, including partial collapse, fail except for D1's documented jitter limitation. The limit provenance audit is done; one limit set by the kernel swap was validated by the third contrast, but flexible interface Fnat 0.55 was chosen after that contrast and has no independent harmless check on the development rows. That is reduced assurance, not an independent validation claim. A fourth contrast such as `--deterministic false` would only change reduction order and would barely test a limit resting on three same-outcome pairs. Two further reduced-assurance items: the `fast_layernorm` contrast reached the near-duplicate cap (3 of 3) on 1a14 unconstrained, where bf16 versus fp32 gave at most 2, and moved D11's mean constrained distance by 0.29 Å against a 0.5 Å tolerance. D3 and D13's `1a14` job are the same input with the same seeds, so the 16 development jobs are 15 distinct results. Negative-control rejections observed while limits were drafted show development-set separation, not independent sensitivity; the partial-collapse statistic, declared before measurement, is the exception. The D14/D15 Zn calibration supports paired stock-relative placement only, not deposited-site correctness. The held-out assignments are approved as predeclared tests, and `accept.py` enforces complete required rows and a conclusive aggregate Zn claim. Its candidate-side end-to-end path still needs exercise once a candidate exists. Held-out predictions remain unseen. A100 needs its own stock measurements and limit check on the provisioned card before A100 tuning or candidate qualification. The qualification protocol holds the comparison rules and remaining tasks:
- OpenDDE has no structure-comparison metrics; `opendde/metrics/` contains only `clash.py`.
- Start from what is already available. biotite 1.4 is a dependency and provides `biotite.structure.lddt` (confirmed in `opendde-dev`). Use it for protein, DNA and RNA LDDT between paired samples. Match atom identities after the one whole-complex chain assignment, select corresponding heavy polymer atoms, and require every selected coordinate to be finite in both structures. Pass both structures as `AtomArray` objects with `symmetric=True`. Set `aggregation="all"`, `inclusion_radius=15`, `distance_bins=(0.5, 1.0, 2.0, 4.0)`, `exclude_same_residue=True` and `exclude_same_chain=False` explicitly. For type-specific scores, restrict both atoms and contact partners to that type; the all-polymer score includes cross-type contacts. Freeze the per-type limits before tuning. Biotite returns NaN when no contacts are found, the same value a broken input produces, so the harness counts the atom pairs within the inclusion radius in either structure under the same masks. Zero pairs makes the score inconclusive, and it follows the declared held-out resolution rule. Missing or non-finite coordinates, or a NaN score with a nonzero pair count, fail the case. Reason: Biotite's default is directional and can miss contacts newly formed in the candidate; implicit atom and contact settings would make the gate hard to reproduce.
- Do not use `opendde/data/core/substructure_perms.py` for a pass/fail symmetry-aware metric. It caps enumeration and randomly subsamples with `np.random.choice` when more than `MaxMatches` permutations remain.
- Use the original sampling index to pair `off` and candidate structures. OpenDDE ranks output filenames by confidence by default, so matching `_sample_0.cif` across runs can compare different draws. The qualification harness should use prediction arrays before ranking, or set `sorted_by_ranking_score=False` in the inference config for paired runs. Test the normal ranked CIF/JSON output and ranking agreement separately; do not change the public default for qualification.
- Verify that paired sample indices received identical stochastic inputs; do not assume it. Diffusion noise, random rotations and guidance noise use dedicated torch and NumPy generators seeded per run (`opendde/model/generator.py`, `opendde/model/utils.py`, `opendde/tfg/engine.py`), so other code using torch's global generator cannot disturb those draws. But samples are drawn in chunks of `sample_diffusion_chunk_size` (default 5), each chunk draws all its noise before the next starts, and a different tensor shape gives different random values on the GPU. Sample *k* therefore gets the same stochastic inputs only when chunk size, sample count, noise dtype and the order of random draws all match.
  - The harness asserts that both runs use the same `N_sample` and `sample_diffusion_chunk_size`.
  - The step 0 trace checks stock repeatability. During candidate qualification, compare every traced stochastic input with `off`. A mismatch makes the sampled-structure comparison unpaired; do not score it against the paired-sample limits until a different gate has been declared and frozen.
  - A candidate that must change chunking, such as `big` or a sampler change, is compared with an `off` run at the same chunk size.
  - Reason: with the defaults (5 samples, chunk size 5) a chunking difference is invisible, so it would surface only with more samples or a mode that changes chunking. Matching the first draw would still miss a later change in draw order.
- Use RDKit's `rdkit.Chem.rdMolAlign.CalcRMS` for symmetry-aware ligand RMSD between paired samples, after fitting a specified anchor and applying that one transform to the whole complex. `CalcRMS` compares coordinates without realigning the ligand and has a configurable `maxMatches` cap (default 1,000,000). RDKit 2025.09.3 is already pinned, and `CalcRMS` is confirmed present in `opendde-dev`. Prediction CIFs omit bonds by default. Take each ligand's bonds from the featurized in-process `atom_array`, which the harness already has when it reads prediction arrays before ranking, or write qualification CIFs with the writer's existing `include_bonds=True`. Rebuild the graph from the input and match it to CIF atom names only if neither is possible, and then check the correspondence. Exclude hydrogens consistently and preserve covalent attachment identities when defining allowed atom mappings.
- `CalcRMS` returns an RMSD, not the number of matches examined or a truncation flag. Have the harness enumerate the matches itself with `GetSubstructMatches(..., uniquify=False, maxMatches=cap + 1)`: more than `cap` results means the search would be truncated, which gives an explicit flag. Pass the enumerated matches to `CalcRMS` through its `map` argument. `CalcRMS` by default treats the terminal atoms of conjugated groups such as carboxylate and nitro as equivalent (`symmetrizeConjugatedTerminalGroups=True`), and a hand-built `map` loses that, so the harness must apply the same equivalence; otherwise a harmless swap of two carboxylate oxygens could appear as a pose error. In step 0, compare the explicit-map result with RDKit's automatic result on carboxylate and nitro fixtures, test a covalently attached ligand whose attachment atom must stay fixed, and test the cap boundary. Do not freeze the symmetry rule until those checks pass. Record the cap and result. If the search is non-exhaustive, an RMSD within the limit can still pass that metric; an RMSD above the limit cannot establish failure because an unexamined mapping may be better. Mark that case inconclusive unless another pose-sensitive test resolves it. Check repeatability on the pinned RDKit stack rather than assuming match enumeration order is a public guarantee.
- Also measure ligand centroid displacement and the set of protein residues within a fixed contact distance, after the same protein alignment and chain assignment. These atom-label-independent checks can catch a shift to another site, but a wrong orientation at the same site may leave both nearly unchanged. They supplement ligand RMSD; they do not settle an inconclusive symmetry-aware pose comparison by themselves.
- For equivalent-chain assignment, `entity_id` groups copies and `sym_id` numbers them; neither proves that every permutation is valid. Enumerate mappings only among chemically equivalent copies that preserve declared covalent bonds and input-specific chain constraints. Choose one whole-complex assignment under a declared anchor criterion and tie-break, then use it for LDDT, contacts and ligand pose checks. Do not choose a different best permutation for each metric. If exhaustive assignment is impractical, record the approximate mapping and treat symmetry-sensitive hard-gate results as inconclusive until an independent method resolves them.
- Declare the anchor before use: the protein chains by default, and for inputs without protein a declared polymer anchor such as the nucleic-acid chains.
- Screen symmetry completeness when the inputs are frozen, not during qualification. The number of ligand atom matches and the number of equivalent-chain permutations depend only on the input, not on any prediction. For every qualification input and every input in the reserved held-out pool, count both. Either exclude inputs that cannot be searched exhaustively, or declare in advance the test that will decide them. Reason: otherwise an inconclusive result can arise in a single-use held-out run, where a retry would use up the set.
- Add DockQ, OpenStructure or similar only if these defaults are not enough.
- Keep any added comparison tooling as a test or dev dependency, not a runtime dependency in `pyproject.toml`.

Compare each candidate with a fresh `off` run from its source commit on the same target GPU and software stack, using paired seeds and original sample indices during development.

Check `off` itself for drift on every candidate commit. Compare the fresh `off` prediction artifacts with the Step 0 baseline (or the latest recorded rebaseline) on the same GPU and software stack under the deterministic recipe. Require byte identity where stock was reproducible; otherwise use the agreed tolerance. A difference stops qualification until it is classified:
- an intended change, such as either unconditional bug fix or an upstream merge, recorded with its reason and followed by a new `off` baseline; or
- a leak of port code into the `off` path, which must be fixed before qualification continues.

Reason: candidate and `off` now come from the same commit, so a change that reaches both would be invisible to the paired comparison while changing what `off` users get.

Then decide per GPU whether the candidate needs a held-out set. The held-out set guards against selection on development results, so it is required only where this port makes a selection (user decision, 2026-10-10; detail in the qualification protocol):
- A dispatch row whose kernel source, configuration and shape range are unchanged from a kit record for that compute capability qualifies on the development set alone. Record it as a development-set pass, a weaker claim than a held-out pass.
- A row chosen or changed in this port is port-chosen. That includes every sm_120 row, since the kit has no sm_120 records, and narrowing a kit row's shape range after seeing development results. A candidate that enables any port-chosen row on a GPU is qualified once on that GPU on a held-out set drawn from the reserved pool.
- A kit row that fails on the development set is rejected and its site keeps the stock operation, with no held-out draw. Changing the row to retry it makes it port-chosen.
- Both routes use the protocol's frozen criteria against the 1.2 `off` baseline, never the kit's own limits.

For held-out qualification:
- A held-out failure rejects that candidate. An inconclusive held-out result is decided only by the resolution test declared at freeze time; if none was declared, it counts as a failure.
- H3's four extra Zn-only seeds are its declared resolution test. An H3 Zn result still unresolved at the cap therefore counts as a held-out failure: the candidate is rejected and the held-out set is used up. "Partial result without Zn" is only a label in the report, never a third acceptance outcome. Expected cost for a harmless candidate: if H3 behaves like D3 (30-50% same-outcome pairs, three bound Zn), at most three same-outcome pairs among its 30 Zn samples has probability below 1% at a 30% rate and about 12% at 20%. Any change to this rule is made before H3 is opened.
- Full sm_120 acceptance is decided by one acceptance command over a declared evidence set, not by `tier2_gates.py evaluate` or other diagnostic commands. The rows required for each GPU and route live in a committed file, never in command-line flags. The command rejects missing rows or jobs, wrong or reused seeds, unmatched recorded draws and a nonqualified Zn claim. It also binds the evidence to its runs (implemented and tested, protocol E5): write a run record (header and wrapper options) for every run, traced or not; compare the development `off` run with a committed digest manifest of the step 0 baseline runs, with no baseline path in the evidence; require an independently recorded candidate repeat; check every trace against rollouts and artifacts expected from the input's jobs, the logged seeds and the frozen recipe, not from the trace itself; bind tier 1 results to declared, separate single-cycle capture runs and their replay source; rebuild each run's expected command line from the manifest row and require `off` to match it exactly and the candidate to add exactly its declared arguments. Each held-out draw entry records the candidate's code commit and declared arguments before any held-out run; every run must come from a clean commit that differs from that code commit only in `held_out_log.json` and contains its seeds, and the log's Git history must only ever be appended to. Add failure tests for self-aliased paths, rollouts or artifacts missing from both sides, substituted tier 1 results or replay sources, a wrong input or override, a candidate run with `off`'s command line, a seed absent from its run commit, an edited or removed log entry, a valid log-only commit sequence and a code change between runs. Acceptance runs come from a dedicated Git clone on gnode2 (user decision). Keep exit 0 exclusive to full acceptance, with distinct nonzero codes for a failed gate and a report that would otherwise pass without Zn. Reason: complete rows and passing metrics are insufficient if the evidence can refer to the wrong or same run, while one immutable commit cannot contain each H3 seed before it is chosen, and a candidate changed after a held-out failure must not be able to rerun the same draw.
- Before treating a diagnostic result as full acceptance (implemented and tested, and exercised on D1 by the GPU smoke test; protocol E5): `tier2_gates.py measure --draw` derives `(seed, seed)` per row from the held-out log and records the pairs on each entry, and `accept.py` requires them to match the logged seeds in order. Each run records the digests of the artifacts and `tier1.pt` captures it wrote, and each tier 1 or tier 2 measurement records the digests of the files it read and its own code revision; `accept.py` requires the two to agree and the measurement code to pass the run-commit rule. A run record marks any uncaught exception as failed and states whether its output and capture directories existed before the run, which `accept.py` requires they did not; `run_rows.py` also refuses existing paths. Protocol E5 specifies the failure tests, a CPU test of the exception path and a development-only D1 GPU smoke test. Reason: swapped seed pairs, stale measurements or an exceptional run with reused output can otherwise produce a passing evidence set, and recording digests on both sides binds measurements to runs without rereading tens of gigabytes at acceptance.
- Freeze the contract as a Git tag, not only as limit hashes (check implemented and tested; sm_120 frozen as `qual-sm120-contract-1`, amended as `qual-sm120-contract-2`, protocol E5). Tag the freeze commit (for example `qual-sm120-contract-1`) and push the tag. `accept.py` requires that the tag is an ancestor of the draw's `code_commit` and that `git diff <tag> <code_commit> -- scripts/qualification` is empty apart from `inputs/held_out_log.json`; the tag name is recorded in `acceptance.json` before tagging. Keep the limit hashes in `acceptance.json` as a readable record. Test a changed gate constant, limits and their hashes changed in one commit, and a `code_commit` that does not descend from the tag. Reason: much of the approved contract is outside the limit files: the same-outcome cutoff (`SAME_OUTCOME`, defined in both `tier2_gates.py` and `zn_sites.py`), the Zn coordination, same-site and pocket distances, contact distances, gate selection, Zn claim rules, `accept.py` itself, `acceptance.json`'s routes and recipe, `baseline_off.json` and the run wrapper. `accept.py` runs from the candidate's own `code_commit`, and the hashes live in a file the same commit can edit, so a candidate branch that also changed qualification code or limits plus hashes would still pass. This guards against accidental change, consistent with the E5 trust boundary; it is not a defence against deliberate tampering.
- Plan for the run wrapper. `run_opendde.py`'s tier 1 hooks capture and substitute tensors inside the model, so a port that renames or restructures hooked modules would also need a wrapper change, which the contract check refuses. Prefer keeping the hooked modules and their outputs stable as a port design constraint. Otherwise amend the contract with a new tag, made before any held-out draw is logged, after rerunning the D1 smoke test at that tag (`off` byte-identical to the step 0 baseline and a self-consistent tier 1 `off` comparison). Reason: the wrapper is part of the contract, and an unplanned change would block acceptance or be made under pressure.
- Before logging any held-out draw, rehearse the full candidate path on D1-D15 through the `dev-rehearsal` route in `acceptance.json` (`"rehearsal": true`, development rows only). Its evidence names `code_commit` and `candidate_args` without a draw; held-out rows and draws are forbidden. It skips only draw-dependent checks and runs the per-run, per-row, tier 1, tier 2, Zn, Git and contract-tag checks on complete development evidence, including an independent repeat. A passing rehearsal exits 4, never 0. Run it in the dedicated acceptance clone. This route was implemented and frozen in `qual-sm120-contract-2`; its candidate-side end-to-end path still awaits a candidate. Cost per candidate: the full development set (`off`, candidate, repeat and both tier 1 runs on 15 rows), several GPU-hours. Reason: the stock-only D1 smoke test cannot exercise the candidate command, candidate-against-`off` traces or repeat.
- Bind the passing rehearsal to the first held-out draw before any held-out seed is used. Record a rehearsal receipt in an append-only Git log commit that precedes the draw commit, with the rehearsal verdict, its development-evidence identity, the exact `code_commit` and the ordered `candidate_args`; do not put held-out seeds or predictions in that receipt. Full acceptance must reject a missing or nonpassing receipt, a receipt first added with or after the draw, a changed development evidence set, or any mismatch in commit or arguments. The rehearsed development runs become the development set of final acceptance, so they are checked again without repeating GPU inference. A changed commit or arguments makes a new candidate that needs a new rehearsal. Add tests for each failure and a passing receipt followed by a draw; keep exit 0 exclusive to full acceptance. The current `accept.py` does not enforce this binding: a synthetic full-pass test succeeds without a rehearsal, and the plan's earlier commit-only match omitted candidate arguments. Implement the binding under `scripts/qualification`, amend `qual-sm120-contract-2` as `qual-sm120-contract-3`, rerun the D1 stock-side GPU smoke test at that tag and confirm the tag in `~/opendde-accept`, all before porting starts; the port then branches from `qual-sm120-contract-3`. The approved numeric limits need not change. Reason: a rehearsal at one commit or mode does not validate a later draw at another, and an unenforced pre-draw step could be skipped. Any held-out run that fails for a harness reason still uses up its set.
  - **The draw names the earlier receipt.** "Earlier" means the receipt is already in `held_out_log.json` at the parent of the commit that first adds the draw. Commit order alone would not do: `code_commit` can be checked out and the development rows run after a draw is logged, so only a receipt committed first, holding digests of the evidence, shows that the evidence existed before the draw.
  - **Evidence digest.** The receipt identifies the development evidence by a canonical sha256 over the sorted `(role, row, file, sha256)` entries of every development run record (`off`, candidate, repeat, tier 1 `off` and candidate), every trace, the tier 1 and tier 2 measurement files and the set declarations. Measurement files alone would not bind the run records (command lines and artifact digests), the traces or the repeat run, which no measurement reads; run records list their artifact digests, so this binds every output. Full acceptance recomputes the digest from the development evidence it reevaluates. The receipt stores only the rehearsal verdict's sha256, status and exit code, not the verdict itself; acceptance reevaluates the evidence rather than trusting it.
  - **Receipts in the log rules.** Receipts are append-only like draws (never edited or removed) and have unique ids; the existing history checks cover them.
  - **One draw per candidate.** Each receipt may be named by at most one draw, and each `(code_commit, candidate_args)` pair may appear in at most one draw across the whole log history. Reason: nothing currently stops a failed candidate from drawing fresh seeds unchanged and trying again, which is held-out shopping; the plan's "retry after changes" was not enforced.
  - **Current contract only.** The contract tag named in `acceptance.json` must be the newest `qual-sm120-contract-N` tag in the acceptance clone. Reason: `accept.py` runs from the candidate's own `code_commit` and checks the tag named there, so a candidate branched before an amendment would carry the older `accept.py`, without the newer checks, and could still pass under it. This makes a stale candidate fail instead of relying on someone merging the amendment.
  - **Protect the rehearsed evidence.** Make the rehearsal output directories read-only once the receipt is committed. Reason: the same files must survive unchanged until full acceptance, possibly days later; the digest check would catch a change only after the held-out runs.
  - **Tests.** In addition to those above: a missing or nonpassing receipt; a receipt added in or after the draw commit; changed development evidence (a run record, a trace, a measurement file, a repeat artifact); a changed code commit or arguments; a valid receipt-then-draw sequence; an edited or removed receipt; a second draw naming a used receipt; a second draw with the same commit and arguments under a new receipt; and a contract tag older than the newest in the clone.
- Before freezing (implemented and tested, protocol E5), require exactly one tier 2 measurement entry per expected row and job, and exactly one tier 1 entry per expected row, job and seed, within each evidence set. The expected tier 2 keys are every row of the set times every job in that row's input; the expected tier 1 keys are the set's rows that are also in the route's `tier1_rows`, times their jobs and that row's seeds. `accept.py` checks each set's entries against these keys before evaluating gates and rejects duplicates and entries outside them. `tier2_gates.py evaluate` also fails a duplicated row/job key or a job not in the row's input with a `unique` gate. The producers refuse the corresponding errors: `measure` rejects duplicate `--rows`, and `tier1_compare.py --rows` excludes other rows in a shared capture root. Tests cover both duplicate orders and unexpected entries. Reason: the earlier last-entry-wins indexing could hide a failed measurement behind a passing duplicate despite valid input digests. This repair did not change calibrated limits and needed no new GPU calibration.
- Once a held-out set has been seen, it is used up: any retry after changes draws fresh seeds from the pool. The pool has only ten inputs, so a retry reuses inputs already seen and is a weaker, seed-only held-out test; the qualification record says so. Reserving further held-out inputs is possible only before the first held-out run.
- Record which held-out sets have been used, and for which candidate.
- If the pool runs out, add new reserved inputs and seeds before tuning continues, not after seeing results.

Record the absolute limits, the native bf16-versus-fp32 contrast, any common-random precision diagnostic, and the stock seed-to-seed variation. A small, medium and large size label alone is not enough to establish the numerical envelope.

For each mode, record:
- cold start and second-run latency;
- per-phase latency;
- peak allocated and reserved VRAM;
- host memory peak;
- output hashes, confidence values and coordinate comparisons;
- guidance satisfaction;
- the trunk-cache setting, which is off for benchmarks.

On gnode2:
- Code lives under the home directory, and outputs go under the user's `/data` directory. GPU commands run through Slurm.
- Weights come from gnode2's existing shared, read-only OpenDDE weights store, the one the shared OpenDDE job template points `OPENDDE_ROOT_DIR` at. Its `opendde.pt` matches the size and SHA-256 in `opendde/config/model_manifest.json` for 1.2.0, and its `common/` holds the CCD and template data files. Do not download a private copy. Recheck the digest if the store changes.
- Small kernel tests can use a GPU share with host RAM proportional to the share.
- Clean end-to-end benchmarks need a whole GPU and normally `--mem=29G`.
- Large host-offload tests may need more RAM. Base the request on measured use, and schedule it so it does not block other GPUs.
- The cards are the Max-Q (power-limited) RTX PRO 6000. Repeat timing runs and report variance, and do not compare absolute timings with other RTX PRO 6000 variants.

For A100, use the same Python, torch 2.7.1+cu128, Triton and cuEquivariance versions, so the GPU architecture is the main variable. Ask for approval before creating either conda environment. Ask the user to provision the A100 VM when the A100 gate is ready; it comes after CPU and sm_120 validation, so cloud time is spent on a stable candidate. Check the VM's driver version when it is provisioned.

## Main risks

- The `exact` contract assumes stock output is reproducible under the deterministic recipe. The deterministic flag does not govern cuEquivariance kernels, so reproducibility with the default `auto` triangle kernels is unverified until step 0.
- The model's built-in phase timers do not synchronize the GPU. Without the synchronized timing from step 0, exact changes that remove host waits would appear to move time between phases.
- Fold-CP is excluded from the optimization modes in this port. Users who combine them get an error, not a speedup.
- The kit's numerical claims are tied to OpenDDE 1.1.1, a `fast_layernorm` and TF32-on baseline, H100 or A100 on cu126, and specific software versions. Treat them as hypotheses until requalified.
- The kit's Triton kernels are unqualified on sm_120 with Triton 3.3.1. Its only sm_120 measurement used Triton 3.6. A failing optional kernel can fall back to the stock path; a dependency-pin change is needed only if a required target path has no suitable fallback.
- The trunk cache can mix results from different checkpoints or numerical paths unless its identity is extended (step 2). It also conflicts with pair offload, which owns the same trunk result.
- Path, size and modification time do not prove checkpoint content, and the loaded model can change after the file is read. Disk-cache identity must follow the model state or the cache must step aside.
- Global LayerNorm is selected at import and model-construction time. Mode-dependent switching needs an explicit construction-time design; the first port keeps that selection fixed.
- CLI defaults do not identify which overlapping options the user supplied. Dispatch must record explicit options before applying those defaults.
- Guidance with `rho>0` needs gradients through the denoiser. Fused sampler kernels without autograd support must step aside, or guidance silently weakens.
- Kernels without backward support must also step aside when only module parameters require gradients; checking inputs alone would miss that case. Normal inference runs under `@torch.no_grad()` (`InferenceRunner.predict`), so check grad mode first to keep the common path cheap. The model's parameters keep `requires_grad=True`, so with guidance `rho>0` every optimized denoiser site falls back to stock.
- Paired-seed comparisons of sampled structures are heavy-tailed: a small numerical change can switch a flexible region to a different outcome. Limits based on the maximum error, or on near-zero fixed-seed repeatability, would reject benign changes. The two-tier criteria use absolute limits; a common-random precision diagnostic can help interpret them without becoming an automatic allowance. A native bf16-versus-fp32 contrast with different draws cannot isolate precision.
- A regression baseline that never runs the changed code proves nothing. The LayerNorm stream fix needs `fast_layernorm` baselines plus a non-default-stream test, and the trunk-cache fix can only be exercised on a single GPU.
- Guidance mutates sampler state between denoising steps. Graph capture and fused schedules must either be proven compatible or step aside.
- Copying more of `opt_core` than needed would bring in tens of thousands of lines and a large maintenance burden. The port needs a dependency inventory and a selected source subset.
- The existing Fold-CP and the overlay's tensor-parallel code solve overlapping problems with different state layouts. Combining them is riskier than benchmarking the native path first.

## Work records to keep in `porting_docs/`

Add these documents as the port advances:

- The step 0 environment check, stock and Fold-CP reproducibility results, prediction-artifact lists, pre-fix regression baselines and stream-test results with their source commit, the native bf16-versus-fp32 contrast, any common-random precision diagnostic, and baseline profiles.
- A source-to-destination map for each imported or rewritten optimization, including provenance and license notes.
- A separate qualification protocol, for example `porting_docs/qualification_protocol.md`, started in step 0. It holds the numerical contract; the two-tier and domain criteria with absolute limits; sample-index pairing and the full random-input trace, with its expected per-site key manifest and comparison-dtype hash; the symmetry-completeness screen and ligand-map parity fixtures; the anchor, equivalent-chain and ligand-symmetry rules and the chosen comparison tooling; the native cross-precision contrast and any common-random replay diagnostic; the named qualification inputs; the reserved held-out pool; and a log of which held-out sets each candidate used. The plan keeps the strategy and points to the protocol for test detail. Reason: the plan is long enough that mixing strategy with test procedure makes both harder to follow.
- GPU qualification tables for sm_80 and sm_120, including refused and fallback shapes.
- Benchmark reports with commands, environment fingerprints, warm or cold status, timings, memory and output comparisons.
- Decisions that change scope, especially go or no-go per group after step 0, the A100 capacity and `big` decision, the trunk-cache identity, any global LayerNorm selection design, dependency-pin changes, any lifting of the Fold-CP restriction, any change to the decision not to upstream the bug fixes, and the Fold-CP versus `rowpair_tp` result.
- A list, refreshed before each upstream merge, of upstream functions the optimized code restates or depends on.
