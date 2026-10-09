# Inline optimization porting plan

Revision 3. Changes and reasons are recorded in `inline_optimizations_plan_changelog.md`.

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
1. Create a conda environment from `~/miniforge3`. Use Python 3.11, `torch==2.7.1` from the cu128 index, `triton==3.3.1`, and the cuEquivariance 0.10.0 packages, with OpenDDE installed editable from a clone under the home directory.
2. Inside a Slurm allocation, check:
   - the CPU test suite;
   - a stock `opendde pred` smoke run;
   - that cuEquivariance 0.10 triangle kernels load and run on sm_120, and if not, what `--triatt_kernel auto` falls back to;
   - that the `fast_layernorm` extension builds for sm_120;
   - that a representative Triton kernel compiles and runs on Triton 3.3.1.
3. Profile unmodified 1.2 with the released checkpoint from the shared weights store (see the validation matrix), fixed seeds and the trunk cache disabled. Use the small, medium and large inputs from the validation matrix. Record:
   - cold and warm time per phase: featurization, trunk, structural expansion, diffusion, confidence, output writing;
   - peak allocated and reserved GPU memory;
   - host memory peak;
   - output digests.

Gate and decision:
- A failing stock smoke run or unusable stock triangle fallback blocks the baseline until its cause is diagnosed. If an optional cuEquivariance, `fast_layernorm` or kit Triton path fails, record the failure and continue the baseline and exact work through the available stock path. Qualify that optimized path only after it works. Propose a dependency-pin change to the user only if a required target path has no suitable fallback and the alternative stack has been tested.
- Use the phase profile to rank the groups by expected gain. Drop or postpone a group whose phase is too small to matter. Decide on `big` after measuring `fast` memory on the provisioned A100 as well as sm_120.

Reason: the kit's numbers come from H100 on another software stack and OpenDDE version. Without a local profile there is no basis for the order or the scope.

### 1. Define the numerical contract and the configuration surface

The modes, all measured against the 1.2 defaults (bf16, TF32 off, `LAYERNORM_TYPE=torch`, triangle kernels `auto`):

- `off`: current OpenDDE 1.2 behavior.
- `exact`: byte-identical prediction artifacts to `off` under the deterministic recipe (`--deterministic true` plus `CUBLAS_WORKSPACE_CONFIG`). It may not change the LayerNorm implementation, precision, TF32 or kernel choice. Any optimization without proof of identity on the running GPU and stack runs the stock operation.
- `fast`: deterministic across repeated runs in the same environment when the underlying kernels are deterministic, finite throughout, and within an agreed validation envelope against `off` and normal seed-to-seed variation. It may select qualified Triton kernels. It does not change the global LayerNorm implementation merely because the mode changed.
- `big`: the same numerical class as `fast`, with lower peak device memory. It exists only if measurements on the target cards justify it.

Add one native configuration object for mode, GPU capability, per-optimization decisions and fallbacks. Expose it through the existing config system and CLI as `--optimization_mode {off,exact,fast,big}`, defaulting to `off`, consistent with flags such as `--triatt_kernel` and `--enable_tf32`.

Record which overlapping options were supplied before applying defaults at both CLI and direct Python entry points. Today the CLI passes `triatt_kernel="auto"` even when the option is absent. An explicit `--triatt_kernel auto` must retain the stock `auto` selection; an omitted flag may allow qualified optimization dispatch. Use an internal unset sentinel for omitted Python arguments, then resolve it to the current effective default so existing callers retain their behavior. Apply this rule to `--trimul_kernel`, `--dtype`, `--enable_tf32` and other overlapping flags. Explicit `LAYERNORM_TYPE`, cache, guidance and Fold-CP choices also take precedence. The resolved value alone does not record the user's intent.

Keep the existing import-time `LAYERNORM_TYPE` selection for the first port, and use local dispatch for qualified pair-row LayerNorm kernels in step 3. `opendde/model/triangular/layers.py` reads the environment variable when it is imported and constructs a concrete LayerNorm module, so changing the mode later cannot switch every existing layer. If whole-model `fast_layernorm` shows a worthwhile gain, make it a separate follow-on design: resolve the choice before model construction, pass it explicitly to the constructors, preserve the environment override, and test mode changes through direct Python use. Do not implement it by changing `LAYERNORM_TYPE` after import.

Allow individual optimizations to be switched off through config keys, for ablation and debugging. Use the existing logger, not the kit's banner protocol.

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
2. **Trunk cache identity.** Audit the inputs and settings read by the trunk, then version the cache format and key it by the feature fingerprint, `N_cycle`, the loaded checkpoint's content identity, relevant trunk configuration, precision and TF32 settings, LayerNorm type, and the effective optimized kernel or row choices. Bypass the disk cache for a call if its numerical path cannot be identified before the lookup. Compute the checkpoint digest once when loading it, or use a verified release-manifest digest; do not rehash the multi-gigabyte file for each call. Keep the disk cache off during numerical qualification until cache hits, misses, custom-checkpoint changes and cross-mode isolation are tested. Reason: today `_trunk_with_cache` hashes only tensor features and `N_cycle`. The CLI also accepts an explicit checkpoint path, so two models or two numerical paths could share one key. Dropping `bond_mask` changes the feature set and hence the key, which is harmless but should be expected.

The 1.2 guidance path is a hard boundary:
- CUDA graph replay stays disabled whenever training-free guidance is enabled. Rigid-contact and epitope guidance require it.
- The other exact changes may stay enabled if their operator-level and end-to-end tests pass with guidance on.

Gate: all existing CPU tests pass; the exact operator tests pass; deterministic `exact` and `off` outputs match on sm_120 before work moves to fast kernels.

Keep exact changes behind the mode gate throughout this port so `off` remains the unmodified 1.2 path for A/B comparison. After each change passes the A100 and sm_120 gates, record whether it is a candidate for default promotion in a later release. Candidates are the host schedule, structural-token sync, JSON encoding, template deduplication and the hoists. A separate promotion decision would update the `off` contract, docs and tests. Reason: keeping two paths for an identical result costs maintenance, but changing the baseline during qualification would invalidate comparisons.

### 3. Bring in only the portable kernels, with a small dispatch

Create a normal package area such as `opendde/model/optimized/` for:
- GPU capability detection;
- per-call-site dispatch;
- the selected Triton kernels;
- the A100 CUDA sources, kept separate.

Do not keep the core's generic provider interfaces, `*_CELLS.json` tables or "tier word" selection. Instead give each call site (triangle attention, triangle multiplication, transition, pair-row LayerNorm, pair-bias attention) a small explicit table keyed by GPU capability, dtype and shape range. It holds only the rows measured in this port. Anything else uses the stock implementation. Reason: the generic machinery serves many models and GPUs, most of it unreachable from OpenDDE on these two GPUs. The tables alone are 31 MB, one of them 17.8 MB.

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
- autograd is needed (`torch.is_grad_enabled()` with inputs that require gradients);
- the device is CPU or MPS;
- the numerical class is unknown.

Order:
1. Triton kernels on sm_120 at gnode2. Start with the stock fallback for every site. Run per-shape correctness and timing sweeps under Triton 3.3.1, and promote only measured winners. The kit's sm_120 entries are hypotheses until then. If a kernel fails with an out-of-resources error at sm_120's ~99 KB shared-memory limit, retune it or leave it on the stock path.
2. The same Triton kernels on sm_80 in the A100 VM, on the same stack.
3. The A100 CUDA triangle-attention and triangle-multiplication kernels, built from source with the cu128 toolkit, as A100-only rows. Building from source avoids depending on the kit's CUDA 13.0 cubins and the R580 driver requirement. Port them only if they beat the Triton rows on A100 at OpenDDE's shapes.

Gate: every promoted row has a correctness record, a run-to-run check, a graph-capture result where applicable, and a measured win over the fallback at the exact call shape.

### 4. Reconcile and port the optimized sampler

Integrate conditioning deduplication, pair-bias attention, the fused token and atom schedules, and low-precision operands into the model classes. Keep the diffusion loop in `generator.py` authoritative, so the 1.2 contact and epitope interventions still run at their current steps and Fold-CP synchronization is unchanged.

Guidance rules:
- When guidance `rho` is non-zero, `tfg/engine.py` calls the denoiser under `torch.enable_grad()` and backpropagates to `x_t`. The kit's fused sampler kernels have no gradient check, and none was confirmed to support autograd. Every fused or low-precision sampler path must route a call that needs gradients to the stock module. Reason: otherwise the guidance gradient may be silently zero or partial; the `allow_unused=True` path in the engine replaces a missing gradient with zeros. The default `rho=0` hides this, so it needs an explicit test.
- Rigid and epitope interventions rewrite `x_l` between steps. Any cached or captured state must be rebuilt from the current `x_l` after an intervention.

Test four cases separately: unguided, generic guidance (with `rho=0` and `rho>0`), rigid contact, and epitope guidance. Exact scheduling must not change intervention timing or random-number consumption. Fast mode must keep guidance success rates.

Gate: end-to-end unguided and guided runs pass the numerical, determinism and constraint-satisfaction checks on sm_120, then on sm_80.

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

## Validation matrix

Use fixed small, medium and large inputs, with thresholds set during step 0 rather than copied from H100. Cover:
- protein-only;
- ligand and covalent-bond;
- template and MSA;
- multi-job and multi-seed;
- unguided, generic guidance with `rho=0` and with `rho>0`, rigid contact, and epitope guidance.

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

- The kit's numerical claims are tied to OpenDDE 1.1.1, a `fast_layernorm` and TF32-on baseline, H100 or A100 on cu126, and specific software versions. Treat them as hypotheses until requalified.
- The kit's Triton kernels are unqualified on sm_120 with Triton 3.3.1. Its only sm_120 measurement used Triton 3.6. A failing optional kernel can fall back to the stock path; a dependency-pin change is needed only if a required target path has no suitable fallback.
- The trunk cache can mix results from different checkpoints or numerical paths unless its identity is extended (step 2). It also conflicts with pair offload, which owns the same trunk result.
- Global LayerNorm is selected at import and model-construction time. Mode-dependent switching needs an explicit construction-time design; the first port keeps that selection fixed.
- CLI defaults do not identify which overlapping options the user supplied. Dispatch must record explicit options before applying those defaults.
- Guidance with `rho>0` needs gradients through the denoiser. Fused sampler kernels without autograd support must step aside, or guidance silently weakens.
- Guidance mutates sampler state between denoising steps. Graph capture and fused schedules must either be proven compatible or step aside.
- Copying more of `opt_core` than needed would bring in tens of thousands of lines and a large maintenance burden. The port needs a dependency inventory and a selected source subset.
- The existing Fold-CP and the overlay's tensor-parallel code solve overlapping problems with different state layouts. Combining them is riskier than benchmarking the native path first.

## Work records to keep in `porting_docs/`

Add these documents as the port advances:

- The step 0 environment check and baseline profile.
- A source-to-destination map for each imported or rewritten optimization, including provenance and license notes.
- The numerical contract and agreed tolerances once baseline results are available.
- GPU qualification tables for sm_80 and sm_120, including refused and fallback shapes.
- Benchmark reports with commands, environment fingerprints, warm or cold status, timings, memory and output comparisons.
- Decisions that change scope, especially go or no-go per group after step 0, the A100 capacity and `big` decision, the trunk-cache identity, any global LayerNorm selection design, dependency-pin changes, and the Fold-CP versus `rowpair_tp` result.
