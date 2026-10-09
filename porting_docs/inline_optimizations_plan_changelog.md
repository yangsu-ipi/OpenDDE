# Inline optimization plan changelog

Changes to `inline_optimizations_plan.md`. Newest first.

## Revision 3 (2026-10-09)

Revision 2 was reviewed against OpenDDE's CLI, LayerNorm construction, and trunk-cache code. These are planning changes only; no environment, GPU job or optimization code was created.

- **Keep global LayerNorm selection fixed during the first port.** Qualify pair-row LayerNorm through local dispatch. Consider mode-dependent whole-model `fast_layernorm` only after measuring a benefit and designing a choice passed before model construction.
  - *Reason:* `triangular/layers.py` reads `LAYERNORM_TYPE` at import and constructs concrete modules. Changing `--optimization_mode` later cannot switch them, and changing the environment after import would give a misleading result.
- **Track which overlapping options the user actually supplied.** Distinguish an explicit `--triatt_kernel auto` from the CLI's implicit `auto`, and use an unset sentinel for omitted direct Python arguments before resolving them to their existing effective defaults.
  - *Reason:* the current CLI and Python entry points pass default values into the configuration even when the user supplied no option. The resulting value alone cannot enforce the plan's rule that explicit choices win.
- **Make the trunk-cache identity cover the checkpoint and effective trunk computation.** Version the format, audit the trunk's inputs, include the loaded checkpoint's content identity and selected numerical path, and test cross-mode isolation. Keep the disk cache off during numerical qualification.
  - *Reason:* the current key covers tensor input features and `N_cycle`, while the CLI accepts a custom checkpoint. Different model weights or kernel decisions could otherwise reuse a result. A verified release digest or a digest computed once at load avoids hashing the checkpoint on every call.
- **Decide on `big` using the actual A100 capacity.** Measure `fast` peak memory on the provisioned A100 and sm_120, and define an acceptable memory margin before the decision.
  - *Reason:* revision 2's 96/80 GB rule excluded a possible 40 GB A100, even though the implementation section proposed 40 GB thresholds. A profile on sm_120 cannot answer whether a 40 GB A100 needs offload.
- **Separate stock failure from optional kernel failure in step 0.** Diagnose a failed stock baseline. Record failures in optional cuEquivariance, `fast_layernorm` or kit Triton paths and continue work that has a stock fallback. Propose a dependency change only after testing an alternative for a required path.
  - *Reason:* PyTorch 2.7 cu128 already includes Blackwell and Triton 3.3 support. The unknown is whether these particular optimized paths work and help on sm_120; one failed optional path need not halt the entire port.
- **Narrow the 1.1.1-to-1.2.0 file comparison to relevant model and inference paths.** Acknowledge the other changed files.
  - *Reason:* revision 2 said "only these files changed," although the full commit diff also includes `runner/template_search.py`, docs, tests and examples.
- **Keep exact changes opt-in for this port.** Record candidates for default promotion as a separate later decision that updates the `off` contract, docs and tests.
  - *Reason:* revision 2 proposed both an unchanged `off` baseline and making qualified exact changes the default. Promoting them during qualification would remove the stable A/B comparator.

## Revision 2 (2026-10-09)

A critical review of revision 1 against the OpenDDE 1.1.1 → 1.2.0 diff, the kit's lever implementations, and the shared core's kernel inventory. Only static reading was done; nothing was run on a GPU.

### Added

- **Step 0: environment check and stock baseline on sm_120.** It verifies that the pinned Triton 3.3.1 and cuEquivariance 0.10 work on sm_120 and that the `fast_layernorm` extension builds there. It also records a per-phase profile that decides which groups are worth porting.
  - *Reason:* the kit's results come from H100 on a different software stack and OpenDDE version. Revision 1 recorded a baseline but did not use it to set scope or order.
- **LayerNorm stream fix as an unconditional bug fix.** `layer_norm_cuda_kernel.cu` launches with `<<<grid, block>>>` on the legacy default stream.
  - *Reason:* this is wrong under any non-default stream or CUDA graph capture, whatever the optimization mode. It is a correctness fix, not an optimization.
- **Trunk cache key extended with the mode and every numerics-changing setting.**
  - *Reason:* `_trunk_with_cache` hashes only input features and `N_cycle`. A trunk computed under `fast` would be served to `off` or `exact`, which breaks the `exact` contract.
- **Gradient rule for the sampler.** Fused or low-precision sampler paths must use the stock module when gradients are needed.
  - *Reason:* with guidance `rho > 0`, `tfg/engine.py` backpropagates through the denoiser. The kit's sampler kernels have no gradient check, and the engine's `allow_unused=True` path turns a missing gradient into zeros, so guidance could weaken silently. The default `rho = 0` hides the problem.
- **Validation-matrix and test cases:**
  - generic guidance with both `rho = 0` and `rho > 0`;
  - trunk cache off for benchmarks;
  - timing variance on the Max-Q (power-limited) cards;
  - a driver check on the A100 VM.
- **Scope rows** for the A100 CUDA kernels (A100 only, built from source) and for excluded sm_90a and other-model kernels.
- **Optional change after qualification:** make byte-identical changes the default code path once they pass both GPU gates.
  - *Reason:* two code paths for identical results are an ongoing maintenance cost.
- **New risks:** Triton 3.3.1 untested on sm_120; trunk-cache mode mixing; guidance gradients through fused kernels.

### Changed

- **The numerical contract now names its baseline:** bf16, TF32 off, `LAYERNORM_TYPE=torch`, triangle kernels `auto`. `exact` may not change the LayerNorm, precision, TF32 or kernel choice; switching to `fast_layernorm` is a `fast`-mode decision.
  - *Reason:* the kit's "stock" used `fast_layernorm` and TF32 on. Its LayerNorm levers do nothing under the 1.2 default, and its fast sampler kernels were tuned against TF32-on stock. Revision 1 mentioned BF16 and TF32 but not the LayerNorm difference.
- **Software stack pinned to `torch==2.7.1` (cu128 build), `triton==3.3.1` and `cuequivariance==0.10.0` on both GPUs.** Revision 1 said "cu128-or-newer PyTorch".
  - *Reason:* these are OpenDDE's `pyproject.toml` pins. torch 2.7.1 has a cu128 build for sm_120, and the kit's A100 tables and prebuilds are keyed to torch 2.7.1 and Triton 3.3. Nothing exists for torch 2.8 or 2.9. Changing a pin is a public-interface decision, left to the user if step 0 shows it is needed.
- **Exact-tier kernels deferred.** Revision 1 implicitly carried them with the Pairformer fast path.
  - *Reason:* the bit-exact sampler attention is an sm_90a-only binary. A100 has bit-exact triangle rows only for some shapes, and sm_120 has none. All records were taken on cu126 and do not carry over to cu128. On the target GPUs, `exact` comes down to the scheduling and data-path changes.
- **Kernel dispatch: small per-site tables instead of the shared core's provider interfaces.**
  - *Reason:* the transitive closure is about 82k lines of Python, 31 MB of tables and 148 MB of binaries. Most of it serves other models or sm_90a and cannot be reached on A100 or sm_120.
- **Kernel qualification order:** Triton kernels on sm_120, then the same Triton kernels on sm_80, then A100 CUDA kernels built from source with the cu128 toolkit, kept only if faster.
  - *Reasons:*
    - No kit kernel has an sm_120 binary, and only one sm_120 Triton configuration was ever measured (on Triton 3.6).
    - The A100 triangle-attention CUDA kernel is sized for 163 KB of shared memory; sm_120 allows about 99 KB.
    - The kit's sm_80 triangle-multiplication cubins need driver R580+, which a from-source build avoids.
- **`big` mode is conditional on the baseline profile.** It was unconditional in revision 1.
  - *Reason:* the kit's offload threshold was 1,400 tokens on 80 GB cards. On a 96 GB card the offload may rarely trigger at target input sizes. Also noted: gnode2's default 29 GB of host RAM per GPU limits pinned pair state.
- **CLI flag renamed from `--optimization-mode` to `--optimization_mode`.**
  - *Reason:* existing OpenDDE flags use underscores (`--triatt_kernel`, `--enable_tf32`).
- **Detail added to the 1.1.1 → 1.2.0 comparison:** exact list of changed files, confirmation that no lever replaces `sample_diffusion` outright, and a list of stock function copies inside the kit that would drift.
- **Optional exact items now state their trade-offs:** prefetch, skipped cache flushes, the allocator setting and graph replay.
- **CUDA graph replay must have its safe token window measured on each GPU.**
  - *Reason:* the kit found A100 replay was not bit-exact above 1,024 tokens.
- **Licensing made specific:**
  - Anthropic's Apache-2.0 copyright line;
  - Protenix notices for the FlashPairformer-derived Triton kernels;
  - PyTorch BSD-3 only if the exact LayerNorm kernel is carried;
  - FlashAttention-3 and CUTLASS obligations avoided by excluding the sm_90a units.
- **Test porting:** carry the kit's behavior tests, not its overlay-mechanics tests.
  - *Reason:* some of the latter check stock line numbers that 1.2 already moved.
- **Step 6 records the `rowpair_tp` and trunk-cache conflict.**
  - *Reason:* ranks would share one cache key while holding different row shards.
- **The overlay-infrastructure exclusion now also names** the `ACTIVE`/`LEVER` banner protocol and the exit-code-3 refusal semantics. Fallbacks use OpenDDE's logger instead.

- **Weights location on gnode2.** Weights now come from gnode2's existing shared, read-only OpenDDE weights store instead of a copy under the user's `/data` directory. Only outputs go there.
  - *Reason:* the store already holds the released `opendde.pt`, and its size and SHA-256 match the 1.2.0 manifest (checked 2026-10-09). It also holds the `common/` data files. A private copy would duplicate about 3.3 GB in a `/data` quota that is already mostly used, and could drift from the shared copy.

### Removed

- The size reference to the "whole 305 MB `opt_core` tree" in the recommendation. The figure was correct for the whole tree, but it was not the right measure of what a port would bring in. It is replaced by the transitive inventory of what the OpenDDE kit actually imports, and the size of a selection limited to A100 and sm_120.

## Revision 1

Initial plan, committed in `7f8b3ef`.
