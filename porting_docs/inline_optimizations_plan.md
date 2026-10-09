# Inline optimization porting plan

## Recommendation

Port the optimization kit in three functional layers: exact scheduling and data-path changes, fast kernels, then large-input memory changes. Keep OpenDDE 1.2.0 behavior as the default and make every optimized path opt-in. Do not copy the overlay launcher, import hooks, monkeypatch framework, stock-package verifier, or the whole 305 MB `opt_core` tree. Move the OpenDDE-specific code and only the kernel/provider pieces it uses into the OpenDDE package, call them from the normal model and runner code, and retain explicit fallbacks to the current PyTorch and cuEquivariance implementations.

This work is isolated on `porting/inline-optimizations`, created from OpenDDE commit `9d0d9c9`. The source optimization kit was inspected at commit `f4f62fa`. This document is a plan only; no optimization code has been ported yet.

## What the comparison showed

- The optimization kit is pinned to OpenDDE 1.1.1 commit `ddfa1df`. The fork's 1.2.0 commit descends directly from that pin.
- Nearly every model component patched by the kit is unchanged between 1.1.1 and 1.2.0: Pairformer, diffusion modules, transformer blocks, structural-token expansion, confidence, triangle operations, LayerNorm, `runner/inference.py`, `runner/dumper.py`, and the featurizer.
- The risky changes are concentrated in `opendde/model/generator.py` and `opendde/model/opendde.py`. Version 1.2 adds rigid contact and epitope interventions inside the diffusion loop, plus an optional trunk cache. The kit wraps or replaces code in both areas.
- OpenDDE 1.2 already defaults to BF16 and disables TF32. The kit's old stock-base launcher forced BF16 against a 1.1.1 FP32 default and assumed TF32 in several qualification notes. That launcher behavior should not be ported, and all numerical claims need to be re-established with the 1.2 defaults.
- The kit has A100-specific rows and tests. It has no qualified OpenDDE rows for sm_120; one adapter explicitly lists only sm_80 and sm_90 as tested. The shared core contains generic or inferred sm_120 fallbacks, but inferred rows are not enough to claim correctness or speed on the RTX Pro 6000.
- The kit's `big --n_gpu P` implementation overlaps with native Fold-CP and refuses native Fold-CP and guidance combinations. On gnode2, multi-GPU communication is PCIe-only. It should not be part of the first port.

## Scope by optimization group

| Group | Disposition | Notes |
| --- | --- | --- |
| Exact host and data-path changes | Port first | Drop unused `bond_mask`, deduplicate padded templates, avoid in-forward cache flushes, remove sampler host synchronizations, hoist invariant pair preparation, reduce structural-token syncs, encode JSON in one pass, optionally prefetch inputs, use the current stream for LayerNorm, and guard CUDA graph replay. |
| Pairformer fast path | Port after exact parity | Add the transpose-free, cast-once Pairformer path plus triangle attention, triangle multiplication, transition, and C=384 LayerNorm provider dispatch. Keep the stock torch and cuEquivariance implementations as named fallbacks. |
| Diffusion and atom fast path | Port after Pairformer | Add conditioning deduplication, optimized pair-bias attention, fused token and atom schedules, and low-precision operands. Reconcile these changes directly with the 1.2 guidance loop. |
| Single-GPU `big` mode | Port last | Add host-resident trunk, structural, and confidence pair tensors; diffusion pair memoization; one-sample chunks; template release; and a large-LayerNorm guard. Make the trunk-cache interaction explicit. |
| Custom `rowpair_tp` | Defer | Benchmark native Fold-CP first. Reconsider this path only if Fold-CP leaves a demonstrated capacity or throughput gap on the target systems. |
| Unshipped or negative-value levers | Exclude initially | `writer_overlap`, `sampler_amp`, `xl_tri_ln`, and the older `dit_attn_bf16` are not part of a shipped mode or were measured as neutral or negative. |
| Overlay infrastructure | Do not port | Do not add `.pth` files, `sitecustomize`, runtime monkeypatches, a separate `opendde-opt` executable, a pinned-wheel verifier, or a clean-subprocess stock runner. |

## Implementation plan

### 1. Establish the baseline and acceptance rules

Record cold and warm phase timings, peak allocated and reserved GPU memory, output digests, and environment metadata for the unmodified 1.2 code. Use the released checkpoint and fixed seeds.

Define the numerical contract before moving code:

- `off`: current OpenDDE 1.2 behavior and current defaults.
- `exact`: byte-identical prediction artifacts to `off` under the deterministic recipe; any unqualified kernel uses the stock operation.
- `fast`: deterministic for repeated runs with the same environment when the underlying kernel is deterministic, finite throughout, and within an agreed validation envelope against `off` and normal seed-to-seed variation.
- `big`: the same numerical class as `fast`, with lower peak device memory.

Add one native configuration object for mode, architecture capability, per-lever decisions, and fallbacks. Expose it through the existing CLI and config system, preferably as `--optimization-mode {off,exact,fast,big}`, defaulting to `off`. Existing `--triatt_kernel`, `--trimul_kernel`, precision, TF32, cache, guidance, and Fold-CP choices must continue to win when the user states them.

### 2. Port the exact, low-risk changes into their call sites

Implement each exact change as ordinary OpenDDE code behind the native mode and config gate. Start with independent changes that can be unit-tested on CPU: unread feature removal, template slot deduplication, JSON encoding, prefetch policy, and allocation or chunk policy. Then integrate the sampler scheduling changes and structural-token synchronization reduction.

Modify the existing LayerNorm CUDA source so launches use PyTorch's current CUDA stream. Build it through OpenDDE's existing extension loader for the active architecture; do not carry the kit's cu126 sm_80 or sm_90 binary. Add CUDA graph capture tests before enabling step replay.

The 1.2 guidance path is a hard boundary: CUDA graph replay stays disabled whenever training-free guidance is active, including contact and epitope guidance. Other exact optimizations may remain enabled if their operator-level and end-to-end tests pass.

Gate: all existing CPU tests pass; exact operator tests pass; deterministic `exact` and `off` outputs match on sm_120 before work proceeds to fast kernels.

### 3. Bring in only the kernel and provider subset OpenDDE uses

Create a normal package area such as `opendde/model/optimized/` for architecture detection, provider selection, kernel census, and the carried Triton or CUDA sources. Preserve the provider interfaces at first, but prune JAX, unrelated-model adapters, unused prebuilt binaries, and generic overlay machinery. Carry the required third-party notices and mark modified files as required by Apache-2.0.

Replace monkeypatched `forward` methods with explicit dispatch inside the existing Pairformer, triangle, transition, LayerNorm, diffusion, and atom modules. A provider refusal, compile failure, unsupported shape, explicit stock kernel flag, autograd, CPU or MPS execution, or unknown numerical class must select the existing implementation and emit a structured fallback record instead of silently changing behavior.

Qualify the architectures separately:

1. On sm_120 at gnode2, begin with stock fallbacks, run per-shape correctness and timing sweeps, and promote only measured rows.
2. On sm_80 in the provisioned A100 VM, validate the carried A100 rows against the new inline call sites and the chosen cu128 stack; do not assume the old cu126 records transfer.

Gate: every promoted row has a correctness record, run-to-run check, graph-capture result where applicable, and a measured win over the fallback for the exact call shape.

### 4. Reconcile and port the optimized sampler

Integrate conditioning deduplication, pair-bias attention, fused DiT blocks, low-precision operands, and fused atom blocks into the model classes. Keep the diffusion loop in `generator.py` authoritative so the 1.2 contact and epitope interventions still run at their current steps and Fold-CP synchronization remains intact.

Test four cases separately: unguided, generic TFG, rigid contact, and epitope guidance. Exact scheduling must not change intervention timing or random-number consumption. Fast mode must preserve guidance success rates and must not reuse stale captured state after a rigid-body intervention.

Gate: end-to-end unguided and guided runs pass numerical, determinism, and constraint-satisfaction checks on sm_120, then on sm_80.

### 5. Port single-GPU large-input memory handling

Convert the overlay's `HostPair` and row-block machinery into native stage implementations for trunk, structural expansion and refinement, diffusion conditioning, confidence, and distogram. Keep the existing PyTorch path available at every stage.

Resolve the 1.2 trunk cache deliberately. Initially, disable the disk trunk cache when pair offload is active and report the reason. Re-enable it only after defining a stable serialization format for host-resident pair state and testing cache hits across guided and unguided inputs. Benchmarks must disable the disk cache so it cannot hide model work.

Tune offload and hoist thresholds independently for 96 GB sm_120 and 40 or 80 GB A100. Do not reuse H100 thresholds. Include host RAM and transfer time in the decision, especially on gnode2 where node RAM is shared.

Gate: `big` has a lower measured CUDA peak than `fast`, handles the target large inputs without corrupting outputs, and does not regress medium inputs enough to justify selecting it below its gate.

### 6. Decide whether custom multi-GPU work remains

Benchmark native Fold-CP on the target workloads after the single-GPU port. Open a follow-on phase for the overlay's `rowpair_tp` only if native Fold-CP cannot meet a concrete capacity or throughput target. That phase needs a separate design because it overlaps Fold-CP internals, currently refuses guidance, and will be sensitive to gnode2's PCIe-only topology.

### 7. Finish the public surface and release hygiene

Update CLI help, config defaults and schema, docs, examples, `opendde doctor`, changelog, package data, and notices together. Add a machine-readable run report with selected rows, fallbacks, software stack, architecture, warm or cold state, phase times, and peak memory. Keep generated binaries, caches, benchmark outputs, checkpoints, and private paths out of git.

## Validation matrix

Use fixed small, medium, and large inputs, with thresholds determined during baseline rather than copied from H100. Cover protein-only, ligand and covalent-bond, template and MSA, multi-job, multi-seed, and all four 1.2 guidance cases. For each mode, record cold start, second-run latency, per-phase latency, peak allocated and reserved VRAM, host-memory peak, output hashes, confidence values, coordinate comparisons, and guidance satisfaction.

On gnode2, code will live under the home directory and checkpoints and outputs under `/data/<user>`. GPU commands must run through Slurm. Small kernel tests can use a GPU share with host RAM proportional to the share; clean end-to-end benchmarks need an exclusive GPU and normally `--mem=29G`. Large host-offload tests may need more RAM, but the request should be based on measured use and scheduled so it does not strand other GPUs. The environment should use `~/miniforge3`, Python 3.11, and a cu128-or-newer PyTorch build because sm_120 cannot run the kit's cu126 stack.

Use the same Python and OpenDDE versions on A100 where practical, with cu128 PyTorch, so architecture is the main variable. Ask for approval before creating either conda environment and ask the user to provision the A100 VM when the A100 gate is ready. The A100 gate comes after CPU and sm_120 validation, so cloud time is spent on a stable candidate.

## Main risks

- Numerical claims from the kit are tied to OpenDDE 1.1.1, older defaults, and specific GPU and software cells. Treat them as hypotheses until requalified.
- The trunk cache and pair offload both own the trunk result's lifecycle; combining them casually can serialize the wrong representation or retain too much memory.
- Guidance mutates sampler state between denoising steps. Graph capture and fused schedules must either prove compatibility or step aside.
- Blindly copying `opt_core` would import hundreds of megabytes of unrelated and prebuilt artifacts, plus a large maintenance burden. The port needs a dependency inventory and a curated source subset.
- The existing Fold-CP implementation and overlay tensor-parallel implementation solve overlapping problems with different state layouts. Combining them is riskier than benchmarking the native path first.

## Work records to keep in `porting_docs/`

Add these documents as the port advances:

- A source-to-destination map for each imported or rewritten optimization, including provenance and license notes.
- The numerical contract and agreed tolerances once baseline results are available.
- Architecture qualification tables for sm_80 and sm_120, including refused and fallback shapes.
- Benchmark reports with commands, environment fingerprints, warm or cold status, timings, memory, and output comparisons.
- Decisions that change scope, especially trunk-cache policy and the Fold-CP versus `rowpair_tp` result.
