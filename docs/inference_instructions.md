# Inference Instructions

Concise reference for installing OpenDDE, preparing runtime data, and running
`opendde` commands.

## Install

OpenDDE supports CPython `3.11`, `3.12`, and `3.13`. We recommend
[`uv`](https://docs.astral.sh/uv/getting-started/installation/) for Python
installations. Choose one of the following methods.

### Install from PyPI

```bash
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu opendde
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 "opendde[gpu]"
```

### Install from source

```bash
git clone https://github.com/aurekaresearch/OpenDDE.git
cd OpenDDE
uv venv --python 3.11
```

CPU:

```bash
uv pip install --python .venv --torch-backend cpu -e .
```

NVIDIA GPU (Linux x86_64, CUDA 12.6):

```bash
uv pip install --python .venv --torch-backend cu126 -e ".[gpu]"
```

After a PyPI or source installation, verify the environment with:

```bash
uv run --no-project --python .venv opendde doctor
```

### Use Docker

The prebuilt image targets NVIDIA GPU inference:

```bash
docker pull aurekaresearch/opendde:v1
```

See the [Docker guide](./docker_installation.md) for GPU setup, runtime-data
mounts, and a complete `docker run` example.

> [!NOTE]
> `--torch-backend` selects the PyTorch build, while `[gpu]` adds the optional
> cuEquivariance kernels. Linux wheels require glibc 2.28 or newer. Apple
> Silicon runs on CPU or on the Metal (MPS) backend with `--device mps`; Intel
> macOS is unsupported, and Windows has not been validated. At runtime,
> `--device auto` uses CUDA when available, then MPS, and otherwise falls back
> to CPU.

## Runtime data

Set `OPENDDE_ROOT_DIR` to the directory that stores checkpoints and runtime data:

```text
$OPENDDE_ROOT_DIR/
├── checkpoint/opendde.pt
├── common/
└── search_database/        # needed for local template/RNA-MSA search
```

The default checkpoint and managed common files come from a release-pinned
asset revision and are verified against their published size and SHA-256.
Checkpoints passed explicitly with `--load_checkpoint_path` are left untouched.

Prepare data from a source checkout:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh
```

For a protein-only prediction that disables MSA, template, and RNA-MSA features,
search databases are not needed:

```bash
bash scripts/download_opendde_data.sh --skip-search-database
```

If you already have a custom checkpoint, keep a descriptive filename and pass
it directly. Use `--skip-model` when preparing only the remaining runtime data:

```bash
mkdir -p "$OPENDDE_ROOT_DIR/checkpoint"
cp /path/to/my_checkpoint.pt \
  "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt"
bash scripts/download_opendde_data.sh --skip-model
opendde pred \
  --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/my_checkpoint.pt" \
  -i examples/cofold/input.json \
  -o ./output
```

The names `opendde.pt` and `opendde_abag.pt` are reserved for released assets.
Their authoritative links, sizes, and digests are in
[supported_models.md](./supported_models.md).

Use `opendde.pt` with `-n opendde_v1` as the default general-purpose
checkpoint. To use the ABAG-optimized checkpoint, keep it as
`opendde_abag.pt` and pass it with `--load_checkpoint_path`, for example
`opendde pred --load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"`.

Install and verify the ABAG checkpoint from the same manifest-backed helper:

```bash
export OPENDDE_ROOT_DIR=/path/to/opendde_data
bash scripts/download_opendde_data.sh \
  --checkpoint opendde_abag.pt \
  --skip-common \
  --skip-search-database
```

Then run general-purpose inference without an explicit checkpoint path. For ABAG
inference, add:

```bash
--load_checkpoint_path "$OPENDDE_ROOT_DIR/checkpoint/opendde_abag.pt"
```

Useful environment variables:

| Variable | Purpose |
| --- | --- |
| `OPENDDE_ROOT_DIR` | Checkpoints, common files, search databases. Defaults to `~/.cache/opendde`. |
| `OPENDDE_DEPENDENCY_URL` | Override checkpoint download root. |
| `OPENDDE_COMMON_URL` | Override common runtime file download root. Falls back to `OPENDDE_DEPENDENCY_URL` when set. |
| `OPENDDE_SEARCH_DATABASE_URL` | Override template/RNA-MSA database download root. |
| `LAYERNORM_TYPE` | LayerNorm backend; defaults to `torch`. Set to `fast_layernorm` to opt into the fused kernel. |

Template/RNA-MSA preprocessing also needs HMMER. Template inference may need
`kalign`:

```bash
apt-get update && apt-get install -y hmmer kalign
```

## Input JSON

OpenDDE input is a top-level list of jobs:

```json
[
  {
    "name": "tiny",
    "modelSeeds": [101],
    "sequences": [
      {
        "proteinChain": {
          "sequence": "ACDEFGHIK",
          "count": 1
        }
      }
    ]
  }
]
```

`covalent_bonds` is optional and may be omitted from a job; include it only to
declare explicit covalent links between entities.

Full schema: [infer_json_format.md](./infer_json_format.md).

Convert a structure file to JSON:

```bash
opendde json -i examples/cofold/structures/7pzb.pdb -o ./output --altloc first
opendde json -i examples/cofold/structures/2lwu.cif -o ./output --altloc first --assembly_id 1
```

## Preprocess optional features

```bash
# Protein MSA
opendde msa -i examples/cofold/input.json -o ./output

# Protein MSA + template search
opendde mt -i examples/cofold/input.json -o ./output

# Protein MSA + template search + RNA MSA when RNA is present
opendde prep -i examples/cofold/input.json -o ./output
```

Notes:

- Protein MSA uses the public ColabFold MMseqs2 API unless A3M paths are already
  present in the JSON.
- Template and RNA-MSA search use local databases under
  `$OPENDDE_ROOT_DIR/search_database/`.
- Generated JSON files are written under
  `<out_dir>/.opendde_preprocessed/<input-hash>/` rather than next to the
  input JSON, so read-only input directories work.

Details: [msa_template_pipeline.md](./msa_template_pipeline.md).

## Run prediction

Standard run:

```bash
opendde pred -i examples/cofold/input.json -o ./output -n opendde_v1
```

Compatibility run with the standard step/cycle counts:

```bash
opendde pred \
  -i examples/cofold/input.json \
  -o ./output \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10
```

Inference defaults to `--device auto`, `bf16` with TF32 off (`--dtype fp32` and
`--enable_tf32 true` opt out), and `auto` triangle kernels.
Device auto-selection uses NVIDIA CUDA when available, then the Apple Metal
(MPS) backend on Apple Silicon, and otherwise CPU.
cuEquivariance is selected only when its Linux CUDA packages import successfully;
otherwise the model uses PyTorch triangle kernels. MPS always uses PyTorch
triangle kernels and follows the same BF16 default and dynamic policy as CUDA:
the trunk uses BF16; by default, diffusion and the confidence head stay FP32
through 2560 tokens, confidence uses BF16 above 2560, and diffusion uses BF16
above 3840 to reduce memory.
BF16 performance varies by Apple GPU and workload, so compare it with FP32 for
your inputs. BF16 autocast needs macOS 14 or newer and is downgraded to FP32
below that.

## Multi-GPU Fold-CP inference

> [!IMPORTANT]
> Fold-CP inference does not currently support cuEquivariance (`cueq`) triangle
> kernels. Select the distributed PyTorch implementations with
> `--triatt_kernel torch --trimul_kernel torch`. In distributed mode, `auto`
> resolves to these PyTorch kernels and an explicit `cueq` request fails before
> model loading. On CUDA BF16, Fold-CP triangle
> attention also uses Triton 3.3.1 from the GPU install extra to fuse
> attention-bias addition; this Triton helper is separate from cuEquivariance.

Fold-CP distributes token-pair-heavy inference work over a `1 x P` mesh, where
`P` can be any available GPU count greater than one. Launch it with `torchrun`
and expose exactly the GPUs you want to use. For example, four GPUs use:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node 4 \
  -m runner.batch_inference pred \
  -i examples/cofold/protein_200.json \
  -o ./output_foldcp \
  -n opendde_v1 \
  --use_msa false \
  --use_template false \
  --use_rna_msa false \
  --sample 1 \
  --step 200 \
  --cycle 10 \
  --trimul_kernel torch \
  --triatt_kernel torch \
  --foldcp_mode distributed \
  --foldcp_size_dp 1 \
  --foldcp_size_cp 4
```

Runtime notes:

- `--nproc_per_node P` must match `--foldcp_size_dp 1` times
  `--foldcp_size_cp P`.
- `--foldcp_size_cp P` creates a `1 x P` context-parallel mesh. The example
  uses `P=4`; other GPU counts require changing both `4` values.
- Keep the input, model, dtype, cycle, step, sample, MSA, and template settings
  identical when comparing single-GPU and Fold-CP outputs. Select
  `--triatt_kernel torch --trimul_kernel torch` for both runs; the distributed
  CUDA BF16 path additionally uses Triton for attention-bias fusion.
- Outputs are written under the requested `-o/--out_dir` just like normal
  inference.
- Optional `--foldcp_metrics_jsonl path/to/metrics.jsonl` records Fold-CP timing
  and memory metrics.

For single-GPU inference, omit the Fold-CP flags or set
`--foldcp_mode single --foldcp_size_cp 1`.

Use prepared features:

```bash
opendde pred -i examples/cofold/template/example_9fm7.json \
  -o ./output -n opendde_v1 \
  --use_msa true --use_template true

opendde pred -i examples/cofold/rna_msa/example_9gmw_2.json \
  -o ./output -n opendde_v1 \
  --use_rna_msa true
```

## Optional TFG Guidance

OpenDDE includes default-off Training-Free Guidance (TFG). TFG refines each
sampled trajectory with geometry potentials while keeping the requested
`--sample` count unchanged.

```bash
opendde pred -i examples/cofold/input.json -o ./output -n opendde_v1 \
  --use_tfg_guidance true
```

Outputs are written to:

```text
<out_dir>/<job_name>/seed_<seed>/predictions/
```

With `--use_tfg_guidance true`, the input JSON `constraint` field (`contact`,
`movable_chains`, `epitope`) additionally steers the sampler towards partial
knowledge of an antibody–antigen interface. See
[TFG constraint guidance](./tfg_constraint_guidance.md), the `constraint` section
of [infer_json_format.md](./infer_json_format.md) and
[`examples/tfg`](../examples/tfg/README.md).

## Common flags

| Flag | Meaning |
| --- | --- |
| `-n`, `--model_name` | Model name. Currently `opendde_v1`. |
| `--load_checkpoint_path` | Explicit checkpoint path. |
| `--seeds` | Comma-separated seeds, e.g. `101,102`. Overrides the job's `modelSeeds`; if unset, `modelSeeds` are used, or a random seed when both are absent. |
| `--use_msa` | Use/generate protein MSA features. |
| `--use_template` | Use/generate template features. |
| `--use_rna_msa` | Use/generate RNA MSA features; requires `--use_msa true`. |
| `--use_tfg_guidance` | Enable Training-Free Guidance. |
| `--foldcp_mode` | `single` or `distributed`; use `distributed` with `torchrun` for multi-GPU Fold-CP inference. |
| `--foldcp_size_dp` | Compatibility option; only `1` is supported. Runtime `2 x 2` topology is not maintained. |
| `--foldcp_size_cp` | Context-parallel degree `P` in the maintained `1 x P` mesh; must match the launched process count. |
| `--foldcp_devices` | Optional visible-device list recorded in Fold-CP metrics; actual GPU visibility is controlled by `CUDA_VISIBLE_DEVICES`. |
| `--foldcp_metrics_jsonl` | Optional JSONL path for Fold-CP timing and memory metrics. |
| `--dtype` | `bf16` or `fp32`. |
| `--device` | `auto`, `cpu`, `cuda`, or `mps`; auto uses CUDA when available, then Apple MPS, and otherwise CPU. |
| `--trimul_kernel`, `--triatt_kernel` | `auto`, `cuequivariance`, or `torch`. |
| `--optimization_mode` | `off` (default, unmodified OpenDDE), `exact` or `fast`. `exact` runs only changes that give byte-identical prediction files to `off` under `--deterministic true`; in default non-deterministic runs it matches `off` in distribution, not bit for bit. Options you set explicitly, such as `--triatt_kernel auto`, are never overridden by an optimization. `exact` keeps the diffusion denoiser's step-invariant tensors for each sampler call instead of recomputing them every step, including the token transformer's pair biases (about 1.5 KB × N_token² of GPU memory, 2.2 GB at 1,200 tokens); the trunk usually sets the peak. On an RTX PRO 6000 Blackwell (compute capability 12.0) or an A100 (8.0) with the pinned torch 2.7, Triton 3.3.1 and cuEquivariance 0.10 stack, `exact` also runs Triton kernels for triangle multiplication and the pair transition that give the stock kernels' bits. On both GPUs it also runs a CUDA triangle-attention kernel that gives cuEquivariance's bits; it is compiled on first use with CUDA 12.8 `nvcc` (from `CUDA_HOME` or `PATH`) for the GPU's compute capability into `$TORCH_EXTENSIONS_DIR` (default `~/.cache/torch_extensions`), and without that compiler triangle attention runs the stock kernel. On other GPUs or stacks those calls run the stock kernels. `fast` runs everything in `exact` plus kernels whose results differ from `off` in rounding only: on the same RTX PRO 6000 stack, a Triton flash kernel for triangle attention, and on the A100 a CUDA triangle-attention kernel (compiled on first use like the exact one), each with an error against an fp32 reference matching the stock kernel's. On the RTX PRO 6000 stack, `fast` also computes the diffusion sampler's single conditioning once instead of once per sample when a run uses 5 samples and the input has 1–64 or 77–1,120 tokens; its rounding differs from stock's, with an error against an fp64 reference within 1.01× of stock's. Outputs under `fast` are not bit-identical to `off`; they are repeatable run to run under `--deterministic true`. Explicitly setting `--trimul_kernel` or `--triatt_kernel` keeps the stock kernels for that site. `big` is not available in this release, and any mode other than `off` is rejected with Fold-CP. |

Run `opendde <command> --help` for the full option list.
