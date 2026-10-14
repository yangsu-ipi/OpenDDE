#!/usr/bin/env python
"""Run the stock ``opendde`` CLI in-process with optional step 0 instrumentation.

Usage:
    python scripts/qualification/run_opendde.py [--phase-timing OUT.json]
        [--draw-trace OUT.jsonl] [--set KEY=VALUE ...] [--run-record OUT.json]
        [--tier1-capture DIR [--tier1-replay DIR]] -- pred -i input.json -o out_dir ...

Everything after ``--`` is passed unchanged to the ``opendde`` CLI. Without
either option this is equivalent to running ``opendde``. The instrumentation
is installed through ``opendde.utils.instrumentation``; it is not reachable
from the CLI, config or environment.

``--phase-timing`` synchronizes the device at every phase boundary and records
wall time and peak device memory per phase. It is for profiling only.

``--draw-trace`` records a hash of every raw random draw in the sampler. It
copies each draw to the host, so it synchronizes every step; use traced runs
only to establish random-input identity, never for timing or for checking
reproducibility of prediction artifacts.

``--set`` overrides a dotted inference config key that ``pred`` has no option
for, such as ``sample_diffusion.guidance.rho=0.1`` or
``sorted_by_ranking_score=false``. The value is parsed as JSON, falling back to
a string. It is applied to the config ``pred`` builds, before ``pred`` applies
its own options and validates the result, so it cannot override a ``pred``
option.

``--tier1-capture`` saves, per job and seed, the non-sampling tensors tier 1
compares (``DIR/<job>/seed_<s>/tier1.pt``): the input embedding and the
single and pair representations passed to the distogram and confidence
stages, the distogram probabilities, and the confidence head's inputs,
logits and pTM/ipTM. With ``--tier1-replay`` (a capture directory from the
reference run) the confidence head runs a second time on the recorded inputs
(``s_inputs``, single and pair representations, ``x_pred_coords``) and that
fixed-input result is saved instead of the native head output. The extra
work happens after each stock stage returns and never touches the tensors the
run writes out; Fold-CP runs are refused. Keep it off timing runs.

``--run-record`` writes the run's provenance on its own, for runs that are
neither traced nor timed: the command line, config overrides, exit code,
fingerprint, the wrapper options, the inference-relevant environment
(``ENV_PREFIXES``, read before the run), whether the output and capture
directories existed before the run, any exception that ended it, and the
SHA-256 of every prediction artifact and tier 1 capture it left
(``provenance.py``). Any exception other than a zero ``SystemExit`` is
recorded as exit code 1. The same fields head the trace and timing files.

A recorded run also records its input provenance (contract-6, protocol E5)
under ``provenance``, and fails closed:

- Before launch it accepts only ``pred`` on a single input JSON file and
  refuses a directory, or a run that would search for a protein MSA,
  template or RNA MSA (``provenance.search_reasons``). A template run must
  name an existing executable ``--kalign_binary_path``, so Kalign cannot
  fall back to ``PATH``, and set ``data.template.fetch_remote=false``,
  ``prot_template_mmcif_dir`` and a ``prot_template_cache_dir`` inside the
  output directory that does not exist yet. A refused run is not started;
  its record has exit code 2 and the reason.
- It records the source JSON's path and SHA-256, the SHA-256 of the bytes
  the inference parser reads (``runner.inference._load_inference_jobs_synchronized``,
  the parser-read hook; the parsed jobs must equal those bytes' jobs), and
  the expected effective digest after ``pred``'s format-only legacy MSA
  conversion.
- It records before and after digests of every MSA, RNA-MSA and template
  file the effective JSON names and, for template runs, of every file in
  the mmCIF directory, and checks at the parser read that the template
  cache is still absent.
- After the run it records the selected runtime assets from the validated
  effective configuration (checkpoint, both CCD files and, for template
  runs, the release-date and obsolete-PDB files and the resolved Kalign
  binary) with fresh SHA-256 digests, and the seeds each job's output layout
  holds.

``accept.py`` compares these with the frozen inputs and the runtime-asset
reference. Before and after digests show files were stable between the two
hashes; they do not prove what each later read returned.
"""

import argparse
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import torch

COMPARISON_DTYPE = torch.bfloat16
ENV_PREFIXES = ("OPENDDE_", "LAYERNORM_TYPE", "CUBLAS_WORKSPACE_CONFIG")


def _sync() -> None:
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.synchronize()


def _versions() -> dict[str, Any]:
    import numpy
    import scipy

    out: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "numpy": numpy.__version__,
        "scipy": scipy.__version__,
    }
    for name in (
        "triton",
        "cuequivariance",
        "cuequivariance_torch",
        "rdkit",
        "biotite",
    ):
        try:
            module = __import__(name)
            out[name] = getattr(module, "__version__", "unknown")
        except Exception as exc:  # optional packages
            out[name] = f"unavailable: {type(exc).__name__}"
    return out


def fingerprint() -> dict[str, Any]:
    """Source commit, host, GPU and software-stack fingerprint for a record."""
    repo = Path(__file__).resolve().parents[2]

    def _git(*args: str) -> str:
        try:
            return subprocess.run(
                ["git", "-C", str(repo), *args],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        except Exception:
            return "unknown"

    def _git_bytes_sha256(*args: str) -> str:
        try:
            out = subprocess.run(
                ["git", "-C", str(repo), *args], capture_output=True, check=True
            ).stdout
        except Exception:
            return "unknown"
        return hashlib.sha256(out).hexdigest()

    gpu: dict[str, Any] = {"available": torch.cuda.is_available()}
    if torch.cuda.is_available():
        index = torch.cuda.current_device()
        gpu.update(
            name=torch.cuda.get_device_name(index),
            capability=list(torch.cuda.get_device_capability(index)),
        )
        try:
            gpu["driver"] = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout.split()[0]
        except Exception:
            gpu["driver"] = "unknown"
    return {
        "commit": _git("rev-parse", "HEAD"),
        "dirty_paths": _git("status", "--porcelain").splitlines(),
        # Same bytes as ``git diff HEAD > source.patch``.
        "dirty_diff_sha256": _git_bytes_sha256("diff", "HEAD"),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "layernorm_type": os.environ.get("LAYERNORM_TYPE"),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "gpu": gpu,
        "versions": _versions(),
        "imports": _import_paths(),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _import_paths() -> dict[str, Any]:
    """Where ``opendde`` and ``runner`` resolve (``provenance.import_paths``)."""
    import provenance

    return provenance.import_paths()


class PhaseTimer:
    """Synchronized wall time and peak device memory per phase.

    Phases nest (model stages run inside ``predict``). Peak memory is tracked
    with a stack so each phase reports its own peak and still contributes to
    its parent's.
    """

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self._stack: list[dict[str, int]] = []
        self._cuda = torch.cuda.is_available()

    def _peaks(self) -> dict[str, int]:
        if not self._cuda:
            return {"allocated": 0, "reserved": 0}
        return {
            "allocated": torch.cuda.max_memory_allocated(),
            "reserved": torch.cuda.max_memory_reserved(),
        }

    def _fold_into_parent(self, peaks: dict[str, int]) -> None:
        if self._stack:
            parent = self._stack[-1]
            for key, value in peaks.items():
                parent[key] = max(parent[key], value)

    @contextmanager
    def stage(self, name: str):
        _sync()
        self._fold_into_parent(self._peaks())
        if self._cuda:
            torch.cuda.reset_peak_memory_stats()
        self._stack.append({"allocated": 0, "reserved": 0})
        start = time.perf_counter()
        try:
            yield
        finally:
            _sync()
            seconds = time.perf_counter() - start
            peaks = self._stack.pop()
            for key, value in self._peaks().items():
                peaks[key] = max(peaks[key], value)
            self._fold_into_parent(peaks)
            if self._cuda:
                torch.cuda.reset_peak_memory_stats()
            self.records.append(
                {
                    **self.context,
                    "phase": name,
                    "depth": len(self._stack),
                    "seconds": seconds,
                    "peak_allocated_bytes": peaks["allocated"],
                    "peak_reserved_bytes": peaks["reserved"],
                }
            )


def _tensor_bytes(value: torch.Tensor) -> bytes:
    value = value.detach().contiguous()
    return value.reshape(-1).view(torch.uint8).cpu().numpy().tobytes()


class DrawTrace:
    """Hash every raw sampler draw per original sample index."""

    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.calls: list[dict[str, Any]] = []
        self.context: dict[str, Any] = {}
        self._ordinals: Counter = Counter()

    def begin_rollout(self, call: dict[str, Any]) -> None:
        self.context["rollout"] = len(self.calls)
        self.calls.append({**self.context, **call})

    def __call__(self, record: dict[str, Any], value: Any) -> None:
        value = torch.as_tensor(value)
        sample_dim = record.pop("sample_dim")
        offset = record.get("sample_offset", 0)
        for index in range(value.shape[sample_dim]):
            draw = value.select(sample_dim, index)
            key = {
                **self.context,
                **{k: v for k, v in record.items() if k != "sample_offset"},
                "sample": offset + index,
            }
            ordinal_key = json.dumps(key, sort_keys=True, default=str)
            key["ordinal"] = self._ordinals[ordinal_key]
            self._ordinals[ordinal_key] += 1
            self.records.append(
                {
                    **key,
                    "dtype": str(draw.dtype).replace("torch.", ""),
                    "shape": list(draw.shape),
                    "raw_sha256": hashlib.sha256(_tensor_bytes(draw)).hexdigest(),
                    "cmp_dtype": str(COMPARISON_DTYPE).replace("torch.", ""),
                    "cmp_sha256": hashlib.sha256(
                        _tensor_bytes(draw.to(COMPARISON_DTYPE))
                    ).hexdigest(),
                }
            )


class Tier1Capture:
    """Wrapper-private capture of tier 1 tensors, keyed by job and seed."""

    def __init__(self, out_dir: Path, replay_dir: Path | None) -> None:
        self.out_dir = out_dir
        self.replay_dir = replay_dir
        self.context: dict[str, Any] = {}
        self.pending: dict[str, Any] = {}
        self.written: list[str] = []

    def key_dir(self, root: Path) -> Path:
        return root / self.context["sample_name"] / f"seed_{self.context['seed']}"


def _host(t: torch.Tensor, dtype: torch.dtype | None = None) -> torch.Tensor:
    t = t.detach()
    if dtype is not None:
        t = t.to(dtype)
    return t.to("cpu", copy=True)


def _confidence_scalars(model: Any, kw: dict[str, Any], pae, pde) -> dict[str, Any]:
    """pTM, ipTM and their ranking combination, computed as OpenDDE's summary
    does, after the same residue-level conversion."""
    from opendde.model import sample_confidence

    pred_dict = kw["pred_dict"]
    residue = model.get_residue_level_confidence_inputs(
        input_feature_dict=kw["input_feature_dict"],
        pae_logits=pae,
        pde_logits=pde,
        contact_probs=pred_dict.get(
            "per_sample_contact_probs", pred_dict["contact_probs"]
        ),
        target_device=pae.device,
    )
    pae_prob = torch.softmax(residue["pae_logits"].float(), dim=-1)
    bins = sample_confidence.get_bin_params(model.configs.confidence.pae)
    ptm = sample_confidence.calculate_ptm(
        pae_prob, has_frame=residue["token_has_frame"], **bins
    )
    iptm = sample_confidence.calculate_iptm(
        pae_prob,
        has_frame=residue["token_has_frame"],
        asym_id=residue["token_asym_id"],
        **bins,
    )
    return {"ptm": _host(ptm, torch.float32), "iptm": _host(iptm, torch.float32)}


def _head_record(model: Any, kw: dict[str, Any], outputs) -> dict[str, Any]:
    plddt, pae, pde, resolved = outputs
    return {
        "plddt_logits": _host(plddt, torch.float32),
        "resolved_logits": _host(resolved, torch.float32),
        "pae_logits": _host(pae, torch.float16),
        "pde_logits": _host(pde, torch.float16),
        **_confidence_scalars(model, kw, pae, pde),
    }


def _install_tier1(capture: Tier1Capture) -> None:
    import opendde.model.opendde as model_module

    model_cls = model_module.OpenDDE

    def distogram_stage(fn):
        def wrapper(model, *, pred_dict, pair_z, pair_z_spec):
            result = fn(
                model, pred_dict=pred_dict, pair_z=pair_z, pair_z_spec=pair_z_spec
            )
            if pair_z_spec is not None:
                raise RuntimeError("tier 1 capture does not support Fold-CP")
            from opendde.utils.torch_utils import autocasting_disable_decorator

            # The stock distogram path runs the head with autocast disabled and
            # fp32 inputs (compute_distogram_contact_probs); do the same.
            logits = autocasting_disable_decorator(True)(model.distogram_head)(pair_z)
            probs = torch.softmax(logits.float(), dim=-1)
            del logits
            capture.pending["distogram_probs"] = _host(probs)
            del probs
            return result

        return wrapper

    def confidence_stage(fn):
        def wrapper(model, **kw):
            outputs = fn(model, **kw)
            if kw.get("z_trunk_spec") is not None:
                raise RuntimeError("tier 1 capture does not support Fold-CP")
            record = {
                "s_inputs": _host(kw["s_inputs"]),
                "s_trunk": _host(kw["s_trunk"]),
                "z_trunk": _host(kw["z_trunk"]),
                **capture.pending,
            }
            capture.pending = {}
            if capture.replay_dir is None:
                record["x_pred_coords"] = _host(kw["x_pred_coords"], torch.float32)
                record["head"] = _head_record(model, kw, outputs)
            else:
                source = capture.key_dir(capture.replay_dir) / "tier1.pt"
                recorded = torch.load(source, map_location="cpu")
                fixed = {}
                for name in ("s_inputs", "s_trunk", "z_trunk", "x_pred_coords"):
                    live = kw[name]
                    if recorded[name].shape != live.shape:
                        raise RuntimeError(f"tier 1 replay: {name} shape differs")
                    fixed[name] = recorded[name].to(
                        device=live.device, dtype=live.dtype
                    )
                head_kw = {k: v for k, v in kw.items() if k != "pred_dict"}
                head_kw.update(fixed)
                head_kw["inplace_safe"] = False
                fixed_outputs = model.run_confidence_head(**head_kw)
                record["head_fixed"] = _head_record(model, kw, fixed_outputs)
                record["replay_source"] = str(source)
                del fixed, fixed_outputs
            target = capture.key_dir(capture.out_dir) / "tier1.pt"
            target.parent.mkdir(parents=True, exist_ok=True)
            torch.save(record, target)
            capture.written.append(str(target))
            return outputs

        return wrapper

    _wrap(model_cls, "run_distogram_contact_stage", distogram_stage)
    _wrap(model_cls, "run_confidence_head_stage", confidence_stage)


def _wrap(owner: Any, name: str, make):
    original = getattr(owner, name)
    setattr(owner, name, make(original))


def _truthy(value: str) -> bool:
    import click

    return bool(click.BOOL.convert(value, None, None))


def _override_values(overrides: list[str]) -> dict[str, Any]:
    return {".".join(path): value for path, value in map(_parse_override, overrides)}


class RunProvenance:
    """Input, referenced-file, runtime-asset and seed provenance of one
    recorded ``pred`` run (contract-6; see the module docstring)."""

    def __init__(self, cli_args: list[str], overrides: list[str]) -> None:
        self.cli_args = cli_args
        self.overrides = _override_values(overrides)
        self.record: dict[str, Any] = {"refused": None}
        self.input: Path | None = None
        self.jobs: list[str] = []
        self.configs: Any = None
        self.cache_dir: Path | None = None
        self.mmcif_dir: Path | None = None

    def _flag(self, name: str, default: bool) -> bool:
        value = _cli_value_last(self.cli_args, name)
        return default if value is None else _truthy(value)

    def preflight(self) -> str | None:
        """Record the source state; return why the run is refused, if it is."""
        import provenance

        args = self.cli_args
        if not args or args[0] != "pred":
            return "a recorded run supports only the pred command"
        source = _cli_value_last(args, "-i") or _cli_value_last(args, "--input")
        out = _cli_value_last(args, "-o") or _cli_value_last(args, "--out_dir")
        if source is None or out is None:
            return "a recorded run needs -i and -o"
        path = Path(source)
        if path.is_dir() or path.suffix != ".json" or not path.is_file():
            return f"a recorded run needs a single input JSON file, got {source}"
        flags = {
            "use_msa": self._flag("--use_msa", True),
            "use_template": self._flag("--use_template", False),
            "use_rna_msa": self._flag("--use_rna_msa", False),
        }
        data = path.read_bytes()
        try:
            effective, jobs = provenance.effective_input(data, flags["use_msa"])
        except (ValueError, TypeError, AttributeError) as error:
            return f"input JSON unreadable: {type(error).__name__}: {error}"
        self.input = path
        self.jobs = [j.get("name") for j in jobs if isinstance(j, dict)]
        references = provenance.referenced_files(jobs)
        self.record.update(
            source={
                "path": source,
                "resolved": os.path.realpath(path),
                "sha256": provenance.sha256_bytes(data),
            },
            flags=flags,
            expected_effective_sha256=provenance.sha256_bytes(effective),
            parser_reads=[],
            referenced={
                p: {**provenance.file_state(p), "before": None, "after": None}
                for p in references
            },
            template_mmcif=None,
            template_cache=None,
            kalign=None,
            assets={},
            seeds={},
        )
        for state in self.record["referenced"].values():
            state["before"] = state.pop("sha256")
        searches = provenance.search_reasons(jobs, **flags)
        if searches:
            return "the run would search: " + "; ".join(searches)
        if flags["use_template"]:
            return self._template_preflight(Path(out))
        return None

    def _template_preflight(self, out: Path) -> str | None:
        import provenance

        kalign = _cli_value_last(self.cli_args, "--kalign_binary_path")
        executable = (
            kalign is not None and Path(kalign).is_file() and os.access(kalign, os.X_OK)
        )
        self.record["kalign"] = {
            "requested": kalign,
            "requested_resolved": None if kalign is None else os.path.realpath(kalign),
            "executable": executable,
        }
        if not executable:
            return f"--kalign_binary_path must name an executable file, got {kalign}"
        if self.overrides.get("data.template.fetch_remote") is not False:
            return "a template run must set data.template.fetch_remote=false"
        cache = self.overrides.get("data.template.prot_template_cache_dir")
        mmcif = self.overrides.get("data.template.prot_template_mmcif_dir")
        if not isinstance(cache, str) or not isinstance(mmcif, str):
            return "a template run must set prot_template_cache_dir and prot_template_mmcif_dir"
        self.cache_dir, self.mmcif_dir = Path(cache), Path(mmcif)
        inside = Path(os.path.realpath(cache)).is_relative_to(os.path.realpath(out))
        self.record["template_cache"] = {
            "path": cache,
            "inside_output": inside,
            "absent_before": not self.cache_dir.exists(),
            "absent_at_parse": None,
        }
        self.record["template_mmcif"] = {
            "dir": mmcif,
            "resolved": os.path.realpath(mmcif),
            "before": provenance.tree_digests(mmcif),
            "after": None,
        }
        if not inside or self.cache_dir.exists():
            return f"the template cache {cache} must be new and inside the output directory"
        return None

    def parser_read(self, fn):
        """The parser-read hook: hash the bytes of the JSON the inference
        parser reads and require the parsed jobs to equal them."""
        import provenance
        from opendde.data.inference.input_validation import validate_inference_jobs

        def wrapper(path, *args, **kwargs):
            data = Path(path).read_bytes()
            cache = self.record.get("template_cache")
            if cache is not None:
                cache["absent_at_parse"] = not self.cache_dir.exists()
                if not cache["absent_at_parse"]:
                    raise RuntimeError(f"template cache {self.cache_dir} prepopulated")
            jobs = fn(path, *args, **kwargs)
            try:
                same = validate_inference_jobs(json.loads(data)) == jobs
            except ValueError:
                same = False
            self.record["parser_reads"].append(
                {
                    "path": str(path),
                    "resolved": os.path.realpath(path),
                    "sha256": provenance.sha256_bytes(data),
                    "matches_parse": same,
                }
            )
            if not same:
                raise RuntimeError(f"parsed jobs differ from the bytes of {path}")
            return jobs

        return wrapper

    def capture_config(self, configs: Any) -> None:
        if self.configs is None:
            self.configs = configs

    def _assets(self) -> dict[str, Any]:
        """Paths of the runtime assets the validated configuration selected."""
        from opendde.utils.download import resolve_checkpoint_path

        c = self.configs
        paths = {
            "checkpoint": resolve_checkpoint_path(c),
            "ccd_components_file": c.data.ccd_components_file,
            "ccd_components_rdkit_mol_file": c.data.ccd_components_rdkit_mol_file,
        }
        if c.use_template:
            template = c.data.template
            paths.update(
                release_dates_path=template.release_dates_path,
                obsolete_pdbs_path=template.obsolete_pdbs_path,
                kalign_binary_path=template.kalign_binary_path,
            )
        return paths

    def finish(self, out_dir: Path) -> None:
        """After the run: second digests, runtime assets and output seeds."""
        import provenance

        for path, state in self.record.get("referenced", {}).items():
            state["after"] = provenance.file_state(path)["sha256"]
        mmcif = self.record.get("template_mmcif")
        if mmcif is not None:
            mmcif["after"] = provenance.tree_digests(mmcif["dir"])
        if self.configs is not None:
            self.record["assets"] = {
                name: {"path": path, **provenance.file_state(path)}
                for name, path in self._assets().items()
            }
            self.record["effective_config"] = {
                "use_template": bool(self.configs.use_template),
                "prot_template_cache_dir": self.configs.data.template.prot_template_cache_dir,
                "prot_template_mmcif_dir": self.configs.data.template.prot_template_mmcif_dir,
            }
        self.record["seeds"] = provenance.job_seeds(out_dir, self.jobs)


def _install_provenance(prov: RunProvenance) -> None:
    import runner.inference as inference_module

    _wrap(inference_module, "_load_inference_jobs_synchronized", prov.parser_read)


def _install(
    timer: PhaseTimer | None,
    trace: DrawTrace | None,
    tier1: Tier1Capture | None = None,
    prov: RunProvenance | None = None,
) -> None:
    import opendde.model.opendde as model_module
    import runner.dumper as dumper_module
    import runner.inference as inference_module
    from opendde.utils import instrumentation

    contexts = [obj for obj in (timer, trace, tier1) if obj is not None]

    def staged(name):
        def make(fn):
            def wrapper(*args, **kwargs):
                if timer is None:
                    return fn(*args, **kwargs)
                with timer.stage(name):
                    return fn(*args, **kwargs)

            return wrapper

        return make

    def prepare_prediction_batch(fn):
        def wrapper(runner, configs, data, *, seed, **kwargs):
            if prov is not None:
                prov.capture_config(configs)
            for obj in contexts:
                obj.context = {"sample_name": str(data["sample_name"]), "seed": seed}
            if timer is not None:
                # Featurization for this batch ran before its name was known.
                for record in reversed(timer.records):
                    if "sample_name" in record:
                        break
                    record.update(timer.context)
                with timer.stage("batch_preparation"):
                    return fn(runner, configs, data, seed=seed, **kwargs)
            return fn(runner, configs, data, seed=seed, **kwargs)

        return wrapper

    def next_batch(fn):
        def wrapper(*args, **kwargs):
            if timer is None:
                return fn(*args, **kwargs)
            timer.context = {}
            with timer.stage("featurization"):
                result = fn(*args, **kwargs)
            if not result[0]:
                timer.records.pop()  # end-of-data probe, not a batch
            return result

        return wrapper

    _wrap(inference_module, "_next_inference_batch_synchronized", next_batch)
    _wrap(inference_module, "_prepare_inference_batch", staged("batch_unpack"))
    _wrap(inference_module, "_prepare_prediction_batch", prepare_prediction_batch)
    _wrap(inference_module.InferenceRunner, "predict", staged("predict"))
    _wrap(dumper_module.DataDumper, "dump", staged("output_writing"))

    if timer is not None:
        instrumentation.install_phase_recorder(timer.stage)

    if tier1 is not None:
        _install_tier1(tier1)

    if prov is not None:
        _install_provenance(prov)

    if trace is not None:

        def sample_diffusion(fn):
            def wrapper(*args, **kwargs):
                guidance = kwargs.get("guidance_configs")
                trace.begin_rollout(
                    {
                        "N_sample": kwargs.get("N_sample"),
                        "diffusion_chunk_size": kwargs.get("diffusion_chunk_size"),
                        "N_step": len(kwargs["noise_schedule"]) - 1,
                        "rollout_seed": kwargs.get("rollout_seed"),
                        "guidance_configs": _jsonable(guidance),
                    }
                )
                return fn(*args, **kwargs)

            return wrapper

        _wrap(model_module, "sample_diffusion", sample_diffusion)
        instrumentation.install_draw_recorder(trace)


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        return _jsonable(to_dict())
    return repr(value)


def _host_peak_bytes() -> dict[str, Any]:
    out: dict[str, Any] = {
        "ru_maxrss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    }
    job = os.environ.get("SLURM_JOB_ID")
    if job:
        for path in Path("/sys/fs/cgroup/system.slice").glob(
            f"*slurmstepd.scope/job_{job}/memory.peak"
        ):
            out["cgroup_job_memory_peak_bytes"] = int(path.read_text())
    return out


def _cli_value(args: list[str], name: str) -> str | None:
    return args[args.index(name) + 1] if name in args[:-1] else None


def _cli_value_last(args: list[str], name: str) -> str | None:
    """The value of the last occurrence of ``name``, as click resolves it."""
    found = [i for i, a in enumerate(args[:-1]) if a == name]
    return args[found[-1] + 1] if found else None


def _parse_override(text: str) -> tuple[list[str], Any]:
    key, sep, raw = text.partition("=")
    if not sep or not key:
        raise SystemExit(f"error: --set expects KEY=VALUE, got {text!r}")
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw
    return key.split("."), value


def _install_overrides(overrides: list[tuple[list[str], Any]]) -> None:
    import runner.batch_inference as batch_module

    def apply(configs: Any) -> None:
        for path, value in overrides:
            node = configs
            for part in path[:-1]:
                node = node[part] if isinstance(node, dict) else getattr(node, part)
            if isinstance(node, dict):
                if path[-1] not in node:
                    raise SystemExit(f"error: unknown config key {'.'.join(path)}")
                node[path[-1]] = value
            else:
                if not hasattr(node, path[-1]):
                    raise SystemExit(f"error: unknown config key {'.'.join(path)}")
                setattr(node, path[-1], value)

    def make(fn):
        def wrapper(*args, **kwargs):
            configs = fn(*args, **kwargs)
            apply(configs)
            return configs

        return wrapper

    _wrap(batch_module, "build_inference_config", make)


def main() -> int:
    argv = sys.argv[1:]
    if "--" not in argv:
        print(__doc__, file=sys.stderr)
        return 2
    split = argv.index("--")
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-timing", type=Path)
    parser.add_argument("--draw-trace", type=Path)
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--tier1-capture", type=Path)
    parser.add_argument("--tier1-replay", type=Path)
    parser.add_argument("--run-record", type=Path)
    options = parser.parse_args(argv[:split])
    if options.tier1_replay and not options.tier1_capture:
        parser.error("--tier1-replay needs --tier1-capture")
    cli_args = argv[split + 1 :]

    timer = PhaseTimer() if options.phase_timing else None
    trace = DrawTrace() if options.draw_trace else None
    tier1 = (
        Tier1Capture(options.tier1_capture, options.tier1_replay)
        if options.tier1_capture
        else None
    )
    prov = RunProvenance(cli_args, options.overrides) if options.run_record else None
    refused = prov.preflight() if prov is not None else None
    if prov is not None:
        prov.record["refused"] = refused
    _install(timer, trace, tier1, prov)
    overrides = [_parse_override(text) for text in options.overrides]
    if overrides:
        _install_overrides(overrides)

    from runner.cli import opendde_cli

    # The fingerprint is taken after the run so it never initializes CUDA
    # ahead of the stock code path.
    meta: dict[str, Any] = {
        "cli_args": cli_args,
        "config_overrides": options.overrides,
        "wrapper": {
            name: None if value is None else str(value)
            for name, value in (
                ("draw_trace", options.draw_trace),
                ("phase_timing", options.phase_timing),
                ("tier1_capture", options.tier1_capture),
                ("tier1_replay", options.tier1_replay),
            )
        },
        "env": {
            k: v for k, v in sorted(os.environ.items()) if k.startswith(ENV_PREFIXES)
        },
    }
    out_dir = _cli_value(cli_args, "-o")
    meta["preexisting"] = {
        "output": None if out_dir is None else Path(out_dir).exists(),
        "tier1_capture": (
            None if options.tier1_capture is None else options.tier1_capture.exists()
        ),
    }
    meta["exception"] = None
    start = time.perf_counter()
    exit_code = 1
    try:
        if refused is not None:
            exit_code = 2
            meta["exception"] = f"refused: {refused}"
            print(f"error: run refused: {refused}", file=sys.stderr)
        else:
            result = opendde_cli.main(
                args=cli_args, prog_name="opendde", standalone_mode=False
            )
            exit_code = result if isinstance(result, int) else 0
    except SystemExit as exc:
        exit_code = exc.code if isinstance(exc.code, int) else int(exc.code is not None)
        if exit_code:
            meta["exception"] = f"SystemExit: {exc.code}"
    except BaseException as exc:
        meta["exception"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        meta["total_seconds"] = time.perf_counter() - start
        meta["exit_code"] = exit_code
        meta["host_memory"] = _host_peak_bytes()
        meta["fingerprint"] = fingerprint()
        if options.run_record is not None:
            import provenance

            meta["outputs"] = {
                "prediction_sha256": (
                    {}
                    if out_dir is None
                    else provenance.prediction_digests(Path(out_dir))
                ),
                "tier1_sha256": (
                    {}
                    if options.tier1_capture is None
                    else provenance.capture_digests(options.tier1_capture)
                ),
            }
            if refused is None and out_dir is not None:
                try:
                    prov.finish(Path(out_dir))
                except Exception as exc:  # recorded; accept.py then fails the run
                    prov.record["finish_error"] = f"{type(exc).__name__}: {exc}"
            meta["provenance"] = prov.record
        if options.run_record is not None:
            options.run_record.parent.mkdir(parents=True, exist_ok=True)
            options.run_record.write_text(json.dumps(meta, indent=1) + "\n")
        if timer is not None:
            options.phase_timing.parent.mkdir(parents=True, exist_ok=True)
            options.phase_timing.write_text(
                json.dumps({**meta, "phases": timer.records}, indent=1)
            )
        if trace is not None:
            options.draw_trace.parent.mkdir(parents=True, exist_ok=True)
            with options.draw_trace.open("w") as handle:
                handle.write(json.dumps({**meta, "rollouts": trace.calls}) + "\n")
                for record in trace.records:
                    handle.write(json.dumps(record, default=str) + "\n")
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
