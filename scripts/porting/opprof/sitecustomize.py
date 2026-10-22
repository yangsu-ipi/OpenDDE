"""Module-family GPU time of one warm OpenDDE model forward (kit-item disposition, plan revision 117).

Usage: put this directory first on PYTHONPATH and set OPENDDE_OPPROF_OUT to an output directory, then run
rows through scripts/qualification/run_rows.py with two seeds; the hook installs only in run_opendde.py
processes and acts on the second OpenDDE.__call__ (the warm seed). Summarize with
scripts/porting/opprof_items.py.

OPENDDE_OPPROF_MODE=events (the default) times every module of the classes in FAMILIES with CUDA events on
the GPU timeline and charges each its span minus the spans of the family modules inside it (exclusive time,
idle gaps included), writing <row>.opevents.json; it holds little host memory and adds about 1% to the
call. A frame whose closing hook never runs (a guided row's sampler blocks) is closed with the enclosing
frame and counted in frames_closed_early.

OPENDDE_OPPROF_MODE=profiler runs torch.profiler over the call with each family module in a
record_function range "<top-level component>/<class>" and charges each GPU kernel to the innermost range
enclosing its launch (<row>.opprof.json: per-family kernel time and the top kernels). Inside the diffusion
module it also charges each kernel to its sampler part (the diffusion module's child the module sits in:
diffusion_transformer, atom_attention_encoder, ..., with each atom transformer as its own part, such as
atom_attention_encoder.atom_transformer) and keeps the per-part kernel times by kernel name. It
keeps every event in host memory, so large or guided rows can exceed a 29 GB job.

OPENDDE_OPPROF_MODE=idle samples NVML utilization of the job's GPU every 20 ms during the warm call and writes
<row>.opidle.json with the mean busy share (the share of time with at least one kernel running), which bounds
what host-overhead items can save.

OPENDDE_TRIAL_KEEP_POOL=1 (independent of the modes) makes torch.cuda.empty_cache a counted no-op inside
every OpenDDE.__call__ of the process, the kit's keep_pool, for timing it.

Development use only: the outputs of profiled runs are not compared with anything.
"""

import os

OUT = os.environ.get("OPENDDE_OPPROF_OUT")
MODE = os.environ.get("OPENDDE_OPPROF_MODE", "events")
KEEP_POOL = os.environ.get("OPENDDE_TRIAL_KEEP_POOL") == "1"

FAMILIES = {
    "TriangleAttention",
    "TriangleMultiplicationIncoming",
    "TriangleMultiplicationOutgoing",
    "Transition",
    "AttentionPairBias",
    "OuterProductMean",
    "MSAPairWeightedAveraging",
    "OpenFoldLayerNorm",
    "FusedLayerNorm",
    "AdaptiveLayerNorm",
    "ConditionedTransitionBlock",
    "AtomTransformer",
    "AtomAttentionEncoder",
    "AtomAttentionDecoder",
    "DiffusionConditioning",
    "DiffusionTransformer",
    "FourierEmbedding",
    "DistogramHead",
    "ConfidenceHead",
    "PairformerStack",
    "MSAModule",
    "TemplateEmbedder",
    "InputFeatureEmbedder",
    "RelativePositionEncoding",
    "StructuralTokenExpander",
    "DiffusionModule",
}


def _install():
    import json
    import sys
    import time
    from collections import defaultdict

    import torch
    from opendde.model import opendde as mod

    row = "unknown"
    if "-o" in sys.argv:
        row = os.path.basename(os.path.normpath(sys.argv[sys.argv.index("-o") + 1]))
    state = {"calls": 0}
    original = mod.OpenDDE.__call__

    def wrap_modules(model):
        handles = []
        for path, m in model.named_modules():
            cls = type(m).__name__
            if cls not in FAMILIES:
                continue
            top = path.split(".")[0] if path else "(root)"
            name = f"M::{top}/{cls}"
            if top == "diffusion_module":
                parts = path.split(".")
                if len(parts) > 2 and parts[2] == "atom_transformer":
                    parts[1] = ".".join(parts[1:3])  # the atom transformers apart
                name += "|" + (parts[1] if len(parts) > 1 else "(self)")

            def pre(module, args, kwargs=None, _name=name):
                rf = torch.profiler.record_function(_name)
                rf.__enter__()
                module.__dict__.setdefault("_opprof_stack", []).append(rf)

            def post(module, args, out, _name=name):
                stack = module.__dict__.get("_opprof_stack")
                if stack:
                    stack.pop().__exit__(None, None, None)

            handles.append(m.register_forward_pre_hook(pre))
            handles.append(m.register_forward_hook(post))
        return handles

    def call(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] != 2:
            return original(self, *args, **kwargs)
        handles = wrap_modules(self)
        acts = [
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ]
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.profiler.profile(activities=acts) as prof:
            with torch.profiler.record_function("M::(root)/OpenDDE"):
                result = original(self, *args, **kwargs)
            torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        for h in handles:
            h.remove()
        fam = defaultdict(float)
        kernels = defaultdict(lambda: [0.0, 0])
        part_kernels = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
        total = 0.0
        for e in prof.events():
            ks = getattr(e, "kernels", None) or []
            if not ks:
                continue
            owner = e
            while owner is not None and not owner.name.startswith("M::"):
                owner = owner.cpu_parent
            key = (
                owner.name[3:].split("|")[0] if owner is not None else "(unattributed)"
            )
            part = (
                owner.name.split("|")[1]
                if owner is not None and "|" in owner.name
                else None
            )
            for k in ks:
                fam[key] += k.duration
                total += k.duration
                kernels[k.name][0] += k.duration
                kernels[k.name][1] += 1
                if part is not None:
                    part_kernels[part][k.name][0] += k.duration
                    part_kernels[part][k.name][1] += 1
        top = sorted(kernels.items(), key=lambda kv: -kv[1][0])[:80]
        rec = {
            "row": row,
            "argv": sys.argv,
            "wall_s_profiled": wall,
            "kernel_us_total": total,
            "families_us": dict(sorted(fam.items(), key=lambda kv: -kv[1])),
            "top_kernels": [{"name": n, "us": v[0], "count": v[1]} for n, v in top],
            "sampler_part_kernels": {
                part: [
                    {"name": n, "us": v[0], "count": v[1]}
                    for n, v in sorted(ks.items(), key=lambda kv: -kv[1][0])
                ]
                for part, ks in part_kernels.items()
            },
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{row}.opprof.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(
            f"[opprof] {row}: kernel time {total / 1e6:.2f} s over a {wall:.2f} s profiled call",
            flush=True,
        )
        return result

    def call_events(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] != 2:
            return original(self, *args, **kwargs)
        frames = []
        stack = []

        skipped = [0]
        early = defaultdict(int)

        def push(name, key=None):
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            frame = {"name": name, "key": key, "start": ev, "end": None, "children": []}
            (stack[-1]["children"] if stack else frames).append(frame)
            stack.append(frame)

        def pop(key=None):
            # Close frames down to this module's own frame: a frame whose forward hook never ran (for
            # example a forward that raised and was caught) is closed here and counted as skipped.
            if not any(f["key"] == key for f in stack):
                return
            ev = torch.cuda.Event(enable_timing=True)
            ev.record()
            while stack:
                frame = stack.pop()
                frame["end"] = ev
                if frame["key"] == key:
                    break
                skipped[0] += 1
                early[frame["name"]] += 1

        handles = []
        for path, m in self.named_modules():
            cls = type(m).__name__
            if cls not in FAMILIES:
                continue
            top = path.split(".")[0] if path else "(root)"
            name = f"{top}/{cls}"
            handles.append(
                m.register_forward_pre_hook(lambda mod_, a, _n=name: push(_n, id(mod_)))
            )
            handles.append(m.register_forward_hook(lambda mod_, a, o: pop(id(mod_))))
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        push("(root)/OpenDDE", "root")
        result = original(self, *args, **kwargs)
        pop("root")
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        for h in handles:
            h.remove()
        excl = defaultdict(float)
        calls = defaultdict(int)
        unclosed = 0

        def total_of(frame):
            if frame["end"] is None:
                return None
            return frame["start"].elapsed_time(frame["end"])

        span = 0.0
        todo = list(frames)
        while todo:  # iterative: guided rows can nest deeply
            frame = todo.pop()
            total = total_of(frame)
            if total is None:
                unclosed += 1
                continue
            if frame in frames:
                span += total
            inner = sum(
                t for t in (total_of(c) for c in frame["children"]) if t is not None
            )
            excl[frame["name"]] += total - inner
            calls[frame["name"]] += 1
            todo.extend(frame["children"])
        rec = {
            "row": row,
            "argv": sys.argv,
            "mode": "events",
            "wall_s": wall,
            "span_ms": span,
            "unclosed_frames": unclosed,
            "frames_closed_early": skipped[0],
            "closed_early_by_name": dict(early),
            "families_ms": dict(sorted(excl.items(), key=lambda kv: -kv[1])),
            "calls": dict(calls),
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{row}.opevents.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(
            f"[opprof] {row}: GPU span {span / 1e3:.2f} s over a {wall:.2f} s call (events)",
            flush=True,
        )
        return result

    def call_idle(self, *args, **kwargs):
        state["calls"] += 1
        if state["calls"] != 2:
            return original(self, *args, **kwargs)
        import threading

        import pynvml

        pynvml.nvmlInit()
        uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
        handle = pynvml.nvmlDeviceGetHandleByUUID(
            uuid if uuid.startswith("GPU-") else "GPU-" + uuid
        )
        samples = []
        stop = threading.Event()

        def poll():
            while not stop.is_set():
                samples.append(pynvml.nvmlDeviceGetUtilizationRates(handle).gpu)
                stop.wait(0.02)

        torch.cuda.synchronize()
        thread = threading.Thread(target=poll, daemon=True)
        t0 = time.perf_counter()
        thread.start()
        result = original(self, *args, **kwargs)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
        stop.set()
        thread.join()
        busy = sum(samples) / len(samples) if samples else None
        rec = {
            "row": row,
            "argv": sys.argv,
            "mode": "idle",
            "wall_s": wall,
            "samples": len(samples),
            "busy_percent_mean": busy,
            "busy_percent_min": min(samples) if samples else None,
            "torch": torch.__version__,
            "device": torch.cuda.get_device_name(),
        }
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, f"{row}.opidle.json"), "w") as f:
            json.dump(rec, f, indent=1)
        print(
            f"[opprof] {row}: GPU busy {busy:.1f}% over a {wall:.2f} s call (idle)",
            flush=True,
        )
        return result

    chosen = (
        {"events": call_events, "idle": call_idle}.get(MODE, call) if OUT else original
    )
    if KEEP_POOL:
        empty_cache = torch.cuda.empty_cache
        inside = {"depth": 0, "skipped": 0}

        def keep_pool_empty_cache():
            if inside["depth"]:
                inside["skipped"] += 1
                return
            empty_cache()

        torch.cuda.empty_cache = keep_pool_empty_cache

        def call_keep_pool(self, *args, **kwargs):
            inside["depth"] += 1
            try:
                return chosen(self, *args, **kwargs)
            finally:
                inside["depth"] -= 1
                print(
                    f"[opprof] keep_pool: {inside['skipped']} in-forward empty_cache calls skipped so far",
                    flush=True,
                )

        mod.OpenDDE.__call__ = call_keep_pool
    else:
        mod.OpenDDE.__call__ = chosen


if OUT or KEEP_POOL:
    try:
        import importlib.util

        if (
            importlib.util.find_spec("opendde") is not None
            and os.path.basename(__import__("sys").argv[0] or "") == "run_opendde.py"
        ):
            _install()
    except Exception as exc:  # never break a run
        print(f"[opprof] not installed: {exc!r}", flush=True)
