#!/usr/bin/env python
"""Targeted fast_layernorm stream diagnostics (step 0, item 6).

Two checks, recorded whether they pass or fail:

* graph capture: capture fast_layernorm forward, and forward plus backward,
  in a CUDA graph, then replay and compare with eager results;
* side-stream correctness: on a non-default stream, enqueue a controlled
  delay before writing the LayerNorm input (and the output gradient), run
  forward and backward on that stream, and compare with the same kernel on
  the default stream and with torch's LayerNorm.

These are diagnostics, not prediction artifacts to match. A pre-fix failure
is not required: capture behavior and race visibility depend on the stack.
"""

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from opendde.model.layer_norm.layer_norm import (
    FusedLayerNorm,
    _load_fast_layer_norm_cuda_v2,
)

SHAPES = [(4096, 384), (128 * 128, 128), (256 * 256, 128)]
DTYPES = [torch.float32, torch.bfloat16]


def _module(c: int, dtype: torch.dtype) -> FusedLayerNorm:
    ln = FusedLayerNorm(c).cuda()
    with torch.no_grad():
        ln.weight.copy_(torch.linspace(0.5, 1.5, c))
        ln.bias.copy_(torch.linspace(-0.1, 0.1, c))
    return ln


def _forward_backward(ln, x, g):
    x = x.detach().requires_grad_(True)
    for p in ln.parameters():
        p.grad = None
    y = ln(x)
    y.backward(g)
    return y.detach(), x.grad.detach(), ln.weight.grad.detach(), ln.bias.grad.detach()


def _summary(a, b) -> dict[str, Any]:
    diff = (a.float() - b.float()).abs()
    return {
        "equal": bool(torch.equal(a, b)),
        "max_abs_diff": float(diff.max()) if diff.numel() else 0.0,
        "nonfinite": int((~torch.isfinite(a.float())).sum()),
    }


def side_stream(shape, dtype, sleep_cycles: int, trials: int) -> dict[str, Any]:
    n, c = shape
    ln = _module(c, dtype)
    gen = torch.Generator(device="cuda").manual_seed(0)
    src = torch.randn(n, c, device="cuda", dtype=dtype, generator=gen)
    grad = torch.randn(n, c, device="cuda", dtype=dtype, generator=gen)
    reference = _forward_backward(ln, src, grad)
    control = _forward_backward(ln, src, grad)
    torch_y = F.layer_norm(
        src.float(), (c,), ln.weight.float(), ln.bias.float(), ln.eps
    ).to(dtype)
    torch.cuda.synchronize()

    results = []
    stream = torch.cuda.Stream()
    for _ in range(trials):
        x = torch.full_like(src, float("nan"))
        g = torch.full_like(grad, float("nan"))
        torch.cuda.synchronize()
        with torch.cuda.stream(stream):
            torch.cuda._sleep(sleep_cycles)
            x.copy_(src)
            torch.cuda._sleep(sleep_cycles)
            g.copy_(grad)
            out = _forward_backward(ln, x, g)
        stream.synchronize()
        torch.cuda.synchronize()
        names = ("output", "grad_input", "grad_weight", "grad_bias")
        results.append({k: _summary(o, r) for k, o, r in zip(names, out, reference)})
    passed = all(all(v["equal"] for v in r.values()) for r in results)
    return {
        "shape": list(shape),
        "dtype": str(dtype),
        "sleep_cycles": sleep_cycles,
        "trials": trials,
        "pass": passed,
        "failed_trials": sum(not all(v["equal"] for v in r.values()) for r in results),
        "first_trial": results[0],
        "default_stream_repeat_equal": all(
            torch.equal(a, b) for a, b in zip(reference, control)
        ),
        "default_stream_vs_torch_output": _summary(reference[0], torch_y),
    }


def graph_capture(shape, dtype, backward: bool) -> dict[str, Any]:
    n, c = shape
    ln = _module(c, dtype)
    gen = torch.Generator(device="cuda").manual_seed(1)
    static_x = torch.randn(n, c, device="cuda", dtype=dtype, generator=gen)
    static_g = torch.randn(n, c, device="cuda", dtype=dtype, generator=gen)
    result: dict[str, Any] = {
        "shape": list(shape),
        "dtype": str(dtype),
        "backward": backward,
    }

    def step():
        if backward:
            return _forward_backward(ln, static_x, static_g)
        with torch.no_grad():
            return (ln(static_x),)

    try:
        warm = torch.cuda.Stream()
        warm.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(warm):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(warm)
        eager = [t.clone() for t in step()]
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = step()
        result["capture"] = "ok"
    except Exception as exc:
        result.update(capture="failed", error=f"{type(exc).__name__}: {exc}"[:400])
        torch.cuda.synchronize()
        return result
    try:
        new = torch.randn(n, c, device="cuda", dtype=dtype, generator=gen)
        static_x.copy_(new)
        graph.replay()
        torch.cuda.synchronize()
        expected = [t.clone() for t in step()]
        result["replay"] = "ok"
        result["replay_vs_eager_new_input"] = [
            _summary(a, b) for a, b in zip(captured, expected)
        ]
        result["pass"] = all(s["equal"] for s in result["replay_vs_eager_new_input"])
        result["eager_before_capture_differs_from_new"] = not torch.equal(
            eager[0], expected[0]
        )
    except Exception as exc:
        result.update(
            replay="failed",
            **{"pass": False},
            error=f"{type(exc).__name__}: {exc}"[:400],
            trace=traceback.format_exc()[-800:],
        )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sleep-cycles", type=int, default=50_000_000)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument(
        "--out", type=Path, required=True, help="JSON report path (stdout is noisy)"
    )
    args = parser.parse_args()

    if _load_fast_layer_norm_cuda_v2() is None:
        print(
            json.dumps({"ok": False, "error": "fast_layernorm extension unavailable"})
        )
        return 1
    report: dict[str, Any] = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "side_stream": [],
        "graph_capture": [],
    }
    for shape in SHAPES:
        for dtype in DTYPES:
            report["side_stream"].append(
                side_stream(shape, dtype, args.sleep_cycles, args.trials)
            )
            for backward in (False, True):
                report["graph_capture"].append(graph_capture(shape, dtype, backward))
    args.out.write_text(json.dumps(report, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
