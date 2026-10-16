"""Every fast_layernorm CUDA launch must run on PyTorch's current stream.

A launch without a stream argument goes to the legacy default stream, which is
wrong under any non-default stream and inside CUDA graph capture.
"""

import re
from pathlib import Path

KERNEL_DIR = (
    Path(__file__).resolve().parents[1] / "opendde" / "model" / "layer_norm" / "kernel"
)
LAUNCH = re.compile(r"<<<(.*?)>>>", re.S)
STREAM_DEF = re.compile(r"\bstream\s*=\s*([^;]+);")


def _top_level_args(config: str) -> list[str]:
    """Split a launch configuration on commas outside parentheses."""
    args, depth, current = [], 0, ""
    for ch in config:
        if ch == "," and depth == 0:
            args.append(current.strip())
            current = ""
            continue
        depth += {"(": 1, ")": -1}.get(ch, 0)
        current += ch
    return args + [current.strip()]


def _sources():
    files = sorted(KERNEL_DIR.glob("*.cu")) + sorted(KERNEL_DIR.glob("*.cuh"))
    assert files, f"no CUDA sources under {KERNEL_DIR}"
    return files


def test_every_launch_names_a_stream():
    total = 0
    for path in _sources():
        text = path.read_text()
        for match in LAUNCH.finditer(text):
            total += 1
            args = _top_level_args(match.group(1))
            line = text.count("\n", 0, match.start()) + 1
            # <<<grid, block, shared_bytes, stream>>>
            assert len(args) == 4 and args[3] == "stream", f"{path.name}:{line}"
    assert total == 28  # the known launch sites; update with the source


def test_stream_is_the_current_pytorch_stream():
    for path in _sources():
        text = path.read_text()
        if not LAUNCH.search(text):
            continue
        definitions = STREAM_DEF.findall(text)
        assert definitions, path.name
        for value in definitions:
            assert value.strip() == "at::cuda::getCurrentCUDAStream().stream()", (
                path.name,
                value,
            )
