#!/usr/bin/env python
"""Summarize ``run_opendde.py --phase-timing`` records as a per-phase table.

Usage:
    python scripts/qualification/summarize_phases.py PHASES.json [...]

For each input and seed, prints synchronized wall time and peak device
memory per phase group. The first seed in a process is the cold run (it
includes kernel loading and JIT work); later seeds are warm.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

GROUPS = {
    "featurization": "featurization",
    "batch_unpack": "featurization",
    "batch_preparation": "featurization",
    "opendde_pairformer_trunk": "trunk",
    "opendde_structural_token_expansion": "structural expansion",
    "opendde_pair_output_branch": "structural expansion",
    "opendde_pre_sample_p2p_warmup": "diffusion",
    "opendde_diffusion_cache": "diffusion",
    "opendde_sample_diffusion": "diffusion",
    "opendde_distogram_contact": "confidence",
    "opendde_confidence_head": "confidence",
    "opendde_post_confidence_outputs": "confidence",
    "output_writing": "output writing",
}
ORDER = [
    "featurization",
    "trunk",
    "structural expansion",
    "diffusion",
    "confidence",
    "predict (other)",
    "output writing",
]
GIB = 1024**3


def summarize(path: Path) -> str:
    data = json.loads(path.read_text())
    seconds: dict[tuple, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    peak: dict[tuple, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    order: list[tuple] = []
    for record in data["phases"]:
        run = (record.get("sample_name"), record.get("seed"))
        if run not in order:
            order.append(run)
        phase = record["phase"]
        if phase == "predict":
            seconds[run]["_predict"] += record["seconds"]
            peak[run]["_predict"] = max(
                peak[run]["_predict"], record["peak_allocated_bytes"]
            )
            continue
        group = GROUPS.get(phase, phase)
        seconds[run][group] += record["seconds"]
        peak[run][group] = max(peak[run][group], record["peak_allocated_bytes"])
    lines = [f"### {path.name}", ""]
    header = "| phase | " + " | ".join(
        f"{name} seed {seed} ({'cold' if i == 0 else 'warm'})"
        for i, (name, seed) in enumerate(order)
    )
    lines += [header + " |", "|---" * (len(order) + 1) + "|"]
    for run in order:
        model = sum(
            v
            for k, v in seconds[run].items()
            if k in {"trunk", "structural expansion", "diffusion", "confidence"}
        )
        seconds[run]["predict (other)"] = seconds[run]["_predict"] - model
    for group in ORDER + ["total"]:
        cells = []
        for run in order:
            if group == "total":
                value = sum(seconds[run][g] for g in ORDER)
                cells.append(f"{value:.1f} s")
            else:
                text = f"{seconds[run][group]:.1f} s"
                if peak[run].get(group):
                    text += f" / {peak[run][group] / GIB:.1f} GiB"
                cells.append(text)
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    host = data.get("host_memory", {})
    lines += [
        "",
        f"Process total {data['total_seconds']:.1f} s; host peak "
        f"{host.get('ru_maxrss_bytes', 0) / GIB:.1f} GiB (ru_maxrss).",
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    for arg in sys.argv[1:]:
        print(summarize(Path(arg)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
