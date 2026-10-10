#!/usr/bin/env python
"""Check and compare sampler random-draw traces from ``run_opendde.py``.

Commands:
    manifest TRACE           check TRACE against the expected draw manifest
    selftest TRACE           confirm the manifest check catches injected faults
    compare A B [--hash H]   compare two traces draw by draw (H: raw or cmp)

The expected manifest is derived from the sampling settings recorded for each
rollout (``N_sample``, chunk size, step count and guidance settings). In the
1.2 sampler every draw count is fixed by configuration:

* one initial Gaussian per sample;
* per step and sample: one step Gaussian, one augmentation rotation and one
  augmentation translation;
* with guidance enabled, per step and sample: one Monte Carlo Gaussian when the
  perturbation standard deviation is nonzero, and one Gaussian per guidance
  outer step.

The rigid-contact and epitope modules draw no random numbers. If a stock
``off`` trace fails this check, treat it as a harness defect: inspect the
source, hooks and manifest before qualifying anything.
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

KEY_FIELDS = (
    "sample_name",
    "seed",
    "rollout",
    "chunk",
    "step",
    "site",
    "kind",
    "outer",
    "sample",
    "ordinal",
)


def load(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    with path.open() as handle:
        meta = json.loads(handle.readline())
        records = [json.loads(line) for line in handle if line.strip()]
    return meta, records


def key_of(record: dict[str, Any]) -> tuple:
    return tuple(record.get(field) for field in KEY_FIELDS)


def _guidance(configs: Any) -> tuple[bool, float, int]:
    if not configs:
        return False, 0.0, 0
    from opendde.tfg import parse_tfg_config

    cfg = parse_tfg_config(configs)
    if not cfg.enable:
        return False, 0.0, 0
    return True, float(cfg.eps_std), int(cfg.outer_steps)


def _chunks(n_sample: int, chunk_size: int | None) -> Iterable[tuple[int, int, int]]:
    if chunk_size is None:
        yield 0, 0, n_sample
        return
    size = max(1, chunk_size)
    for index, offset in enumerate(range(0, n_sample, size)):
        yield index, offset, min(size, n_sample - offset)


def expected_keys(meta: dict[str, Any]) -> Counter:
    expected: Counter = Counter()
    for call in meta["rollouts"]:
        enabled, eps_std, outer_steps = _guidance(call.get("guidance_configs"))
        base = {
            "sample_name": call.get("sample_name"),
            "seed": call.get("seed"),
            "rollout": call["rollout"],
        }

        def add(**fields: Any) -> None:
            record = {**base, "outer": None, "ordinal": 0, **fields}
            expected[key_of(record)] += 1

        for chunk, offset, size in _chunks(
            call["N_sample"], call.get("diffusion_chunk_size")
        ):
            for sample in range(offset, offset + size):
                common = {"chunk": chunk, "sample": sample}
                add(**common, step=None, site="sampler.initial_noise", kind="gaussian")
                for step in range(call["N_step"]):
                    add(**common, step=step, site="sampler.step_noise", kind="gaussian")
                    add(
                        **common,
                        step=step,
                        site="augmentation.rotation",
                        kind="rotation",
                    )
                    add(
                        **common,
                        step=step,
                        site="augmentation.translation",
                        kind="gaussian",
                    )
                    if enabled and eps_std != 0.0:
                        add(**common, step=step, site="tfg.eps", kind="gaussian")
                    if enabled:
                        for outer in range(outer_steps):
                            add(
                                **common,
                                step=step,
                                outer=outer,
                                site="tfg.outer_noise",
                                kind="gaussian",
                            )
    return expected


def check_manifest(meta: dict[str, Any], records: list[dict[str, Any]]) -> dict:
    expected = expected_keys(meta)
    observed = Counter(key_of(record) for record in records)
    missing = expected - observed
    extra = observed - expected
    duplicates = [key for key, count in observed.items() if count > 1]

    def by_site(counter: Counter) -> dict[str, int]:
        sites: Counter = Counter()
        for key, count in counter.items():
            sites[key[KEY_FIELDS.index("site")]] += count
        return dict(sites)

    return {
        "ok": not missing and not extra and not duplicates,
        "expected": sum(expected.values()),
        "observed": sum(observed.values()),
        "missing_by_site": by_site(missing),
        "extra_by_site": by_site(extra),
        "duplicate_keys": len(duplicates),
        "expected_by_site": by_site(expected),
    }


def self_test(meta: dict[str, Any], records: list[dict[str, Any]]) -> dict:
    results: dict[str, Any] = {"baseline_ok": check_manifest(meta, records)["ok"]}
    sites = sorted({record["site"] for record in records})
    for site in sites:
        kept = [record for record in records if record["site"] != site]
        results[f"omit:{site}"] = not check_manifest(meta, kept)["ok"]
    if records:
        results["duplicate"] = not check_manifest(meta, records + [records[0]])["ok"]
        swapped = records[1:] + [records[-1]]
        results["omit_one_and_duplicate_one"] = not check_manifest(meta, swapped)["ok"]
    expected_sites = set(check_manifest(meta, records)["expected_by_site"])
    results["all_expected_sites_seen"] = expected_sites <= set(sites)
    results["ok"] = all(value for value in results.values() if isinstance(value, bool))
    return results


def compare(
    a: list[dict[str, Any]], b: list[dict[str, Any]], hash_field: str
) -> dict[str, Any]:
    index_a = {key_of(record): record for record in a}
    index_b = {key_of(record): record for record in b}
    only_a = set(index_a) - set(index_b)
    only_b = set(index_b) - set(index_a)
    mismatched: Counter = Counter()
    first: dict[str, Any] = {}
    for key in set(index_a) & set(index_b):
        ra, rb = index_a[key], index_b[key]
        same = ra[hash_field] == rb[hash_field] and ra["shape"] == rb["shape"]
        if hash_field == "raw_sha256":
            same = same and ra["dtype"] == rb["dtype"]
        if not same:
            site = key[KEY_FIELDS.index("site")]
            mismatched[site] += 1
            first.setdefault(site, dict(zip(KEY_FIELDS, key)))
    dtypes = {
        site: sorted(
            {f"{r['dtype']}" for r in records if r["site"] == site}
            for records in (a, b)
        )
        for site in sorted({r["site"] for r in a + b})
    }
    return {
        "ok": not only_a and not only_b and not mismatched,
        "hash": hash_field,
        "compared": len(set(index_a) & set(index_b)),
        "only_in_a": len(only_a),
        "only_in_b": len(only_b),
        "mismatched_by_site": dict(mismatched),
        "first_mismatch_by_site": first,
        "dtypes_by_site": {k: [sorted(x) for x in v] for k, v in dtypes.items()},
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("manifest", "selftest"):
        sub.add_parser(name).add_argument("trace", type=Path)
    cmp = sub.add_parser("compare")
    cmp.add_argument("a", type=Path)
    cmp.add_argument("b", type=Path)
    cmp.add_argument("--hash", choices=("raw", "cmp"), default="raw")
    args = parser.parse_args()

    if args.command == "compare":
        _, a = load(args.a)
        _, b = load(args.b)
        result = compare(a, b, f"{args.hash}_sha256")
    else:
        meta, records = load(args.trace)
        check = check_manifest if args.command == "manifest" else self_test
        result = check(meta, records)
    print(json.dumps(result, indent=1, default=str))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
