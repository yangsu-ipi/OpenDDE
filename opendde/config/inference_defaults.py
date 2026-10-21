# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
# pylint: disable=C0114
import os
from typing import Any

from opendde.config.data import default_root_dir
from opendde.config.extend_types import ListValue, RequiredValue
from opendde.config.model_registry import DEFAULT_MODEL_NAME

OPENDDE_ROOT_DIR = os.environ.get("OPENDDE_ROOT_DIR", default_root_dir())
inference_configs: dict[str, Any] = {
    "model_name": DEFAULT_MODEL_NAME,  # inference model selection
    # Empty = "unset"; resolved at run time to CLI > JSON modelSeeds > random seed.
    "seeds": ListValue([], dtype=int),
    "dump_dir": "./output",
    "need_atom_confidence": True,
    "sorted_by_ranking_score": True,
    "device": "auto",
    "input_json_path": RequiredValue(str),
    "load_checkpoint_dir": os.path.join(OPENDDE_ROOT_DIR, "checkpoint"),
    "num_workers": 0,
    "use_msa": True,
    "enable_tf32": False,
    "enable_efficient_fusion": True,
    "enable_diffusion_shared_vars_cache": True,
    "msa_pair_as_unpair": True,
    "use_template": False,
    "use_rna_msa": False,
    # "single" keeps the original single-card path. "distributed" enables the
    # maintained one-dimensional 1 x P context-parallel path.
    "foldcp_mode": "single",
    "foldcp_size_dp": 1,
    "foldcp_size_cp": 1,
    "foldcp_devices": "",
    "foldcp_metrics_jsonl": "",
    # Optimization mode (opendde.config.optimization): "off" runs unmodified
    # OpenDDE. The entry points record which overlapping options the user
    # supplied explicitly; those take precedence over any optimization.
    "optimization_mode": "off",
    "explicit_options": ListValue([], dtype=str),
    # Per-optimization switches for ablation and debugging; an optimization
    # runs only under a mode that includes it and while its switch is true.
    "optimization": {
        "drop_bond_mask": True,
        "template_dedup": True,
        "json_oneshot": True,
        "structural_token_sync": True,
        "host_noise_schedule": True,
        "dit_hoist": True,
        "zprep_hoist": True,
        "cond_dedupe": True,
        "trimul_triton": True,
        "transition_triton": True,
        "triattn_exact": True,
        "triattn_k2b": True,
        "triattn_native": True,
        "dit_fused": True,
    },
}
