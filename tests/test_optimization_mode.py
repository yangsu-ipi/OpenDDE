"""Optimization mode, explicit-option record and Fold-CP rejection (CPU)."""

import pytest
import torch
from click.testing import CliRunner

import runner.batch_inference as batch_inference
import runner.inference as inference
from opendde.config import optimization as opt
from opendde.config.inference import build_inference_config

# ------------------------------------------------------------- resolution


def test_omitted_options_take_defaults_and_are_not_explicit():
    supplied = {name: opt.UNSET for name in opt.OVERLAPPING_OPTION_DEFAULTS}
    values, explicit = opt.resolve_overlapping_options(supplied)
    assert values == dict(opt.OVERLAPPING_OPTION_DEFAULTS)
    assert explicit == []


def test_a_passed_default_value_is_explicit():
    supplied = {name: opt.UNSET for name in opt.OVERLAPPING_OPTION_DEFAULTS}
    supplied["triatt_kernel"] = "auto"
    values, explicit = opt.resolve_overlapping_options(supplied)
    assert values["triatt_kernel"] == "auto"
    assert explicit == ["triatt_kernel"]


def test_given_explicit_names_replace_the_markers():
    supplied = dict(opt.OVERLAPPING_OPTION_DEFAULTS)  # concrete, as from Click
    _, explicit = opt.resolve_overlapping_options(supplied, ["dtype"])
    assert explicit == ["dtype"]
    with pytest.raises(ValueError, match="not overlapping"):
        opt.resolve_overlapping_options(supplied, ["use_msa"])
    with pytest.raises(ValueError, match="not overlapping"):
        opt.resolve_overlapping_options({"use_msa": True})


def test_unset_marker():
    assert opt.UNSET is opt._Unset()
    assert not opt.UNSET
    assert repr(opt.UNSET) == "<default>"


def test_dotted_overrides_and_environment_count_as_explicit():
    arg_str = "--dtype fp32 --model.N_cycle 4 --triangle_attention=torch"
    assert opt.explicit_from_arg_str(arg_str) == ["dtype", "triatt_kernel"]
    assert opt.explicit_from_arg_str("--model.N_cycle 4") == []
    env = {"LAYERNORM_TYPE": "torch", "OPENDDE_TRUNK_CACHE": "", "OTHER": "x"}
    assert opt.explicit_environment(env) == ["env:LAYERNORM_TYPE"]


# ------------------------------------------------------------- validation


@pytest.mark.parametrize("mode", ["off", "exact"])
def test_available_modes_pass(mode):
    assert opt.validate_optimization_mode(mode, foldcp_enabled=False) == mode


@pytest.mark.parametrize("mode", ["fast", "big"])
def test_unavailable_modes_are_rejected(mode):
    with pytest.raises(ValueError, match="not available in this release"):
        opt.validate_optimization_mode(mode, foldcp_enabled=False)


def test_unknown_mode_is_rejected():
    with pytest.raises(ValueError, match="must be one of"):
        opt.validate_optimization_mode("turbo", foldcp_enabled=False)


def test_fold_cp_accepts_only_off():
    assert opt.validate_optimization_mode("off", foldcp_enabled=True) == "off"
    with pytest.raises(ValueError, match="not supported with Fold-CP"):
        opt.validate_optimization_mode("exact", foldcp_enabled=True)


# ------------------------------------------------------------- plan


def test_plan_with_no_registered_optimizations_runs_stock(monkeypatch):
    monkeypatch.setattr(opt, "OPTIMIZATIONS", {})
    configs = build_inference_config(fill_required_with_null=True)
    configs.optimization_mode = "exact"
    configs.explicit_options = ["dtype"]
    plan = opt.build_optimization_plan(configs, device=None)
    assert plan.mode == "exact" and plan.capability is None
    assert plan.decisions == {} and not plan.enabled("anything")
    assert "no optimizations apply" in plan.summary()
    assert "explicit options: dtype" in plan.summary()


def test_registered_exact_optimizations_run_only_outside_off():
    configs = build_inference_config(fill_required_with_null=True)
    configs.optimization_mode = "exact"
    plan = opt.build_optimization_plan(configs, device=None)
    assert plan.enabled("drop_bond_mask")
    configs.optimization_mode = "off"
    assert not opt.build_optimization_plan(configs, device=None).enabled(
        "drop_bond_mask"
    )
    configs.optimization_mode = "exact"
    configs.optimization = {"drop_bond_mask": False}
    assert not opt.build_optimization_plan(configs, device=None).enabled(
        "drop_bond_mask"
    )


def test_published_decisions_drive_optimization_enabled():
    from types import SimpleNamespace

    assert not opt.optimization_enabled(SimpleNamespace(), "drop_bond_mask")
    on = SimpleNamespace(optimization_decisions={"drop_bond_mask": True})
    assert opt.optimization_enabled(on, "drop_bond_mask")
    with pytest.raises(KeyError):
        opt.optimization_enabled(on, "not_registered")


def test_feature_optimizations_drop_bond_mask_only_when_enabled():
    from types import SimpleNamespace

    from opendde.data.inference.infer_dataloader import apply_feature_optimizations

    def features():
        return {"bond_mask": torch.zeros(3, 3), "token_bonds": torch.zeros(2, 2)}

    off = features()
    apply_feature_optimizations(SimpleNamespace(), off)
    assert set(off) == {"bond_mask", "token_bonds"}
    on = features()
    apply_feature_optimizations(
        SimpleNamespace(optimization_decisions={"drop_bond_mask": True}), on
    )
    assert set(on) == {"token_bonds"}


def test_plan_decisions_follow_mode_and_switches(monkeypatch):
    monkeypatch.setattr(
        opt, "OPTIMIZATIONS", {"a": ("exact", "fast"), "b": ("exact",), "c": ("fast",)}
    )
    configs = build_inference_config(fill_required_with_null=True)
    configs.optimization_mode = "exact"
    configs.optimization = {"b": False}
    plan = opt.build_optimization_plan(configs, device=None)
    assert plan.enabled("a")
    assert (
        not plan.enabled("b") and "optimization.b=false" in plan.decisions["b"].reason
    )
    assert (
        not plan.enabled("c") and "not part of mode exact" in plan.decisions["c"].reason
    )
    configs.optimization_mode = "off"
    off = opt.build_optimization_plan(configs, device=None)
    assert not any(off.enabled(n) for n in "abc")


def test_config_defaults():
    configs = build_inference_config(fill_required_with_null=True)
    assert configs.optimization_mode == "off"
    assert configs.explicit_options == []
    assert configs.optimization == {name: True for name in opt.OPTIMIZATIONS}
    assert not hasattr(configs, "optimization_decisions")


# ------------------------------------------------------------- entry points


class _StopRunner(Exception):
    pass


@pytest.fixture
def captured_runner(monkeypatch):
    seen = {}

    def fake_runner(configs, foldcp_config=None):
        seen["configs"] = configs
        raise _StopRunner

    monkeypatch.setattr(batch_inference, "InferenceRunner", fake_runner)
    monkeypatch.delenv("LAYERNORM_TYPE", raising=False)
    monkeypatch.delenv("OPENDDE_TRUNK_CACHE", raising=False)
    return seen


def test_python_api_records_only_passed_options(captured_runner):
    with pytest.raises(_StopRunner):
        batch_inference.get_default_runner(device="cpu")
    configs = captured_runner["configs"]
    assert configs.optimization_mode == "off"
    assert configs.explicit_options == []
    assert configs.triangle_attention == "auto" and configs.dtype == "bf16"


def test_python_api_explicit_default_and_environment(captured_runner, monkeypatch):
    monkeypatch.setenv("LAYERNORM_TYPE", "torch")
    with pytest.raises(_StopRunner):
        batch_inference.get_default_runner(
            device="cpu", triatt_kernel="auto", optimization_mode="exact"
        )
    configs = captured_runner["configs"]
    assert configs.optimization_mode == "exact"
    assert configs.explicit_options == ["triatt_kernel", "env:LAYERNORM_TYPE"]


def test_python_api_rejects_a_mode_under_fold_cp_before_the_runner(captured_runner):
    with pytest.raises(ValueError, match="not supported with Fold-CP"):
        batch_inference.get_default_runner(
            device="cpu",
            foldcp_mode="distributed",
            foldcp_size_cp=2,
            optimization_mode="exact",
        )
    assert "configs" not in captured_runner


def test_runner_rejects_an_unavailable_mode_before_initialization(monkeypatch):
    def no_init(self):
        raise AssertionError("init_env must not run")

    monkeypatch.setattr(inference.InferenceRunner, "init_env", no_init)
    configs = build_inference_config(fill_required_with_null=True)
    configs.optimization_mode = "fast"
    with pytest.raises(ValueError, match="not available in this release"):
        inference.InferenceRunner(configs)


def _cli(monkeypatch, *args):
    seen = {}
    monkeypatch.setattr(
        batch_inference, "inference_jsons", lambda *a, **kw: seen.update(kw)
    )
    result = CliRunner().invoke(
        batch_inference.predict, ["-i", "in.json", "-o", "out", *args]
    )
    assert result.exit_code == 0, result.output
    return seen


def test_cli_records_explicit_options_from_parameter_sources(monkeypatch):
    seen = _cli(monkeypatch)
    assert seen["explicit_options"] == []
    assert seen["optimization_mode"] == "off"
    assert seen["triatt_kernel"] == "auto"
    seen = _cli(monkeypatch, "--triatt_kernel", "auto", "--dtype", "bf16")
    assert sorted(seen["explicit_options"]) == ["dtype", "triatt_kernel"]
    seen = _cli(monkeypatch, "--optimization_mode", "EXACT")
    assert seen["optimization_mode"] == "exact"


def test_cli_rejects_an_unknown_mode():
    result = CliRunner().invoke(
        batch_inference.predict,
        ["-i", "in.json", "-o", "out", "--optimization_mode", "turbo"],
    )
    assert result.exit_code != 0
