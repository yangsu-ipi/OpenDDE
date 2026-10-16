# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""dit_hoist / zprep_hoist: step-invariant denoiser tensors made once, same samples."""

import pytest
import torch

from opendde.model import step_invariant
from opendde.model.generator import InferenceNoiseScheduler, sample_diffusion
from opendde.model.modules.diffusion import DiffusionModule
from opendde.model.opendde import update_input_feature_dict
from opendde.model.step_invariant import StepInvariantMemo, step_invariant_memo
from opendde.tfg.potentials import CLASS_REGISTRY, Potential, register

if "_StepInvariantGeoPotential" not in CLASS_REGISTRY:

    @register
    class _StepInvariantGeoPotential(Potential):
        def _eval(self, coords, feats, params, need_grad):
            energy = (coords**2).flatten(start_dim=-2).sum(dim=-1)
            return (energy, 2.0 * coords) if need_grad else energy


# ------------------------------------------------------------------ the memo


class _Owner(torch.nn.Module):
    pass


def _memo():
    return StepInvariantMemo(frozenset({"dit_hoist"}))


def _counted(fn):
    calls = []

    def producer():
        calls.append(1)
        return fn()

    return producer, calls


def test_same_inputs_reuse_the_recorded_result():
    memo, owner, x = _memo(), _Owner(), torch.randn(3, 4)
    producer, calls = _counted(lambda: x * 2)
    first = memo.get(owner, "site", (x, 1, None), producer)
    second = memo.get(owner, "site", (x, 1, None), producer)
    assert second is first and len(calls) == 1
    assert (memo.records, memo.hits) == (1, 1)


def test_a_view_of_the_same_storage_is_the_same_input():
    memo, owner, x = _memo(), _Owner(), torch.randn(3, 4)
    producer, calls = _counted(lambda: x.sum())
    memo.get(owner, "site", (x.unsqueeze(0),), producer)
    memo.get(owner, "site", (x.unsqueeze(0),), producer)  # a new view object
    assert len(calls) == 1


@pytest.mark.parametrize(
    "change",
    ["input_in_place", "result_in_place", "equal_copy", "setting", "owner", "site"],
)
def test_anything_else_recomputes(change):
    memo, owner, x = _memo(), _Owner(), torch.randn(3, 4)
    producer, calls = _counted(lambda: x * 2)
    result = memo.get(owner, "site", (x, 1), producer)
    inputs, who, site = (x, 1), owner, "site"
    if change == "input_in_place":
        x.add_(1.0)
    elif change == "result_in_place":
        result.add_(1.0)
    elif change == "equal_copy":
        inputs = (x.clone(), 1)
    elif change == "setting":
        inputs = (x, 2)
    elif change == "owner":
        who = _Owner()
    elif change == "site":
        site = "other"
    memo.get(who, site, inputs, producer)
    assert len(calls) == 2


def test_one_entry_per_site_and_tuple_results():
    memo, owner = _memo(), _Owner()
    a, b = torch.ones(2), torch.zeros(2)
    memo.get(owner, "site", (a,), lambda: (a + 1, a - 1))
    memo.get(owner, "site", (b,), lambda: (b + 1, b - 1))  # replaces, not adds
    assert len(memo._entries) == 1
    pair = memo.get(owner, "site", (b,), lambda: None)
    assert torch.equal(pair[0], b + 1) and memo.hits == 1


def test_inference_tensors_are_not_memoized():
    memo, owner = _memo(), _Owner()
    with torch.inference_mode():
        x = torch.ones(2)
    producer, calls = _counted(lambda: x + 1)
    memo.get(owner, "site", (x,), producer)
    memo.get(owner, "site", (x,), producer)
    assert len(calls) == 2 and memo.records == 0


def test_unsupported_inputs_are_rejected():
    with pytest.raises(TypeError):
        _memo().get(_Owner(), "site", ([1, 2],), lambda: torch.ones(1))


def test_scope_names_and_grad_mode():
    assert step_invariant.active("dit_hoist") is None
    with torch.no_grad():
        with step_invariant_memo(["zprep_hoist"]):
            assert step_invariant.active("dit_hoist") is None
            memo = step_invariant.active("zprep_hoist")
            assert memo is not None
            memo.get(_Owner(), "site", (torch.ones(1),), lambda: torch.ones(1))
            with torch.enable_grad():
                assert step_invariant.active("zprep_hoist") is None
        assert step_invariant.active("zprep_hoist") is None
        assert memo._entries == {}  # released when the sampler call ends
        with step_invariant_memo([]):
            assert step_invariant._MEMO.get() is None
    with pytest.raises(ValueError, match="not a step-invariant memo"):
        with step_invariant_memo(["drop_bond_mask"]):
            pass


# ------------------------------------------------------------- the sampler

N_TOKEN, ATOMS_PER_TOKEN, N_STEP = 6, 2, 5
N_TOK_BLOCKS, N_ATOM_BLOCKS = 2, 1


def _module():
    torch.manual_seed(0)
    module = DiffusionModule(
        c_atom=8,
        c_atompair=4,
        c_token=16,
        c_s=8,
        c_z=8,
        c_s_inputs=10,
        atom_encoder={"n_blocks": N_ATOM_BLOCKS, "n_heads": 2},
        transformer={"n_blocks": N_TOK_BLOCKS, "n_heads": 4},
        atom_decoder={"n_blocks": N_ATOM_BLOCKS, "n_heads": 2},
    ).eval()
    with torch.no_grad():
        for p in module.parameters():  # zero inits would make every test pass
            p.normal_(0, 0.3)
    return module


def _features(module, structural_bias):
    g = torch.Generator().manual_seed(1)
    n_atom = N_TOKEN * ATOMS_PER_TOKEN
    feats = {
        "atom_to_token_idx": torch.arange(N_TOKEN).repeat_interleave(ATOMS_PER_TOKEN),
        "ref_pos": torch.randn(n_atom, 3, generator=g),
        "ref_charge": torch.randn(n_atom, generator=g),
        "ref_mask": torch.ones(n_atom),
        "ref_element": torch.rand(n_atom, 128, generator=g),
        "ref_atom_name_chars": torch.rand(n_atom, 4 * 64, generator=g),
        "ref_space_uid": torch.arange(N_TOKEN).repeat_interleave(ATOMS_PER_TOKEN),
        "asym_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        "residue_index": torch.tensor([0, 1, 2, 0, 1, 2]),
        "entity_id": torch.tensor([0, 0, 0, 1, 1, 1]),
        "token_index": torch.arange(N_TOKEN),
        "sym_id": torch.zeros(N_TOKEN, dtype=torch.long),
    }
    if structural_bias:
        feats["structural_pair_attn_bias"] = torch.randn(N_TOKEN, N_TOKEN, generator=g)
    update_input_feature_dict(feats)
    module.diffusion_conditioning.relpe.generate_relp(feats)
    return feats


def _inputs(module, structural_bias):
    g = torch.Generator().manual_seed(2)
    feats = _features(module, structural_bias)
    s_inputs = torch.randn(N_TOKEN, 10, generator=g)
    s_trunk = torch.randn(N_TOKEN, 8, generator=g)
    z_trunk = torch.randn(N_TOKEN, N_TOKEN, 8, generator=g)
    with torch.no_grad():
        pair_z = module.diffusion_conditioning.prepare_cache(
            feats["relp"], z_trunk, False
        )
        p_lm, c_l = module.atom_attention_encoder.prepare_cache(
            ref_pos=feats["ref_pos"],
            ref_charge=feats["ref_charge"],
            ref_mask=feats["ref_mask"],
            ref_element=feats["ref_element"],
            ref_atom_name_chars=feats["ref_atom_name_chars"],
            atom_to_token_idx=feats["atom_to_token_idx"],
            d_lm=feats["d_lm"],
            v_lm=feats["v_lm"],
            pad_info=feats["pad_info"],
            r_l=True,
            z=pair_z,
            inplace_safe=False,
        )
    return dict(
        denoise_net=module,
        input_feature_dict=feats,
        s_inputs=s_inputs,
        s_trunk=s_trunk,
        z_trunk=None,
        pair_z=pair_z,
        p_lm=p_lm,
        c_l=c_l,
        noise_schedule=InferenceNoiseScheduler()(N_step=N_STEP),
        rollout_seed=7,  # the sampler's own generators, as the runner passes
    )


def _guidance():
    return {
        "enable": True,
        "rho": 0.5,  # denoiser-path guidance: grad-enabled denoiser calls
        "mu": 0.1,
        "mc": {"std": 0.0, "batch": 1},
        "steps": {
            "tfg_outer": 1,
            "tfg_inner": 1,
            "projection_outer": 0,
            "projection_inner": 0,
        },
        "terms": {"_StepInvariantGeoPotential": {"interval": 1, "weight": 0.01}},
    }


def _sample(names, *, structural_bias=True, **kwargs):
    module = _module()
    inputs = _inputs(module, structural_bias)
    cached = {k: inputs[k].clone() for k in ("pair_z", "p_lm", "c_l")}
    torch.manual_seed(3)
    with torch.no_grad(), step_invariant_memo(names):
        memo = step_invariant._MEMO.get()
        out = sample_diffusion(N_sample=2, **inputs, **kwargs)
    for name, value in cached.items():  # the sampler's cache is left as it was
        assert torch.equal(inputs[name], value), name
    return out, memo


def _sites(fusion, inplace_safe):
    atom = 6 * 2 * N_ATOM_BLOCKS  # 3 AdaLNs, 2 gates, 1 local bias per block
    dit = int(inplace_safe) + 1 + N_TOK_BLOCKS + 1 + atom
    return dit + int(fusion)


@pytest.mark.parametrize("fusion", [True, False])
@pytest.mark.parametrize("structural_bias", [True, False])
def test_samples_are_identical_and_every_site_is_reused(fusion, structural_bias):
    kwargs = dict(
        enable_efficient_fusion=fusion,
        inplace_safe=True,
        structural_bias=structural_bias,
    )
    stock, none = _sample([], **kwargs)
    hoisted, memo = _sample(["dit_hoist", "zprep_hoist"], **kwargs)
    assert none is None
    assert torch.equal(stock, hoisted)
    sites = _sites(fusion, inplace_safe=True)
    assert memo.records == sites
    assert memo.hits == sites * (N_STEP - 1)


def test_sample_chunks_reuse_the_same_entries():
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=True)
    stock, _ = _sample([], diffusion_chunk_size=1, **kwargs)
    hoisted, memo = _sample(
        ["dit_hoist", "zprep_hoist"], diffusion_chunk_size=1, **kwargs
    )
    assert torch.equal(stock, hoisted)
    sites = _sites(True, inplace_safe=True)
    assert memo.records == sites and memo.hits == sites * (2 * N_STEP - 1)


def test_without_inplace_safety_the_memo_follows_the_changing_cache():
    # Stock writes the atom context into the cached p_lm in place when
    # inplace_safe is false, so it changes every step: the memo sees the new
    # version and recomputes, as stock does.
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=False)
    module = _module()
    inputs = _inputs(module, True)
    results = []
    for names in ([], ["dit_hoist", "zprep_hoist"]):
        fresh = {k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)}
        torch.manual_seed(3)
        with torch.no_grad(), step_invariant_memo(names):
            results.append(
                sample_diffusion(N_sample=2, **{**inputs, **fresh}, **kwargs)
            )
    assert torch.equal(results[0], results[1])


def test_dit_hoist_alone_is_identical():
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=True)
    stock, _ = _sample([], **kwargs)
    hoisted, memo = _sample(["dit_hoist"], **kwargs)
    assert torch.equal(stock, hoisted)
    # Without zprep_hoist the token pair input is a new tensor every step, so
    # the token pair biases computed from it are recomputed every step.
    sites = _sites(False, inplace_safe=True)
    assert memo.records == sites + N_TOK_BLOCKS * (N_STEP - 1)
    assert memo.hits == (sites - N_TOK_BLOCKS) * (N_STEP - 1)


def test_zprep_hoist_alone_follows_a_fresh_pair_clone():
    # Without dit_hoist the conditioning clones the pair every step, so the
    # token pair input has a new source each time and is recomputed.
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=True)
    stock, _ = _sample([], **kwargs)
    hoisted, memo = _sample(["zprep_hoist"], **kwargs)
    assert torch.equal(stock, hoisted)
    assert memo.records == N_STEP and memo.hits == 0


def test_guided_sampling_is_identical_and_grad_calls_run_stock():
    kwargs = dict(enable_efficient_fusion=True, inplace_safe=True)
    stock, _ = _sample([], guidance_configs=_guidance(), **kwargs)
    hoisted, memo = _sample(
        ["dit_hoist", "zprep_hoist"], guidance_configs=_guidance(), **kwargs
    )
    assert torch.equal(stock, hoisted)
    assert memo.hits > 0
