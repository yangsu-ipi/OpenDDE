# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""trimul_triton / transition_triton: dispatch, fallbacks, and (on a GPU with
a measured row) bitwise equality with the stock modules."""

import pytest
import torch

from opendde.model.modules.primitives import Transition
from opendde.model.optimized import dispatch
from opendde.model.triangular.triangular import (
    TriangleMultiplicationIncoming,
    TriangleMultiplicationOutgoing,
)


@pytest.fixture(autouse=True)
def _fresh_dispatch(monkeypatch):
    monkeypatch.setattr(dispatch, "_DISABLED", {})
    monkeypatch.setattr(dispatch, "_LOGGED", set())
    dispatch._row_reason.cache_clear()
    yield
    dispatch._row_reason.cache_clear()


def _randomized(module):
    torch.manual_seed(0)
    with torch.no_grad():
        for p in module.parameters():  # zero inits would hide differences
            p.normal_(0, 0.05)
            if p.dim() == 1:
                p.add_(1.0)
    return module.eval()


# ------------------------------------------------------------- dispatch (CPU)


def test_flags_default_off():
    assert TriangleMultiplicationOutgoing(c_z=8, c_hidden=8).use_triton_kernel is False
    assert Transition(c_in=8, n=4).use_triton_kernel is False


def test_rows_match_only_the_measured_stack(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (12, 0))
    stack = {"torch": "2.7.1+cu128", "triton": "3.3.1", "cuequivariance_ops": "0.10.0"}
    monkeypatch.setattr(dispatch, "_stack", lambda: stack)
    cuda = torch.device("cuda", 0)
    assert dispatch.available("trimul_triton", cuda)
    assert dispatch.available("transition_triton", cuda)
    assert ("trimul_triton", "in use") in dispatch._LOGGED
    for key, value, reason in [
        ("triton", "3.4.0", "triton 3.4.0"),
        ("cuequivariance_ops", "0.11.0", "cuequivariance-ops 0.11.0"),
        ("torch", "2.8.0", "torch 2.8.0"),
    ]:
        dispatch._row_reason.cache_clear()
        monkeypatch.setattr(dispatch, "_stack", lambda: {**stack, key: value})
        assert not dispatch.available("trimul_triton", cuda)
        assert ("trimul_triton", reason) in dispatch._LOGGED
    dispatch._row_reason.cache_clear()
    monkeypatch.setattr(dispatch, "_stack", lambda: stack)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    assert not dispatch.available("trimul_triton", cuda)
    assert ("trimul_triton", "compute capability 8.0") in dispatch._LOGGED
    assert not dispatch.available("trimul_triton", torch.device("cpu"))


def test_guarded_disables_on_failure_but_raises_out_of_memory(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (12, 0))
    monkeypatch.setattr(
        dispatch,
        "_stack",
        lambda: {"torch": "2.7.1", "triton": "3.3.1", "cuequivariance_ops": "0.10.0"},
    )

    def broken():
        raise RuntimeError("triton build failed")

    assert dispatch.guarded("trimul_triton", broken) is None
    assert not dispatch.available("trimul_triton", torch.device("cuda", 0))
    assert dispatch.available("transition_triton", torch.device("cuda", 0))

    def oom():
        raise torch.OutOfMemoryError("out of memory")

    with pytest.raises(torch.OutOfMemoryError):
        dispatch.guarded("transition_triton", oom)
    assert dispatch.available("transition_triton", torch.device("cuda", 0))


def test_cpu_calls_with_the_flag_on_run_the_stock_code():
    transition = _randomized(Transition(c_in=8, n=4))
    trimul = _randomized(TriangleMultiplicationOutgoing(c_z=8, c_hidden=8))
    x = torch.randn(5, 5, 8)
    with torch.no_grad():
        expected = transition(x), trimul(x, triangle_multiplicative="cuequivariance")
        transition.use_triton_kernel = trimul.use_triton_kernel = True
        got = transition(x), trimul(x, triangle_multiplicative="cuequivariance")
    assert all(torch.equal(a, b) for a, b in zip(expected, got))


def test_transition_eligibility_is_limited_to_the_measured_cell():
    from opendde.model.optimized import transition as T

    pair = Transition(c_in=384, n=4)
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        assert not T.eligible(pair, torch.zeros(4, 4, 384))  # not under CUDA autocast
    narrow = Transition(c_in=384, n=2)  # the diffusion and refiner width
    assert not T.eligible(narrow, torch.zeros(1))
    with torch.enable_grad():
        assert not T.eligible(pair, torch.zeros(1))


# ------------------------------------------------------- bitwise (GPU row)


def _gpu_row(site):
    if not torch.cuda.is_available():
        return False
    return dispatch.available(site, torch.device("cuda", 0))


def _stock_and_triton(module, call):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module.use_triton_kernel = False
        stock = call()
        module.use_triton_kernel = True
        fast = call()
        again = call()
    return stock, fast, again


@pytest.mark.parametrize("n", [101, 256, 517, 1001, 1536, 2047])
@pytest.mark.parametrize("mask_kind", ["none", "ones", "pad"])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_trimul_is_bitwise_the_stock_op(n, mask_kind, outgoing, dtype):
    if not _gpu_row("trimul_triton"):
        pytest.skip("no measured trimul_triton row on this machine")
    cls = TriangleMultiplicationOutgoing if outgoing else TriangleMultiplicationIncoming
    module = _randomized(cls(c_z=384, c_hidden=384)).cuda()
    g = torch.Generator(device="cuda").manual_seed(n)
    z = torch.randn(n, n, 384, device="cuda", generator=g).to(dtype)
    mask = None
    if mask_kind != "none":
        mask = torch.ones(n, n, device="cuda")
        if mask_kind == "pad":
            keep = n - max(1, n // 15)
            mask[keep:, :] = 0
            mask[:, keep:] = 0
    for residual in (True, False):

        def call():
            return module(
                z,
                mask=mask,
                inplace_safe=residual,
                _add_with_inplace=residual,
                triangle_multiplicative="cuequivariance",
            )

        stock, fast, again = _stock_and_triton(module, call)
        assert fast.dtype == stock.dtype and fast.shape == stock.shape
        assert torch.equal(fast, stock), (n, mask_kind, outgoing, dtype, residual)
        assert torch.equal(fast, again)


@pytest.mark.parametrize("n", [256, 517, 1001])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_transition_is_bitwise_the_stock_op(n, dtype):
    if not _gpu_row("transition_triton"):
        pytest.skip("no measured transition_triton row on this machine")
    module = _randomized(Transition(c_in=384, n=4)).cuda()
    g = torch.Generator(device="cuda").manual_seed(n)
    x = torch.randn(n, n, 384, device="cuda", generator=g).to(dtype)
    stock, fast, again = _stock_and_triton(module, lambda: module(x))
    assert fast.dtype == stock.dtype and fast.shape == stock.shape
    assert torch.equal(fast, stock)
    assert torch.equal(fast, again)


def _pad_mask(n):
    mask = torch.ones(n, n, device="cuda")
    keep = n - max(1, n // 15)
    mask[keep:, :] = 0  # fully masked rows: stock averages v there
    mask[:, keep:] = 0
    return mask


def _rel(x, ref):
    return ((x.float() - ref.float()).norm() / ref.float().norm()).item()


@pytest.mark.parametrize(
    "n, chunk", [(128, None), (300, None), (517, None), (1001, None), (1218, 512)]
)
@pytest.mark.parametrize("mask_kind", ["ones", "pad"])
def test_triattn_matches_stock_accuracy_and_repeats(n, chunk, mask_kind):
    if not _gpu_row("triattn_k2b"):
        pytest.skip("no measured triattn_k2b row on this machine")
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    g = torch.Generator(device="cuda").manual_seed(n)
    z = torch.randn(n, n, 384, device="cuda", generator=g).to(torch.bfloat16)
    mask = _pad_mask(n) if mask_kind == "pad" else torch.ones(n, n, device="cuda")

    def call(kernel="cuequivariance"):
        return module(z, mask=mask, triangle_attention=kernel, chunk_size=chunk)

    stock, fast, again = _stock_and_triton(module.mha, call)
    with torch.no_grad():
        ref = module(z.float(), mask=mask, triangle_attention="torch", chunk_size=32)
    assert fast.dtype == stock.dtype and fast.shape == stock.shape
    assert torch.isfinite(fast).all()
    assert torch.equal(fast, again)
    assert not torch.equal(fast, stock)  # the kernel did run
    assert _rel(fast, ref) <= 1.05 * _rel(stock, ref)
    assert (fast.float() - ref).abs().max() <= 1.5 * (stock.float() - ref).abs().max()


def test_triattn_small_calls_run_the_stock_kernel():
    if not _gpu_row("triattn_k2b"):
        pytest.skip("no measured triattn_k2b row on this machine")
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    z = torch.randn(100, 100, 384, device="cuda").to(torch.bfloat16)
    stock, fast, _ = _stock_and_triton(
        module.mha, lambda: module(z, triangle_attention="cuequivariance")
    )
    assert torch.equal(fast, stock)
