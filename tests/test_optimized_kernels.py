# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""Optimized kernels: dispatch, fallbacks, and (on a GPU with a measured row)
bitwise equality with the stock modules, or for triattn_k2b accuracy."""

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
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    assert not dispatch.available("trimul_triton", cuda)
    assert ("trimul_triton", "compute capability 9.0") in dispatch._LOGGED
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


# ------------------------------------------------------------ triattn_exact


def test_triattn_exact_row_ignores_triton(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "NVIDIA A100 80GB PCIe"
    )
    stack = {"torch": "2.7.1+cu128", "triton": None, "cuequivariance_ops": "0.10.0"}
    monkeypatch.setattr(dispatch, "_stack", lambda: stack)
    cuda = torch.device("cuda", 0)
    assert dispatch.available("triattn_exact", cuda)
    assert not dispatch.available("trimul_triton", cuda)  # a Triton kernel
    dispatch._row_reason.cache_clear()
    monkeypatch.setattr(
        dispatch, "_stack", lambda: {**stack, "cuequivariance_ops": "0.11.1"}
    )
    assert not dispatch.available("triattn_exact", cuda)
    dispatch._row_reason.cache_clear()
    monkeypatch.setattr(dispatch, "_stack", lambda: stack)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index: "NVIDIA A30")
    assert not dispatch.available("triattn_exact", cuda)
    assert ("triattn_exact", "device NVIDIA A30") in dispatch._LOGGED
    dispatch._row_reason.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (12, 0))
    assert dispatch.available("triattn_exact", cuda)  # any 12.0 device
    dispatch._row_reason.cache_clear()
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (9, 0))
    assert not dispatch.available("triattn_exact", cuda)


def test_triattn_exact_requires_the_measured_nvcc(monkeypatch):
    from opendde.model.optimized import triattn_exact as TE

    release = "Cuda compilation tools, release {0}, V{0}.85"
    monkeypatch.setattr(TE, "_nvcc_version", lambda: release.format("12.8"))
    assert TE._toolchain_reason() is None
    monkeypatch.setattr(TE, "_nvcc_version", lambda: release.format("12.6"))
    assert TE._toolchain_reason() == "nvcc 12.6"
    monkeypatch.setattr(TE, "_nvcc_version", lambda: None)
    assert TE._toolchain_reason() == "nvcc not found"


def test_triattn_exact_cache_must_be_private(monkeypatch, tmp_path):
    from opendde.model.optimized import triattn_exact as TE

    monkeypatch.setenv("TORCH_EXTENSIONS_DIR", str(tmp_path))
    path = TE._cache_dir()
    assert path == tmp_path / "opendde_triattn_exact"
    assert path.stat().st_mode & 0o777 == 0o700
    path.chmod(0o775)
    with pytest.raises(PermissionError):
        TE._cache_dir()


def test_triattn_exact_launch_plan():
    from opendde.model.optimized import triattn_exact as TE

    assert TE._plan_rows(300, 128) == (256, 44, 0)  # 44 tail rows: a BM64 launch
    assert TE._plan_rows(256, 128) == (256, 0, None)
    assert TE._plan_rows(250, 128) == (250, 0, None)  # under 257 rows: no split
    assert TE._plan_rows(1030, 128) == (1024, 6, 9)
    for n in (192, 300, 1001, 2047):
        assert TE._pick_cfg(1, n, 4, n) in TE._CFG


def test_triattn_exact_configs_fit_the_shared_memory():
    from opendde.model.optimized import triattn_exact as TE

    sm120 = TE._SMEM_OPTIN[(12, 0)]
    # fp32 bias: the two largest-tile configurations exceed sm_120's 99 KB
    assert TE._smem_bytes(4, 1000, 4, True) > sm120
    assert TE._smem_bytes(0, 1000, 4, True) > sm120
    assert TE._smem_bytes(4, 1000, 2, True) <= sm120  # bf16 bias fits
    for n in (192, 300, 517, 1001, 1218, 2047):
        for bias_bytes in (2, 4):
            for rows in (8, 256):
                cfg = TE._pick_cfg(1, rows, 12, n, bias_bytes, True, sm120)
                assert TE._smem_bytes(cfg, n, bias_bytes, True) <= sm120
                _, _, tail = TE._plan_rows(
                    n,
                    TE._CFG[cfg][1],
                    lambda c: TE._smem_bytes(c, n, bias_bytes, True) <= sm120,
                )
                assert (
                    tail is None or TE._smem_bytes(tail, n, bias_bytes, True) <= sm120
                )
    assert TE.nvcc_flags((8, 0))[-1] == "arch=compute_80,code=sm_80"
    assert TE.nvcc_flags((12, 0))[-1] == "arch=compute_120,code=sm_120"


def test_triattn_exact_skips_ineligible_calls():
    from opendde.model.optimized import triattn_exact as TE

    q = torch.zeros(4, 4, 256, 32)
    biases = [torch.zeros(4, 1, 1, 256), torch.zeros(1, 4, 256, 256)]
    with torch.no_grad():
        assert TE.serve(q, q, q, biases, 32) is None  # no CUDA autocast
    with torch.enable_grad():
        assert TE.serve(q, q, q, biases, 32) is None
    assert ("triattn_exact", "autograd enabled") in dispatch._LOGGED
    assert ("triattn_exact", "not under bf16 autocast") in dispatch._LOGGED


def _exact_row():
    if not _gpu_row("triattn_exact"):
        pytest.skip("no measured triattn_exact row on this machine")
    from opendde.model.optimized import triattn_exact as TE

    if TE._toolchain_reason() is not None:
        pytest.skip(TE._toolchain_reason())


def _stock_and_exact(module, call, k2b=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module.use_triton_kernel = module.use_exact_kernel = False
        stock = call()
        module.use_triton_kernel = k2b
        module.use_exact_kernel = True
        exact = call()
        again = call()
    return stock, exact, again


@pytest.mark.parametrize(
    "n, chunk",
    [
        (192, None),
        (245, None),
        (300, None),
        (517, None),
        (1001, None),
        (1218, 512),
        (2047, 256),
    ],
)
@pytest.mark.parametrize("mask_kind", ["ones", "pad"])
def test_triattn_exact_is_bitwise_the_stock_op(monkeypatch, n, chunk, mask_kind):
    _exact_row()
    from opendde.model.optimized import triattn_exact as TE
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    g = torch.Generator(device="cuda").manual_seed(n)
    z = torch.randn(n, n, 384, device="cuda", generator=g).to(torch.bfloat16)
    mask = _pad_mask(n) if mask_kind == "pad" else torch.ones(n, n, device="cuda")

    def call():
        return module(
            z, mask=mask, triangle_attention="cuequivariance", chunk_size=chunk
        )

    served = []
    serve = TE.serve
    monkeypatch.setattr(TE, "serve", lambda *a: served.append(serve(*a)) or served[-1])
    stock, exact, again = _stock_and_exact(module.mha, call)
    assert served and all(o is not None for o in served)  # the kernel ran
    assert exact.dtype == stock.dtype and exact.shape == stock.shape
    assert torch.equal(exact, stock), (n, chunk, mask_kind)
    assert torch.equal(exact, again)


def test_triattn_exact_serves_fast_where_k2b_has_no_row():
    _exact_row()
    if _gpu_row("triattn_k2b"):
        pytest.skip("triattn_k2b has a row on this machine")
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    z = torch.randn(300, 300, 384, device="cuda").to(torch.bfloat16)
    stock, exact, _ = _stock_and_exact(
        module.mha, lambda: module(z, triangle_attention="cuequivariance"), k2b=True
    )
    assert torch.equal(exact, stock)


def test_triattn_exact_small_calls_run_the_stock_kernel():
    _exact_row()
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    z = torch.randn(150, 150, 384, device="cuda").to(torch.bfloat16)
    stock, exact, _ = _stock_and_exact(
        module.mha, lambda: module(z, triangle_attention="cuequivariance")
    )
    assert torch.equal(exact, stock)
    assert ("triattn_exact", "150 tokens outside 192-2047") in dispatch._LOGGED


# ----------------------------------------------------------- triattn_native


def test_triattn_native_row_is_a100_only(monkeypatch):
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda index: (8, 0))
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda index: "NVIDIA A100-SXM4-80GB"
    )
    stack = {"torch": "2.7.1+cu128", "triton": None, "cuequivariance_ops": None}
    monkeypatch.setattr(dispatch, "_stack", lambda: stack)
    cuda = torch.device("cuda", 0)
    assert dispatch.available("triattn_native", cuda)
    for capability in ((12, 0), (9, 0)):
        dispatch._row_reason.cache_clear()
        monkeypatch.setattr(
            torch.cuda, "get_device_capability", lambda index, c=capability: c
        )
        assert not dispatch.available("triattn_native", cuda)


def test_triattn_native_skips_ineligible_calls():
    from opendde.model.optimized import triattn_native as TN

    q = torch.zeros(4, 4, 256, 32)
    biases = [torch.zeros(4, 1, 1, 256), torch.zeros(1, 4, 256, 256)]
    with torch.no_grad():
        assert TN.serve(q, q, q, biases, 32) is None
    with torch.enable_grad():
        assert TN.serve(q, q, q, biases, 32) is None
    assert ("triattn_native", "autograd enabled") in dispatch._LOGGED
    assert ("triattn_native", "not under bf16 autocast") in dispatch._LOGGED


def _native_row():
    if not _gpu_row("triattn_native"):
        pytest.skip("no measured triattn_native row on this machine")
    from opendde.model.optimized import triattn_exact as TE

    if TE._toolchain_reason() is not None:
        pytest.skip(TE._toolchain_reason())


def _stock_and_native(module, call, exact=False):
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        module.use_native_kernel = module.use_exact_kernel = False
        stock = call()
        module.use_native_kernel = True
        module.use_exact_kernel = exact
        fast = call()
        again = call()
    return stock, fast, again


@pytest.mark.parametrize(
    "n, chunk", [(192, None), (300, None), (517, None), (1001, None), (1218, 512)]
)
@pytest.mark.parametrize("mask_kind", ["ones", "pad"])
def test_triattn_native_matches_stock_accuracy_and_repeats(n, chunk, mask_kind):
    _native_row()
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    g = torch.Generator(device="cuda").manual_seed(n)
    z = torch.randn(n, n, 384, device="cuda", generator=g).to(torch.bfloat16)
    mask = _pad_mask(n) if mask_kind == "pad" else torch.ones(n, n, device="cuda")

    def call():
        return module(
            z, mask=mask, triangle_attention="cuequivariance", chunk_size=chunk
        )

    stock, fast, again = _stock_and_native(module.mha, call)
    with torch.no_grad():
        ref = module(z.float(), mask=mask, triangle_attention="torch", chunk_size=32)
    assert fast.dtype == stock.dtype and fast.shape == stock.shape
    assert torch.isfinite(fast).all()
    assert torch.equal(fast, again)
    assert not torch.equal(fast, stock)  # the kernel did run
    # Kernel benchmark on A100 (2026-10-18): 1.03-1.06x stock's relative error.
    assert _rel(fast, ref) <= 1.10 * _rel(stock, ref)
    assert (fast.float() - ref).abs().max() <= 1.5 * (stock.float() - ref).abs().max()


def test_triattn_native_small_calls_fall_through_to_exact():
    _native_row()
    from opendde.model.triangular.triangular import TriangleAttention

    module = _randomized(TriangleAttention(c_in=384, c_hidden=32, no_heads=12)).cuda()
    z = torch.randn(150, 150, 384, device="cuda").to(torch.bfloat16)
    stock, fast, _ = _stock_and_native(
        module.mha, lambda: module(z, triangle_attention="cuequivariance"), exact=True
    )
    assert torch.equal(fast, stock)  # 150 tokens: below both kernels' rows
    assert ("triattn_native", "150 tokens outside 192-2047") in dispatch._LOGGED
