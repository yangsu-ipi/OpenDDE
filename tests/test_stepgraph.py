# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
"""stepgraph: the capturable atom-to-token mean and the eligibility rules."""

import pytest
import torch

from opendde.model.optimized import stepgraph
from opendde.model.utils import aggregate_atom_to_token
from opendde.utils.scatter_utils import scatter_mean


@pytest.mark.parametrize("sorted_index", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_mean_is_bitwise_scatter_mean(sorted_index, dtype):
    g = torch.Generator().manual_seed(0)
    n_token = 37
    idx = torch.randint(0, n_token - 3, (500,), generator=g)  # 3 tokens without atoms
    if sorted_index:
        idx = idx.sort().values
    x = torch.randn(2, 500, 16, generator=g).to(dtype)
    stock = scatter_mean(x, idx, dim=-2, dim_size=n_token)
    table = stepgraph.segment_table(idx, n_token)
    assert torch.equal(stepgraph.segment_mean(x, table, n_token), stock)


def test_aggregation_uses_the_segment_mean_only_inside_a_graphed_call():
    g = torch.Generator().manual_seed(1)
    idx = torch.randint(0, 20, (200,), generator=g).sort().values
    x = torch.randn(3, 200, 8, generator=g)
    stock = aggregate_atom_to_token(x, idx, n_token=20)
    with stepgraph.segments() as tables:
        inside = aggregate_atom_to_token(x, idx, n_token=20)
        again = aggregate_atom_to_token(x, idx, n_token=20)
        assert len(tables) == 1  # built once per sampler call
    assert torch.equal(inside, stock) and torch.equal(again, stock)


def test_not_eligible_on_cpu_or_with_guidance():
    assert not stepgraph.eligible(torch.device("cpu"), False, True)
    with torch.enable_grad():
        assert not stepgraph.eligible(torch.device("cuda", 0), False, True)
    with torch.no_grad():
        assert not stepgraph.eligible(torch.device("cuda", 0), True, True)
        assert not stepgraph.eligible(torch.device("cuda", 0), False, False)


cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@pytest.fixture
def deterministic():
    previous = torch.are_deterministic_algorithms_enabled()
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(previous)


@cuda
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_segment_mean_is_bitwise_deterministic_scatter_mean_on_cuda(
    dtype, deterministic
):
    g = torch.Generator().manual_seed(2)
    n_token = 300
    idx = torch.randint(0, n_token - 5, (4000,), generator=g).sort().values.cuda()
    x = torch.randn(5, 4000, 128, generator=g).to(dtype).cuda()
    stock = scatter_mean(x, idx, dim=-2, dim_size=n_token)
    table = stepgraph.segment_table(idx, n_token)
    assert torch.equal(stepgraph.segment_mean(x, table, n_token), stock)


class _Denoiser(torch.nn.Module):
    """A small denoiser with the sampler's call signature."""

    def __init__(self, item: bool = False):
        super().__init__()
        torch.manual_seed(3)
        self.lin = torch.nn.Linear(16, 16).cuda()
        self.item = item

    def forward(self, x_noisy, t_hat_noise_level, scale):
        h = self.lin(x_noisy) * torch.sigmoid(t_hat_noise_level)[..., None, None]
        if self.item:
            h = h * float(h.abs().max().item())
        return h * scale + x_noisy


def _calls(n_calls=6):
    g = torch.Generator(device="cuda").manual_seed(4)
    for i in range(n_calls):
        x = torch.randn(5, 40, 16, device="cuda", generator=g)
        t = torch.tensor([10.0 / (i + 1)], device="cuda").expand(5)
        yield x, t


@cuda
@torch.no_grad()
def test_step_graph_replays_bitwise_the_eager_step():
    net = _Denoiser()
    step = stepgraph.StepGraph()
    for x, t in _calls():
        out = step(net, x, t_hat_noise_level=t, scale=0.5)
        assert torch.equal(out, net(x, t_hat_noise_level=t, scale=0.5))
    assert step.graph is not None and not step.disabled
    assert step.replays == 4  # calls 3-6


@cuda
@torch.no_grad()
def test_step_graph_falls_back_when_the_step_cannot_be_captured():
    net = _Denoiser(item=True)
    step = stepgraph.StepGraph()
    for x, t in _calls():
        out = step(net, x, t_hat_noise_level=t, scale=0.5)
        assert torch.equal(out, net(x, t_hat_noise_level=t, scale=0.5))
    assert step.disabled and step.replays == 0
