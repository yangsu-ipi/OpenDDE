# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Aureka AI Research
"""structural_token_sync: role-pair projections with one host read, bit-identical."""

from unittest import mock

import pytest
import torch

from opendde.model.modules.structural_tokens import StructuralTokenExpander

C = 8


@pytest.fixture
def expander():
    torch.manual_seed(0)
    module = StructuralTokenExpander(
        c_s=C, c_z=C, c_s_inputs=C, pair_projection_mode="full"
    ).eval()
    with torch.no_grad():
        for proj in module.pair_block_proj:  # zero init would make every test pass
            proj.weight.normal_(0, 0.5)
    return module


def _role(n, roles, seed=1):
    g = torch.Generator().manual_seed(seed)
    choices = torch.tensor(roles)
    return choices[torch.randint(0, len(roles), (n,), generator=g)]


def _z(*shape, seed=2):
    return torch.randn(*shape, C, generator=torch.Generator().manual_seed(seed))


def _both(module, method, *args):
    out = {}
    for flag in (False, True):
        module.one_read_role_pairs = flag
        with torch.no_grad():
            out[flag] = getattr(module, method)(*args)
    return out[False], out[True]


@pytest.mark.parametrize("roles", [[0, 1, 2], [1], [0, 3]])
def test_full_projection_is_bit_identical(expander, roles):
    n = 7
    role = _role(n, [r for r in roles if r < expander.n_roles])
    stock, fast = _both(expander, "_pair_project_by_role_full", _z(n, n), role)
    assert torch.equal(stock, fast)


def test_chunk_and_tile_projections_are_bit_identical(expander):
    n = 9
    role = _role(n, [0, 1, 2])
    rows = torch.tensor([1, 4, 5])
    cols = torch.tensor([0, 2, 7, 8])
    stock, fast = _both(
        expander, "_pair_project_by_role_full_chunk", _z(len(rows), n), role, rows
    )
    assert torch.equal(stock, fast)
    stock, fast = _both(
        expander,
        "_pair_project_by_role_full_tile",
        _z(len(rows), len(cols)),
        role,
        rows,
        cols,
    )
    assert torch.equal(stock, fast)


def test_batch_dimensions_are_kept(expander):
    n = 5
    role = _role(n, [0, 2])
    stock, fast = _both(expander, "_pair_project_by_role_full", _z(2, n, n), role)
    assert stock.shape == fast.shape == (2, n, n, C)
    assert torch.equal(stock, fast)


def test_one_host_read_instead_of_one_per_role_pair(expander):
    n = 6
    role = _role(n, [0, 1])
    anys, reads = [], []
    real_any, real_tolist = torch.any, torch.Tensor.tolist

    def counted_any(*a, **k):
        anys.append(1)
        return real_any(*a, **k)

    def counted_tolist(self):
        reads.append(1)
        return real_tolist(self)

    with (
        mock.patch.object(torch, "any", counted_any),
        mock.patch.object(torch.Tensor, "tolist", counted_tolist),
    ):
        _both(expander, "_pair_project_by_role_full", _z(n, n), role)
    assert len(anys) == expander.n_roles**2  # the stock loop, flag off
    assert len(reads) == 1  # the one read, flag on


def test_default_is_off():
    module = StructuralTokenExpander(c_s=C, c_z=C, c_s_inputs=C)
    assert module.one_read_role_pairs is False
