# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Anthropic, PBC
"""Optimized kernels selected by the optimization modes.

Each module serves one call site of the stock model with a kernel that was
measured at that site, and only on the GPUs and software stacks listed in
``dispatch.ROWS``. Every other call, and any call the kernel cannot serve,
runs the stock code. Triton is imported only when a kernel is first used.
"""
