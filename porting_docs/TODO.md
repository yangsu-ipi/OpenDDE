# Porting TODO

Open items that wait on something outside the current work. Each names what it needs before it can start.

## Test `k2b` triangle attention on a B200 or B300 GPU

- **Added:** 2026-10-17, after acceptance line `accept/sm120-fast-25267ec` was rejected (draw `sm120-fast-1`; see [the line ledger](qualification_line_ledger.md)).
- **Why:** in the optimization kit, `k2b` is the measured first choice of `fast` triangle attention mainly on Blackwell datacenter parts: 24 of 28 bf16, head-dim-32 cells on B200 (cc 10.0) and 23 of 25 on B300 (cc 10.3), in `common/opt_core/opt_core/kernels/triattn/TRIATTN_CELLS.json` (kit commit `f4f62fa`). On A100 and H100 the kit's `fast` mostly uses prebuilt CUDA kernels, and on the RTX PRO 6000 (cc 12.0) it has no measured column and serves `k2b` only by inheritance. This port's sm_120 `k2b` row was its own choice and failed its held-out draw.
- **Needs:** a cloud VM with a B200 or B300, provisioned by the user (as for the A100 VM; see `porting_docs/step0_results_sm80.md` for that setup). Register its store in `run_store_registry.json` before its first run.
- **Test:** the kernel benchmark and tile sweep used on sm_120 (scripts in `/mnt/compbio-data/yang.su/opendde-porting/kernel-bench-sm120` on gnode2: `bench_stock.py`, `bench_kit.py`, `k2b_sweep.py`, `k2b_rank.py`), at OpenDDE's shapes (12 heads of 32, bf16 pair, the model's chunking, 128-2,047 tokens, all-ones and padding masks), against stock cuEquivariance, with the error against an fp32 reference beside stock's. Development rows only.
- **Not covered by the test:** a `fast` claim on B200 or B300. Neither GPU is in the declared claims (sm_120 and sm_80) or has a qualification contract: a claim would need the user's declaration, its own step 0 baseline and calibration, a limit audit and freeze, and then a rehearsal and held-out draw on that GPU.
