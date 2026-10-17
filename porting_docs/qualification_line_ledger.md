# Qualification line ledger

One entry per acceptance line (qualification_protocol.md, E5: held-out log continuity across candidates). An entry is opened, and pushed, when the line's receipt commit is pushed as `accept/<gpu>-<receipt-id>`, and before its draw. On closure it records the draw ID if any, the status (`accepted`, `rejected` or `abandoned`), the verdict if any, the final branch-tip commit, the held-out `evidence_digest` and the hashes of `record.json`, `gate.json`, `verdict.json` and the archive manifest. At most one line is open at a time, across all GPUs. Every closed line is merged into `porting/inline-optimizations` before a later candidate's code commit.

## accept/sm120-fast-25267ec

| Field | Value |
| --- | --- |
| Status | **open** (2026-10-17) |
| GPU | sm_120 (RTX PRO 6000 Blackwell Max-Q, gnode2) |
| Branch | `accept/sm120-fast-25267ec` on `yangsu-ipi/OpenDDE` |
| Receipt ID | `fast-25267ec` |
| Candidate | `fast-k2b`: code commit `25267ec49a720355c91c6f08c452397b3fa4407e`, candidate arguments `--optimization_mode fast` |
| Contract tag | `qual-sm120-contract-6` (commit `e69506e055634dbb23ce828050a3f83391330395`) |
| Pushed receipt commit | `9340115f712e61d27b93ba1e129b30a680cd646b` (changes only `scripts/qualification/inputs/held_out_log.json`) |
| Acceptance checkout | `~/opendde-accept` on gnode2, at the receipt commit |
| Draw | none logged |

Rehearsal (2026-10-17, `dev-rehearsal` route through `qual_gate.py --accept` from the clean `qual-sm120-contract-6` checkout `~/opendde-c6`):

- Development evidence: `/data/yang.su/opendde-accept-sm120/rehearsal-1/runs` (D1-D15, development seeds 101 and 102; `off` traced, candidate traced, candidate repeat, tier 1 `off` capture and candidate replay; jobs 52502-52505; measurements and verdict job 52506). Preceded by the paired D1/D9 candidate smoke test in `/data/yang.su/opendde-accept-sm120/smoke` (job 52497).
- Verdict: exit 4, "rehearsal passed (not an acceptance)", 1,486 checks with none failed, Zn claim qualified; the gate passed and the candidate was unchanged after the verdict. `record.json` has `ok: false`, as for every rehearsal (`ok` means exit 0).
- Evidence digest (`evidence_sha256` in the receipt): `ced7fde847d5621a554acdceec3e9821b1a41008fc77bb64f52db094633b0a77`.
- Record hashes (SHA-256): `verdict.json` `778afa0929bece63a8b7869cd38440b529a7e77b762cc7193d321d8fafbbd1b4`; `gate.json` `9c43bb171424f833457fcf5e4448bb7522393bea851e04eaefd6f80e1ddf36ca`; `record.json` `7332c627d98f0514ec09b43b7bc7195e0e3078d96448d41fb0b55122637f0bba`; `receipt.json` `38c5ab1dee77f05a9cb33f79880d55a3933ecc767614ff826ed2215ad4a63d7b`; `evidence.json` `50b66e04dde4c95d966f6246e460faec455b86cdb1b4740fa9543dcd1956504e`; gate preflight with the receipt (11 of 11 checks passed) `gate-preflight-receipt.json` `4f9b31df88f22196c766a9874f498e2eb2cbf8ad7fc47ba4e19b73c0c83cd5e2`.
- Pre-receipt checks: GitHub rulesets rechecked ([record](qualification_records/ruleset_recheck_receipt_fast-25267ec_2026-10-17.json)); disk preflight of the held-out block passed ([operator record](qualification_records/line_sm120-fast-25267ec_preflight_receipt_2026-10-17.json), SHA-256 `a34426f8f873af9f23524de5e8bc037276ea6e854f8a28c93fe507ef1ee8629f`): every held-out row dominated by its comparator (H7 and H10 by identical entities), 58.1 GB of bounds plus a 20 GB margin against 269.8 GB available and 334.3 GB of soft-quota headroom, 40.9 GB already held for the line.
- The rehearsal outputs and record were made read-only after the receipt commit.
