# Qualification line ledger

One entry per acceptance line (qualification_protocol.md, E5: held-out log continuity across candidates). An entry is opened, and pushed, when the line's receipt commit is pushed as `accept/<gpu>-<receipt-id>`, and before its draw. On closure it records the draw ID if any, the status (`accepted`, `rejected` or `abandoned`), the verdict if any, the final branch-tip commit, the held-out `evidence_digest` and the hashes of `record.json`, `gate.json`, `verdict.json` and the archive manifest. At most one line is open at a time, across all GPUs. Every closed line is merged into `porting/inline-optimizations` before a later candidate's code commit.

## accept/sm120-fast-25267ec

| Field | Value |
| --- | --- |
| Status | **rejected** (closed 2026-10-17) |
| GPU | sm_120 (RTX PRO 6000 Blackwell Max-Q, gnode2) |
| Branch | `accept/sm120-fast-25267ec` on `yangsu-ipi/OpenDDE` |
| Receipt ID | `fast-25267ec` |
| Candidate | `fast-k2b`: code commit `25267ec49a720355c91c6f08c452397b3fa4407e`, candidate arguments `--optimization_mode fast` |
| Contract tag | `qual-sm120-contract-6` (commit `e69506e055634dbb23ce828050a3f83391330395`) |
| Pushed receipt commit | `9340115f712e61d27b93ba1e129b30a680cd646b` (changes only `scripts/qualification/inputs/held_out_log.json`) |
| Final branch tip | `9def88b9c6d0ed8d1abecee200d3c312d4b9f401` (pushed; draw commit `0879771`, then H3's extra Zn seed `9def88b`) |
| Acceptance checkout | `~/opendde-accept` on gnode2, at the final branch tip |
| Draw | `sm120-fast-1`, draw 1 of at most 3 for the declared `fast` sm_120 claim; seeds H1 201-202 through H10 219-220 (lowest unused seeds in row order), H3 extra Zn seed 221 |
| Verdict | **fail**, `accept.py` exit 1 through `qual_gate.py --accept --receipt fast-25267ec` (gate passed, candidate unchanged); 2,538 checks, 4 failed; Zn claim `fail` |

Rehearsal (2026-10-17, `dev-rehearsal` route through `qual_gate.py --accept` from the clean `qual-sm120-contract-6` checkout `~/opendde-c6`):

- Development evidence: `/data/yang.su/opendde-accept-sm120/rehearsal-1/runs` (D1-D15, development seeds 101 and 102; `off` traced, candidate traced, candidate repeat, tier 1 `off` capture and candidate replay; jobs 52502-52505; measurements and verdict job 52506). Preceded by the paired D1/D9 candidate smoke test in `/data/yang.su/opendde-accept-sm120/smoke` (job 52497).
- Verdict: exit 4, "rehearsal passed (not an acceptance)", 1,486 checks with none failed, Zn claim qualified; the gate passed and the candidate was unchanged after the verdict. `record.json` has `ok: false`, as for every rehearsal (`ok` means exit 0).
- Evidence digest (`evidence_sha256` in the receipt): `ced7fde847d5621a554acdceec3e9821b1a41008fc77bb64f52db094633b0a77`.
- Record hashes (SHA-256): `verdict.json` `778afa0929bece63a8b7869cd38440b529a7e77b762cc7193d321d8fafbbd1b4`; `gate.json` `9c43bb171424f833457fcf5e4448bb7522393bea851e04eaefd6f80e1ddf36ca`; `record.json` `7332c627d98f0514ec09b43b7bc7195e0e3078d96448d41fb0b55122637f0bba`; `receipt.json` `38c5ab1dee77f05a9cb33f79880d55a3933ecc767614ff826ed2215ad4a63d7b`; `evidence.json` `50b66e04dde4c95d966f6246e460faec455b86cdb1b4740fa9543dcd1956504e`; gate preflight with the receipt (11 of 11 checks passed) `gate-preflight-receipt.json` `4f9b31df88f22196c766a9874f498e2eb2cbf8ad7fc47ba4e19b73c0c83cd5e2`.
- Pre-receipt checks: GitHub rulesets rechecked ([record](qualification_records/ruleset_recheck_receipt_fast-25267ec_2026-10-17.json)); disk preflight of the held-out block passed ([operator record](qualification_records/line_sm120-fast-25267ec_preflight_receipt_2026-10-17.json), SHA-256 `a34426f8f873af9f23524de5e8bc037276ea6e854f8a28c93fe507ef1ee8629f`): every held-out row dominated by its comparator (H7 and H10 by identical entities), 58.1 GB of bounds plus a 20 GB margin against 269.8 GB available and 334.3 GB of soft-quota headroom, 40.9 GB already held for the line.
- The rehearsal outputs and record were made read-only after the receipt commit.

Held-out draw and closure (2026-10-17):

- Pre-draw: logged draws per claim counted (0 before this draw); audit of all registered stores (1,697 runs, no stop, no deviation; the retired A100 VM store is the only gap) and a supplementary scan of unregistered OpenDDE-related directories and the porting checkouts (no runs); GitHub rulesets rechecked before each held-out batch and the final verdict.
- Runs, from `~/opendde-accept` at the logged commits with the frozen asset copy, serially behind the gate preflight with the receipt, the asset and input-digest preflights and a per-run headroom guard (operator records in `draw-1/operator`): `off` and the candidate traced (job 52799), H3's extra Zn seed (job 52827; 6 eligible ions after the primary seeds, 12 after seed 221), the candidate repeat and both tier 1 runs (job 52832). All 50 runs exited 0; the checkout stayed clean.
- Verdict (job 52903): the development set passed as in the rehearsal; every held-out tier 1, trace, repeat, provenance and seed check passed. Failed tier 2 gates: H3 `zn_same_site` 0.667 (limit 0.85); H6 (`9xqn_contact`) `outcome_count` 7 (9), `lddt_polymer` and `lddt_protein` 0.946 (0.95), `pair_interface_fnat` 0.694 (0.70), `guidance_contacts` 3 (5), `ensemble_ref_fnat_median` 0.086 (0.05), `ensemble_ref_fnonnat_median` 0.024 (0.02), `ensemble_ref_fnat_min_pair_median` 0.214 (0.05); H7 (`9lh2_contact`) `outcome_count` 1 (2), `ensemble_ref_lddt_polymer_lowest` and `ensemble_ref_lddt_protein_lowest` 0.052 (0.02); H10 (`7wux`) `p95_error` 1.85 (1.0). H1, H2, H4, H5, H8 and H9 passed.
- The candidate differs from `off` only through `triattn_k2b`: every `exact` change is byte-identical on D1-D15. The failures were not diagnosed on held-out data, so that no later candidate is chosen from them.
- Held-out evidence digest (`accept.evidence_digest` of the held-out and Zn sets): `154f710b3c7d82b39b6161c46b2d4adeab73fa4c8a9b8d1173947273997f03cc`; of all sets: `f689ab8e227ca1565e0b70eecd3c74f72c8258e0c1351e8d722b04fc60ae4b08`; development: `ced7fde847d5621a554acdceec3e9821b1a41008fc77bb64f52db094633b0a77`, equal to the receipt's.
- Record hashes (SHA-256, `draw-1/record`): `record.json` `baf9b37ed03e941da762ff827b4b091f807d34be2b3e5330437060b7097ae5c4`; `gate.json` `8b3ce51f365dba04878c3c537e2940d0f86c28d17d3de7a41857d06650155009`; `verdict.json` `42809ee2e6dca5c12bf990043bc8a8f44f47605f5daf48b02080e07a51ee9f10`; `evidence.json` `bee73c9823c321e0cf0454ceecdadf80e39c00bd528665f7f82c80036062ea76`.
- Archive: the whole line store (rehearsal, smoke test, draw and operator records; 3,349 files, 75.1 GB, 2 symlinks linking H3's seed-221 directories into the held-out runs, kept as links) copied to `/mnt/compbio-data/yang.su/opendde-qualification-archive/accept-sm120-fast-25267ec` and verified from gnode2 against the source manifest (every file matched, none missing or extra, symlinks equal; job 53258). Archive manifest `manifest-sha256.txt` SHA-256 `86c96f8143cc68f86938fb1649a7f82e439938976a55385dacc7256cc9a68aaa` (symlink manifest `8d9328f5b0f0869738ac07b63cfb0284fc14d6a34f9c04465050daf903191293`). The copy and the originals are read-only by file permission; this is not a NAS-protected immutable copy, so the originals on `/data` are kept until the storage-admin procedure protects and independently verifies the archive.
- Merged into `porting/inline-optimizations` after this entry.
- Follow-up: [TODO](TODO.md), testing `k2b` on a B200 or B300 GPU.

