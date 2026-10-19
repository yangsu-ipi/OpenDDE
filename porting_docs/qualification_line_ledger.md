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


## accept/sm80-fast80-87526cb

| Field | Value |
| --- | --- |
| Status | open (opened 2026-10-18) |
| GPU | sm_80 (A100 80GB PCIe, A100 VM `/ephemeral/porting`) |
| Branch | `accept/sm80-fast80-87526cb` on `yangsu-ipi/OpenDDE` |
| Receipt ID | `fast80-87526cb` |
| Candidate | `fast-native80`: code commit `87526cb52a76d8729a35ec1d24cca5063674aeff`, candidate arguments `--optimization_mode fast` |
| Contract tag | `qual-sm120-contract-7` (commit `37c02cc9fa13ce3c61119769de181747b5322475`) |
| Pushed receipt commit | `97a875d04c16242ae7bdb02068c099772bb5cb4e` (changes only `scripts/qualification/inputs/held_out_log.json`) |
| Final branch tip | not yet recorded |
| Acceptance checkout | `/ephemeral/porting/opendde-accept` on the A100 VM, on branch `accept/sm80-fast80-87526cb` at the receipt commit |
| Draw | none logged; at most 3 draws for the declared `fast` sm_80 claim |

Rehearsal (2026-10-18, `dev-rehearsal` route through `qual_gate.py --accept` from the clean `qual-sm120-contract-7` checkout `/ephemeral/porting/opendde-c7-gate`):

- Line check before the rehearsal: `scripts/porting/line_check.py` (SHA-256 `b79b91ef15dff0293a4f95dad95dee8c3f67ceb783ac6bd9d8364d882ca63ac3`) on the acceptance checkout at 87526cb passed 12 of 12 checks ([record](qualification_records/line_check_sm80-fast80-87526cb_rehearsal_2026-10-18.json)).
- Smoke test at the tag: the sm_80 D1 smoke test of `qual-sm120-contract-7` (job 337; `/ephemeral/porting/runs/c7-smoke-sm80/smoke.json`, SHA-256 `3e8a9a4aa1f22fa2aebf7000002b13e9df2bd20caf320e64d1044b722efea5b7`) is recorded in the receipt, as contract-7 requires. Preceded by the paired D1 candidate smoke test in `/ephemeral/porting/accept-sm80/smoke` (job 350).
- Development evidence: `/ephemeral/porting/accept-sm80/rehearsal-1/runs` (D1-D15, development seeds 101 and 102; `off` traced, candidate traced, candidate repeat, tier 1 `off` capture and candidate replay; jobs 351, 352, 353 and 355; measurements and verdict job 356). The first tier 1 job (354) stalled in the NVIDIA driver while creating a CUDA context for row D7 in mode `off` (`cuDevicePrimaryCtxRetain`, before any model code ran) and was cancelled after 3 h 9 min; its partial outputs were moved to `rehearsal-1/aborted-tier1-354` and are not part of the evidence. Job 355 reran the whole tier 1 block; every run exited 0 and the checkout stayed clean.
- Verdict: exit 4, "rehearsal passed (not an acceptance)", 1,487 checks with none failed, Zn claim qualified; the gate passed (9 of 9 checks) and the candidate was unchanged after the verdict. `record.json` has `ok: false`, as for every rehearsal (`ok` means exit 0).
- Evidence digest (`evidence_sha256` in the receipt): `f5563339450a9b21a3226cb79f0039f8d40925fc78f5289c723c1b42057c3f64`.
- Record hashes (SHA-256): `verdict.json` `8b7e875d8f069a366a8d7f4c95af05a9fe93ccb021d131abc136d804c248a29d`; `gate.json` `619a8ab804dc34c11c825e9718e1b81c085a4474cb5bf907851584b8cb93d13b`; `record.json` `22da4f171b91464631ea4dcfb335141f02fcdc769cef2aca24a8b2d4f7251d0e`; `receipt.json` `1b96361dd2a202fafe9e8be46dd3b9de723bbd8d431281af80cb8dbbc1f1f87d`; `evidence.json` `c5153aa977147b45d85a76040522cc2761ae9e6560c2ad34f5fe4cf62629f681`; gate preflight with the receipt after the push (11 of 11 checks passed) `gate-preflight-receipt.json` `e5e7c9bca54425b95036c2bcbcc919598ea20fea20ad470c908441e142b285fa`.
- Pre-receipt checks: GitHub rulesets rechecked, unchanged since 2026-10-14, with the line branch and `accept/sm80-example` both under deletion and non-fast-forward protection ([record](qualification_records/ruleset_recheck_receipt_fast80-87526cb_2026-10-18.json)); the line check rerun against the remote (12 of 12 checks passed, [record](qualification_records/line_check_sm80-fast80-87526cb_receipt_2026-10-18.json)); disk preflight of the held-out block passed ([operator record](qualification_records/line_sm80-fast80-87526cb_preflight_receipt_2026-10-18.json), SHA-256 `c2ce87feff562fd3651363df0d60764d0647ec59b6175d1017b0ab7a3a35f811`, from the sm_120 line's method with A100 paths): every held-out row dominated by its comparator (H7 and H10 by identical entities), 58.1 GB of bounds plus a 20 GB margin against 802.1 GB available on `/ephemeral`, which has no user quota; 45.3 GB already held for the line.
- The rehearsal outputs, the line's smoke test and the tag smoke record were made read-only after the receipt commit.
