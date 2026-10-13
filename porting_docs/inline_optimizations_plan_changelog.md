# Inline optimization plan changelog

Changes to `inline_optimizations_plan.md`. Newest first.

## Revision 86 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Narrow the one-time capacity rerun to a documented filesystem or quota failure before any prediction artifact was written. Preserve the failed tree, scheduler evidence and any run record; a forced kill can leave no run record. Keep the candidate and all declared arguments fixed, and abandon the line if the failure or artifact state is ambiguous. Reason: a broad "capacity" exception could permit rerunning a model failure or discarding an unfavorable partial result, while missing run metadata cannot prove that no prediction was produced.
- Require a tested mapping-aware verifier before deleting archived held-out outputs. It must reproduce the ledger's held-out `evidence_digest` from unchanged original set declarations and relocated run records, traces and measurements, using the recorded path map. Reason: `evidence_digest` includes original paths in each declaration and dereferences those paths, so a simple recomputation after relocation would fail even for an intact archive.
- Scope the D5 cross-machine digest to the complete model-consumed template feature set, define its canonical serialization, and check repeatability on gnode2 before comparing the VM. A differing Kalign binary cannot be described as an identical asset; differing features make D5 a different preprocessing workload. Reason: an equal digest of an unspecified feature subset cannot prove broad CPU determinism or equivalence of every preprocessing step.

## Revision 85 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Replace "do not reuse seeds" and "if the contract permits" in the low-space rule with an explicit capacity-rerun rule: a run that failed or was cancelled before writing predictions may be rerun once with its declared seeds in a fresh output directory, recorded as a capacity rerun with the failed run's record kept; a completed run is never rerun, and the evidence names the first completed run. Reason: contract-5 already permits this (a failed run cannot be evidence; per-run checks require a fresh directory, exit 0 and logged seeds; seed uniqueness is per log entry, not per run), while the literal no-reuse reading forced abandonment, spending a draw and, for the first draw, the only fresh-input test.
- Let active jobs finish only if headroom covers their estimated remaining output from the rehearsal's per-row high-water marks. Reason: "capacity remains adequate" had no threshold.
- At line closure, push the held-out `evidence_digest` and the hashes of `record.json`, `gate.json`, `verdict.json` and the archive manifest in the ledger entry. Reason: the receipt binds only development evidence, and held-out run records sit beside the artifacts they hash, so after local removal an auditor had no off-machine reference for held-out bytes; `evidence_digest` already accepts any evidence sets.
- Recompute `record.json`'s `evidence_sha256` with the gate's own function. Reason: `qual_gate.py` hashes decoded text with normalised newlines, so `sha256sum` of the raw bytes matches only by accident.
- Make the sm_80 D5 preprocessing comparison concrete: compare the contract-6 D5 CPU preprocessing test's canonical template-feature digest between gnode2 and the VM. State that D5 is the only template row, so a Kalign difference cannot affect a held-out row. Reason: sm_80 predictions differ numerically anyway and cannot reveal a preprocessing difference.

## Revision 84 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Separate the archive prechecks by what each digest binds: `record.json` hashes the evidence JSON text, the receipt holds a canonical digest of development evidence, and run records hash prediction artifacts and tier 1 captures. Verify each against its own source; give other files a new archive-time manifest baseline. Reason: comparing unlike digests, or treating an archive-time manifest as an earlier verdict binding, would misstate which bytes were accepted.
- Require an operator-writable destination with a storage-admin durable-write and independent-verification procedure before the first receipt. Keep local originals if the only destination is the current `soft` NFS mount; a same-client read after `sync` or direct I/O is insufficient proof of server-side persistence. Reason: verification must justify removing the local evidence, not just show that the producing client can read cached or server-returned bytes.
- When headroom falls below the remaining estimate plus margin, stop launching jobs, preserve the open-line evidence, assess active jobs, and never reuse declared draw seeds. Resume only after a fresh preflight under the same contract, or close the line as abandoned. Reason: a blanket instruction to continue into likely disk exhaustion risks invalid evidence and the shared filesystem.
- Exercise D5 during sm_80 stock calibration whether the copied Kalign runs or a VM-specific binary is needed. Reason: byte-identical executables do not by themselves establish preprocessing behavior on the VM.

## Revision 83 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Measure quota headroom to the soft limit and compare in bytes. Reason: on gnode2 the `/data` quota is 570 GiB soft and 600 GiB hard, with 350 GiB used; past the soft limit a grace period starts and writes then fail, so revision 82's 250 GiB figure (measured to the hard limit) overstated usable headroom by 30 GiB.
- Check the whole remaining line budget before the development-evidence run, not only at the receipt commit. Reason: by the receipt the 36 GB rehearsal already exists, so that check needs only 56 GB, and a line could spend a full rehearsal before finding no room for its draw.
- Require a writable archive destination before the first receipt and add it to the receipt preflight. Reason: `/mnt/nas-backups` is root-owned with mode 711 and not writable by the operator, and no per-user directory was found; at 92 GB per line the current headroom allows about two lines before archiving is needed.
- Verify NAS copies with cache-bypassing reads after `sync`, replacing the representative read-back; confirm the read-only mode with `stat`; check local files against the run records' artifact digests and the recorded evidence digest before writing the manifest; and record the local-to-archive path mapping. Reason: the NAS is a `soft` NFSv3 mount, a hash read straight after copying can come from the client page cache, and a manifest taken only at archive time would preserve files changed after the verdict.
- Replace revision 82's GPU-scoped fallback for differing portable inputs. The input JSONs, MSAs, references and D5's template mmCIFs are already checked by `accept.py`'s `check_digests` against the shared `digests.json` and `external_digests.json`; run those checks and the runtime-asset comparison as a preflight before sm_80 stock calibration, treat a mismatch as a copy error, and stop for a separate contract decision if bytes cannot be matched. Reason: under the shared digest files, sm_80 acceptance on different inputs fails, so the fallback had no implementable path without a further amendment.
- Copy gnode2's frozen Kalign to the A100 VM and require digest equality, falling back to a VM-specific Kalign, D5 stock calibration and a stated difference only if it cannot run there. Reason: Kalign links only libc, so one binary makes D5's workload identical on both GPUs at little cost.
- Say what to do when free space falls during a batch (free the operator's own files outside open lines; do not abort, since the seeds are spent), and update the paragraph labels and the protocol's remaining steps. Splitting the long seed-budget bullet into sub-bullets is left for a later revision.

## Revision 82 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Replace the approximate 70 GB line budget with an initial 92 GB minimum: 36 GB of observed rehearsal output, 36 GB provisionally for held-out output and a 20 GB margin. Before the receipt and each held-out batch, record filesystem and quota headroom and stop if either cannot cover the estimated remainder plus the greater of 20 GB or 25%. Reason: the former estimate was below the sum of its own components, and an undefined margin cannot support a repeatable stop decision on a shared filesystem.
- Archive closed lines and unneeded step 0 outputs only after checking the NAS mount and capacity, verifying destination files against a path/size/SHA-256 manifest and testing a read-back. Keep the archive location, manifest digest and qualification records locally. Reason: a pathname on the NAS does not establish that evidence can be audited or restored.
- For a combined same-workload cross-GPU claim, require equal digests for portable source inputs, referenced data and template metadata as well as the checkpoint and CCD files. Freeze Kalign separately and exercise D5 during sm_80 stock calibration; a difference in portable inputs requires a separately approved GPU-scoped claim and affected-row recalibration. Reason: the revision 81 rule could describe two different evaluation workloads as one cross-GPU result.
- Update the protocol's revision label and remaining-step wording to match these rules.

## Revision 81 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Add a disk-headroom rule: budget about 70 GB per acceptance line (the stand-in rehearsal's evidence was 36 GB, and a held-out batch should be similar), check free space on the filesystem and the quota before the receipt and each held-out batch, archive closed lines to `/mnt/nas-backups`, and archive rather than delete unneeded step 0 artifacts. Reason: on 2026-10-13 the quota had about 250 GiB and the shared `/data` filesystem 215 GB free (95% full), and a held-out run that fails for lack of disk is a harness failure that still spends its seeds.
- Require plain `cp` or `rsync` without link options for the private asset copy. Reason: `/data/shared` and `/data/yang.su` are one filesystem, so a hard link is possible; revision 80's device-and-inode check already rejects one, and this states how to avoid creating it.
- State the cross-GPU asset-identity rule: the sm_80 copy's checkpoint and both CCD files must match the sm_120 freeze-time digests, with Kalign recorded per machine. Reason: independent freezes could otherwise qualify two different checkpoints under one claim.

## Revision 80 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Make the sm_120 qualification source unambiguous: contract-6 uses a frozen private copy; ordinary porting can still use the shared store. Copy files independently, reject links back to the shared store, freeze resolved paths and digests, and preflight them before a held-out batch. Reason: the older store paragraph contradicted revision 79, and a link or stale asset could defeat the copy or waste held-out seeds.
- Check Kalign's resolved effective-configuration path and digest against the frozen executable, and reject a missing or nonexecutable requested binary before launch. Add a fallback test. Keep the mid-run replacement limit explicit. Reason: `resolve_kalign_binary` can choose `kalign` from `PATH` when the requested binary is absent; hashing the environment variable's path would not establish what was selected at configuration time.
- Scope the private copy to sm_120 and require an independent, later sm_80 freeze on the A100 VM. Describe the step 0 drift comparison as output equality, including the template-enabled D5 row, without claiming identity or individual use of each asset. Reason: the gnode2 path is not available on the A100 VM, and equal predictions cannot identify which asset bytes step 0 read.
- Update the protocol's freeze-record prose to distinguish contract-5's shared assets from the planned contract-6 copy. Reason: it still described the shared CCD and Kalign locations as if they remained the qualification recipe.

## Revision 79 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Run every qualification run from a private, read-only copy of the runtime assets under `/data/yang.su/` (selected checkpoint, `common/` and the kalign binary, about 3.3 GB), with `OPENDDE_ROOT_DIR` and `OPENDDE_QUAL_KALIGN` pointed at it; its digests are the freeze-time reference, and creating it is a contract-6 release step. Ordinary porting runs may keep using the shared directory. Reason: `/data/shared/models/opendde` is owned and writable by the administrators, the two variables are unchecked per-host settings, and an asset refresh while a line is open would invalidate the evidence set and burn the draw's seeds.
- Add the kalign binary to the template-row assets. Reason: D5's template alignment runs it, and neither its path nor its bytes were checked.
- Link the freeze-time assets to step 0 through the byte-identical drift check (D1 at contract-6, all 15 development rows at the first rehearsal, D5 covering template metadata and kalign), leaving file identity explicitly unclaimed. Reason: the drift check is the retained evidence revision 78 asked for, at the level of outputs.
- Describe the template-cache absence check as defence in depth. Reason: the cache lives under `{out}`, which `accept.py` already requires not to exist before the run.

## Revision 78 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Freeze runtime-asset digests at contract-6, and call them freeze-time references rather than step 0 values. The committed step 0 baseline records prediction outputs, not the assets used to produce them. Historical identity needs independent retained evidence. Reason: a later hash cannot prove which bytes an earlier run read.
- Record each runtime asset's path from the effective configuration and require a frozen digest for that selected path, including the checkpoint, both CCD files and the template metadata files when used. Refuse an unlisted override, and hash the checkpoint in every run rather than caching its digest by path, size and modification time. Reason: configuration can select files outside `$OPENDDE_ROOT_DIR`, and those cache keys do not identify contents.
- Treat matching before/after hashes of referenced data as stability checks, not direct evidence of read bytes. Keep the low-complexity bracket approach within the accident-only trust boundary, with an explicit no-mid-run-replacement assumption and a fresh-template-cache check for D5; fail if the cache is already populated. Reason: a file can change between hashes, and the template reader can load a parsed cache instead of an mmCIF.
- Add tests for an overridden asset path, a prepopulated D5 template cache and a changed checkpoint with the same path, size and modification time. Reason: these cases distinguish the amended provenance rule from revision 77's weaker checks.

## Revision 77 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Record why the expected effective digest matters and test it: the legacy-MSA conversion adds MSA paths only when `os.path.exists` succeeds on repository-relative paths resolved against the working directory, so a run from elsewhere silently omits the MSAs; H2 has no baseline, so `off` and the candidate would both run without MSAs and pass the paired comparison. Add a wrong-working-directory failure test. Reason: the check closes a held-out-only failure that the drift check cannot see.
- Digest the `$OPENDDE_ROOT_DIR` assets each run reads (checkpoint, CCD data and, for template rows, `release_date_cache.json` and `obsolete_to_successor.json`), record them in the run record, require one value per evidence set (equal on sm_120 to step 0 values recorded in contract-6) and test a changed asset. Reason: these are the largest unfrozen inputs; D5's template prefilter depends on the release-date cache, and an asset change between rehearsal and held-out runs would otherwise go unnoticed. Revision 76's fail-closed reader rule would otherwise have failed every run.
- Replace data-reader hooks with bracket digests (before and after the run, equal and matching the frozen digests) for referenced MSA, RNA-MSA and template files, and drop the template-cache hook because D5's verified overrides already fix a fresh `{out}/template_cache` and `fetch_remote=false`. Reason: within E5's accident-only trust boundary this shows the bytes read, and it keeps contract-6 to one new hook target, the JSON parser read.

## Revision 76 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Permit deterministic legacy-MSA format conversion while refusing directory inputs and MSA, template or RNA-MSA searches. D2, D4, D13 and H2 contain `proteinChain.msa`; the runner writes a derived JSON even though their MSA data are precomputed. Contract-6 must record the source-to-effective mapping and check the expected effective digest instead of requiring source and read-point digests to match. Reason: revision 75's equality rule would reject four frozen rows; changing their inputs would disturb the freeze and baseline.
- Compare individual protected protein and nucleic-acid sequences as well as whole-complex signatures, and flag any guidance restraint on a 9lh2-containing input. Apply the reserved-seed stop to changed 7wux and 9lh2 siblings too. Edited, truncated or homologous sequences require manual classification. Reason: an exact whole-complex signature or source-byte match misses a protected chain moved into a different assembly, while exact matching cannot establish that every variant is absent.
- Capture referenced MSA, RNA-MSA and template bytes at their data-reader reads, including D5's frozen template-mmCIF files; reject an unexpected template-cache read and fail closed if a reader is not covered. Reason: a pre-run digest does not prove which bytes the model consumed.
- Add D4 legacy-conversion, synthetic H2-format and D5 template provenance tests, plus a change between preflight and data read. Keep the D1 stock GPU smoke and report its coverage limit. Reason: D1 has neither legacy conversion nor templates, so it cannot validate those contract-6 paths.

## Revision 75 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Define protection by what is fresh: complexes never in development (9sat, 9xqn, 7pzb, H1, H8, H9) are protected in any form, as are development complexes carrying a held-out restraint (the 9lh2 contact and pocket inputs). D8's restraint-free 9lh2 input and D4's 7wux input are free at seeds below 200 with any flags or step counts; either at a reserved seed, or any protected input outside a draw, is the stop. The entity signature records restraints for complexes seen in development. Reason: revision 74's exact-recipe exemption would have stopped qualification on any profiling, memory or debugging run of D4, the only input above 1,000 tokens, and its restraint-free signature could not tell D8 from H7.
- Make contract-6 provenance fail closed: support only single-file inputs with precomputed MSA and template paths (all 25 rows), refuse directory inputs and runs that would generate MSA or template data, record source, read-point and referenced-file digests and actual seeds, and require identical input digests for paired roles; tests listed. Reason: no qualification row or realistic porting run needs directory or MSA-search provenance, and contract-6 now gates porting.
- Add contract-6's read-point hook target to the harness-hook list. Reason: it wraps a function in the candidate's own tree, like the stage methods.
- Commit the signature inventory, including protected siblings missing from `digests.json`, in contract-6 under `scripts/qualification/inputs`. Reason: the audit's reference data should be frozen with the code that checks against it.

## Revision 74 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Make contract-6 and its seed-budget decision a prerequisite for development porting, and remove the older statements that porting may start before the decision or that the first candidate commit is the latest deadline. Reason: the new run-provenance rule must cover development inference from its first run; the amendment and tests have their own cost.
- Treat a reserved seed on D8 as a stop, alongside D4, and limit their development exemption to the exact frozen input digest, row recipe and seeds below 200. Reason: the revised do-not-run rule already stopped on both inputs at reserved seeds, while the older budget text exempted all development inputs except D4; a changed 9lh2 sibling must not inherit D8's exemption.
- Specify contract-6 provenance for file and directory inputs: inventory every discovered source JSON, capture the effective JSON at the inference read point after preprocessing, map both to jobs and actual seeds, and compare paired effective inputs with path-independent handling for generated files. Add the corresponding failure tests. Reason: a single pre-run path hash neither covers directory input nor proves what inference read after MSA or template preprocessing.
- Define the audit match as a canonical entity signature with types, sequences or ligand identities, and counts, alongside input digests; retain signatures and digests for protected siblings not in `digests.json`. Correct 9lh2 and 7wux from "two chains" to two protein sequence entries with count two each, plus ligands. Reason: those JSONs expand to more than two chains, and a sequence-only or manifest-digest-only check misses entity changes and unlisted siblings.

## Revision 73 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Remove the development row D8 (`9lh2_unconstrained.json`) from the do-not-run list, keeping `9lh2_contact.json` (H7) and `9lh2_pocket.json`, and exempt D8 and D4 from the list and from sequence grouping when they run as their development rows through the harness at development seeds; either input at a reserved seed outside a draw is the stop condition. Reason: revision 72's directory-wide 9lh2 entry and sequence grouping covered D8 and D4, so every D1-D15 rehearsal would break the list and the pre-draw audit would always stop on step 0 and stand-in runs.
- Correct the freshness claim: the first draw can test H1-H6, H8 and H9 as unseen inputs; H7 is constraint-fresh only (its complex is seen through D8) and H10 is seed-only. Reason: H7 is the D8 complex plus a contact restraint.
- Record the sequence baseline (only H7/D8 and H10/D4 share chain sequences; all 25 row inputs are in `digests.json`), so the grouping is an exact-sequence plus digest match, and check new development inputs against it. Reason: a reproducible check instead of judgement.
- Land contract-6 before development porting starts. Reason: it supplies the run provenance the audit relies on, and it must precede the first candidate anyway.
- Scope the wrapper rule to end-to-end inference runs; allow direct Python-entry runs on development or synthetic inputs with a minimal record, and exempt operator- and kernel-level tests. Reason: step 1 needs `runner.inference.run()` checks, which the wrapper does not cover.
- Give the first, only potentially full-strength, draw to sm_120 `fast`, never to `exact`. Reason: it is the claim that gains most from fresh inputs.

## Revision 72 (2026-10-13)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Treat the first H1-H10 draw across both GPUs as the only draw that can test H1-H9 as unseen inputs under the serial workflow, subject to the pre-draw audit; H10 is always seed-only. Later modes and GPUs use seen inputs and get seed-only claims; a separate full-strength A100 or mode claim needs inputs reserved before the first draw. Reason: sm_120 results can inform later A100 choices, even if A100 uses fresh seeds.
- Extend the do-not-run family to `9lh2_unconstrained.json` and compare target sequences and entities as well as paths and available digests. Reason: it is the same H7 complex without the contact constraint; a copied or reformatted input evades a byte-digest match.
- Make the historical run audit best-effort, and add resolved input path, pre-run input digest and actual per-job seeds to future run records in the planned contract-6 amendment, with verifier checks against frozen inputs and declared seeds. Require the wrapper and a retained record for future non-held-out runs; re-estimate the CPU effort and test the new fields. Reason: current run records store arguments, not input bytes or resolved seeds, so a source-file digest cannot establish what an old run consumed.
- Require a separate held-out draw for each full `exact` GPU claim; until then, describe the mode as development-validated. Reason: the validation matrix requires a draw for any port-chosen mode, and deterministic identity tests do not silently waive that rule.

## Revision 71 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Adopt the seed-budget plan (user request): widen the held-out seed range to 201-999 as `qual-sm120-contract-6` before the first held-out candidate's code commit, with the files and release steps listed (rulesets verified, run-store audit, tests, tag, D1 smoke and gate; about half a day and 10 GPU-minutes); implementation and tagging await the user's go-ahead. Reason: widening is cheap and removes the risk that one failure plus the A100 claim exhausts a four-draw pool.
- Declare the intended claims before that code commit (`fast` on sm_120, then A100 `fast`, `big` only if capacity measurements justify it) and settle whether `exact` needs a held-out draw. Reason: the validation-matrix rule ties a draw to any port-chosen sm_120 row, so `exact` decides whether the full claim set needs four draws or six.
- State that widening buys draws, not strength: retries and a second mode's draw on the same GPU reuse seen inputs, so only each GPU's first draw is full strength; new held-out inputs are the only way to buy strength, cost days and must be decided before the first held-out run. Reason: the budget decision should not treat extra seeds as fresh holdouts.
- Correct revisions 65-70: widening needs no input digest changes, because `digests.json` does not cover `manifest.json`.

## Revision 70 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Add a held-out do-not-run list, applying from the start of porting: H1 (`examples/cofold/input.json`, the default example of `inference_demo.sh` and several documents), all `examples/tfg/9sat/` and `9xqn/` files, `9lh2_contact.json` and its guided sibling `9lh2_pocket.json`, and the H2, H8 and H9 qualification inputs; D4's input (H10) only at reserved seeds. Porting, profiling, benchmarking, smoke, debugging and documentation runs use development inputs only. Before every draw, scan registered run stores for these inputs by path and by `digests.json` digest; a hit is the stop condition. Reason: running a held-out input at any seed exposes it, the default example is H1, and nothing previously named the files or detected such a run.
- Refocus the seed audit on seed and input pairs: require explicit `--seeds` below 200, read the seed actually used from `seed_<n>` output directories or run records, treat a reserved seed on a development input other than D4 as a recorded deviation, and stop only for a held-out input or D4's input at a reserved seed. Reason: OpenDDE picks a random seed from 1 to 65536 when none is given, and a seed carries holdout information only on a held-out input; H10 shares D4's input.
- State each serial line's cost: the whole development set reruns, about 4 GPU-hours on sm_120 from the stand-in timings, because ledger and documentation commits change the code commit. Budget lines in GPU-hours as well as seeds. Reason: the draw-budget decision should see both costs.

## Revision 69 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Require a fresh descendant `code_commit` for each serial acceptance line, including a second mode or GPU using unchanged model code. The clean merge tip can serve as that commit; rehearse its exact arguments. Reason: distinct mode arguments satisfy the one-draw check, but an old commit cannot contain a prior line's merged held-out log.
- Apply the below-200 seed policy to every future non-held-out run on both GPUs, register new run stores, and audit all registered stores before each draw. Stop and re-budget if a reserved seed appears. Reason: auditing only the original stores would miss later calibration or ad hoc runs that consumed held-out seeds.
- Check ruleset target, include and exclude patterns, required rule types and a present, empty `bypass_actors` field; require a caller with ruleset write access to see that field. Treat the tag-pattern check as configuration evidence, retain API results with the relevant record, and repeat before every contract-tag push and acceptance step. Reason: GitHub omits bypass actors from lower-privilege reads, and a static tag match cannot demonstrate server enforcement at push time.

## Revision 68 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- State that the sm_80 freeze is a qualification-code amendment under either contract design: `accept.py` reads fixed `tier1_limits.json` and `tier2_limits.json` paths and one top-level `baseline_off.json` of sm_120 digests, so A100 needs per-GPU limit and baseline paths (and possibly recipe values such as `diffusion_chunk_size`), synthetic tests and an sm_120 D1 smoke test. Estimate that shared cost for both designs. Reason: revision 67 implied only the GPU-scoped design needed verifier changes, while A100's own limits and drift baseline cannot coexist with sm_120's in the current files.
- Specify ruleset verification through the GitHub API (ruleset `enforcement`, `conditions`, rules and empty `bypass_actors`; the branch-rules endpoint for an `accept/` name; a static check of the tag pattern), forbid test pushes of matching refs, save the JSON in each record directory and repeat the check at every protected step. State that the owner can still edit rulesets, so this guards against accidents only. Reason: a matching test tag would become undeletable, one check proves only that moment, and the retained JSON must not be read as tamper-proof.
- Make the seed-range reservation concrete: development and tuning runs use seeds below 200, and the seed audit is repeated over the same stores before each draw. Reason: the revision 67 reservation had no rule behind it.

## Revision 67 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Keep the existing `qual-sm120-contract-N` family as the default shared sm_120/sm_80 contract family. If a new shared family is chosen instead, retire the old one with a higher-numbered tag at the transition and test that a contract-5 gate refuses an old candidate. Reason: the gate searches only the tag family named by its own checkout, so an unretired old family could still approve an old candidate.
- Separate the current directory-wide coupling from the proposed GPU-scoped alternative. The shared design re-tags and D1-smoke-tests both GPUs after every amendment. A scoped design first needs shared/private file boundaries, per-GPU handling of `acceptance.json`, changes to the gate and verifier, and a tested transition; only then can GPU-private changes avoid re-tagging the other GPU. Reason: the previous text promised independent changes while also requiring both GPUs to re-tag for every change.
- Require proof that the GitHub tag and acceptance-branch rulesets are active, match the intended refs and have no unapproved bypass before the next contract amendment or first receipt push. Retain the rule IDs and verification result. Reason: a configured but ineffective rule does not protect the newest tag or the line-checker's ref inventory.
- Bound the seed-history claim to the records actually audited, record search scope and gaps, and reserve any expanded range against later development use. Reason: scanning known run stores cannot prove that an unrecorded ad hoc run never used a seed.

## Revision 66 (2026-10-12)

Documentation only; contract-5, qualification code and limits are unchanged. No GPU work or porting began.

- State that an sm_80 contract always changes the sm_120 contract directory and that the coupling runs both ways (shared `acceptance.json` with top-level `recipe` and `baseline`; each contract check covers all of `scripts/qualification`), so every amendment for either GPU re-tags the other and needs D1 smoke tests on both. Choose a shared tag family or a GPU-scoped contract check before A100 qualification work, build it into the sm_80 freeze amendment, and schedule that freeze between sm_120 lines. Reason: revision 65 called the coupling conditional, and an unplanned coupled re-tag would invalidate candidates and force A100 provisioning.
- Require GitHub rulesets before the first receipt is pushed: tags `qual-*-contract-*` blocked from update and deletion, branches `accept/**` blocked from deletion and force-pushes (user action; the public fork had none). Reason: besides hiding acceptance lines, deleting the newest contract tag on the remote would make the gate approve a candidate at an older contract.
- Frame the budget decision as widening the seed range (one amendment, seed-only retries; first scan step 0 and stand-in run records to confirm the new range was never used) or narrowing the claims, and state that new held-out inputs are a substantial selection, preparation and route task. Reason: the three options are not equally cheap, and the documents alone cannot prove a seed range unused.

## Revision 65 (2026-10-12)

Documentation only; contract-5, qualification code and limits are unchanged. No GPU work or porting began.

- Move the held-out budget decision from the start of porting to before fixing the first held-out candidate's code commit and rehearsal. Budget the intended GPU/mode claims and retries; leave `big` conditional on capacity measurements. Reason: development can proceed without spending holdout seeds, but a contract amendment after rehearsal would invalidate the candidate's tag-bound receipt.
- Distinguish seed-range expansion from reserving new inputs. A wider range needs changes to both the frozen manifest and `accept.py`'s hard-coded range, plus digests and tests. A new input does not become an alternate set under routes that require H1-H10; distinct future sets need their own predeclared rows, limits and route selection. Reason: merely reserving inputs or editing the manifest would not deliver the intended retry capacity.
- Treat cross-GPU seed uniqueness as pending until the sm_80 contract inherits the shared log, prior line history and ledger. Amend the sm_120 contract between draws if sm_80 qualification changes its frozen directory. Reason: the current acceptance recipe contains only sm_120, so separate A100 checkouts would not automatically see spent seeds.
- Open a ledger entry when the receipt branch is first pushed, before the draw. Have the planned line checker reconcile ledger entries with the approved remote's acceptance refs, verify closed ref tips and reject open entries; test missing, unlisted and moved refs along with the merge cases. Protect the refs or retain independent push records. Reason: a checker that examines only listed lines cannot find an omitted one, while a deleted and never-recorded ref remains outside its visibility.
- Replace the instruction to add held-out inputs after the pool runs out with a stop and a separately approved campaign. Reason: inputs chosen after held-out results have been seen cannot be represented as a fresh holdout under this qualification.

## Revision 64 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Record the held-out seed budget as a user decision needed before porting: seeds are single-use across the whole log for every GPU, each draw uses 20 seeds plus up to four H3 extras, and the 99-seed pool allows at most four draws across sm_120, A100 and retries, while each mode is a separate candidate. More inputs can be reserved only before the first held-out run, and a wider seed range is a contract amendment. Reason: the shortage would otherwise surface only after the first draw, when inputs can no longer be added.
- Allow at most one open acceptance line at a time across all GPUs; a later candidate's code commit waits until every earlier line is merged. Reason: `accept.py` walks the log's history across both sides of a merge, so concurrent lines would interleave versions that fail the append-only check for every later candidate, and their seeds could collide.
- Define line closure: a final verdict, or abandonment after any pushed log commit (a stopped draw or a dropped receipt), each recorded in the ledger as `accepted`, `rejected` or `abandoned` and merged. Reason: revision 63 covered only draws that reach a verdict, while an abandoned draw's seeds are still spent.
- Plan a read-only line checker outside `scripts/qualification` (ledger ancestry plus `_extends` against each recorded tip) and CPU merge-topology tests, before the second candidate's rehearsal. Reason: a merge resolved toward the porting branch passes the ancestry check and `accept.py`, and only the log comparison catches it.
- Run the paired smoke test from a clean checkout of the code commit with `PYTHONPATH` set, and add D9 when the diff touches the guidance path. Reason: D1 never reaches the TFG draw sites, and the editable install can otherwise load another checkout's code.

## Revision 63 (2026-10-12)

Documentation only; contract-5, qualification code and numeric limits are unchanged. No GPU work or porting began.

- Record each closed acceptance line's final pushed commit ID and verdict in a version-controlled `porting_docs/qualification_line_ledger.md` (created at the first closure), outside the acceptance clone. Before a later candidate's rehearsal, require those commits in its ancestry and its held-out log to extend each recorded version under `_extends`; recover an unavailable commit before proceeding. Reason: branch names can move or disappear, and checking their current tips alone cannot establish continuity from an earlier draw.
- Make the D1 candidate smoke a paired `off`/candidate check from the candidate code commit, with draw-trace and tier 1 comparison, and include the runner hook targets in the diff review. Compare `instrumentation.py` with the active contract tag. Reason: a candidate-only run has no stock comparison, runner hooks are also outside the frozen qualification directory, and a fixed contract-5 comparison would reject a later approved amendment.
- Use one contract-amendment rule throughout the live plan and protocol: before the first held-out draw or between completed draws, never during one. If hook changes alter measurement semantics, recalibrate after the stock smoke test. Reason: old pre-first-draw wording contradicted revision 62's between-draw policy.

## Revision 62 (2026-10-12)

Operator rules and wording; contract-5, qualification code and numeric limits are unchanged, and no GPU work or porting began.

- Add held-out log continuity across candidates: push each acceptance line as `accept/sm120-<receipt-id>`, merge every such branch into `porting/inline-optimizations` before a later candidate's code commit, and check that the later candidate's log extends each branch tip; plan and documentation commits never go into `~/opendde-accept`. Reason: `accept.py` checks seed reuse, one draw per candidate and append-only history only along the acceptance checkout's own history, which must differ from the code commit only in the held-out log, so acceptance lines diverge from the porting branch and a later candidate without their log commits would see an empty log. Gate enforcement is left as a possible amendment between draws.
- Add the model-side harness hooks as a port constraint: `opendde/utils/instrumentation.py` and its call sites unchanged, the wrapped stage methods' names, signatures and roles kept, an empty `instrumentation.py` diff against the tag and a review of the hook sites before each rehearsal, and a one-row D1 candidate smoke test before the full rehearsal. Reason: these are in the candidate's own tree, outside the gate and contract check; most breakage fails only after GPU-hours, and a bypassed stage could change what tier 1 compares.
- Name the accepted `origin` URL forms and check them in the retained gate report's `remote` field; state that unreadable `--accept` evidence raises whether or not the gate passed (exit 1, no `record.json`) and that the command itself refuses an existing record directory; allow contract amendments between draws, never while a draw is open. Reason: wording found against the code in the revision 61 review.

## Revision 61 (2026-10-12)

Documentation only. Contract-5, qualification code and numeric limits are unchanged; no GPU work or porting began.

- Replace the plan's and protocol's future-tense gate repair and tag-amendment instructions with the completed contract-5 status. The first optimized candidate still needs its own full development rehearsal through the frozen entrypoint before a held-out draw; the earlier stand-in rehearsal preceded the final fixes, and the post-tag D1 smoke tested the stock side. Reason: leaving the old steps live could prompt an unnecessary contract-6 amendment or imply that the first candidate's rehearsal was already covered.
- State that the existing `porting/inline-optimizations` branch descends from contract-5 and its qualification files match the tag, and correct the amendment order. Reason: telling an operator to create that already-existing branch from the tag is ambiguous, while the out-of-order tag list obscured the freeze history.
- Require an operator check of the controlled gate checkout's expected `origin` URL and retention of the URL in each gate report. Treat missing or unreadable `--accept` evidence as a CLI-input error to correct and rerun, not as a completed failed-gate record. Reason: clean Git status does not authenticate remote configuration, and contract-5 hashes the evidence before it can return a failed-gate record; these are bounded operator inputs, not grounds for another contract amendment.
- Correct the protocol's old statement that sm_120 absolute limits still need to be set. Reason: they have been approved and frozen since contract-1.

## Revision 60 (2026-10-12)

Status update recording the contract-5 amendment (user go-ahead); no requirement changes.

- Record `qual-sm120-contract-5` (commit `881cf69`, annotated tag pushed): it freezes the held-out readiness repair (the gate's own remote, the `qual_gate.py --accept` entrypoint, the receipt's gate record) and the stand-in rehearsal fixes (import paths required inside the acceptance checkout, unreadable measurements as failed checks, the memory note); the limit files and hashes are unchanged. Earlier tags are superseded and the port branches from `qual-sm120-contract-5`.
- Record the D1 GPU smoke test at the tag with `PYTHONPATH` set (67 of 67 checks, including six import-path checks, and 206 qualification tests; job 49876) and the real-repository gate runs: the repository and a candidate commit on the tag pass; a candidate at `qual-sm120-contract-4` is refused.

## Revision 59 (2026-10-12)

Status update; no requirement changes.

- Record the stand-in end-to-end rehearsal (user request): on scratch repositories with a scratch contract tag, the torch triangle-kernel swap as candidate ran on D1-D15 (jobs 49509-49512) and passed through `qual_gate.py --accept` (job 49578): exit 4, 1016 checks, none failed, Zn qualified, `off` byte-identical to the step 0 baseline on all 15 rows; the receipt was written, committed to the scratch log, and the gate preflight with `--receipt` passed. The first post-processing attempt ran out of memory at 8 CPUs (job 49513).
- Record the three findings as fixed and tested (206 qualification tests, job 49872), not yet frozen: import paths recorded and required inside the acceptance checkout (the editable install otherwise runs `~/OpenDDE`'s code; use `PYTHONPATH`), unreadable measurement files reported as failed checks, and about 30 GB (16 CPUs) for post-processing.
- The next step is unchanged: freeze the branch's qualification changes under a new tag with D1 smoke, before porting if approved.

## Revision 58 (2026-10-12)

Status update; no requirement changes.

- Record that the held-out readiness repair is implemented and tested (`cac9131`; 198 qualification tests pass on gnode2, job 49502): `qual_gate.py` lists and fetches the approved tag through the gate checkout's own remote; `qual_gate.py --accept` runs the gate, then the candidate's `accept.py` only on a pass, then checks that the candidate is unchanged, writing a combined record (exit 5 on a gate failure, 1 on a changed candidate); rehearsal receipts record the passing gate report, and full acceptance checks that it ran at the contract tag. The branch's `scripts/qualification` now differs from `qual-sm120-contract-4`; the amendment under a new tag, with D1 smoke, remains. A stand-in end-to-end rehearsal through the entrypoint before that freeze was recommended and awaits a decision.

## Revision 57 (2026-10-12)

Documentation only. Contract-4, its numeric limits, held-out log and qualification code are unchanged; no porting or GPU run was started.

- Hold held-out draws until the independent gate discovers the newest contract tag from the controlled acceptance checkout's authoritative remote, then compares the candidate's fetched tag to that commit. Add a differing-remotes test with a newer tag absent from the candidate's remote. Reason: contract-4 currently calls `git ls-remote` in the candidate checkout; its `origin` can be stale or different, so it can conceal a newer tag while the gate checkout remains at the older one.
- Require a trusted entrypoint to run the gate before the candidate verifier, stop on failure, bind the gate's candidate commit to the checkout it verifies, and retain both reports. A gate preflight is also recorded before each held-out batch. Reason: `accept.py` does not consume a gate report and remains directly callable, so its exit 0 alone cannot establish that the independent gate ran. The plan now states this procedural gap instead of treating a standalone verdict as full qualification.
- On approval, test those changes, freeze a narrow qualification-code amendment under a new pushed tag, and repeat D1 smoke before a draw; prefer doing so before porting begins to avoid rebasing candidates. Keep the numeric limits fixed. Reason: the repair changes frozen qualification code, and a tag created during an open draw would invalidate its contract binding.
- Replace the protocol's stale instruction to implement the already-frozen receipt and recast its future-tense contract-4 specification as historical rationale. Reason: those lines contradicted the current status and could send an operator through the wrong amendment sequence.

## Revision 56 (2026-10-12)

Status update recording the contract-4 amendment (user go-ahead); no requirement changes.

- Record `qual-sm120-contract-4` (commit `3e6284a`, annotated tag pushed): relative to `qual-sm120-contract-3`, only `qual_gate.py` (new), `accept.py` and `acceptance.json` changed; the limit files and hashes are unchanged. Earlier tags are superseded, the port branches from `qual-sm120-contract-4`, and no further amendment is planned before a first candidate exists.
- Record the D1 GPU smoke test at the tag (61 of 61 checks and 188 qualification tests, job 49476) and the real-repository gate runs: the repository at the tag and a candidate commit on it pass; a candidate at `qual-sm120-contract-3` is refused on the named tag, descent and qualification files.

## Revision 55 (2026-10-12)

Status update; no requirement changes.

- Record that `qual_gate.py` and the receipt's `contract_tag` (with `rehearsal receipt names the contract tag`) are implemented and tested (`ca5be9a`; 188 qualification tests pass on gnode2, job 49475). The branch's `scripts/qualification` now differs from `qual-sm120-contract-3`, so the next step is the amendment `qual-sm120-contract-4`, with the D1 smoke test and a real-repository gate run at that tag, before porting starts.

## Revision 54 (2026-10-12)

Documentation only. The frozen contract, limits and held-out log are unchanged.

- Make the contract-version gate compare files instead of running a second verifier against the candidate: `qual_gate.py`, run from a checkout of the approved tag, takes the newest `qual-sm120-contract-N` on `origin` as approved, requires the candidate's `acceptance.json` to name it, `scripts/qualification` to match it apart from the held-out log, and the tag to be an ancestor.
  - *Reason:* if the files match, the candidate's `accept.py` is the approved verifier; running a verifier from another checkout would mean refactoring `accept.py`'s single-checkout assumptions in frozen code. A contract-2 candidate fails the gate without any old verifier being run.
- Record the contract tag in each receipt and compare the final gate against it; create no contract tag while a draw is open; run the gate at rehearsal, before each batch of held-out runs and at final acceptance.
  - *Reason:* a tag pushed between a receipt and its final verdict would make the final gate, and contract-3's newest-tag check, fail after the held-out set was spent.
- Implement the gate and the receipt field, amend the contract as `qual-sm120-contract-4` and rerun the D1 smoke test before porting starts; then make no further amendment before a first candidate exists.
  - *Reason:* both change `scripts/qualification`; amending first means the port never needs a merge, and the contract should settle before anything depends on it.
- Test the gate's refusal on a synthetic repository with two tags instead of running an actual older `accept.py`.
  - *Reason:* the refusal is what matters; running old code adds only a dependency on it.
- Acknowledge that revisions 50 and 52 overstated contract-3's newest-tag check, which protects only runs of contract-3 code.

## Revision 53 (2026-10-12)

Documentation only. The frozen contract, limits and held-out log are unchanged; no porting or GPU run was started.

- Add an independent contract-version gate before the first held-out draw and at final acceptance. Use the approved verifier from a trusted checkout, check the candidate's declared tag against the approved tag, and test an actual contract-2 verifier/candidate against contract-3. Reason: a candidate branched from contract-2 carries an `accept.py` that has no newest-tag or receipt check; adding contract-3 to its tag list cannot make that older code enforce either rule. The current unit test calls contract-3 code with a stale tag, which does not cover this path.
- Replace the obsolete instructions to implement the receipt binding and create contract-3 with its completed status and the remaining candidate-side rehearsal. Correct the protocol's current test count to 175. Reason: future-tense instructions beside completed work could cause a second amendment or obscure the actual readiness gate.
- State that the logged receipt is a pre-draw record, not independent proof that `dev-rehearsal` ran. Its verdict hash is recorded but not checked against a retained verdict file; final acceptance reevaluates the same development evidence. Reason: the procedure should describe the assurance the current verifier provides without claiming more than its checks establish.

## Revision 52 (2026-10-12)

Status update recording the contract-3 amendment (user go-ahead); no requirement changes.

- Record `qual-sm120-contract-3` (commit `e9d18f6`, annotated tag pushed): `contract_tag` now names it; relative to `qual-sm120-contract-2`, only `accept.py`, `acceptance.json` and the held-out log's description and empty `receipts` changed; the limit files and hashes are unchanged. `qual-sm120-contract-1` and `qual-sm120-contract-2` are superseded, and the port branches from `qual-sm120-contract-3`.
- Record the D1 GPU smoke test at the tag: 61 of 61 checks (job 49468), including `contract tag is the newest` and `off` byte-identical to the step 0 baseline. Its own test run hit gnode2's shared `/tmp` inode limit (about a million inodes from retained pytest directories of earlier qualification test runs, since removed); all 175 tests then passed at the tag with pytest's temporary directory on `/data` (job 49470).

## Revision 51 (2026-10-12)

Status update; no requirement changes.

- Record that the rehearsal-to-draw binding is implemented and tested (`742d5e8`): held-out log receipts, `accept.py --receipt`, the evidence digest, receipt and one-draw-per-candidate checks, append-only receipts and the newest-contract check; 175 qualification tests pass on gnode2 (job 49467). The branch's `scripts/qualification` now differs from `qual-sm120-contract-2`, so the next step is the amendment `qual-sm120-contract-3` with the D1 smoke test at it, before porting starts.

## Revision 50 (2026-10-12)

Documentation only. `qual-sm120-contract-2` remains the current frozen tag; no qualification code, limits or held-out data changed.

- Amend the contract with the rehearsal binding as `qual-sm120-contract-3` before porting starts, not only before the first draw, and branch the port from it.
  - *Reason:* `accept.py` runs from the candidate's own `code_commit` and checks the tag named there, so a port branched from contract-2 would carry contract-2's `accept.py`, without the receipt check, and could pass full acceptance without a rehearsal. This is the problem revision 46 fixed for contract-2.
- Require the tag named in `acceptance.json` to be the newest `qual-sm120-contract-N` tag in the acceptance clone.
  - *Reason:* amendments may recur; a candidate built on a superseded contract should fail rather than rely on someone merging the amendment.
- Allow at most one draw per receipt and per `(code_commit, candidate_args)` pair across the log history.
  - *Reason:* `check_log` checks only seeds, so a failed candidate could draw fresh seeds unchanged and retry; the plan's "retry after changes" was not enforced.
- Define the evidence digest: every development run record, trace and measurement file plus the set declarations, as a canonical sha256 over sorted `(role, row, file, sha256)` entries; the receipt stores only the verdict's sha256, status and exit code.
  - *Reason:* measurement files alone miss the run records, the traces and the repeat run, which no measurement reads.
- Define "earlier": the receipt is in the log at the parent of the draw's first commit. Receipts are append-only with unique ids under the existing history checks.
  - *Reason:* only a receipt holding digests, committed before the draw, shows the evidence existed then; the runs' own commit (`code_commit`) can be checked out after the draw.
- Make the rehearsal outputs read-only once the receipt is committed.
  - *Reason:* the digest check would catch a change only after the held-out runs.
- Add tests for the new rules: edited or removed receipts, reused receipts, a repeated commit and arguments, each kind of changed evidence and a stale contract tag.

## Revision 49 (2026-10-12)

Documentation only. `qual-sm120-contract-2` remains the current frozen tag; no qualification code, limits or held-out data changed.

- Require a passing development rehearsal to be recorded in an append-only Git commit before the held-out draw. Full acceptance must reject a missing, late or nonpassing receipt and must match the receipt's development evidence, code commit and ordered candidate arguments to the draw. Reason: the current full-acceptance path can exit 0 without a rehearsal, so a written pre-draw rule alone can be skipped.
- Require an exact argument match as well as a commit match. Reason: the same code commit can run `fast` in rehearsal and `big` in qualification; the first run would not test the second mode.
- Plan a new contract tag, a D1 stock-side GPU smoke test and tag availability in the acceptance clone before the first draw. The tier 2 numeric limits stay fixed. Reason: the receipt and history checks will change frozen qualification code, while the existing tag-2 smoke test says nothing about that change.
- Correct the missing conjunction in the plan's acceptance-command summary and distinguish the implemented rehearsal route from the unimplemented rehearsal-to-draw binding.

## Revision 48 (2026-10-12)

Status update recording the contract amendment (user go-ahead); no requirement changes.

- Record `qual-sm120-contract-2` (commit `0be6daa`, annotated tag pushed): `acceptance.json`'s `contract_tag` now names it; relative to `qual-sm120-contract-1`, only `accept.py` and `acceptance.json` changed (the rehearsal route); the limit files and hashes are unchanged. `qual-sm120-contract-1` is superseded, and the port branches from `qual-sm120-contract-2`.
- Record the D1 GPU smoke test at the tag: 60 of 60 checks (job 49458), including `off` byte-identical to the step 0 baseline, both limit hashes and the three contract checks; the 156 qualification tests pass at the tag and the checkout stayed clean.

## Revision 47 (2026-10-12)

Status update; no requirement changes.

- Record that the `dev-rehearsal` route and exit code 4 are implemented and tested (`c113fb2`; 156 qualification tests pass on gnode2, job 49450). The branch's `scripts/qualification` now differs from `qual-sm120-contract-1`, so the next step is the amendment: tag `qual-sm120-contract-2`, rerun the D1 smoke test at it and record contract-1 as superseded, before porting starts.

## Revision 46 (2026-10-12)

Documentation only. The approved sm_120 limits, held-out assignments and frozen tag are unchanged.

- Specify the rehearsal as a `dev-rehearsal` route in `acceptance.json` (`rehearsal: true`, D1-D15), not a command-line mode, with its own exit code 4 ("rehearsal passed, not an acceptance").
  - *Reason:* required rows must come from the file, not flags, and exit 3 already means "contract not frozen".
- Say where a rehearsal gets `code_commit` and `candidate_args` (the evidence, since there is no draw) and which checks it skips (only those needing a draw); everything else runs unchanged, including the contract-tag check, and a rehearsal requires a declared freeze.
  - *Reason:* `accept.py` takes both values from the held-out draw, so revision 45's rehearsal had no defined source for them.
- Bind the rehearsal to the held-out run: the rehearsed development runs are the development set of the final acceptance, so the draw's `code_commit` must be the rehearsed commit; a fixed candidate is rehearsed again.
  - *Reason:* without this, a candidate could pass rehearsal at one commit and be held-out tested at another, and the development GPU pass would be paid twice.
- Run the rehearsal in the acceptance clone instead of a manual tag-availability check.
  - *Reason:* the contract checks then fail if the tag or commit was not fetched there.
- Amend the contract as `qual-sm120-contract-2` before porting starts, and branch the port from it; record `qual-sm120-contract-1` as superseded.
  - *Reason:* `accept.py` checks the tag in the candidate's own `acceptance.json`, so a candidate branched from contract-1 would lack the rehearsal route and could still pass full acceptance under contract-1.
- State the rehearsal's cost: the full development set per candidate, several GPU-hours.

## Revision 45 (2026-10-12)

Documentation only. The approved sm_120 limits, qualification code, held-out assignments and frozen tag are unchanged; no porting or candidate run began.

- Separate contract approval and freeze from readiness to open held-out inputs. Require a development-only acceptance route or mode for D1-D15, with the production per-run, per-row and tier checks, complete candidate evidence, an independent repeat, no held-out seed use and no exit 0. Test its success and representative failures, then rehearse the first candidate on development rows.
  - *Reason:* the current `accept.py` has only the `port-chosen` route, which requires H1-H10 and a held-out draw. The proposed rehearsal cannot run as written, while candidate-side behavior has only synthetic coverage. A harness failure on held-out data would consume the set.
- Require a new contract tag and D1 stock-side smoke test after adding that route or mode, before any held-out draw. Check that the tag and freeze commit are present in the dedicated acceptance clone.
  - *Reason:* the freeze rejects changes under `scripts/qualification`, including this necessary addition; a local tag alone does not establish that the acceptance clone can resolve it.
- Replace stale claims that the sm_120 limit hashes still need freezing with the recorded `qual-sm120-contract-1` status in the plan and protocol.
  - *Reason:* the hashes were recorded at Revision 44, so those future-tense instructions contradicted the freeze status.

## Revision 44 (2026-10-12)

Status update recording the sm_120 freeze (user go-ahead); no requirement changes.

- Freeze the sm_120 contract: `acceptance.json` records `tier1_limits_sha256` `b4b8fdd3bbc0…`, `tier2_limits_sha256` `d0294133b8cf…` and `contract_tag` `qual-sm120-contract-1`; both limit files' `status` name the tag. The commit holding these values and this revision is tagged `qual-sm120-contract-1` (annotated) and the tag is pushed. All 149 qualification tests pass on gnode2 with the frozen inputs (job 49442).
  - *Reason:* the approved contract, the tag check and its tests were in place; the freeze pins the code, constants, inputs, limits, recipe, baseline and wrapper together before any sm_120 tuning.
- Qualification no longer blocks sm_120 tuning; porting still needs the user's go-ahead. Any change under `scripts/qualification` is now a contract amendment with a new tag.

## Revision 43 (2026-10-12)

Status update; no requirement changes.

- Record that the contract-tag check is implemented and tested (`0dfbca0`): `acceptance.json` gains `frozen.contract_tag`; once a freeze is declared, `accept.py` requires both limit hashes, the tag, the draw's code commit descending from it and no change under `scripts/qualification` apart from the held-out log; a partial freeze fails. The test fixture uses a contract commit and clears the copied `frozen` block. All 149 qualification tests pass on gnode2 (job 49437). Only the freeze itself remains before sm_120 tuning.

## Revision 42 (2026-10-12)

Documentation only. The approved sm_120 contract, limits, held-out assignments, qualification code and GPU results are unchanged; nothing is frozen yet.

- Freeze the contract as a pushed Git tag that `accept.py` checks the candidate's `code_commit` against (descends from the tag; no change under `scripts/qualification` except the held-out log), keeping the limit hashes in `acceptance.json` as a readable record. Add tests for a changed gate constant, limits and hashes changed together, and a `code_commit` not descending from the tag. Implement this before the freeze.
  - *Reason:* approved values and rules live outside the limit files (`SAME_OUTCOME` in two modules, Zn and contact distances, gate selection, Zn claim rules, `accept.py`, `acceptance.json`, `baseline_off.json`, the run wrapper). `accept.py` runs from the candidate's own commit, and the hashes sit in a file the same commit can edit, so hashing the limit files alone would not catch accidental changes to the rest.
- Correct the protocol's claim that the same-outcome cutoff is "pinned with the other limits": it is a code constant, pinned by the tag.
- Plan for run-wrapper changes: keep the tier 1 hooked modules stable as a port design constraint, or amend the contract with a new tag and a rerun D1 smoke test before any held-out draw.
  - *Reason:* the wrapper hooks into model modules the port may restructure, and the contract check would otherwise block acceptance.
- Require a development-only rehearsal of the full candidate path through `accept.py` before any held-out draw is logged; a held-out run failing for harness reasons still uses up its set.
  - *Reason:* candidate-side checks have only synthetic coverage; finding a harness bug during a held-out run would spend the held-out set.
- Say that the 144 qualification tests pass (job 49426), not that they were "reported" passing.

## Revision 41 (2026-10-12)

Documentation only. The sm_120 tier 2 limits, held-out assignments, qualification code and GPU results are unchanged; no limit hash was frozen and no porting began.

- Record approval of the full sm_120 tier 2 contract, including the reduced-assurance limits and the acceptance command's implementation choices and trust boundary.
  - *Reason:* the duplicate and unexpected-entry repair closes the last identified fail-open evidence path. The completed stock-side D1 smoke test and 144 reported passing qualification tests support approving the decision rules, while the remaining limitations are explicit.
- Separate contract approval from the hash freeze and from accepting a future candidate. `acceptance.json` still needs the approved sm_120 limit hashes before tuning; candidate-side end-to-end behavior must be exercised once a candidate exists. A100 requires its own stock checks and freeze.
  - *Reason:* approval fixes the rules but does not pin their files, validate a candidate run or transfer sm_120 tolerances to sm_80.
- Update the protocol's current-status sections and remove stale claims that full-set completeness, aggregate Zn enforcement or duplicate-entry tests are still pending. Keep the historical calibration decisions and their reduced-assurance caveats.
  - *Reason:* those requirements are implemented, and the old present-tense wording contradicted revision 40's status. The trust boundary remains deliberate: acceptance binds the measurement inputs and source revision but does not detect a hand-edited measurement value.

## Revision 40 (2026-10-12)

Status update; no requirement changes.

- Record that the duplicate and unexpected measurement-entry rejection, standalone `evaluate`'s `unique` gate and the producer refusals (`measure` repeated rows, `tier1_compare.py --rows`) are implemented and tested (`fdcbb46`; 144 qualification tests pass on gnode2, job 49426). Only the user's approval of the full sm_120 contract remains before the freeze.

## Revision 39 (2026-10-12)

Documentation only. The tier 2 numbers, held-out assignments, qualification code and GPU results are unchanged. Full sm_120 contract approval remains withheld.

- Make the measurement tools refuse what acceptance rejects: `tier2_gates.py measure` refuses duplicate `--rows`, and `tier1_compare.py` gains `--rows` and compares only those rows.
  - *Reason:* `measure --rows D1,D1` writes duplicate entries, and `tier1_compare.py` compares every row under its capture root, so a root shared between sets would give a valid file with unexpected entries.
- `evaluate` reports a duplicated row/job key, or a job not in the row's input, as a failing `unique` gate instead of raising; `accept.py` adds its own per-set uniqueness and expected-key checks before evaluation.
  - *Reason:* a failing gate keeps the standalone report printable with a nonzero exit, as for an unmeasured job, and reaches `accept.py` through its normal results.
- State the expected key sets: tier 2 is every row of the set times its input's jobs; tier 1 is the set's rows in the route's `tier1_rows`, times jobs and that row's seeds.
  - *Reason:* "each expected row and job" could be read as requiring tier 1 entries for rows outside `tier1_rows`.
- Add tests for a tier 2 entry with a job the row lacks, `measure` refusing duplicate rows and `tier1_compare.py --rows` excluding other rows.
- Record the trust boundary among E5's implementation choices: the digest chain binds what a measurement read and the provenance check binds its code, but the measured values are trusted as written; a hand-edited result file is not detected, and recomputing measurements inside `accept.py` is not done for sm_120.
  - *Reason:* the user approves E5's choices at the freeze and should see this scope stated.

## Revision 38 (2026-10-12)

Documentation only. The tier 2 numbers, held-out assignments, qualification code and GPU results are unchanged. Full sm_120 contract approval remains withheld.

- Add a pre-freeze requirement to reject duplicate and unexpected tier 1 and tier 2 measurement entries in each acceptance evidence set; standalone tier 2 evaluation must also reject duplicates. Test a failing entry on either side of a passing duplicate, and an unexpected entry.
  - *Reason:* `accept.py` and `tier2_gates.evaluate` index entries with dictionary comprehensions. The last entry for a row/job (or row/job/seed in tier 1) silently replaces the first, so a passing duplicate can mask a failed measurement even when both entries have valid input digests. The D1 smoke test and 132 existing qualification tests do not cover malformed measurement files.
- Move the approval request after this small evidence-validation repair and its tests; no new calibration or held-out prediction is required.
  - *Reason:* the numeric limits and measured stock behavior do not change, but the acceptance command must reject ambiguous evidence before its full-contract verdict is trusted.

## Revision 37 (2026-10-12)

Status update; no requirement changes.

- Record that revision 36's repairs are implemented and tested (commit `3d26223`): derived per-row seed pairs, the run-to-measurement digest chain, and exception and pre-existing-directory reporting; 132 qualification tests pass.
  - *Reason:* the status line described them as outstanding.
- Record the passing D1 GPU smoke test (job 49415; protocol E5, commit `4d18355`), including `off` byte-identical to the step 0 baseline, and that candidate-side checks remain synthetic until a candidate exists.
  - *Reason:* revision 35 made this run a condition for asking for approval.
- State that only the user's approval of the full tier 2 contract, then recording the limit hashes, remains before the sm_120 freeze.
  - *Reason:* every condition set in revisions 32-36 is met.

## Revision 36 (2026-10-11)

Documentation only. Full sm_120 tier 2 approval remains withheld; no limit, acceptance code, GPU output or held-out result changed. Revision 35's three findings were checked against `tier2_gates.py`, `accept.py` and `run_opendde.py` and hold; this revision sharpens the repairs.

- Derive tier 2 seed pairs per row from the held-out log (`measure --draw ID`) and record them on every entry, instead of only checking a hand-typed list.
  - *Reason:* `measure` takes one `--seed-pairs` list per call, so beyond crossed pairs, one measurement cannot cover the held-out set, whose rows each have their own seeds; the synthetic acceptance tests wrote the held-out `tier2` file directly and hid this.
- Bind measurements to runs by digests recorded on both sides: the producing run records what it wrote, the measurement records what it read and its code revision, and `accept.py` compares them. Drop recomputation as an alternative.
  - *Reason:* this proves the measurement read exactly that run's outputs, makes a missing `tier1.pt` a failure, and avoids rereading tens of gigabytes of captures; `compare_structures` needs at least 12 CPUs and about 17 GB per pass.
- Record in the run record whether the output and capture directories existed before the run, and require that they did not; keep `run_rows.py`'s refusal as well.
  - *Reason:* a refusal in `run_rows.py` alone is a procedure the acceptance command cannot see.
- Test the exceptional-exit path on CPU with a monkeypatched CLI, and note that killed jobs write no record.
  - *Reason:* a controlled exception on GPU would need a fault-injection hook in the wrapper; the CPU test covers the same code deterministically.
- Specify the GPU smoke test as an end-to-end D1 slice (traced `off`, untraced repeat, tier 1 capture, measurement, per-row acceptance checks including drift against the step 0 baseline), with candidate-side checks stated as covered only synthetically. Add an implementation order.
  - *Reason:* the drift check on a fresh run is the most informative real check available before a candidate exists.

## Revision 35 (2026-10-11)

Documentation only. Full sm_120 tier 2 approval remains withheld. No numeric limit, acceptance code, GPU output or held-out result changed.

- Require `accept.py` to check the exact same-seed mapping in each tier 2 measurement and to retain the per-job pairing IDs. Add a swapped-pair failure test.
  - *Reason:* checking only the `off` and candidate seed sets does not detect `101:102,102:101`, although those cross-seed comparisons redefine the same-outcome pairs.
- Bind tier 1 and tier 2 measurement results to the bytes of their expected capture and prediction inputs, and to the measurement code revision. Add missing-capture and stale-result tests.
  - *Reason:* the current acceptance check validates result paths and pass/fail fields but can pass without `tier1.pt` files, or with results computed before artifacts at the same paths changed.
- Make run records report uncaught exceptions as failures, and prevent per-row output directories from being reused. Add exception and stale-output tests.
  - *Reason:* `run_opendde.py` currently writes a default zero exit code in `finally` if an exception escapes; old artifacts in a reused directory could complete an otherwise failed run.
- Require one development-only GPU smoke test of the new run-record path, then request the user's full-contract approval before recording sm_120 limit hashes.
  - *Reason:* the 115 passing qualification tests and real step 0 read-only checks have not exercised a run-record write on a GPU. The smoke test and repairs do not require new stock calibration or held-out predictions.

## Revision 34 (2026-10-11)

Status update; no requirement changes.

- Record that the acceptance command's evidence bindings are implemented and tested (commit `667a03f`): run records, rebuilt command lines with the candidate's declared difference required, an external rollout and artifact expectation, committed step 0 baseline digests, bound tier 1 captures, and draws bound to a code commit with an append-only log history. List what is exercised only at the first acceptance run.
  - *Reason:* revision 33's repairs are done; the status line described them as outstanding.
- Record the user's decision to run acceptance from a dedicated clone, `~/opendde-accept`.
  - *Reason:* the decision was open in revision 33.
- State that what remains before the sm_120 freeze is the user's approval of the full tier 2 contract, then recording the limit hashes.
  - *Reason:* the bindings were the condition revision 32 set for asking.

## Revision 33 (2026-10-11)

Documentation only. The full sm_120 tier 2 contract remains unapproved; no limit, acceptance code, GPU run or held-out output changed. Revision 32's findings were checked against `accept.py`, `run_opendde.py`, `trace_tools.py` and `compare_outputs.py` and hold; this revision adds what they missed and simplifies two repairs.

- Add a run record written on every `run_opendde.py` invocation: the existing header plus the wrapper options.
  - *Reason:* the header is written only into trace or timing files, so the trace-off repeat and the tier 1 captures record no provenance, and no run records its `--tier1-replay` source.
- Derive the expected rollouts and prediction artifacts from the input's jobs, the logged seeds and the frozen recipe, and require the trace to match them before `check_manifest()`.
  - *Reason:* `check_manifest()` takes its expectation from the trace's own `rollouts`, so a rollout missing from both header and records, or a wrong `N_sample` or `N_step`, passes; a job missing from both a candidate and its repeat passes the byte comparison.
- Rebuild each run's expected command line from the manifest row with `run_rows.py`'s own construction, and require `off` to match it exactly and the candidate to add exactly its declared arguments; check `LAYERNORM_TYPE` and `CUBLAS_WORKSPACE_CONFIG` from the fingerprint.
  - *Reason:* nothing today requires the candidate to differ from `off`, so a candidate that silently ran `off` passes every check. The code commit pins defaults, so rebuilding the command line covers the effective settings without dumping the resolved config.
- Bind each held-out draw entry to the candidate's `code_commit` and declared arguments; require every run (not only H3's) to come from a clean commit differing from `code_commit` only in `held_out_log.json` and containing its seeds; require the log's Git history to be append-only, with seeds single-use across every version. This replaces revision 32's ordered-descent rule.
  - *Reason:* development and tier 1 runs may also precede the draw's log commit; without `code_commit` a candidate changed after a held-out failure could rerun the same logged draw; an edited or removed entry is invisible to a check of the current file. The H3 seed order is already checked through `by_seed`, so an ancestry chain adds nothing.
- Name the baseline manifest's source runs (tier 1 `off`, jobs 48900-48903 at `44e1fd1`, for D1-D13; the Zn calibration `off` for D14 and D15) and their recipe, and remove `off_baseline` from the evidence set.
  - *Reason:* once the digests are committed, an evidence-supplied path adds nothing but an aliasing risk; the source runs must share the acceptance recipe for the drift check to mean anything.
- State that acceptance runs come from a Git checkout, and propose a dedicated acceptance clone on gnode2 for the user's decision.
  - *Reason:* the scratch copy carries no commit, and H3's log commits require updating the checkout between jobs, which the no-sync rule for `~/OpenDDE` would block while other jobs are queued.
- Add the corresponding failure tests and an implementation order (protocol E5).

## Revision 32 (2026-10-11)

Documentation only. The full sm_120 tier 2 contract remains unapproved; no limit, acceptance code, GPU run or held-out output changed.

- Require the acceptance command to compare a new `off` with frozen step 0 artifact digests and to identify the candidate repeat as a separate run with its own provenance. Add self-alias and wrong-baseline tests.
  - *Reason:* the current evidence file supplies both comparison paths, so `off_baseline=off` or `cand_repeat=cand` can pass a byte comparison without testing drift or determinism.
- Require the expected random-draw manifest on both sides, including H3 extra-seed runs, before accepting a raw-hash match.
  - *Reason:* `accept.py` currently checks that recorded draws match, but identical omissions remain invisible unless each trace is checked for completeness.
- Bind tier 1 results to declared capture roots and the separate single-cycle run recipe, with their own provenance and fixed-input replay source. Correct E5's suggestion that tier 1 captures should come from the full-cycle traced runs.
  - *Reason:* tier 1 is intentionally run with `--cycle 1 --sample 1`; a passing JSON for the right row, job and seed can otherwise be substituted from another capture pair.
- Check the frozen input and effective recipe for each run, and verify that each held-out seed existed in that run's source commit. Permit ordered H3 log-only commits while holding executable code, inputs and limits fixed. Add wrong-input, undeclared-override, missing-at-commit, valid log-only sequence and mid-draw code-change tests.
  - *Reason:* checking the current input digests, current log and a few trace-header flags does not bind measured outputs to the declared input, configuration or precommitted held-out seeds. The current single-commit rule also conflicts with H3's instruction to commit each extra seed just before its run.
- Keep the numeric limits and disclosed reduced-assurance items unchanged. Ask for full-contract approval only after these evidence-binding repairs and tests pass; then record the limit hashes.
  - *Reason:* the review found acceptance-path defects, not new stock-calibration evidence. No new predictions are needed for the repairs, although existing step 0 artifacts must be read to freeze their digests.

## Revision 31 (2026-10-11)

Status update only; no requirement changes.

- Record that the acceptance command required by revision 30 is built and tested (`accept.py`, 17 tests; 88 qualification tests pass on gnode2), and that the reduced-assurance items are recorded.
  - *Reason:* the status line listed both as open.
- State what remains before the sm_120 freeze: the user's approval of the full contract, including the command's implementation choices and stated gaps (protocol E5), then recording the limit hashes in `acceptance.json`.
  - *Reason:* those choices (tier 1, determinism and trace checks on held-out rows; seeds single-use across the log) add GPU runs per candidate and limit future draws, so they belong in the approval.

## Revision 30 (2026-10-11)

Documentation only. No GPU run, limit, evaluator or held-out output changed.

- Make an H3 Zn result unresolved at its four-extra-seed cap a held-out failure (candidate rejected, set used up); "partial without Zn" is a report label only. State the expected cost for a harmless candidate.
  - *Reason:* revision 29 recorded the unresolved result separately without saying whether the candidate is rejected, while the existing rule counts an inconclusive held-out result after its declared resolution test as a failure. The extra seeds are that test. If H3 behaves like D3, at most three same-outcome pairs in 30 samples has probability below 1% at a 30% same-outcome rate and about 12% at 20%.
- Replace the row-completeness requirement with one acceptance command over a declared evidence set: required rows per GPU and route in a committed file; input digests, commit and limits hash; per row the random-input comparison, tier 1, determinism, tier 2 measurements and held-out log entries; seed provenance checks; exit 0 only for full acceptance; tests for each way a pass could be faked.
  - *Reason:* the evaluator checks only seed and sample counts, so a held-out row run with seeds 101/102 would pass, making H10 a copy of D4; trace, tier 1 and determinism results come from separate commands; development and held-out evidence arrive in separate measurements.
- State that a retry after a held-out failure uses fresh seeds on inputs already seen, a weaker seed-only test, and that extra held-out inputs can be reserved only before the first held-out run.
  - *Reason:* the pool has ten inputs; "a fresh set" cannot mean unseen inputs.
- Record reduced assurance for flexible interface Fnat 0.55 without a fourth contrast, and add the near-duplicate count at its cap and D11's constrained-distance change to the reduced-assurance list. Count 15 distinct development results.
  - *Reason:* `--deterministic false` changes only reduction order and would barely test a limit resting on three same-outcome pairs. D3 and D13's `1a14` job are the same input with the same seeds, so the near-duplicate cap was reached on one input, not two.

## Revision 29 (2026-10-11)

Documentation only. The held-out classes are accepted as predeclared sm_120 tests, but the full tier 2 contract remains unapproved. No held-out prediction, GPU run, limit file or evaluator was changed.

- Keep H1-H10's encoded assignments. Mark H1, H8 and H9 as extrapolations without matched stock dispersion data; H6's stable class is a deliberate stringent test; H10 tests new seeds on D4; and H2's PCG pose gate applies only in same-outcome pairs, possibly two of ten. Do not reclassify a row or change its limit after its output is seen.
  - *Reason:* the held-out set measures generalization under fixed rules. A result on an unfamiliar input is evidence about the candidate, not calibration data for that input. The PCG rule cannot support an all-sample ligand-pose claim.
- Require a complete, predeclared development and held-out row/job set in the full-acceptance path, with a test that removes one held-out row entirely.
  - *Reason:* `tier2_gates.py evaluate` defaults to the rows found in the measurement. Its existing missing-job check cannot catch a row absent from that set.
- Require a `qualified` aggregate Zn claim for full sm_120 acceptance, including conclusive D14, D15 and H3 results. An H3 result unresolved after four extra Zn-only seeds may be reported as partial without Zn, but it cannot pass the full gate; test that outcome and a conclusive pass.
  - *Reason:* the current H3 row can show `PASS` and the evaluator can exit zero at the extra-seed cap while its Zn claim is `unresolved`. The per-row gate and final acceptance claim must not disagree.
- Record flexible interface Fnat 0.55 as a limit chosen after the third harmless contrast failed 0.70. An independent harmless check remains optional, but without one the freeze must state the reduced assurance rather than call the third contrast independent validation of that gate.
  - *Reason:* the third contrast validated the 0.06 reference fnonnat limit, but it also influenced the final Fnat value.
- Reconcile protocol wording for the completed Zn fallback, held-out review and remaining freeze steps.
  - *Reason:* historical premeasurement instructions read as open tasks in revision 28.

## Revision 28 (2026-10-11)

Status update; records user decisions taken since revision 27. No port work.

- Record that the revision 27 pre-freeze work is done: limit provenance audit (`limit_audit.py`), third harmless contrast (`LAYERNORM_TYPE=fast_layernorm`, declared before the run), unpaired Zn fallback (declared, calibrated and rejected: both D14 Zn controls passed its slack and seed-to-seed variation matched the controls), and the Zn scope with `zn_rule` and `zn_claim` in the evaluator.
  - *Reason:* the status line listed these as open.
- Add the user's limit decisions (median reference lDDT drop 0.008; fnonnat 0.06 kept; flexible interface Fnat 0.55) and Zn scope decision (H3 up to four extra Zn-only seeds; H5 report-only) to the recorded decisions.
  - *Reason:* they change the acceptance contract the freeze will approve.
- State what remains before the sm_120 freeze: the user's held-out assignment review and approval of the full tier 2 contract.
  - *Reason:* no other sm_120 stock work is planned before the freeze.

## Revision 27 (2026-10-11)

Documentation only; no calibration run, limit change or port work.

- Make the limit-provenance audit mechanical: tabulate, per limit, the bf16-versus-fp32 value, the kernel-swap value, the chosen limit and the binding contrast; rederive only kernel-swap-bound limits, by a written rule (the gate's existing limit-to-precision ratio), not fresh judgement. Record the near-duplicate gate, the Zn threshold and D10's paired limits as already clean.
  - *Reason:* "rederive" without a rule is a second judgement that can drift toward whatever passes. The near-duplicate gate was declared from bf16 versus fp32 only; bf16 versus fp32 alone gives 0.95 Zn same-site on both rows; D10's kernel swap had no same-outcome pair.
- Run a third harmless same-precision stock contrast that set no limit (preferred `LAYERNORM_TYPE=fast_layernorm`, else `--deterministic false`), recorded before measurement.
  - *Reason:* it gives an independent false-fail estimate regardless of the audit outcome.
- State that negative-control rejections were observed while the limits were drafted, so they show separation on the development set, not independent sensitivity; partial collapse is the exception.
  - *Reason:* the same circularity revision 26 identified for the positive control applies in the other direction.
- Add an unpaired Zn coordination fallback (fraction of Zn with at least two polymer donors within 2.8 Å, relative to `off`), calibrated on existing D14/D15 outputs and controls, for rows where the paired check is inconclusive. Allow capped, predeclared extra seed pairs for H3 as an alternative.
  - *Reason:* H5 is an epitope row with a zero floor, and D10 had no same-outcome pair under the kernel swap, so extra draws may never make the paired check conclusive there.
- Define the Zn claim outcome as a row-level `zn: qualified | unresolved | fail` separate from the gate's `n/a`, and fail an inconclusive Zn result on D14 and D15.
  - *Reason:* those rows exist only to test Zn and stock gives 39 and 20 eligible ions there.
- State the paired Zn check's detectable fault size: with independent ions as an approximation, a fault moving one ion in four passes 0.85 about 24-37% of the time at 10-40 ions, while one moving half the ions almost never passes; a harmless candidate fails about 9% of the time at 10 ions.
  - *Reason:* the 10-ion minimum is a feasibility floor, and D14's moved-Zn failure at 0.79 is not a robust margin.
- Wording: restore "the C1 count floor still applies" to the not-applicable rule; H3's floor counts pairs, not samples; the Zn minimum counts `off`-bound ions in same-outcome pairs (`same_site_of`), not ions within 2.0 Å.
- Record the order of the remaining sm_120 work: audit, third contrast, Zn fallback, Zn claim outcome and tests, user scope decision and held-out review, freeze.

## Revision 26 (2026-10-11)

Documentation only. The sm_120 tier 2 numeric proposal is acceptable for planning; the acceptance contract is not frozen, and no optimization port or new calibration run was started.

- Require a per-limit provenance audit before treating the stock `torch` triangle-kernel swap as an independent positive control. If it influenced a limit, rederive that limit without it and retest, or use a different independent harmless same-precision control. Keep a numeric value only if the independent derivation and check support it.
  - *Reason:* C8 called the kernel swap independent, while C9 said both the precision contrast and kernel swap set the new limits. A control used to choose its own threshold cannot validate that threshold independently.
- Keep 0.85 at 2.0 Å with at least 10 matched ions as a proposed stock-relative Zn placement check, not a claim about deposited-site occupancy or coordination chemistry. Make a Zn `n/a` an unresolved aggregate outcome that blocks a full Zn-qualified pass for H3/H5; test the low-match and zero-same-outcome cases before freeze.
  - *Reason:* D14/D15 stock outputs reproduce their own Zn sites, not most deposited sites. H3 can meet its two-same-outcome floor with at most six ion comparisons, and H5 can meet its zero floor with none. The current evaluator records Zn `n/a` but still reports the row as passing.
- Update status and remove stale statements that Zn stock calibration and partial-collapse testing remain to be done. Keep held-out assignments under explicit freeze review and A100 limits separate.
  - *Reason:* the development Zn controls and near-duplicate results are recorded in step 0 results; stale text obscured the remaining provenance and Zn-applicability tasks.

## Revision 25 (2026-10-11)

Status update only; no requirement changes.

- Record that the sm_120 tier 2 stock work required by revision 24 is complete: the evaluator fails closed and encodes H1-H10, the near-duplicate gate rejects the partial-collapse control at ten samples, and D14/D15 support a proposed paired Zn gate. The remaining step before the sm_120 freeze is the user's review and approval of the tier 2 limits, held-out assignments and Zn gate.
  - *Reason:* the status line and the validation-matrix paragraph still listed this work as outstanding (protocol E2-E4 are done; step 0 results record the jobs).

## Revision 24 (2026-10-11)

Review of revision 23 against the evaluator code and the stock ensemble measurements. Documentation only; the C9 numbers remain proposals.

- Add a within-seed ensemble statistic (median same-seed pairwise anchor RMSD, or a per-row near-duplicate count) for the partial-collapse test, alongside the whole-set spread.
  - *Reason:* with two seeds of five, 25 of the 45 sample pairs cross seeds, so the whole-set median cannot move even under a complete collapse within each seed. Retuning the 0.1-10 window cannot fix this.
- Define the partial-collapse control exactly: 3 of 5 samples per seed replaced by perturbed copies of that seed's sample 0, at two perturbation sizes tied to the row's stock closest same-seed pair, with the rows required to reject it declared before measuring. Expect flexible rows to need more samples.
  - *Reason:* "perturbed enough to clear 0.01 Å" lets the control be made trivially detectable or undetectable. Harmless stock contrasts already resemble a partial collapse on flexible rows: D3 bf16 versus fp32 cut the whole-set spread from 10.47 to 2.10 Å and the closest same-seed pair from 1.30 to 0.31 Å; the kernel swap brought D11's to 0.34 Å and D10's to 0.41 Å. On D7 (spread 0.25 Å) a partial collapse is not a meaningful failure.
- Replace "exactly two seeds and five samples" with the seeds and sample count declared for the row's class.
  - *Reason:* C2 and C8 allow raising the sample count; a fixed count would contradict them.
- Make every gate report pass, fail or a declared not-applicable, and name the evaluator's current silent passes as fixture cases: missing ranking-score and pLDDT differences default to 0.0, guidance and ligand gates are skipped when their entries are absent, the spread gate is skipped when `off`'s spread is 0, and contact frequency returns no value when `off` has no interchain contacts.
  - *Reason:* a general "absent gates fail" rule is easy to implement incompletely; listing the known cases makes the tests specific.
- Note that `score_sample` already computes per-type reference lDDT, reference fnonnat and per-chain-pair Fnat; `measure_row` discards them. Calibrate a gate on the minimum per-chain-pair reference Fnat, or record why it stays reported.
  - *Reason:* per-chain-pair Fnat addresses the plan's concern that an all-polymer average can hide a minority-chain interface.
- Make ipTM report-only on reference-free rows.
  - *Reason:* among development rows without a reference, only D7 has two chains, so a gate would rest on one row.
- Zn: allow case selection from H3/H5's inputs and deposited sites (never their predictions); assign equivalent Zn ions to sites after the chain assignment; add a wrong-site control alongside the moved-Zn control. Align the plan's status line with E's scope-decision fallback.
  - *Reason:* without site assignment, two swapped ions look like displacements; displacement alone is the easy control.
- Reorder the remaining work: Zn case selection first, then the fail-closed evaluator, then the partial-collapse test, then any extra flexible-row samples.
  - *Reason:* the Zn case needs user approval and an input amendment, the longest wait; the evaluator and partial-collapse work rescore existing outputs on CPU; extra samples need GPU time on six of the 14 development jobs.

## Revision 23 (2026-10-11)

The tier 2 proposal was reviewed against the development control results and the current evaluator. The numerical values in C9 remain proposals; this revision changes documentation only. It does not amend the frozen input manifest, implement gates, run stock calibration or start the optimization port.

- Require the evaluator to fail on missing held-out class rules, an incomplete ten-sample job, an unsuccessful reference mapping or an absent applicable gate. Add the per-polymer-type reference lDDT and new-interface-contact gates promised by the protocol, set ipTM's status as gated or report-only, calibrate any new limits from stock, then rerun both harmless stock contrasts and all negative controls.
  - *Reason:* the 14-of-14 pass demonstrates the draft development checks, but `tier2_limits.json` has no H1-H10 mappings, the reference ensemble calculation filters unsuccessful samples, and C9 omits new-contact and ipTM checks named earlier in the protocol. A held-out pass would not yet mean that every declared criterion ran.
- Keep the C9 numbers as provisional and add a perturbed partial-collapse control. If the ensemble gate cannot reject it at ten samples, recalibrate the statistic or sample count before freezing.
  - *Reason:* the 0.01 Å duplicate gate catches exact copies, while the 0.1-10 flexible-row spread ratio can admit an ensemble that has lost modes but retains slight variation.
- Accept D1 as a low-confidence, no-MSA path check with an explicit atom-level noise limitation. A local-geometry gate is conditional on needing that narrower claim.
  - *Reason:* D1 passes the 0.5 Å-per-coordinate jitter control because its stock bf16/fp32 contrast is larger; jitter fails on every other development row. Raising D1's paired sensitivity beyond its stock behavior would be an unsupported limit.
- Choose a Zn gate, calibrated on a distinct development-only case. The exact case needs approval and a recorded amendment to the frozen inputs; H3/H5 remain unseen. Keep P4G and D4 Mg report-only, and request a scope decision if stock Zn proves too mobile for a discriminating coordination or pocket gate.
  - *Reason:* H3/H5 contain three Zn ions, and the D4 Mg observation cannot justify an ion-placement claim for Zn. A held-out Zn limit cannot be set after opening those rows.
- Update the Step 0 status: sm_120 single-cycle tier 1 is approved, reference feasibility and existing controls have run, and the remaining work is tier 2 enforcement, partial-collapse testing and Zn calibration.
  - *Reason:* the older text still presented completed stock work as future work and described now-resolved D6/D8 controls as open.

## Revision 22 (2026-10-10)

Revision 21's acceptance changes were checked against the stock data already on gnode2 (item 7 outputs and calibration job 48816). Four points did not hold up as written. This revision changes documentation only; no hook, reference fetch or GPU run was started.

- Replace the nearest-sample ensemble gate with a distribution-based one: lDDT or DockQ against frozen deposited structures where they exist, intrinsic ensemble statistics (mean pLDDT, ipTM, within-ensemble spread) elsewhere. Raise the sample count for a class if ten samples cannot separate negative controls from stock; a row that still lacks a discriminating gate stays unresolved. Keep nearest-sample distances as a report.
  - *Reason:* stock bf16 versus fp32 already gives nearest-sample distances of 13.8 Å (D3), 15.6 Å (D8), 11.0 Å (D10) and 18 Å (D11, D12) at ten samples. Any limit stock passes would be near 20 Å and could not detect a displaced chain, so revision 21's rule would have left about half the rows unresolved.
- Make lDDT (all polymer, per type, interface) the main same-outcome gate; gate the p95 per-atom error only on D4-D7, whose stock same-outcome pairs sit at most 0.34 Å from each other. Add a concrete interface-contact metric (Fnat-style preserved and new interchain residue contacts within 5 Å).
  - *Reason:* in same-outcome pairs the p95 error tracks anchor RMSD (3.7-4.2 Å at 1.7-1.94 Å on D1 and D8), so near the cutoff a p95 limit only restates the 2.0 Å classifier.
- Give D1 a low-confidence class: count floor 7, an lDDT limit from stock, no tight per-atom check.
  - *Reason:* D1's per-atom pLDDT never exceeds 0.62 (mean 0.49), so no confident region exists for the region-specific check revision 21 suggested, and one stock pair sat at 1.94 Å, which makes a 9-of-10 floor likely to fail a harmless candidate.
- Set confidence limits per row from stock, and record the D9 reconciliation: a near-constant per-seed `ranking_score` offset (about 0.022 on seed 101, about 0.005 on seed 102) with near-identical structures. The earlier 0.004 stable-row claim omitted D9.
  - *Reason:* guided rows carry seed-dependent confidence offsets; tier 1's fixed-input head test isolates head arithmetic, so the end-to-end gate is coarser.
- Require controls before freezing: the stock `torch`-triangle-kernel run must pass every tier 2 gate, used only for validation, and constructed negative controls (displaced minority chain, collapsed ensemble, ligand moved out of its pocket) must fail.
  - *Reason:* without candidates, these are the only evidence that limits are neither too strict nor vacuous; using the positive control to set limits would make it circular.
- Call the 2.0 Å cutoff a proposal frozen with the other limits.
  - *Reason:* revision 21's plan called it frozen while the protocol froze nothing in the criteria.

## Revision 21 (2026-10-10)

Step 0 has frozen inputs and stock contrasts, but acceptance limits are still proposals. This revision updates the plan and `qualification_protocol.md` after a critical review of those proposals. It changes documentation only; no tier 1 hook, ported kernel or GPU run was started.

- Keep the hard-requirement, tier 1 and tier 2 structure, but add a bidirectional ensemble gate that applies even when no paired samples are same-outcome. Require per-polymer-type lDDT and a separate interface-contact gate where meaningful.
  - *Reason:* D10 and H5 had `N_min = 0`, so every pair could switch and all same-outcome structure checks could be skipped. An all-polymer score can also hide a minority-chain or polymer-type regression.
- Keep 2.0 Å as an outcome classifier, replace the blanket `stock count minus 3` count rule with proposed class floors, and demote the twice-the-stock-maximum `E` and `L` values to diagnostic drafts pending stock-only review.
  - *Reason:* the 2.0 Å cutoff separates the observed stable and switched pairs, but the old count rule allowed three switches on rows with 10 of 10 stock matches and all ten on D10. D1's draft 8.5 Å p95 allowance and 0.76 lDDT floor are too weak to assume they preserve its structure. The former 1% binomial calculation relied on unverified sample independence.
- Make the three-configuration tier 1 stock calibration the next Step 0 task. Require artifact-neutral private hooks and fixed confidence-head inputs; set tensor-specific floors and ceilings after stock measurements, not by an automatic multiple or fp32 cap. Check A100 stock limits separately when that GPU is provisioned.
  - *Reason:* tier 1 has no measured limits yet. A zero stock contrast can make an automatic cap impossible for a numerically different valid kernel, while a large contrast can make it too permissive; sm_120 measurements cannot establish sm_80 behavior. Freeze each GPU's limits before tuning on that GPU so the unprovisioned A100 does not block unrelated sm_120 work.
- Retain 6OI, ATR, NAG and PCG as gated ligands and P4G and D4 Mg as report-only. Flag Zn as a separate pre-freeze scope decision; reconcile the D9 confidence-score claim with the Step 0 report.
  - *Reason:* D4 Mg mobility does not establish that the three Zn ions in H3/H5 are disposable, and the proposed 0.004 stable-row ranking-score maximum conflicts with D9's reported 0.02-0.03 contrast.
- Keep H1/H2/H3/H4/H7 flexible, H6 stable by analogy with D9, and H10 tied to D4. Give H8/H9 a provisional uncertain-outcome class and H5 a mandatory ensemble gate; define held-out draws as two seeds with five samples each.
  - *Reason:* H8's distinct strands and H9's lack of RNA MSA weaken the stable-outcome analogy to D7/D6. Count limits are undefined without a fixed held-out sample count, and an epitope count floor of zero cannot be the only structure gate.

## Revision 20 (2026-10-10)

Step 0 is under way. This revision records the user's decision on when a held-out draw is required, matching the qualification protocol (commit `7c832df`). No porting code was written.

- Require a held-out draw only for dispatch rows chosen or changed in this port, including every sm_120 row. Rows whose kernel source, configuration and shape range are unchanged from a kit record for that card qualify on the development set and are recorded as development-set passes. A failing kit row is rejected to stock without a held-out draw; changing it to retry makes it port-chosen. Updated the recorded decisions, the step 4 gate and the validation matrix.
  - *Reason:* the held-out set protects against selecting configurations on development results. A row taken unchanged from the kit was not selected on this port's results, so a held-out draw adds little for it, while the kit has no sm_120 records and its A100 records came from a different stack, so rows re-chosen here still need it.

## Revision 19 (2026-10-10)

Revision 18 was checked against the augmentation code: rotations come from SciPy's `Rotation.random` driven by the NumPy generator and are converted to fp32 on the CPU, and the initial Gaussian can be captured before scaling by splitting one expression without changing the arithmetic. These edits guard the same-commit baseline and tidy item 7. No porting code was written.

- Add an `off` drift check on every candidate commit: compare the fresh `off` with the Step 0 baseline (or the latest recorded rebaseline) on the same GPU and stack, and stop qualification on any difference until it is classified as an intended, rebaselined change or a leak to fix.
  - *Reason:* with candidate and `off` from the same commit, a change that leaks into the `off` path moves both and is invisible to the paired comparison. Nothing previously enforced that `off` stays the unmodified 1.2 path.
- Require a fresh bf16 `off` per candidate, but reuse the Step 0 fp32 run while that `off` stays byte-identical to the Step 0 baseline; rerun the fp32 contrast only after a legitimate rebaseline.
  - *Reason:* revision 18 asked for both stock precision runs on every candidate commit, but the fp32 run is context only.
- Name NumPy and SciPy, alongside the driver, CUDA, torch, Triton and cuEquivariance, in the trace's software-stack fingerprint.
  - *Reason:* those two libraries determine the rotation draws.
- Split item 7 into native contrast, conditional replay, reporting and later-candidate sub-bullets, with the reason for conditional replay moved to the end of its bullet.
  - *Reason:* the revision 17 insertion put a reason sentence mid-paragraph, ahead of further replay requirements.

## Revision 18 (2026-10-10)

Revision 17 left three Step 0 trace assumptions too loose. This revision fixes them before the harness is built. No porting code was written.

- Treat cross-precision translation equality as a measured result, not a guarantee. The translation draw is fp32 in both paths, but it shares a torch generator with an earlier dtype-dependent Gaussian.
  - *Reason:* the earlier draw may advance the generator differently, so matching translation dtype alone does not establish matching values.
- Capture and replay Gaussians before their schedule, translation or guidance scaling. Capture the augmentation rotation matrix before it is applied to coordinates.
  - *Reason:* a scaled perturbation is not the raw random input. Replaying it at the wrong boundary can apply the scale twice or mix arithmetic differences into the trace.
- Require a fresh `off` baseline from the candidate's source commit on the same target GPU and software stack for numerical qualification. On a stock trace-manifest mismatch, inspect the source, hooks and manifest, then rerun the self-tests and stock traces. Rebaseline after upstream merges.
  - *Reason:* a valid manifest proves record coverage, not equivalence of model code across commits; a mismatch can reflect changed sampler behavior as well as a harness defect.

## Revision 17 (2026-10-10)

Revision 16 was checked against the sampler, guidance, rigid-contact and epitope code. Every random draw count is fixed by configuration, so the manifest can be derived from settings. These edits narrow the replay diagnostic and keep the manifest valid over time. No porting code was written.

- Run the native traced bf16-versus-fp32 contrast first. Build the replay diagnostic only if draws differ and an isolated precision estimate is still needed. Capture replay draws from the fp32 run, and keep the replay injector a private in-process callback.
  - *Reason:* if the native traces match after casting, the native contrast already shares random inputs, and a second injection hook would add work without information. Capturing from fp32 gives the fp32 path exact values and the bf16 path a rounded copy.
- Name RNG call sites with stable symbolic names rather than file and line numbers, and record each trace's source commit. Compare a candidate only with an `off` trace from the same commit or with a re-validated manifest.
  - *Reason:* line-based keys would change with unrelated edits and break comparison with older `off` traces.
- Treat a stock `off` mismatch with the manifest as a harness defect, and re-validate the manifest against fresh stock traces after each upstream merge (added to "Keeping up with upstream").
  - *Reason:* the manifest re-encodes the sampler's control flow, so an upstream change that adds or moves a draw would make it stale.

## Revision 16 (2026-10-10)

Revision 15 closed gaps in the random-input trace, but two comparisons still needed clearer limits. No porting code was written.

- Treat native bf16-versus-fp32 stock runs as a cross-precision contrast. If their traced draws differ, the distance is descriptive and cannot inform numerical limits as a precision estimate. When an isolated precision estimate is needed, capture one complete draw set and replay it into both stock precision paths, casting at each draw site. Keep replay out of timing.
  - *Reason:* `torch.randn` uses the requested dtype in the sampler, so equal seeds need not give equal stochastic inputs. A native run may measure both a noise change and an arithmetic change.
- Replace the trace's total-count check with an expected key manifest and per-RNG-site, per-kind counts. Check skipped and repeated guidance draws, reject duplicate keys, and self-test that omitting a hook or duplicating a record fails.
  - *Reason:* one missing record and one duplicate can leave the total unchanged. Qualification needs to detect both before calling samples paired.

## Revision 15 (2026-10-10)

Revision 14 was checked against `biotite.structure.lddt` in biotite 1.4.0 and the augmentation code. The LDDT parameters and per-sample random draws are as the plan assumes; these edits close gaps in the trace and LDDT rules. No porting code was written.

- Require the random-input trace to have an expected record count derived from the sampling settings; a missing or extra record fails. Implement the trace as a private in-process callback, not an environment variable, CLI option or config key.
  - *Reason:* a rewritten sampler that stops calling the hook would produce an empty or partial trace that trivially agrees, and an exposed switch would add public surface outside step 7.
- Store a raw hash and a comparison-dtype hash for each traced draw, and use the comparison-dtype hashes for the bf16-versus-fp32 comparison of later draws.
  - *Reason:* raw hashes of bf16 and fp32 draws always differ, so revision 14's trace could not compare any draw after the first between precisions.
- Have the harness count LDDT contact pairs itself: zero pairs is inconclusive, while a NaN with a nonzero count fails.
  - *Reason:* Biotite returns NaN for zero contacts (confirmed), the same value an invalid input produces, so the score alone cannot distinguish the two outcomes.
- Run reproducibility and artifact checks with the trace off, using traced runs only for random-input identity.
  - *Reason:* hashing copies each draw to the host and synchronizes every step, which can hide the stream races the step 0 diagnostics look for.

## Revision 14 (2026-10-10)

Revision 13 made sample-index pairing and initial-noise checks explicit. This revision completes the sampled-structure comparison rules and makes the new nucleic-acid metrics testable. No porting code was written.

- Add an opt-in trace of every sampler random input, keyed by original sample, diffusion step and chunk. Step 0 checks traced stock repeatability and confirms that tracing leaves prediction artifacts unchanged; candidates are checked during qualification. For bf16 versus fp32, record raw initial-noise hashes before schedule scaling and compare values after casting the fp32 draw to bf16.
  - *Reason:* matching initial noise does not prove that later per-step noise, augmentation or guidance draws match. A changed draw sequence invalidates the paired-sample gate even when the seed and first draw agree.
- Specify symmetric, heavy-atom polymer LDDT, explicit contact settings, per-polymer-type scores and treatment of zero-contact and non-finite cases. Change the two-tier gate from protein to polymer LDDT.
  - *Reason:* Biotite's default directional score counts contacts in the reference but can miss new candidate contacts. An undefined atom subset or missing coordinate could otherwise be mistaken for a valid pass.
- Require step 0 parity fixtures for explicit RDKit ligand maps: carboxylate and nitro versus automatic matching, a covalent attachment that cannot be remapped, and the match-cap boundary.
  - *Reason:* supplying `map` bypasses automatic match generation. The manual enumeration and conjugated-terminal-group equivalence must be checked before their result is used as a hard gate.
- Require DNA-only and RNA-only qualification inputs, with an explicit user decision if stock cannot run either class.
  - *Reason:* revision 13 defined a non-protein anchor and nucleic-acid LDDT but left both optional in the validation matrix, so neither path was guaranteed to be exercised.

## Revision 13 (2026-10-10)

Revision 12 was checked against the sampler, dumper and CIF writer code and the RDKit `CalcRMS` signature in `opendde-dev`. These edits affect the step 0 protocol; no porting code was written.

- Require a check that paired sample indices received identical noise: matching `N_sample` and `sample_diffusion_chunk_size`, plus a step 0 test that hashes each sample's initial noise. A candidate that changes chunking is compared with `off` at the same chunk size. Step 0 item 7 records whether the bf16 and fp32 runs share initial noise.
  - *Reason:* the sampler's generator is seeded per run and isolated from global RNG use, but each chunk draws all its noise before the next, and a different shape gives different GPU random values. Sample-index pairing alone does not guarantee identical noise, and the default 5 samples with chunk size 5 would hide the problem. Noise is drawn in the dtype of `s_inputs`, so the precision comparison may not share noise.
- Screen symmetry completeness (ligand atom matches and equivalent-chain permutations) when the inputs and held-out pool are frozen; exclude inputs that cannot be searched exhaustively or declare their resolution test in advance. An undeclared inconclusive held-out result counts as a failure.
  - *Reason:* both counts depend only on the input, so they can be known before any run. Discovering an inconclusive case during a single-use held-out run left its outcome undefined, and a retry would use up the set.
- Have the harness enumerate ligand matches with `GetSubstructMatches(..., maxMatches=cap + 1)` and pass them to `CalcRMS` through `map`, applying the same conjugated-terminal-group equivalence that `CalcRMS` uses by default.
  - *Reason:* this gives an explicit truncation flag. Without the equivalence, a harmless carboxylate oxygen swap would appear as an error of about 2 Å.
- Take ligand bonds from the in-process featurized `atom_array` or from CIFs written with the existing `include_bonds=True`, instead of reconstructing graphs from input and CIF atom names.
  - *Reason:* the harness already reads in-process prediction arrays, and reusing OpenDDE's own bonds avoids a separate, error-prone atom correspondence.
- Generalize the alignment anchor beyond protein (a declared polymer anchor for inputs without protein), and extend LDDT to DNA and RNA when the qualification set includes them.
  - *Reason:* the protein-only anchor and LDDT left nucleic-acid-only inputs undefined.

## Revision 12 (2026-10-10)

Revision 11 named comparison tools but left three ways for qualification to compare the wrong structures or overstate a result. These edits affect the step 0 protocol; no porting code was written.

- Pair `off` and candidate samples by their original sampling index, using unsorted prediction arrays or `sorted_by_ranking_score=False` in the qualification harness. Check the normal ranked outputs and ranking agreement separately.
  - *Reason:* OpenDDE ranks `_sample_*.cif` filenames by confidence by default. A change in ranking can make matching filenames refer to different draws even when the seeds match.
- Build RDKit ligand graphs from the input and attach predicted coordinates through verified atom correspondence. Define the protein alignment and covalent attachment constraints before calling `CalcRMS`.
  - *Reason:* prediction CIFs omit bonds, and ligand pose RMSD without a shared protein frame or a valid atom graph does not measure the intended difference.
- Check whether the RDKit symmetry search is complete; record its `maxMatches` cap and test repeatability on the pinned stack. A capped RMSD above the limit is inconclusive unless another pose-sensitive check resolves it. Keep centroid and contact checks as supplements, not substitutes.
  - *Reason:* `CalcRMS` returns no truncation flag. A partial symmetry search can miss a lower-RMSD mapping, while a ligand can rotate within one site without materially changing its centroid or residue-contact set.
- Restrict equivalent-chain permutations to mappings that preserve covalent bonds and input-specific constraints. Use one declared whole-complex assignment for all metrics; an approximate assignment does not hard-pass an ambiguous case.
  - *Reason:* OpenDDE's `sym_id` is a copy ordinal, not proof of interchangeability. Choosing the best permutation separately for each metric can hide a coherent structural difference.

## Revision 11 (2026-10-10)

Revision 10 was checked against the validation-matrix layout and the ligand and chain comparison tooling available in the pinned stack. These are planning changes; no porting code was written.

- **Fixed a formatting defect introduced in revision 9.** The held-out-set rules had been merged into the comparison-tooling bullet list, with two unrelated sentences sharing one bullet. They are now a separate paragraph. No content changed.
  - *Reason:* the merged list made the held-out procedure read as a tooling choice.
- **Named the default ligand tools.**
  - RDKit's `rdMolAlign.CalcRMS` gives symmetry-aware, deterministic ligand RMSD without realignment, with its `maxMatches` cap recorded. A comparison that reaches the cap is non-exhaustive and is decided by the mapping-free checks.
  - The mapping-free checks are ligand centroid displacement and the set of protein residues in contact with the ligand.
  - *Reasons:*
    - Revision 10 ruled out the random permutation helper but did not name a replacement.
    - RDKit is already pinned (2025.09.3), and `CalcRMS` was confirmed in `opendde-dev`.
    - Mapping-free checks catch a ligand that moves to the wrong site whatever the atom mapping.
- **Named the default chain mapping:** an exhaustive search over permutations of chains that share an entity, using OpenDDE's entity and symmetry identifiers, with a recorded greedy fallback if there are too many copies.
  - *Reason:* paired predictions of the same input share chain identifiers, so only equivalent copies can swap, and the search is small for typical inputs.
- **DockQ or OpenStructure remain the fallback** if these defaults are not enough.

## Revision 10 (2026-10-10)

Revision 9 was checked against the public Python runner and the ligand-permutation helper. These edits clarify step 0; no porting code was written.

- The pre-fix `fast_layernorm` identity claim now applies to stock CLI baselines run on the default CUDA stream. OpenDDE does not switch streams internally, but a Python caller can invoke the runner under an external stream. The targeted graph-capture and side-stream checks now record actual pre-fix behavior instead of requiring a failure that has not been verified on the target stack. The corrected kernel must pass those checks where supported.
- The structure-comparison protocol can no longer use `get_substructure_perms` unchanged as a pass/fail oracle. The helper caps permutation enumeration and calls `np.random.choice` when more than `MaxMatches` remain. Qualification needs a deterministic, exhaustive mapping where tractable; a capped comparison must be reported as non-exhaustive and checked another way before accepting or rejecting the candidate.
- Numerical limits are fixed before candidate qualification outputs are inspected, after stock calibration. Step 0's representative Triton compile-and-run check remains a compatibility smoke test and cannot set those limits. The previous phrase, "before any optimized output has been seen," conflicted with that smoke test.

## Revision 9 (2026-10-10)

Revision 8 was checked against OpenDDE's stream usage, the LayerNorm kernel source, the available structure-comparison tooling and the cache guard. These are planning changes; no porting code was written.

- **Pre-fix `fast_layernorm` baselines are identity references.** Revision 8's branch for a baseline "affected by the bug" is removed.
  - *Reason:* nothing in `opendde/` or `runner/` creates or switches CUDA streams, including Fold-CP, so every current prediction runs on the default stream and cannot trigger the bug. It becomes reachable only once the port adds graph capture or a side stream.
- **The targeted stream checks now fail reliably before the fix.** One captures LayerNorm forward and backward in a CUDA graph; one runs a side-stream race with a deliberate delay such as `torch.cuda._sleep`.
  - *Reason:* launching onto the legacy default stream during capture fails outright. Without a delay, a race test may pass by chance.
- **The LayerNorm fix covers all 28 launch sites, with a check that none is left without a stream argument.**
  - *Reason:* the backward is half-converted. `HostLayerNormGradient` passes the current stream to `LayerNormParamGradStep2`, while `LayerNormParamGradStep1` and others in the same function still use the default stream. On a side stream that is a race inside a single operation.
- **Structure-comparison tooling is chosen in step 0.** Start from biotite's `lddt` (biotite 1.4 is already a dependency; confirmed in `opendde-dev`) and OpenDDE's ligand-symmetry enumeration, plus a small chain-mapping step. Add DockQ or OpenStructure only if needed, and keep any of them as test or dev dependencies.
  - *Reason:* OpenDDE has no structure-comparison metrics (`opendde/metrics/` has only `clash.py`), and the plan did not say how its paired-structure criteria would be computed.
- **The trunk-cache guard holds tensor references, or weak references, instead of `id()` values. Its design is frozen at this level.**
  - *Reasons:*
    - Python reuses `id()` values after an object is freed, so a replacement tensor could match.
    - The cache is opt-in and off during qualification, so further elaboration has little value.
- **Absolute limits are set after the step 0 calibration and before any optimized output is seen.**
  - *Reason:* revision 8 said only "before tuning", which left open whether limits could be set before calibration data existed.
- **Status note: the plan is ready for step 0.** The next revision should come from step 0 results, or from a defect that would change what step 0 does.
  - *Reason:* recent rounds have mostly added edge cases. The questions that decide scope can only be answered by measurement.

## Revision 8 (2026-10-09)

Revision 7 was checked against the confidence-head inputs, dtype handling, LayerNorm launches and the proposed cache guard. These are planning changes; no porting code was written.

- The numerical gate now has two pass/fail tiers with absolute limits. The bf16-versus-fp32 comparison remains in the report but no longer acts as a hard ceiling. A candidate close to stock bf16 can be slightly farther from fp32 and fail the old rule; a large stock precision gap could also permit too much drift. The runner disables autocast for fp32, so that run is an empirical comparison, not a pure arithmetic-precision control.
- Trunk and distogram outputs are checked on identical inputs. The confidence head gets a separate fixed-coordinate test, while its end-to-end outputs are assessed after sampling. The head consumes `x_pred_coords`, so treating its normal output as independent of sampling would misattribute a diffusion change to the head.
- Paired-structure criteria now fix alignment, equivalent-chain assignment and ligand atom-symmetry rules before comparison. They also check ligand pose or interface contacts separately from internal ligand geometry. Otherwise equivalent structures can appear different, and a misplaced ligand can appear acceptable when its internal geometry is intact.
- The LayerNorm stream fix gains non-default-stream forward and backward checks, plus graph capture where supported. Ordinary prediction baselines can execute `fast_layernorm` on the default stream without exposing the bug. If the pre-fix bug affected a baseline, that output is diagnostic, not an identity oracle; the corrected kernel is checked against the torch reference and end-to-end criteria instead.
- The trunk-cache guard now checks the named tensor set and object identities alongside tensor metadata and version counters, and invalidates identity on runner-managed loads or replacements. A replacement tensor can have the same version counter as the old one. Raw storage writes cannot be certified by that counter, so the cache remains limited to a runner-managed inference lifecycle and is bypassed when external mutation cannot be ruled out.

## Revision 7 (2026-10-09)

Revision 6 was reviewed against the sampler's behavior, the LayerNorm and trunk-cache code paths, checkpoint loading and gradient handling. These are planning changes; no porting code was written.

- **`fast` and `big` acceptance criteria now have three tiers, plus a precision yardstick measured in step 0 (new item 7):**
  1. tight limits on the trunk and confidence outputs;
  2. robust paired-sample metrics for sampled structures (a high-percentile coordinate error and paired LDDT, not the maximum);
  3. no further from stock `--dtype fp32` than stock bf16 is.
  - *Reasons:*
    - Under the deterministic recipe, fixed-seed `off` repeatability is expected to be zero, so revision 6's limits based on it would reject any rounding change.
    - Even with identical injected noise, a 200-step diffusion sampler can send a flexible loop, a ligand pose or a chain-symmetry choice to a different outcome after a small numerical change. Paired errors are therefore heavy-tailed.
    - The bf16-versus-fp32 drift is a measured, non-zero reference for how much stock moves when only arithmetic precision changes.
- **Held-out rule fixed.** A held-out failure rejects the candidate. A held-out set that has been seen is used up, and a retry draws a fresh set from a reserved pool. Used sets are logged, and the pool is topped up before tuning continues, not after seeing results.
  - *Reason:* revision 6 said to "reuse it as a holdout" after it failed, which contradicts holding it out. Once seen, it no longer gives an unbiased test.
- **Pre-fix regression baselines now exercise the changed code.** Step 0 item 6 adds `LAYERNORM_TYPE=fast_layernorm` baselines on one GPU and under Fold-CP, plus a single-GPU trunk-cache miss-then-hit baseline. The step 2 gate names which baselines each fix is checked against.
  - *Reasons:*
    - The LayerNorm stream fix changes a kernel that only runs with `fast_layernorm`, so the default-LayerNorm baselines never reach it.
    - `_trunk_with_cache` skips the cache under Fold-CP, so the Fold-CP regression cannot test the cache fix.
- **Trunk-cache identity mechanisms made concrete:**
  - When the cache is enabled, read the checkpoint once, hash those bytes and load from the same buffer.
  - Detect post-load weight changes, including a later `load_state_dict`, with PyTorch's per-tensor `_version` counters.
  - *Reasons:*
    - Hashing the file and then reading it again leaves a window for it to change between the two reads.
    - Revision 6 required detecting in-place weight changes without saying how. The version counters are cheap and cover both cases.
- **Gradient-rule confirmation recorded under risks.**
  - Normal inference runs under `@torch.no_grad()`, so grad mode is checked first to keep the common path cheap.
  - Parameters keep `requires_grad=True`, so with `rho>0` every optimized denoiser site falls back to stock.
  - *Reason:* this confirms revision 6's parameter-gradient rule works with the current runner, and states its cost and effect.
- **Qualification protocol to live in its own document** (for example `porting_docs/qualification_protocol.md`), started in step 0. The plan keeps the strategy.
  - *Reason:* the plan now mixes strategy with test procedure, which makes both harder to follow.

## Revision 6 (2026-10-09)

Revision 5 was reviewed against checkpoint loading, kernel fallback and the proposed validation gates. These changes refine the plan; no porting code was written.

- **Tie trunk-cache identity to a strictly loaded, unchanged model, not file metadata alone.** Non-strict loads, direct model construction and later weight changes bypass the cache unless the current state has a verified identity. Test same-metadata checkpoint replacement. A new process must rehash without a trusted content-addressed manifest.
  - *Reason:* path, size and modification time can stay the same when checkpoint contents change; a non-strict load or post-load mutation can also leave the model holding weights the file digest does not describe. Any case could make the disk cache return another model's trunk result.
- **Check parameter gradients at optimized dispatch sites.** A kernel without backward support falls back when gradients are required for inputs or module parameters; parameter-only backward gets a test.
  - *Reason:* checking inputs alone misses a call with constant inputs and trainable weights.
- **Set `fast` and `big` acceptance criteria before tuning.** Name the inputs, pair seeds with `off`, set coordinate, confidence and guidance limits from fixed-seed repeatability and domain-specific criteria, and reserve a held-out qualification set. Seed-to-seed variation is context, not a license for large paired-seed errors.
  - *Reason:* choosing limits after inspecting optimized outputs would make the qualification circular. Size labels alone do not establish numerical safety across the input classes in the matrix.
- **Move the Fold-CP `off` regression to the step 2 gate.** Save a two-GPU stock baseline in step 0, before either unconditional bug fix, and compare after each fix. Hold the gate if two GPUs are unavailable.
  - *Reason:* the plan promises unchanged Fold-CP `off` behavior, and testing each fix separately makes a failure easier to attribute.

## Revision 5 (2026-10-09)

Records two user decisions; no change in scope or approach.

- **Added a "Recorded decisions" table at the top of the plan.**
  - *Reason:* decisions were spread across sections and the changelog. One table makes the current state easy to find.
- **Fold-CP rejection marked as approved.** `--optimization_mode` other than `off` is rejected under Fold-CP for this port.
  - *Reason:* the user approved the policy proposed in revision 4.
- **The two bug fixes stay in the fork.** The LayerNorm stream fix and the trunk-cache identity fix will not be offered to `aurekaresearch/OpenDDE` for now. The upstream-sync section now says they may need re-applying when upstream changes those files.
  - *Reason:* the user's decision. It replaces revision 4's open question about upstreaming them.

## Revision 4 (2026-10-09)

Revision 3 was reviewed against OpenDDE's determinism settings, phase timers, LayerNorm modules, CLI and config entry points, and Fold-CP handling. These are planning changes; no optimization code was written.

- **Step 0 checks that stock output is reproducible before `exact` relies on it.** Run `off` twice under the deterministic recipe and compare artifacts byte for byte. If the default `auto` triangle kernels are not reproducible, retry with `torch` kernels and use that configuration for `exact` comparisons. Stop and ask if neither is reproducible. Step 0 also lists the files that count as prediction artifacts and confirms they contain no run-specific values.
  - *Reason:* `--deterministic true` sets `torch.use_deterministic_algorithms` and `CUBLAS_WORKSPACE_CONFIG`, which do not govern cuEquivariance's custom kernels. Byte identity between `exact` and `off` cannot hold if `off` does not reproduce itself.
- **The `exact` contract now says that default, non-deterministic runs match `off` in distribution, not bit for bit.** The user docs must say so.
  - *Reason:* the byte-identity guarantee is tested only under the deterministic recipe.
- **Step 0 adds opt-in synchronized phase timing, used for all baseline and comparison numbers.** One-time compilation is reported separately.
  - *Reason:* `_main_inference_loop` times phases with `time.time()` and never synchronizes the GPU, so queued work is charged to the next phase that forces a host wait. Exact changes remove such waits, so the built-in timers would show time moving between phases without any real speedup.
- **Fold-CP policy: `--optimization_mode` other than `off` is rejected under Fold-CP in this port.** A Fold-CP regression is added to the validation matrix: stock Fold-CP output is unchanged by the port. Lifting the restriction for individual exact changes is a later, recorded decision.
  - *Reason:* revision 3 did not say what the modes do under Fold-CP. The kit's levers step aside or refuse there, and none was qualified with Fold-CP's sharded state. A mode that silently ran stock would mislead users and benchmarks.
- **Pair-row LayerNorm dispatch moved to `OpenFoldLayerNorm.forward`,** for the bf16 path at width 384. It is measured against PyTorch's bf16 `layer_norm`, qualified last, and skipped unless the profile shows LayerNorm time matters.
  - *Reason:* with the default `LAYERNORM_TYPE=torch`, the model builds `OpenFoldLayerNorm`. The kit's `ln_core` wraps only `FusedLayerNorm.forward`, so it would never run. Its speed and accuracy were measured against the fused extension, a different reference.
- **Trunk-cache key adds the GPU compute capability and device name, the torch, CUDA, Triton and cuEquivariance versions, and the OpenDDE version.** The checkpoint digest is computed only when `OPENDDE_TRUNK_CACHE` is set, and memoized by real path, size and modification time. Previously the plan said to compute it at every load.
  - *Reasons:*
    - Stock trunk results differ between GPU models and library versions, so a shared or long-lived cache could serve results from another platform.
    - Hashing the 2.6 GB checkpoint at every model load would slow every run, including runs that never use the cache. Checking against the release digest also needs a hash.
- **Explicit-option detection is concrete.**
  - The CLI uses Click's `ctx.get_parameter_source()`, so option defaults and help text are unchanged.
  - Python entry points use an unset sentinel.
  - Dotted config overrides on the `runner.inference.run()` path, and kernel or precision environment variables, count as explicit.
  - *Reason:* Click (8.5 in the dev environment) already records whether a value came from the command line, the environment or a default. Replacing the defaults with sentinels would change help output and callers for no gain. Revision 3 did not cover the dotted-override entry point.
- **New section: keeping up with upstream OpenDDE.**
  - Edits to upstream files are kept to thin dispatch calls, config plumbing and the two bug fixes, with the implementations in new files.
  - A list of restated or depended-on upstream functions is refreshed before each upstream merge, and the gates are re-run afterwards.
  - Offering the two bug fixes to `aurekaresearch/OpenDDE` is listed as a user decision.
  - *Reason:* the fork will need future upstream releases, and inline changes spread across model files raise merge cost. Revision 3 had no strategy for this.
- **Step 0 names the environment `opendde-dev`** and records that the production `opendde` environment must not be modified.
  - *Reason:* the user's instruction when the environment was created.
- **Risks and work records updated** for stock reproducibility, unsynchronized timers, the Fold-CP restriction, the upstream bug-fix decision and the upstream dependency list.

## Revision 3 (2026-10-09)

Revision 2 was reviewed against OpenDDE's CLI, LayerNorm construction, and trunk-cache code. These are planning changes only; no environment, GPU job or optimization code was created.

- **Keep global LayerNorm selection fixed during the first port.** Qualify pair-row LayerNorm through local dispatch. Consider mode-dependent whole-model `fast_layernorm` only after measuring a benefit and designing a choice passed before model construction.
  - *Reason:* `triangular/layers.py` reads `LAYERNORM_TYPE` at import and constructs concrete modules. Changing `--optimization_mode` later cannot switch them, and changing the environment after import would give a misleading result.
- **Track which overlapping options the user actually supplied.** Distinguish an explicit `--triatt_kernel auto` from the CLI's implicit `auto`, and use an unset sentinel for omitted direct Python arguments before resolving them to their existing effective defaults.
  - *Reason:* the current CLI and Python entry points pass default values into the configuration even when the user supplied no option. The resulting value alone cannot enforce the plan's rule that explicit choices win.
- **Make the trunk-cache identity cover the checkpoint and effective trunk computation.** Version the format, audit the trunk's inputs, include the loaded checkpoint's content identity and selected numerical path, and test cross-mode isolation. Keep the disk cache off during numerical qualification.
  - *Reason:* the current key covers tensor input features and `N_cycle`, while the CLI accepts a custom checkpoint. Different model weights or kernel decisions could otherwise reuse a result. A verified release digest or a digest computed once at load avoids hashing the checkpoint on every call.
- **Decide on `big` using the actual A100 capacity.** Measure `fast` peak memory on the provisioned A100 and sm_120, and define an acceptable memory margin before the decision.
  - *Reason:* revision 2's 96/80 GB rule excluded a possible 40 GB A100, even though the implementation section proposed 40 GB thresholds. A profile on sm_120 cannot answer whether a 40 GB A100 needs offload.
- **Separate stock failure from optional kernel failure in step 0.** Diagnose a failed stock baseline. Record failures in optional cuEquivariance, `fast_layernorm` or kit Triton paths and continue work that has a stock fallback. Propose a dependency change only after testing an alternative for a required path.
  - *Reason:* PyTorch 2.7 cu128 already includes Blackwell and Triton 3.3 support. The unknown is whether these particular optimized paths work and help on sm_120; one failed optional path need not halt the entire port.
- **Narrow the 1.1.1-to-1.2.0 file comparison to relevant model and inference paths.** Acknowledge the other changed files.
  - *Reason:* revision 2 said "only these files changed," although the full commit diff also includes `runner/template_search.py`, docs, tests and examples.
- **Keep exact changes opt-in for this port.** Record candidates for default promotion as a separate later decision that updates the `off` contract, docs and tests.
  - *Reason:* revision 2 proposed both an unchanged `off` baseline and making qualified exact changes the default. Promoting them during qualification would remove the stable A/B comparator.

## Revision 2 (2026-10-09)

A critical review of revision 1 against the OpenDDE 1.1.1 → 1.2.0 diff, the kit's lever implementations, and the shared core's kernel inventory. Only static reading was done; nothing was run on a GPU.

### Added

- **Step 0: environment check and stock baseline on sm_120.** It verifies that the pinned Triton 3.3.1 and cuEquivariance 0.10 work on sm_120 and that the `fast_layernorm` extension builds there. It also records a per-phase profile that decides which groups are worth porting.
  - *Reason:* the kit's results come from H100 on a different software stack and OpenDDE version. Revision 1 recorded a baseline but did not use it to set scope or order.
- **LayerNorm stream fix as an unconditional bug fix.** `layer_norm_cuda_kernel.cu` launches with `<<<grid, block>>>` on the legacy default stream.
  - *Reason:* this is wrong under any non-default stream or CUDA graph capture, whatever the optimization mode. It is a correctness fix, not an optimization.
- **Trunk cache key extended with the mode and every numerics-changing setting.**
  - *Reason:* `_trunk_with_cache` hashes only input features and `N_cycle`. A trunk computed under `fast` would be served to `off` or `exact`, which breaks the `exact` contract.
- **Gradient rule for the sampler.** Fused or low-precision sampler paths must use the stock module when gradients are needed.
  - *Reason:* with guidance `rho > 0`, `tfg/engine.py` backpropagates through the denoiser. The kit's sampler kernels have no gradient check, and the engine's `allow_unused=True` path turns a missing gradient into zeros, so guidance could weaken silently. The default `rho = 0` hides the problem.
- **Validation-matrix and test cases:**
  - generic guidance with both `rho = 0` and `rho > 0`;
  - trunk cache off for benchmarks;
  - timing variance on the Max-Q (power-limited) cards;
  - a driver check on the A100 VM.
- **Scope rows** for the A100 CUDA kernels (A100 only, built from source) and for excluded sm_90a and other-model kernels.
- **Optional change after qualification:** make byte-identical changes the default code path once they pass both GPU gates.
  - *Reason:* two code paths for identical results are an ongoing maintenance cost.
- **New risks:** Triton 3.3.1 untested on sm_120; trunk-cache mode mixing; guidance gradients through fused kernels.

### Changed

- **The numerical contract now names its baseline:** bf16, TF32 off, `LAYERNORM_TYPE=torch`, triangle kernels `auto`. `exact` may not change the LayerNorm, precision, TF32 or kernel choice; switching to `fast_layernorm` is a `fast`-mode decision.
  - *Reason:* the kit's "stock" used `fast_layernorm` and TF32 on. Its LayerNorm levers do nothing under the 1.2 default, and its fast sampler kernels were tuned against TF32-on stock. Revision 1 mentioned BF16 and TF32 but not the LayerNorm difference.
- **Software stack pinned to `torch==2.7.1` (cu128 build), `triton==3.3.1` and `cuequivariance==0.10.0` on both GPUs.** Revision 1 said "cu128-or-newer PyTorch".
  - *Reason:* these are OpenDDE's `pyproject.toml` pins. torch 2.7.1 has a cu128 build for sm_120, and the kit's A100 tables and prebuilds are keyed to torch 2.7.1 and Triton 3.3. Nothing exists for torch 2.8 or 2.9. Changing a pin is a public-interface decision, left to the user if step 0 shows it is needed.
- **Exact-tier kernels deferred.** Revision 1 implicitly carried them with the Pairformer fast path.
  - *Reason:* the bit-exact sampler attention is an sm_90a-only binary. A100 has bit-exact triangle rows only for some shapes, and sm_120 has none. All records were taken on cu126 and do not carry over to cu128. On the target GPUs, `exact` comes down to the scheduling and data-path changes.
- **Kernel dispatch: small per-site tables instead of the shared core's provider interfaces.**
  - *Reason:* the transitive closure is about 82k lines of Python, 31 MB of tables and 148 MB of binaries. Most of it serves other models or sm_90a and cannot be reached on A100 or sm_120.
- **Kernel qualification order:** Triton kernels on sm_120, then the same Triton kernels on sm_80, then A100 CUDA kernels built from source with the cu128 toolkit, kept only if faster.
  - *Reasons:*
    - No kit kernel has an sm_120 binary, and only one sm_120 Triton configuration was ever measured (on Triton 3.6).
    - The A100 triangle-attention CUDA kernel is sized for 163 KB of shared memory; sm_120 allows about 99 KB.
    - The kit's sm_80 triangle-multiplication cubins need driver R580+, which a from-source build avoids.
- **`big` mode is conditional on the baseline profile.** It was unconditional in revision 1.
  - *Reason:* the kit's offload threshold was 1,400 tokens on 80 GB cards. On a 96 GB card the offload may rarely trigger at target input sizes. Also noted: gnode2's default 29 GB of host RAM per GPU limits pinned pair state.
- **CLI flag renamed from `--optimization-mode` to `--optimization_mode`.**
  - *Reason:* existing OpenDDE flags use underscores (`--triatt_kernel`, `--enable_tf32`).
- **Detail added to the 1.1.1 → 1.2.0 comparison:** exact list of changed files, confirmation that no lever replaces `sample_diffusion` outright, and a list of stock function copies inside the kit that would drift.
- **Optional exact items now state their trade-offs:** prefetch, skipped cache flushes, the allocator setting and graph replay.
- **CUDA graph replay must have its safe token window measured on each GPU.**
  - *Reason:* the kit found A100 replay was not bit-exact above 1,024 tokens.
- **Licensing made specific:**
  - Anthropic's Apache-2.0 copyright line;
  - Protenix notices for the FlashPairformer-derived Triton kernels;
  - PyTorch BSD-3 only if the exact LayerNorm kernel is carried;
  - FlashAttention-3 and CUTLASS obligations avoided by excluding the sm_90a units.
- **Test porting:** carry the kit's behavior tests, not its overlay-mechanics tests.
  - *Reason:* some of the latter check stock line numbers that 1.2 already moved.
- **Step 6 records the `rowpair_tp` and trunk-cache conflict.**
  - *Reason:* ranks would share one cache key while holding different row shards.
- **The overlay-infrastructure exclusion now also names** the `ACTIVE`/`LEVER` banner protocol and the exit-code-3 refusal semantics. Fallbacks use OpenDDE's logger instead.

- **Weights location on gnode2.** Weights now come from gnode2's existing shared, read-only OpenDDE weights store instead of a copy under the user's `/data` directory. Only outputs go there.
  - *Reason:* the store already holds the released `opendde.pt`, and its size and SHA-256 match the 1.2.0 manifest (checked 2026-10-09). It also holds the `common/` data files. A private copy would duplicate about 3.3 GB in a `/data` quota that is already mostly used, and could drift from the shared copy.

### Removed

- The size reference to the "whole 305 MB `opt_core` tree" in the recommendation. The figure was correct for the whole tree, but it was not the right measure of what a port would bring in. It is replaced by the transitive inventory of what the OpenDDE kit actually imports, and the size of a selection limited to A100 and sm_120.

## Revision 1

Initial plan, committed in `7f8b3ef`.
