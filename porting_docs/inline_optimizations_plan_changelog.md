# Inline optimization plan changelog

Changes to `inline_optimizations_plan.md`. Newest first.

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
