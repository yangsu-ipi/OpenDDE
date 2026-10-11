# Qualification protocol

Status: **inputs frozen on 2026-10-10 (user approval); acceptance criteria not frozen.** The qualification inputs, paired development seeds, sample-index pairing, reserved held-out pool and symmetry-completeness screen below are fixed; `scripts/qualification/inputs/manifest.json` and `digests.json` record them. The next Step 0 tasks are stock-only: the reference-structure feasibility check and tier 1 calibration with private hooks (section E). The plan requires tier 1, tier 2 and domain limits to be frozen for a target GPU before a `fast` kernel is tuned there. No held-out output is inspected to set a limit.

## Development set

Used for tuning and development comparisons against `off`. All inputs come from the repository's examples except D7, which is synthetic. Every row's input file, job name, flags, environment and config overrides are in `scripts/qualification/inputs/manifest.json`; rows that use one job of a multi-job example, and the synthetic inputs, have their own files in that directory. Runs use the default `--sample 5 --step 200 --cycle 10`, bf16, TF32 off and torch LayerNorm unless a row says otherwise. MSA and template paths are the precomputed files that ship with each example, so no search runs.

| ID | Input | Covers | Size | Flags beyond the defaults |
|---|---|---|---|---|
| D1 | `examples/cofold/protein_200.json` | protein only, small, no MSA | 200 tokens | `--use_msa false` |
| D2 | `7r6r` from `examples/cofold/example.json` (`inputs/d2_7r6r.json`) | protein–DNA, MSA, small | 245 tokens | `--use_msa true` |
| D3 | `examples/tfg/1a14/1a14_unconstrained.json` | three protein chains, paired MSA, medium | 612 tokens | `--use_msa true` |
| D4 | `7wux` from `examples/cofold/example.json` (`inputs/d4_7wux.json`) | heterotetramer with ligands and ions, MSA, large | 1,218 tokens | `--use_msa true` |
| D5 | `examples/cofold/template/example_9fm7.json` | template and MSA, protein dimer with ligands | 582 residues plus ligands | `--use_msa true --use_template true` |
| D6 | `examples/cofold/rna_msa/example_9gmw_2.json` | RNA only, RNA MSA | 86 nt | `--use_msa true --use_rna_msa true` (1.2 requires both) |
| D7 | synthetic DNA duplex: `CGCGAATTCGCG` (Dickerson–Drew dodecamer), `dnaSequence` with `count: 2` | DNA only, non-protein anchor | 24 nt | `--use_msa false` |
| D8 | `examples/tfg/9lh2/9lh2_unconstrained.json` | covalent bond (N-glycan NAG on Asn), ligands | 704 residues plus glycans | `--use_msa true` |
| D9 | `examples/tfg/1a14/1a14_contact.json` | rigid-contact guidance | 612 tokens | `--use_tfg_guidance true --enable_tf32 false` |
| D10 | `examples/tfg/1a14/1a14_pocket.json` | epitope guidance | 612 tokens | as D9 |
| D11 | D9 input with `OPENDDE_RIGID_CONTACT=off` | generic TFG potential guidance, `rho=0` (scoped; see below) | 612 tokens | as D9 |
| D12 | D11 with `rho=1.0` | guidance gradients through the denoiser (scoped; see below) | 612 tokens | as D11, run through `scripts/qualification/run_opendde.py --set sample_diffusion.guidance.rho=1.0` |
| D13 | D2 and D3 in one input file (`inputs/d13_7r6r_1a14.json`) | multi-job, multi-seed | — | `--seeds 101,102` |

- **Paired seeds:** 101 and 102 for every development row. Pair `off` and candidate samples by original sample index (see the plan's validation matrix), not by output filename.
- **Sizes:** D1 and D2 are small, D3, D5 and D8 medium, D4 large.
- **Repeat runs:** item 3 already showed D1 and D4 byte-reproducible under the deterministic recipe, and D9 as well for guided runs.

### Scope of D11 and D12

User decision, 2026-10-10. `pred` has no option for `rho`, so D12 sets it through the qualification wrapper's `--set`, which edits the config `pred` builds before validation. Paired D11 and D12 runs also set `sorted_by_ranking_score=false`, so output `sample_k` is original sample index k.

`rho=1.0` was chosen from a stock sweep on sm_120 (step 0 results, rho sweep). Against `rho=0`, it moved paired samples by 0.67–1.30 Å Cα RMSD and pulled the restrained Cα–Cα contacts from the 8 Å limit in to 3.5–6.5 Å, with every coordinate finite and ranking scores unchanged. At `rho=0.1` the shift was 0.06–0.34 Å, which may not exceed the differences a `fast` candidate is allowed, so a silently zeroed gradient could pass; `rho=0.01` barely moved the samples.

Stock generic guidance breaks the backbone on this input, at `rho=0` as well: with rigid contact off, the atom-level contact restraint pulls single atoms out of the chain, leaving consecutive Cα atoms up to 17 Å apart at and next to the restrained residues (L29–31, L91–94, N247–248, H54–56). `rho=1.0` widens these gaps to up to 35 Å. The checked unguided D3 and rigid-contact D9 samples show no gap. So D11 and D12 test only that the generic guidance path matches `off`, not structure quality:
- Structure comparisons for D11 and D12 exclude the residues named in the input's contact constraints and the residues within 3 positions of them in the same chain. The exclusion is fixed from the input, not from any prediction.
- D12 adds a gradient check: for each paired sample, every constrained Cα–Cα distance of the candidate must stay within a limit, frozen with the other criteria, of the `off` sample's distance. A zeroed or partial gradient leaves these distances near the 8 Å limit, as at `rho=0`.
- Structure quality and backbone geometry are not judged on D11 or D12.

## Reserved held-out pool

Never used for tuning or inspected during development. A qualification that needs a held-out draw (see the next section) draws a set from this pool once per GPU, and a used set is retired (plan, validation matrix). Each held-out row uses two single-use seeds drawn from 201-299, with five samples per seed, so its count gates apply to ten pairs. Record the seeds before running `off` or the candidate. Reason: `N_min` is a count of ten pairs; without a fixed held-out sample count, that gate has no defined meaning.

| ID | Input | Covers | Size |
|---|---|---|---|
| H1 | `examples/cofold/input.json` (`user_test`) | protein only, no MSA, medium | 693 residues |
| H2 | `7pzb` from `examples/cofold/example.json` (`inputs/h2_7pzb.json`) | protein–DNA with ligands, MSA | 554 residues plus ligands |
| H3 | `examples/tfg/9sat/9sat_unconstrained.json` | three protein chains with Zn ions, MSA | 707 residues |
| H4 | `examples/tfg/9xqn/9xqn_unconstrained.json` | three protein chains, MSA | 650 residues |
| H5 | `examples/tfg/9sat/9sat_pocket.json` | epitope guidance | 707 residues |
| H6 | `examples/tfg/9xqn/9xqn_contact.json` | rigid-contact guidance | 650 residues |
| H7 | `examples/tfg/9lh2/9lh2_contact.json` | covalent glycan with guidance | 704 residues |
| H8 | synthetic DNA duplex, two distinct strands: `GTCAGCTTACGGAT` and its reverse complement `ATCCGTAAGCTGAC` (`inputs/h8_dna_duplex.json`) | DNA only, no equivalent chains | 28 nt |
| H9 | synthetic RNA hairpin `GGCAUCGAGUCACUUCGGUGACUCGAUGCC`: 13 bp stem closed by C–G around a UUCG loop (`inputs/h9_rna_hairpin.json`), run with `--use_msa false --use_rna_msa false` | RNA only, no MSA | 30 nt |
| H10 | D4 (`7wux`) with held-out seeds only | large | 1,218 tokens |

H10 is the large held-out case (user decision, 2026-10-10): the repository has no second input above 1,000 tokens. Because D4 is tuned on, H10 tests only new seeds, not a new input, and is a weaker check than the other held-out rows. Its seeds come from the 201–299 held-out range like every other held-out run, and are never used for development.

D7, H8 and H9 are synthetic nucleic acids, accepted by the user on 2026-10-10. They have no reference structure, so they test the non-protein anchor and polymer LDDT between paired `off` and candidate samples, not prediction accuracy. The H8 and H9 sequences were approved by the user on 2026-10-10: H8 has two different strands, unlike D7's self-complementary pair, and no palindromic stretch of 6 or more bases; H9 is a stable, well-characterized C(UUCG)G hairpin. H8 and H9 themselves are not run before use, because a held-out input must stay unseen until it is used. Instead D7, D6 and D6 without its RNA MSA (`--use_msa false --use_rna_msa false`, as H9 will run) were checked through stock `off` on sm_120: all three run, give finite coordinates and repeat byte for byte under the deterministic recipe (step 0 results, nucleic-acid-only runnability). Both input classes stay in.

## Freeze record

- **Files.** `scripts/qualification/inputs/manifest.json` lists every row; `digests.json` holds the SHA-256 of each input JSON and each MSA or template file it references (46 files). Run `python scripts/qualification/freeze_inputs.py check` from the repository root before any qualification run; a changed, missing or added file stops qualification until the change is reviewed. CCD data come from the shared, read-only `$OPENDDE_ROOT_DIR` store and are not digested here; the plan's store check covers only the weights.
- **D5 template data.** Added 2026-10-10 (user decision, option 1). D5's four template mmCIF files (5ocm, 3zgy, 5a9s, 5a9r, the hits OpenDDE selects under its 2021-09-30 template date cutoff) were fetched once from PDBe and are kept in a read-only directory named by the host variable `OPENDDE_QUAL_DATA`; `external_digests.json` holds their SHA-256, checked with `freeze_inputs.py check --external` on the GPU host. D5 runs with `fetch_remote=false`, a per-run template cache and kalign 2.04 named by `OPENDDE_QUAL_KALIGN` (on gnode2, the shared colabfold env's binary). A host without these, such as the A100 VM, must provide the same four files (checked by digest) and kalign 2.04 before running D5.
- **Reference structures.** Added 2026-10-11 for the tier 2 ensemble gate (revision 22). The deposited mmCIFs 7r6r, 1a14, 7wux, 9fm7, 9gmw, 9lh2, 7pzb, 9sat and 9xqn were fetched once from PDBe into the read-only `${OPENDDE_QUAL_DATA}/references/`; manifest rows name theirs under `reference`, and `external_digests.json` holds their SHA-256 (13 external files in all). Fetching a held-out row's reference reads only the deposited file; no held-out input was run.
- **Seeds and pairing.** Development seeds are 101 and 102. Each held-out row draws two seeds from 201-299; seeds are single-use and recorded in the held-out log. Both routes use five samples per seed. Pair `off` and candidate samples by original sample index, with `sorted_by_ranking_score=false`.
- **Symmetry-completeness screen.** `scripts/qualification/symmetry_screen.py` counts, from the inputs alone, the whole-complex equivalent-chain assignments (chains with identical entity content permuted together, keeping inter-chain covalent bonds and constraint-named chains fixed) and each ligand's heavy-atom automorphisms with covalent attachment atoms fixed, with bond orders and with elements only. Run on gnode2 (job 48760); every row is exhaustively searchable:

| Rows | Chain assignments | Ligand matchings (bond orders / elements only) |
|---|---|---|
| D1, D2, D3, D6, D9–D13, H1, H4, H6, H8, H9 | 1 | no ligands |
| D4, H10 (`7wux`) | 48 of 48 | P4G 2 / 2; 6OI 2 / 12; Mg 1 |
| D5 (`9fm7`) | 4 of 4 | ATR 4 / 72 |
| D7 | 2 of 2 | no ligands |
| D8 (`9lh2`, covalent NAG) | 4 of 8 | NAG 1 / 1 (attachment fixed) |
| H2 (`7pzb`) | 16 of 16 | PCG 1 / 2 |
| H3, H5 (`9sat`) | 6 of 6 | Zn 1 |
| H7 (`9lh2`, guided) | 1 of 8 | NAG 1 / 1 (attachment fixed) |

  The largest searches are 48 chain assignments and 72 ligand matchings, far below the limits of 100,000 assignments and RDKit's 1,000,000 matches, so no input needs a declared resolution test for a non-exhaustive search. The comparison harness must still enumerate all assignments and matchings itself and flag truncation (plan, validation matrix).

## When a held-out draw is required

User decision, 2026-10-10. This narrows the plan's validation-matrix rule that every candidate is qualified once per GPU on a held-out set. The held-out set guards against selection on development results, so it is required only where this port makes a selection.

- **Kit-unchanged rows.** A dispatch row qualifies on the development set alone when the kit ships a record for that compute capability and the row keeps the kit's kernel source, configuration and shape range unchanged. Nothing about such a row is chosen from this port's results, so a held-out draw adds little. The pass is recorded as a development-set pass, a weaker claim than a held-out pass.
- **Port-chosen rows.** Any row chosen or changed in this port needs a held-out draw on that GPU. This includes every sm_120 row, since the kit has no sm_120 records, and any A100 row whose kernel, configuration or shape range differs from the kit's. Narrowing a kit row's shape range after seeing development results counts as a change.
- **Rejecting a kit row.** If a kit-unchanged row fails on the development set, it is rejected and that site keeps the stock operation. Rejection needs no held-out draw. Changing the row to retry it makes it port-chosen.
- **End-to-end candidates.** A candidate needs a held-out draw on a GPU if any row it enables on that GPU is port-chosen. In practice this means every sm_120 candidate.
- **Criteria.** Both routes use this protocol's frozen criteria against the 1.2 `off` baseline. The kit's own limits and numerical claims are not used: they were set for OpenDDE 1.1.1 on H100 with torch 2.7.1+cu126 and a `fast_layernorm`, TF32-on baseline.
- **Record.** The GPU qualification tables state, for each row, whether it was kit-unchanged or port-chosen and which route qualified it. The held-out log records the draws used.

## Comparison rules and tooling

Fixed on 2026-10-10, before any candidate output exists. The plan's validation matrix sets the requirements; this section records how they are met. Tooling lives in `scripts/qualification/`: `run_rows.py` runs manifest rows through `run_opendde.py`, `compare_structures.py` compares two runs, `qual_structures.py` holds the rules below, and `trace_tools.py` compares random-input traces. `tests/test_qualification_compare.py` covers the rules (10 tests, passing on gnode2).

- **Pairing.** Both runs use the same manifest row, seeds, sample count and chunk size, and write outputs with `sorted_by_ranking_score=false`, so `sample_k` is original sample index k in both. The comparison fails if the two runs' sample sets differ. Candidate qualification also compares every traced random input with `off` (`trace_tools.py compare --hash raw`); a mismatch makes the pair unpaired, and it is not scored against paired-sample limits.
- **Atom correspondence.** Each prediction CIF must list exactly the atoms of the featurized input, `SampleDictToFeatures(job).get_atom_array()`, in the same order, with the same chain, residue number, atom name and element; otherwise the comparison fails. Ligand bonds come from that featurized array. This is the plan's fallback of rebuilding the graph from the input, and the exact correspondence check is its required check. On gnode2 the CIF order matched the featurized order exactly for `7wux`.
- **Equivalent-chain assignment.** Candidates are the valid assignments from the symmetry screen: permutations within groups of identical entities that keep inter-chain covalent bonds and constraint-named chains fixed. The anchor is the protein Cα atoms, or the nucleic-acid C4′ atoms when an input has no protein. The chosen assignment minimizes anchor RMSD after superposing the candidate's anchor onto the reference's. Exact ties are broken by the mean squared deviation of all non-anchor heavy atoms under that superposition, with each ligand at its best symmetry map, and then by enumeration order with identity first. One assignment serves every metric for that sample pair.
- **Superposition.** One Kabsch fit of the anchor atoms, applied to the whole candidate complex. lDDT does not use it; RMSDs, ligand poses, pockets and contact distances do.
- **lDDT.** `biotite.structure.lddt` on heavy polymer atoms, with both structures as `AtomArray`s and `symmetric=True`, `aggregation="all"`, `inclusion_radius=15`, `distance_bins=(0.5, 1.0, 2.0, 4.0)`, `exclude_same_residue=True` and `exclude_same_chain=False`. Scores are reported for all polymer atoms and for each polymer type present, with atoms and partners restricted to that type. The harness counts atom pairs within 15 Å in either structure under the same masks; zero pairs makes the score inconclusive. Non-finite coordinates fail the case.
- **Ligand pose.** Each ligand chain's heavy-atom graph is built with bond orders, aromatic bonds as one type. Covalent attachment atoms are held fixed by giving each a unique pseudo-element. Conjugated terminal O and N atoms are made equivalent with RDKit's `symmetrizeConjugatedTerminalGroups` pattern. All automorphisms are enumerated with `GetSubstructMatches(uniquify=False, useChirality=False, maxMatches=cap + 1)`, cap 1,000,000, and passed to `CalcRMS` through `map`; more than `cap` matches marks the ligand inconclusive. Single-atom ligands (ions) use their displacement. The tests check the following before this rule was frozen: explicit-map results equal RDKit's automatic `CalcRMS` on carboxylate, benzoic acid, nitro and phosphate fixtures with swapped terminal oxygens; a fixed attachment atom stops a symmetric swap; the cap boundary is exact (24 maps pass at cap 24, cap 23 truncates); and enumeration repeats identically on the pinned RDKit 2025.09.3.
- **Ligand placement.** Centroid displacement after the anchor fit, and the Jaccard overlap of polymer residues with a heavy atom within 4.0 Å of the ligand in each structure. These supplement ligand RMSD and cannot by themselves resolve an inconclusive pose.
- **Constraint distances.** For inputs with contact constraints, each constrained atom pair's distance in both structures, whether each lies within its bounds, and their difference. For D12 this is the gradient check.
- **D11 and D12 exclusion.** Rows with `comparison.exclude_constraint_window` in the manifest (D11 and D12, window 3) drop the constraint residues and their same-chain neighbours within 3 positions from anchor, RMSD and lDDT masks.
- **Confidence and ranking.** Absolute differences of each sample's scalar confidence values (`ranking_score`, `ptm`, `iptm`, `plddt`, `gpde`, `has_clash` where present), and, per seed, whether the top-ranked sample is the same and Kendall's tau of the ranking scores.

## Calibration context (step 0 item 7)

Measured on sm_120 before any candidate exists (step 0 results, item 7; table there). Stock `off`, bf16 versus fp32, with identical random inputs on every development row (all draws are fp32 and every trace matched), so the contrast isolates numerical precision. Stock seed-to-seed variation is measured alongside. Neither grants an automatic error allowance; they inform the absolute limits, which are still to be set.

- **Stable inputs** (D1, D2, D4 polymer, D5, D6, D7, D9): bf16 versus fp32 stays within 1.94 Å anchor RMSD and lDDT 0.81 or higher, mostly far tighter (D4, D6, D7 ≤ 0.34 Å), and below seed-to-seed variation.
- **Flexible multi-chain inputs** (D3, D8, D10, D11, D12): precision alone moves about half the paired samples to a different chain arrangement, up to 16–20 Å anchor RMSD, as far as seed-to-seed variation. Paired-sample limits on these inputs need a form that tolerates such switches, such as limits on medians or on the fraction of pairs beyond a threshold, rather than per-sample maxima.
- **Ligands:** the specific ligand 6OI moved at most 0.93 Å; the PEG fragment P4G and Mg²⁺ ions in D4 jump sites under a precision change and between seeds; D8's NAG follows its chain. Pose limits should distinguish specific ligands from additives and ions.
- **D12 gradient check:** constrained distances differ by up to 3.7 Å between precisions, about as much as switching `rho` from 1.0 to 0, so the check needs an aggregate form (for example the mean constrained distance over all samples and seeds) rather than a per-pair limit.
- **Confidence:** `ranking_score` differences are at most 0.002 on D1, D2, D4, D6 and D7, 0.004 on D5, 0.02-0.03 on D9 and D10, and up to 0.17 on flexible rows. The top-ranked sample changes on some flexible rows. Reconciled for D9 (revision 22): every seed-101 pair differs by 0.021-0.024 and every seed-102 pair by 0.001-0.005, while the structures differ by only 0.1-0.9 Å anchor RMSD. That is a near-constant confidence offset per seed, not structural drift. The earlier draft's 0.004 stable-row maximum omitted D9 and was wrong.
- **D5** (templates) is stable: 0.04 Å anchor RMSD and ATR within 0.45 Å between precisions.

## Acceptance criteria (revised proposal, not frozen)

Revised 2026-10-10 from stock calibration only (item 7 and job 48816); no candidate exists. Applies to `fast` and `big` against `off` on the same commit, GPU and stack. Every row uses two seeds with five samples each, giving ten paired samples per job. The hard requirements, tier 1 and tier 2 must all pass; one tier cannot compensate for a failure in another. Keep sm_120 and sm_80 stock calibration records separate. Do not transfer an sm_120 numerical limit to A100 without checking it on stock A100 runs before A100 candidate qualification.

### A. Hard requirements (every row)

1. Every coordinate and confidence value is finite.
2. Every traced random input equals `off`'s under the raw hash (`trace_tools.py compare --hash raw`), so pairs are truly paired.
3. Two candidate runs under the deterministic recipe give byte-identical prediction artifacts.
4. `off` passes its drift check against the step 0 baseline (plan, validation matrix).
5. Every comparison is conclusive. The symmetry screen showed every input exhaustively searchable, so an inconclusive result is a failure; no resolution test is declared.

### B. Tier 1: non-sampling outputs

Compared on identical inputs, per row and seed:
- **Trunk:** the single and pair representations the trunk passes to the distogram and confidence stages, by relative Frobenius error and by the 99.9th percentile of absolute error divided by the tensor's RMS.
- **Distogram:** the largest absolute difference in bin probabilities, and the fraction of token pairs whose most likely bin changes.
- **Confidence head on fixed inputs:** the candidate's confidence head is fed `off`'s recorded `s_inputs`, single and pair representations and `x_pred_coords`. Compared: the largest absolute difference in pLDDT, PAE, PDE and resolved probabilities, the per-atom expected pLDDT, and `ptm`, `iptm` and `ranking_score`.

Limits are not set yet: no stock measurement of these tensors exists. Do this stock-only calibration next, before tuning a `fast` kernel. Capture the development rows in three stock configurations: bf16 with the default cuEquivariance triangle kernels, bf16 with stock `torch` triangle kernels, and fp32. Use the same recorded `off` inputs for every fixed-input confidence-head comparison; a native fp32 run with different head inputs is a separate contrast. Add two wrapper-private hooks in `run_opendde.py`, one to capture tensors and one to substitute recorded head inputs. Check that enabling the capture hook leaves stock prediction artifacts byte-identical, and keep the hooks off timing runs. Budget about three GPU-hours on sm_120 as an estimate, not a guaranteed runtime. Do not run held-out inputs for calibration.

Set each tier 1 limit from the stock measurements before inspecting a candidate. Record the kernel-swap and precision contrasts, but do not make twice the swap maximum or the fp32 difference an automatic limit: a zero or tiny contrast could demand impossible bit identity, while a large contrast could permit a weak gate. Predeclare a nonzero numerical floor and a meaningful ceiling for each tensor quantity, with a rule for a zero reference RMS. Treat changed distogram argmax bins near a tie separately from changes with a material top-two probability margin; freeze that margin and its limit from stock results. Reason: the argmax fraction alone can change under negligible probability differences.

### C. Tier 2: sampled structures

A pair is **same-outcome** when its anchor RMSD is at most 2.0 Å. This is a proposed classification cutoff, frozen together with the other limits: every bf16-versus-fp32 pair on the stable rows was within 1.94 Å, while switched pairs on flexible rows were about 9-20 Å apart. The classification alone does not establish that a structure is acceptable.

**Row classes** (development; held-out classes in D):
- **Stable:** D2, D4, D5, D6, D7, D9 and D13's `7r6r` job. All ten stock precision pairs were same-outcome, with margin: anchor RMSD at most 1.17 Å.
- **Low-confidence:** D1. Its per-atom pLDDT never exceeds 0.62 (mean 0.49; a synthetic sequence without MSA), so it has no confident region to anchor a tight structure gate, and one stock pair sat at 1.94 Å, just under the cutoff. D1 stays for coverage of the no-MSA path, small shapes and Fold-CP.
- **Flexible:** D3 and D13's `1a14` job, D8, D11, D12. About half the stock precision pairs switched.
- **Epitope:** D10. One of ten stock precision pairs matched.

1. **Outcome agreement.** Proposed count floors: 9 of 10 for stable rows, 7 of 10 for the low-confidence class, 2 of 10 for flexible rows and 0 of 10 for epitope rows. Stable rows matched 10 of 10 with margin; D1's floor allows for its pair near the cutoff. The lower floors avoid treating precision-like switches as automatic failures and never waive the ensemble, guidance or other gates. The former `stock count minus 3` rule and its 1% binomial claim are withdrawn: the ten samples share two seeds and are not independent.
2. **Ensemble agreement, on every row.** Compare the candidate's and `off`'s sample sets as distributions, not sample by sample:
   - **Rows with a deposited structure:** the distribution of lDDT (all polymer atoms, and per polymer type) of each sample against the deposited reference, plus interface metrics where the reference has interfaces (DockQ if available in the pinned tooling, otherwise the interface-contact metric in C4). Gate the shift of the median and of the lower quantile between candidate and `off`. References: D2 `7r6r`, D3 and D9-D13 `1a14`, D4 `7wux`, D5 `9fm7`, D6 `9gmw`, D8 `9lh2`; held-out H2 `7pzb`, H3 and H5 `9sat`, H4 and H6 `9xqn`, H7 `9lh2`, H10 `7wux`. Each reference mmCIF is fetched once and frozen by digest, like D5's templates. Freezing a held-out row's reference reads only the deposited file, never a held-out prediction. Atoms are matched after per-chain sequence alignment and the declared chain assignment; residues absent from the deposited structure are excluded.
   - **Rows without a deposited structure** (D1, D7; held-out H1, H8, H9): intrinsic ensemble statistics, namely the distributions of mean pLDDT and, where defined, ipTM, and the within-ensemble spread (median pairwise anchor RMSD).
   - **Limits** come from the stock precision and seed contrasts on development rows only. If ten samples cannot separate the negative controls (C8) from stock, raise the sample count for that class (for example four seeds of five) before freezing, rather than loosening the gate. A row that still has no discriminating, non-vacuous ensemble gate stays unresolved and is not qualified.
   - **Feasibility (job 48887, step 0 results).** At ten samples, stock bf16 and fp32 medians agree within 0.004 in reference lDDT and 0.010 in Fnat, while a displaced minority chain lowers them by at least 0.026 and 0.39 on D2, D3, D4, D5 and D9-D12. Draft limits, to be validated with the C8 controls: |Δ median reference lDDT| ≤ 0.01 (all polymer and per type), |Δ lowest-sample reference lDDT| ≤ 0.02 (stock at most 0.011), |Δ median Fnat| ≤ 0.05, and candidate spread between 0.1 and 10 times `off`'s (stock ratios 0.20-5.0). D8 is not resolved by this gate: stock reproduces almost none of 9lh2's interfaces (Fnat about 0.07), so the displaced-chain control does not move it. D6 (single chain) needs a different negative control. Rows without references (D1, D7) rely on spread, pLDDT distribution and their paired gates.
   - Nearest-sample distances (`compare_structures.py --set-level`) are reported only. At ten samples they cannot discriminate: stock bf16 versus fp32 already gives 13.8 Å (D3), 15.6 Å (D8), 11.0 Å (D10) and 18 Å (D11, D12).
3. **Same-outcome accuracy.** lDDT is the main local gate, because it needs no superposition: all polymer atoms, each polymer type present, and the interface (C4). The 95th-percentile per-atom error `E` is gated only on rows whose same-outcome stock pairs sit far below the cutoff (D4, D5, D6, D7: anchor RMSD at most 0.34 Å; draft `E` 1.0 Å). Elsewhere it is reported: in same-outcome pairs it tracks anchor RMSD (pairs at 1.7-1.94 Å have p95 errors of 3.7-4.2 Å on D1 and D8), so a limit there would only restate the 2.0 Å cutoff. The maximum atom error is reported, never gated alone. D1 gets an lDDT limit set from its stock contrast, without a tight per-atom check.
4. **Interface contacts**, on multichain rows in same-outcome pairs: the fraction of `off`'s interchain residue contacts (any heavy-atom pair within 5 Å) that the candidate preserves, and the fraction of the candidate's interchain contacts that `off` lacks, in the manner of DockQ's Fnat. Limits from development stock contrasts.
5. **Specific ligands**, in same-outcome pairs: the proposed symmetry-aware RMSD limit is 2.0 Å and pocket Jaccard minimum is 0.6. The gated ligands are D4's 6OI, D5's ATR, D8's NAG and H2's PCG. PCG uses these predeclared limits despite having no held-out-specific calibration; do not adjust them after H2 is opened. Report P4G and D4's Mg ions without a pose gate: both moved sites under the stock precision or seed contrast. H3 and H5's Zn ions are also reported for now, but the D4 Mg observation is not evidence that Zn placement is expendable. Before freezing, either add a development-only Zn case and a stock-calibrated coordination or pocket gate, or state explicitly that ion placement is outside the qualified claim. Keep H3 and H5 unseen in either case.
6. **Confidence.** Limits are set per row from that row's stock contrast, with a floor, because guided rows carry seed-dependent offsets (D9: about 0.022 on seed 101, about 0.005 on seed 102, with near-identical structures). Tier 1's fixed-input head test isolates arithmetic changes in the confidence head, so this end-to-end gate is a coarser check. Draft starting points: |Δ `ranking_score`| ≤ 0.01 on D2, D4, D5, D6 and D7, ≤ 0.05 on D9, ≤ 0.10 in same-outcome pairs elsewhere, and |Δ mean pLDDT| ≤ 2.0 in same-outcome pairs. Report sample ranking order without gating near-tied scores.
7. **Guidance.**
   - D9 rigid contact: all contacts within bounds in at least 9 of 10 samples; stock 10/10.
   - D10 epitope: OpenDDE's rule is met in at least as many samples as `off` minus 1; stock 10/10.
   - D11: the mean constrained distance over the 10 samples is within 0.5 Å of `off`'s; stock 8.00 Å in both precisions. The restraint holds distances at its 8 Å bound, so a success count is meaningless here.
   - D12, the gradient check: the mean constrained distance is within 1.0 Å of `off`'s and at most 7.0 Å. Stock: 5.76 Å in bf16, 6.14 Å in fp32. A zeroed gradient gives 8.0 Å, as at `rho=0`.
   - Held-out guided rows: success in at least as many samples as `off` minus 1.
8. **Controls before freezing.** Set the limits from the stock precision and seed contrasts, then validate them without any candidate:
   - **Positive control:** the stock bf16 run with `torch` triangle kernels, which the tier 1 calibration produces, is a harmless same-precision change and must pass every tier 2 gate. It validates the limits and is not used to set them; using it for both would make the check circular. If it fails, revisit the limits before freezing.
   - **Negative controls**, built from stock outputs, must fail: a displaced minority chain, a collapsed ensemble (every candidate sample replaced by one `off` sample), and a specific ligand moved out of its pocket.

Stock calibration and draft limits for the development rows. `E` and `L` are **diagnostic drafts, not pass limits**; they are reassessed with the per-type, interface and ensemble gates before the freeze. `E` applies only where marked.

| Row | Class | Proposed `N_min` | Draft `E` (p95 error) | Draft `L` (lDDT) | Stock bf16 vs fp32: same-outcome pairs, max p95 / min lDDT |
|---|---|---|---|---|---|
| D1 | low-confidence | 7 | reported | from stock, to set | 10, 4.04 Å / 0.810 |
| D2, D13 `7r6r` | stable | 9 | reported | 0.94 | 10, 1.97 Å / 0.994 |
| D3, D13 `1a14` | flexible | 2 | reported | 0.92 | 5, 1.78 Å / 0.975 |
| D4 | stable | 9 | 1.0 Å (gated) | 0.94 | 10, 0.25 Å / 0.999 |
| D5 | stable | 9 | 1.0 Å (gated) | 0.94 | 10, 0.10 Å / 0.997 |
| D6 | stable | 9 | 1.0 Å (gated) | 0.93 | 10, 0.29 Å / 0.982 |
| D7 | stable | 9 | 1.0 Å (gated) | 0.94 | 10, 0.25 Å / 0.998 |
| D8 | flexible | 2 | reported | 0.84 | 5, 4.18 Å / 0.890 |
| D9 | stable | 9 | reported | 0.92 | 10, 1.82 Å / 0.973 |
| D10 | epitope | 0 | reported | 0.92 | 1, 2.96 Å / 0.973 |
| D11 | flexible | 2 | reported | 0.94 | 5, 1.38 Å / 0.990 |
| D12 | flexible | 2 | reported | 0.94 | 5, 1.30 Å / 0.992 |

### D. Held-out rows

Held-out inputs have no calibration of their own, and running stock on them now would expose them. Freeze their class assignments and absolute limits from development data before any held-out run. Every class must pass the ensemble gate: reference-based for H2-H7 and H10, intrinsic for H1, H8 and H9.
- **Stable:** H6 (rigid contact, like D9), with proposed `N_min` 9 and draft `L` 0.92. H10 is the D4 input with new seeds, so it uses D4's final limits; it tests new seeds rather than a new large input.
- **Flexible:** H1 (693 residues, no MSA), H2 (protein-DNA with ligands, two copies of each), H3 and H4 (three chains), H7 (covalent glycan with guidance), with proposed `N_min` 2 and draft `L` 0.84.
- **Nucleic-acid outcome uncertain:** H8 (two distinct DNA strands) and H9 (RNA hairpin without RNA MSA), with proposed `N_min` 2 and draft nucleic-acid lDDT `L` 0.92. D7 is self-complementary and D6 uses RNA MSA, so their 10-of-10 stock precision match does not establish that H8 or H9 will have stable outcome counts.
- **Epitope:** H5, like D10, with proposed `N_min` 0 and draft `L` 0.84. Its ensemble and guidance gates are mandatory even if no pairs match.

### E. Work remaining before acceptance is frozen

1. **Reference feasibility check (stock only): done** (job 48887). Ten samples suffice for a reference-based ensemble gate on D2, D3, D4, D5 and D9-D13. Open: a discriminating ensemble gate for D8 (more samples, or an interface statistic that does not rely on the deposited interfaces), a negative control for single-chain D6, and confirmation of the draft limits against the C8 controls.
2. **Tier 1 calibration (stock only).** Build the wrapper-private hooks, prove capture is artifact-neutral, and run the three stock configurations on the development set. Freeze tensor-specific floors, ceilings and tie handling from stock results. The `torch`-kernel run also serves as the tier 2 positive control.
3. **Tier 2 limits.** From development stock contrasts, set the ensemble limits, per-type and interface lDDT limits, the interface-contact limits, D1's lDDT limit and per-row confidence limits. Build the negative controls and confirm that every gate rejects them and the positive control passes. If a row has no discriminating non-vacuous gate, its acceptance remains unresolved.
4. **Zn decision:** gate Zn coordination using a development-only calibration case, or state that ion placement is outside the qualified claim. P4G and D4 Mg remain report-only; H3 and H5 remain held out.
5. **Freeze per GPU.** Freeze the held-out class rules and sm_120 limits before any sm_120 candidate is tuned or qualified; do not use held-out runs to set limits. Repeat the stock calibration and freeze the applicable limits on the provisioned A100 before A100 tuning or qualification. Sm_120 work need not wait for A100 provisioning, but sm_120 results alone do not establish sm_80 limits.

## Notes

- **Fold-CP.** The Fold-CP regression input is D1 on two GPUs (plan, validation matrix), independent of this set.
