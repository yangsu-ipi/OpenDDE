# Qualification protocol

Status: **inputs frozen on 2026-10-10 (user approval).** The qualification inputs, paired seeds, sample-index pairing, reserved held-out pool and symmetry-completeness screen below are fixed; `scripts/qualification/inputs/manifest.json` and `digests.json` record them. The plan (`inline_optimizations_plan.md`, step 0 gate) also requires the acceptance criteria and guidance-success criteria to be frozen before any `fast` kernel is tuned; those, with the numerical contract, comparison rules and held-out log, will be added as the remaining step 0 work lands.

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

Never used for tuning or inspected during development. A qualification that needs a held-out draw (see the next section) draws a set from this pool once per GPU, and a used set is retired (plan, validation matrix). Seeds for held-out runs come from 201–299 and are also single-use.

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

- **Files.** `scripts/qualification/inputs/manifest.json` lists every row; `digests.json` holds the SHA-256 of each input JSON and each MSA or template file it references (46 files). Run `python scripts/qualification/freeze_inputs.py check` from the repository root before any qualification run; a changed, missing or added file stops qualification until the change is reviewed. Template structures and CCD data come from the shared, read-only `$OPENDDE_ROOT_DIR` store and are not digested here; the plan's store check covers only the weights.
- **Seeds and pairing.** Development seeds are 101 and 102. Held-out seeds come from 201–299, are single-use and are recorded in the held-out log. Pair `off` and candidate samples by original sample index, with `sorted_by_ranking_score=false`.
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

## Notes

- **Fold-CP.** The Fold-CP regression input is D1 on two GPUs (plan, validation matrix), independent of this set.
