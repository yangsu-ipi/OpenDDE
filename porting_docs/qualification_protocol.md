# Qualification protocol

Status: **draft for approval. Nothing here is frozen yet.** The plan (`inline_optimizations_plan.md`, step 0 gate) requires the qualification inputs, paired seeds, sample-index pairing, symmetry-completeness screen, acceptance criteria, guidance-success criteria and reserved held-out pool to be frozen before any `fast` kernel is tuned. This first section proposes the inputs, seeds and held-out pool. The numerical contract, two-tier limits, comparison rules and held-out log will be added as the remaining step 0 work lands.

## Proposed development set

Used for tuning and development comparisons against `off`. All inputs come from the repository's examples except D7, which is synthetic. Runs use the default `--sample 5 --step 200 --cycle 10`, bf16, TF32 off and torch LayerNorm unless a row says otherwise. MSA and template paths are the precomputed files that ship with each example, so no search runs.

| ID | Input | Covers | Size | Flags beyond the defaults |
|---|---|---|---|---|
| D1 | `examples/cofold/protein_200.json` | protein only, small, no MSA | 200 tokens | `--use_msa false` |
| D2 | `7r6r` from `examples/cofold/example.json` | protein–DNA, MSA, small | 245 tokens | `--use_msa true` |
| D3 | `examples/tfg/1a14/1a14_unconstrained.json` | three protein chains, paired MSA, medium | 612 tokens | `--use_msa true` |
| D4 | `7wux` from `examples/cofold/example.json` | heterotetramer with ligands and ions, MSA, large | 1,218 tokens | `--use_msa true` |
| D5 | `examples/cofold/template/example_9fm7.json` | template and MSA, protein dimer with ligands | 582 residues plus ligands | `--use_msa true --use_template true` |
| D6 | `examples/cofold/rna_msa/example_9gmw_2.json` | RNA only, RNA MSA | 86 nt | `--use_rna_msa true` |
| D7 | synthetic DNA duplex: `CGCGAATTCGCG` (Dickerson–Drew dodecamer), `dnaSequence` with `count: 2` | DNA only, non-protein anchor | 24 nt | `--use_msa false` |
| D8 | `examples/tfg/9lh2/9lh2_unconstrained.json` | covalent bond (N-glycan NAG on Asn), ligands | 704 residues plus glycans | `--use_msa true` |
| D9 | `examples/tfg/1a14/1a14_contact.json` | rigid-contact guidance | 612 tokens | `--use_tfg_guidance true --enable_tf32 false` |
| D10 | `examples/tfg/1a14/1a14_pocket.json` | epitope guidance | 612 tokens | as D9 |
| D11 | D9 input with `OPENDDE_RIGID_CONTACT=off` | generic TFG potential guidance, `rho=0` | 612 tokens | as D9 |
| D12 | D11 with `--sample_diffusion.guidance.rho` > 0 | guidance gradients through the denoiser | 612 tokens | value to choose (see open questions) |
| D13 | D2 and D3 in one input file | multi-job, multi-seed | — | `--seeds 101,102` |

- **Paired seeds:** 101 and 102 for every development row. Pair `off` and candidate samples by original sample index (see the plan's validation matrix), not by output filename.
- **Sizes:** D1 and D2 are small, D3, D5 and D8 medium, D4 large.
- **Repeat runs:** item 3 already showed D1 and D4 byte-reproducible under the deterministic recipe, and D9 as well for guided runs.

## Proposed reserved held-out pool

Never used for tuning or inspected during development. Each qualification draws a held-out set from this pool once per GPU, and a used set is retired (plan, validation matrix). Seeds for held-out runs come from 201–299 and are also single-use.

| ID | Input | Covers | Size |
|---|---|---|---|
| H1 | `examples/cofold/input.json` (`user_test`) | protein only, no MSA, medium | 693 residues |
| H2 | `7pzb` from `examples/cofold/example.json` | protein–DNA with ligands, MSA | 554 residues plus ligands |
| H3 | `examples/tfg/9sat/9sat_unconstrained.json` | three protein chains with Zn ions, MSA | 707 residues |
| H4 | `examples/tfg/9xqn/9xqn_unconstrained.json` | three protein chains, MSA | 650 residues |
| H5 | `examples/tfg/9sat/9sat_pocket.json` | epitope guidance | 707 residues |
| H6 | `examples/tfg/9xqn/9xqn_contact.json` | rigid-contact guidance | 650 residues |
| H7 | `examples/tfg/9lh2/9lh2_contact.json` | covalent glycan with guidance | 704 residues |
| H8 | synthetic DNA duplex, a sequence different from D7 | DNA only | about 24 nt |
| H9 | synthetic RNA hairpin, no MSA | RNA only | about 30 nt |

## Open questions for approval

1. **Large held-out input.** The repository has only one input above 1,000 tokens (D4). Either supply an external large complex, for example a PDB entry of 1,000–1,500 tokens with a precomputed MSA, or accept new seeds on D4 as the large held-out case. New seeds on a tuned input are a weaker test.
2. **`rho` for D12.** The default is `rho=0`. A value is needed that exercises gradients through the denoiser without making the sampler diverge.
3. **Synthetic inputs.** D7, H8 and H9 are synthetic nucleic acids with no reference structure. They test the non-protein anchor and polymer LDDT between paired samples, not accuracy. Accept them, or name deposited DNA-only and RNA-only entries instead.
4. **Fold-CP.** The Fold-CP regression input is D1 on two GPUs (plan, validation matrix), independent of this set.
