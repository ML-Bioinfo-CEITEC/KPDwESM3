# Revision 2 generated knot candidates

This directory contains the 32 distinct generated protein structures used in
the Revision 2 plausibility and Foldseek screen.

## Files

- `candidate_01.pdb` through `candidate_32.pdb`: restrained-minimized structures.
- `candidates.fasta`: generated amino-acid sequences.
- `candidates.json`: full source-confidence, initial post-relaxation topology,
  minimization, and geometry metrics.
- `validation_summary.json`: compact manuscript-facing evidence, including
  Foldseek/PDB hits and independent 500-closure checks for Candidates 1 and 2.

## Processing

Candidate structures were predicted with ESM3-SM, screened for source mean
pLDDT above 0.50 and knot score above 0.50, completed with PDBFixer, and
restrained-minimized using OpenMM with the Amber14 all-atom force field.
All deposited structures pass the continuity and clash checks described in the
manuscript and retain knot score above 0.50 after minimization.

These are computational candidates, not experimentally validated proteins.

## Metric notes

- `candidates.json` reports the post-relaxation knot scores used in the initial
  screen, calculated from 100 stochastic closures.
- `validation_summary.json` additionally reports independent 500-closure
  checks for Candidates 1 and 2. The two estimates can differ because closures
  are resampled.
- `dominant_resolved_topology` excludes the trivial topology (`0_1`) and
  unresolved results above Topoly's crossing limit (`TMC`). It is reported as
  `unresolved` when no resolved nontrivial type is available.
- The numerical strict Foldseek screen did not include topology concordance.
  Candidate 2 is topology-concordant with the experimentally determined
  `3_1`-knotted TrmH/SpoU structure 2I6D. Candidate 1's 7WIW nucleotide-binding
  domain match is topology-discordant and supports fold similarity, not the
  generated knot.

## Minimal R2 workflow code

- [`src/overnight_candidate_search.py`](../../src/overnight_candidate_search.py)
- [`src/high_confidence_seed_conversion.py`](../../src/high_confidence_seed_conversion.py)
- [`src/relax_knot_candidates.py`](../../src/relax_knot_candidates.py)
- [`src/foldseek_batch_api.py`](../../src/foldseek_batch_api.py)
- [`src/freeze_r2_evidence.py`](../../src/freeze_r2_evidence.py)

## External resources

- [Sequence dataset on Hugging Face](https://huggingface.co/datasets/EvaKlimentova/Diffusion-all_knots)
- [Source PDB archive on Zenodo](https://doi.org/10.5281/zenodo.21206269)
- [KnotProt 2.0 entry for 2I6D chain A](https://knotprot.cent.uw.edu.pl/viewFS/2i6d/A/0/)
