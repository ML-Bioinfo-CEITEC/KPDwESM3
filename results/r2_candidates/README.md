# Revision 2 generated knot candidates

This directory contains the 32 distinct generated protein structures used in
the Revision 2 plausibility and Foldseek screen.

## Files

- `candidate_01.pdb` through `candidate_32.pdb`: restrained-minimized structures.
- `candidates.fasta`: generated amino-acid sequences.
- `candidates.json`: source confidence, topology, minimization, and geometry metrics.

## Processing

Candidate structures were predicted with ESM3-SM, screened for source mean
pLDDT above 0.50 and knot score above 0.50, completed with PDBFixer, and
restrained-minimized using OpenMM with the Amber14 all-atom force field.
All deposited structures pass the continuity and clash checks described in the
manuscript and retain knot score above 0.50 after minimization.

These are computational candidates, not experimentally validated proteins.
