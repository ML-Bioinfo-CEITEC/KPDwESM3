# New Hypotheses and Possible Next Steps

Based on all results so far, here are observations and ideas for further investigation.

## Key Observations

### 1. ESM3 "knows" topology from context
The targeted masking experiment shows ESM3 can reconstruct the knot core from flanking
regions alone (66% success when entire core is masked). This suggests:
- Topological information is DISTRIBUTED across the sequence, not localized
- The model has learned long-range dependencies that specify topology
- **Hypothesis**: There exist specific residue patterns in the flanking regions that
  "encode" the knot type. Could we identify them?

### 2. The phase transition is sharp, not gradual
68% of proteins show a single-step drop >0.3 in knot probability.
- **Hypothesis**: There is a minimal "topological information content" threshold.
  Below it, the model can't reconstruct the knot; above it, it almost always can.
- Could relate to information-theoretic ideas about minimum description length for topology.

### 3. Structure drifts but topology persists
RMSD increases substantially before topology is lost.
- **Hypothesis**: ESM3 generates structures in a "topological equivalence class" --
  many different 3D arrangements share the same knot type.
- Could we measure the diversity of generated structures that share the same topology?

### 4. Contiguous masking is more disruptive but not dramatically so
Only 0.11-0.15 more disruptive than random at 50-75%.
- **Hypothesis**: The topological information is distributed enough that even a large
  contiguous gap can be bridged by the remaining sequence.

### 5. Unknotted-to-knotted conversion is hard (~10%, not 31%)
- **Hypothesis**: Most unknotted proteins have sequences that are strongly "anti-knotted" --
  the model's prior pulls them back to unknotted topology.
- The proteins that DO convert might share properties (shorter? specific folds?)
- Could we identify what makes a protein "convertible"?

## Possible New Experiments

### A. Topological fingerprinting
Use topoly matrix mode to compute the full knot fingerprint for generated proteins.
Compare fingerprints of original vs masked-regenerated proteins. Do they share the
same fingerprint pattern even when the specific 3D structure differs?

### B. Minimal information for topology
Binary search for the exact breaking point per protein (instead of fixed grid).
Map the distribution more precisely. Correlate with sequence entropy or conservation.

### C. Cross-model comparison
Run the same masking experiments with ESM-C embeddings or a simpler model to see
if the topological understanding is specific to ESM3's multi-modal training.

### D. Functional domain masking
Instead of random/contiguous/core masking, mask specific functional domains
(e.g., active sites identified by InterPro) to see if functional information
and topological information are correlated.

### E. What makes a protein "convertible"?
Analyze the ~10% of unknotted proteins that converted to knotted.
Do they share sequence motifs, lengths, secondary structure patterns,
or proximity to knotted proteins in embedding space?

### F. Embedding space topology
Visualize ESM3 embeddings of knotted vs unknotted proteins (t-SNE/UMAP).
Are they cleanly separable? Is there a "boundary" in embedding space
that corresponds to the topological transition?
