# ESM3 Knotted Protein Paper - Progress Notes

## LATEST: Third Round of Experiments (April 5 night)

### New experiments this session:

1. **Knot-Type-Specific Generation** (~$5, 13 min)
   - Can we target a SPECIFIC knot type with modified scoring function?
   - Trefoil (3_1) targeting: 90% success, 100% of knotted proteins had correct type
   - Figure-eight (4_1): 20% hit rate, model defaults to trefoils
   - 5_1: 30% hit rate, 5_2: 20% hit rate
   - Conclusion: partial topological control, trefoils act as attractor

2. **Anti-Memorization Figure** (local, free)
   - Per-protein scatter of sequence identity vs knot probability
   - Visually compelling: at 50-70% masking, sequences are very different but knots persist
   - Strong evidence against memorization for the paper

3. **Breaking Point vs Length** (local, free)
   - r = -0.044: essentially zero correlation
   - Knot robustness is independent of protein length

### Paper updated:
- Added knot-type-specific generation as new subsection + figure
- Added anti-memorization scatter plot figure
- Recompiled PDF (585 KB, ~14 pages)

### All new figures:
- `fig_typed_gen.pdf` -- type-specific generation success rates
- `fig_identity_vs_knot.pdf` -- anti-memorization evidence
- `fig_bp_vs_length.pdf` -- breaking point vs length (for supplementary)
- `fig_sliding_window.pdf` -- positional vulnerability profile
- `fig_length_gen.pdf` -- length-dependent success rate

## Complete Results Table

| Experiment | N | Key Result |
|---|---|---|
| Guided generation | 100 | 89% success, 15+ knot types |
| Type-specific generation | 40 | 90% for trefoils, 20-30% for complex types |
| Length-dependent gen | 60 | 20% at 100aa, 100% at 350aa |
| Masking stability | 250 | Mean 84% breaking point |
| RMSD analysis | 80 | Topology outlasts structure |
| Contiguous masking | 50 | 0.07-0.15 more disruptive |
| Targeted core masking | 40 | 66% reconstruction from context |
| Sliding window | 40 | Core region 0.796 vs non-core 0.910 |
| Classifier | 5000 | 97.1% accuracy |
| UMAP visualization | 5000 | Partial separation, convertibles NOT close to knotted |
| Unknotted-to-knotted | 99 | 17% with gentle params |
| Breaking point vs length | 250 | No correlation (r=-0.044) |
| Anti-memorization | 80 | Novel sequences maintain topology |

## Total project cost: ~$65
