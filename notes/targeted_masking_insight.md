# Key Finding: Targeted Masking (Core vs Non-Core)

## The question
Does it matter WHERE you mask relative to the knot core?

## The answer
Yes, but less than you'd expect!

| What's masked | % of total seq | Knot probability |
|---|---|---|
| 100% of core | ~47% | **0.659** |
| 100% of non-core | ~53% | **0.828** |
| 100% random (entire seq) | 100% | **0.084** |

## Interpretation
1. Even masking the ENTIRE knot core (47% of seq), the model reconstructs the knot 66% of the time. The flanking regions carry enough topological information.
2. Masking outside the core barely affects the knot (0.828), confirming the core IS the critical region.
3. But the core-only masking effect is surprisingly WEAK -- the model's learned representations can infer the knot from context.

## Implication for biosecurity
Even targeted redaction of the knot core is insufficient. The model "knows" what topology should go there from the flanking sequence.

## Implication for protein engineering  
ESM3 has a deep implicit understanding of how protein topology relates to sequence -- it can reconstruct complex topological features from partial information.

## For the paper
This is a novel finding not in either workshop paper. It strengthens both the protein design and biosecurity narratives.
