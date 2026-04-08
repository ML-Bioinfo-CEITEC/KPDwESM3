# Evidence Against Memorization

## 1. Regenerated sequences are different from originals
At 50% masking: only 73% sequence identity, yet 85% knot probability.
At 70% masking: only 58% identity, yet 76% knot probability.
The model generates NOVEL sequences that fold into the same topology.

## 2. De novo generated proteins are diverse
Pairwise identity among 20 generated knotted proteins: mean 6.1% (range 2-10%).
This is at random-protein baseline level -- no evidence of copying from training data.

## 3. For the paper
Include a table/figure showing sequence identity vs knot probability.
The key argument: if ESM3 were memorizing, we'd expect high sequence identity
in the regenerated regions. Instead, the sequences are substantially different
while maintaining the same topology. This is generalization, not memorization.
