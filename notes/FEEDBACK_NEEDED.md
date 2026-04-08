# Feedback Needed From You

Please add your comments below each section. You can write directly in this file,
or tell me in chat -- either way works.

---

## 1. Paper Draft (tex/main.pdf)

The paper is at `tex/main.pdf` (~10 pages, 9 figures, 6 tables).

### Overall structure:
1. Intro: knotted proteins + ESM3 + our contributions
2. Methods: dataset, guided generation, knot score, masking strategies, RMSD, classifier
3. Results: guided gen (89%), masking stability (90% median BP), RMSD, contiguous masking, classifier (97%), masking ablation, targeted core masking
4. Discussion: protein engineering + biosecurity implications
5. Conclusion

### Questions for you:
- [ ] Is the balance between protein design and biosecurity framing right?
      Or should one angle dominate?
- [ ] The targeted core masking is our strongest novel finding. Should it be
      even more prominent (e.g., in the abstract, as main contribution #1)?
- [ ] Should we include the unknotted-to-knotted results at all? Currently
      mentioned in limitations. We could either:
      (a) Drop it entirely
      (b) Keep as minor result with honest numbers
      (c) Wait for the final 100-protein run to get better numbers
- [ ] Target venue? Options:
      - ML venue (NeurIPS/ICML) -- would need to strengthen ML novelty
      - Bioinformatics (Bioinformatics journal) -- good fit for methods
      - Biology (Nature Methods, Protein Science) -- focus on biological insights
      - Biosecurity workshop again -- easy acceptance but low impact
- [ ] Missing: the paper has no "Related Work" section. Should we add one?
- [ ] The second author (Eva Marsalkova) -- should she still be listed?
      Any other co-authors to add?

YOUR COMMENTS:

For writing style - look at the original papers and try to mimic it. you are overusing list, refering to reviewer the reader never heard about... 

Try to use more references from the original papers. Particularly, from Joanna Sulkowska lab

Add one of the generated knotted protein figure - from the original paper

At this stage it is ok to add rather more than less. We can always cut it out

Speaking about cutting out - add Supplementaty Materials and start to think which Figures should be moved there

---

## 2. New Hypotheses (notes/new_hypotheses.md)

Most interesting follow-up directions, ranked by what I think is most publishable:

### A. Topological information is distributed (strongest)
ESM3 reconstructs knots from flanking context alone. WHY?
- Could identify specific residue patterns that "encode" topology
- Information-theoretic analysis: how much sequence = minimum for topology?
- Compare with conservation patterns from evolutionary analysis

Or it could memorize the proteins - this is dangerous direction if not handeled properly

### B. Embedding space topology
Visualize ESM3 embeddings with UMAP/t-SNE, color by knot type.
- Is there a clear boundary in embedding space?
- Can we see the "convertible" unknotted proteins near the boundary?
- Quick experiment (~$1, 5 min)

Interesting. I do not much belive in it. But quick experiment seems good - let us try

Are 3_1s sitting together. are those convertible somehow close to them

### C. What makes proteins "convertible"?
The ~10-50% that convert unknotted→knotted -- what's special about them?
- Shorter? Specific folds? Closer to knotted proteins in embedding space?
- Could help explain the variable success rate

yes, please try

### D. Cross-model comparison
Run masking experiments with a non-ESM3 model (e.g., ESMFold, or even
a simple sequence model) to test if topological understanding is unique
to ESM3's multi-modal training.

no

### Which ones interest you? Any other ideas?

YOUR COMMENTS:


---

## 3. Current Results Summary

| Experiment | N | Result | Confidence |
|---|---|---|---|
| Guided generation | 100 | **89%** | High |
| Masking stability | 250 | **90% median BP** | High |
| Classifier | 5000 | **97.1%** | High |
| RMSD analysis | 80 | **Topology > structure** | High |
| Contiguous masking | 50 | **0.07-0.15 more disruptive** | Medium |
| Targeted core masking | 40 | **66% reconstruction from context** | Medium-High |
| Unknotted-to-knotted | 14 completed | **50% of completed** | Low (running 100 more) |
| Knot location | 80 | **No correlation core size vs BP** | High |

### Any results that surprise you or seem wrong?

YOUR COMMENTS:

I feel bad about  Masking stability | 250 | **90% median BP**. Feels too strong and suspicious

---

## 4. How to Add Your Comments

Three options:
1. **Edit this file directly** -- I'll read your changes
2. **Tell me in chat** -- I'll incorporate
3. **Annotate the PDF** -- I can't read annotations, so summarize in chat

I recommend option 1 or 2. Just write naturally, I'll handle formatting.

Regarding more experiments

### D. Functional domain masking
Instead of random/contiguous/core masking, mask specific functional domains
(e.g., active sites identified by InterPro) to see if functional information
and topological information are correlated.

Seems nice but hard - were to get domains, how to correlate anything with limited data.


Different idea - give me more movies from those convertible proteins