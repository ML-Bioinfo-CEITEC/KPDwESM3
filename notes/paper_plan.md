# Unified Paper Plan

## Working Title
"Topological Robustness and Guided Design of Knotted Proteins with ESM3:
Implications for Protein Engineering and Biosecurity"

## Key Thesis
Multi-modal protein language models (ESM3) capture deep topological information
that enables both precise generation and robust reconstruction of knotted proteins.
This has dual implications: powerful tools for protein engineering AND challenges
for sequence-based biosecurity measures.

## Proposed Structure

### 1. Introduction
- Knotted proteins: rare (<1%), stable, functionally important
- Previous work: RFdiffusion+ProteinMPNN at ~0.5% success rate
- ESM3: multi-modal model unifying seq/struct/function
- Our contributions (updated from both papers):
  1. Guided generation at 88% success rate (confirmed, n=73)
  2. Continuous knot score with sharp-transition property
  3. 97% classification accuracy from embeddings alone
  4. Biosecurity implications: knots survive 85% sequence redaction
  5. NEW: contiguous vs random masking analysis
  6. NEW: knot location and breaking pattern analysis

### 2. Methods
- Dataset: 1000 knotted + 4000 unknotted real proteins (characterize: mean len ~400, median ~340)
- ESM3-SM (1.4B) on Modal A10G GPUs
- Knot detection: topoly Alexander polynomials
- Guided generation: ESM3GuidedDecoding with knot scoring function
- Continuous knot score: Algorithm 1 (mask X%, regenerate, check topology, N trials)
- Masking strategies: random vs contiguous

### 3. Results

#### 3.1 Guided Generation (87.7% success, n=73)
- De novo generation from fully masked sequences
- 170x improvement over RFdiffusion baseline
- Diverse topologies: TMC, 3_1, 5_1, 4_1, 8_19, etc.
- NEW: knot type distribution analysis

#### 3.2 Knot Stability Under Masking (n=80)
- Median breaking point: 85%
- Sharp transition: 68% of proteins have a single-step drop >0.3
- Steepest drops cluster at 80-90% masking
- Avg curve: flat until 70%, then rapid decline
- NEW: contiguous masking breaks knots ~0.11-0.15 more easily than random
- Biosecurity framing: redaction of even 75% of sequence doesn't prevent reconstruction

#### 3.3 Classification (97.1% accuracy)
- Simple MLP on ESM3 mean-pooled 1536-d embeddings
- Suggests ESM3 implicitly learns topological features

#### 3.4 Masking Percentage Ablation
- Justification for X=10%: 98% of knotted proteins still detected
- Sensitivity degrades gradually: 94% at 20%, 91% at 50%, 81% at 70%

#### 3.5 Unknotted-to-Knotted (8-14%, n=70+)
- Lower than originally reported (31%)
- Still non-trivial: converting ANY unknotted protein to knotted is meaningful
- Successful cases needed 3-4 iterations, scored >0.9
- Report honestly with confidence intervals

### 4. Discussion
- Combine protein design + biosecurity angles
- Why ESM3 is so good at this: multi-modal understanding
- Sharp transition has implications for knot evolution
- Biosecurity: partial disclosure enables reconstruction
- Limitations: ESM3-SM only, need larger models, wet-lab validation
- Address reviewer concerns explicitly

### 5. Conclusion

## Figures Needed
1. Example knotted protein (existing Fig 1)
2. Knot probability vs masking % -- aggregated curve with error bars (NEW, from our data)
3. Sharp transition: overlay individual protein curves (NEW)
4. Guided generation example + knot type distribution (update existing Fig 5)
5. Contiguous vs random masking comparison (NEW)
6. Confusion matrix for classifier (update existing Table 1)
7. Knot location analysis (NEW, pending experiment)
8. Breaking point histogram (update existing Fig 2)

## Reviewer Concerns Addressed
- [JmYJ] X=10%, N=16 justification: Section 3.4 ablation
- [JmYJ] Scale-up discussion: addressed in discussion
- [Qboz] ESM3 vs guided sampling contribution: shown by high success rate of guided gen
- [ehvs] Contiguous masking: Section 3.2
- [ehvs] Non-linear threshold for multiple proteins: Section 3.2 aggregated
- [ehvs] Dataset description: Section 2 with full stats
- [ehvs] RedactBench details: expanded in discussion

## Target Venue
- Could aim for a main ML venue (NeurIPS/ICML main track)
- Or a biology-focused venue (Bioinformatics, Nature Methods)
- Discuss with Petr
