# Papers Summary

## Paper 1: knotter_esm4.tex (ICML 2025 Workshop)
**Title:** Advancing Knotted Protein Design with ESM3: Guided Generation and Topological Insights

**Framing:** Protein design / engineering perspective.

**Contributions:**
1. Continuous knot score via randomized masking (Algorithm 1: mask X% of seq, regenerate with ESM3, check topology, repeat N times)
2. ESM3 guided generation achieves 87% success rate (vs 0.5% baseline from RFdiffusion+ProteinMPNN)
3. ESM3 embeddings -> 93% classification accuracy (knotted vs unknotted)
4. Unknotted-to-knotted conversion: 31% success rate

**Key parameters:** X=10% masking, N=16 trials, lambda=1.0 for guidance

## Paper 2: biosecurity.tex (NeurIPS 2025 Biosecurity Workshop)
**Title:** Structural Persistence Despite Sequence Redaction: A Biosecurity Evaluation of Protein Language Models

**Framing:** Biosecurity - redaction of protein sequences doesn't prevent reconstruction.

**Same results reframed as:**
- 85% of sequence must be redacted before topology is lost
- Guided generation recovers rare structural features from partial info
- Proposes "RedactBench" as evaluation framework

**Additional content:** Threat model, policy recommendations, Table 2 (reconstruction at different redaction levels)

## Shared Data
- Dataset: EvaKlimentova/Diffusion-all_knots (HuggingFace)
- 1000 knotted + 4000 unknotted real proteins
- Model: ESM3-SM (1.4B params, open source)

## Reviewer Concerns (4 reviews total)
1. No ablation on X=10%, N=16 choices
2. No comparison with other models on same data
3. Only random masking tested (need contiguous, domain-based)
4. Sharp threshold shown for 1 protein only - need aggregated evidence
5. Dataset poorly described (lengths, diversity, homology)
6. Unclear if ESM3 embeddings or guided sampling drives improvement
7. Need biophysical metrics beyond topology
