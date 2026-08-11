# Advancing Knotted Protein Design with ESM3

Code and experiments for the paper *"Advancing Knotted Protein Design with ESM3"*.

We investigate how multimodal protein language models interact with topological complexity, using knotted proteins as a test case. Using ESM3's guided generation, we achieve an 89% success rate in producing knotted proteins (compared to ~0.5% for unguided approaches), reveal that knot topology is remarkably robust to sequence perturbation (mean breaking point: 84%), and show that structural drift precedes topological disruption.

## Repository Structure

```
├── src/                    # Experiment scripts (Modal GPU compute)
└── results/                # Experiment outputs and R2 candidate structures
```

## Experiment Scripts (`src/`)

All experiments run on [Modal](https://modal.com) serverless GPUs using the ESM3-SM (1.4B) model.

| Script | Description |
|--------|-------------|
| `smoke_test.py` | Minimal validation: load ESM3, generate structure, run topoly |
| `benchmark.py` | Per-operation timing benchmarks on Modal A10G |
| `guided_gen_run.py` | De novo guided generation of knotted proteins (n=100) |
| `masking_experiment.py` | Knot stability under random masking (n=250, 10 levels) |
| `rmsd_analysis.py` | RMSD structural drift analysis (n=80) |
| `embeddings_classifier.py` | ESM3 embedding extraction + MLP classifier (n=5000) |
| `contiguous_masking.py` | Contiguous vs random masking comparison (n=50) |
| `targeted_masking.py` | Core vs non-core targeted masking (n=40) |
| `sliding_window.py` | Position-resolved vulnerability profiles (n=40) |
| `unknot_final.py` | Unknotted-to-knotted conversion (n=99) |
| `typed_gen.py` | Knot-type-specific guided generation (4 types × 10) |
| `length_gen.py` | Length-dependent generation success (6 lengths × 10) |
| `extract_embeddings.py` | Full embedding extraction for UMAP visualization |
| `restyle_figures.py` | Generate all paper figures from result JSONs (3 selectable themes) |
| `overnight_candidate_search.py` | Confidence/topology generation and geometry QC |
| `high_confidence_seed_conversion.py` | Iterative conversion of high-confidence generated seeds |
| `relax_knot_candidates.py` | PDBFixer/OpenMM restrained minimization |
| `foldseek_batch_api.py` | Rate-limited Foldseek API screening |
| `freeze_r2_evidence.py` | Freeze exact Candidate 1/2 R2 metrics |

## Setup

Requires [uv](https://docs.astral.sh/uv/) and a Modal account.

```bash
# Install dependencies
uv sync

# Set Modal token (one-time)
uv run modal token set --token-id <ID> --token-secret <SECRET>

# Create HuggingFace secret on Modal
uv run modal secret create huggingface-secret HF_TOKEN=<TOKEN> --force
```

## Running Experiments

```bash
# Smoke test (~5 min)
uv run modal run src/smoke_test.py

# Guided generation (n=100, ~10 min with 10 GPUs)
uv run modal run src/guided_gen_run.py --n-attempts 100

# Masking stability (n=300, ~90 min with 10 GPUs)
uv run modal run src/masking_experiment.py \
  --n-proteins 300 --n-trials 8 \
  --levels "10,20,30,40,50,60,70,80,85,90"

# Generate all figures locally (Okabe-Ito colorblind-safe theme used in the paper)
uv run python src/restyle_figures.py --theme bold

# Or render all three themes side-by-side for comparison (calm / minimal / bold)
uv run python src/restyle_figures.py
```

Add `--detach` to keep jobs running if your terminal disconnects.

## Key Results

| Experiment | N | Result |
|---|---|---|
| Guided generation | 100 | 89% success (95% CI: 81–94%) |
| Masking breaking point | 250 | Mean 84% (±1.2% SE) |
| RMSD at 50% masking | 80 | 3.24 Å median, knot prob 0.85 |
| Embedding classifier | 5000 | 97.1% accuracy |
| Max seq identity to known | 89 | 14.5% (random baseline: 12.5%) |
| Unknotted-to-knotted | 99 | 17% (95% CI: 10–26%) |
| Expanded plausibility screen | 32 | 2 numerical Foldseek/PDB passes; 1 topology-concordant match |

## Revision 2 structural validation

The minimal public R2 package is in
[`results/r2_candidates/`](results/r2_candidates/). It contains:

- 32 restrained-minimized PDB structures;
- the corresponding FASTA sequences and candidate-level metrics; and
- a compact validation summary with Foldseek/PDB evidence and independent
  500-closure checks for Candidates 1 and 2.

All 32 candidates pass the reported post-relaxation geometry checks, and 24
have initial post-relaxation knot scores above 0.90. Candidate 2 is the primary
topology-concordant example: its generated `3_1` topology agrees with the
experimentally determined, `3_1`-knotted TrmH/SpoU structure 2I6D. Candidate 1
is reported as topology-discordant because its 7WIW nucleotide-binding-domain
match is unknotted.

Only the five non-plotting scripts needed to document the R2 generation,
minimization, Foldseek, and evidence-freezing workflow are included. Submission
files, TeX revisions, raw API payloads, plotting outputs, and internal working
logs are intentionally excluded.

## Dataset

[EvaKlimentova/Diffusion-all_knots](https://huggingface.co/datasets/EvaKlimentova/Diffusion-all_knots) on HuggingFace Hub. 15,000 proteins (Real, RFdiffusion, EvoDiff), 1,000 knotted + 4,000 unknotted each.

## Dependencies

- **ESM3** (`esm==3.2.3`): EvolutionaryScale multimodal protein language model
- **Topoly**: Alexander polynomial knot detection
- **Modal**: Serverless GPU compute
- **AlphaKnot 2.0**: Knot core position ground truth

## Citation

```
@article{marsalkova2026advancing,
  title={Advancing Knotted Protein Design with ESM3: Guided Generation and Topological Insights},
  author={Marsalkova, Eva and Simecek, Petr},
  journal={bioRxiv},
  pages={2026--05},
  year={2026},
  publisher={Cold Spring Harbor Laboratory}
}
```

## Acknowledgments

Supported by the Czech Science Foundation, project no. 23-04260L ("Biological code of knots"). Computational resources provided by Modal serverless GPU compute.
