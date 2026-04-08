"""TM-score structural comparison: are generated knotted proteins structurally similar to known ones?

For each generated knotted protein, predict its structure and compute TM-score
against structures of known knotted proteins. This tests whether the model is
doing structural template retrieval even when sequences are dissimilar.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-tm-score")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("esm==3.2.3", "torch==2.5.1", "huggingface-hub", "datasets", "biotite")
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=30 * MINUTES,
)
def compute_tm_scores(
    generated_seqs: list[str],
    known_seqs: list[str],
) -> dict:
    """Generate structures for both sets and compute pairwise TM-scores."""
    import os
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    import numpy as np
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from esm.utils.structure.aligner import Aligner

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    def get_chain(seq):
        protein = ESMProtein(sequence=seq)
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        return protein.to_protein_chain()

    # Generate structures
    print(f"Generating structures for {len(generated_seqs)} generated proteins...")
    gen_chains = []
    for i, seq in enumerate(generated_seqs):
        try:
            gen_chains.append(get_chain(seq))
        except Exception as e:
            gen_chains.append(None)
            print(f"  Gen {i} failed: {e}")

    print(f"Generating structures for {len(known_seqs)} known proteins...")
    known_chains = []
    for i, seq in enumerate(known_seqs):
        try:
            known_chains.append(get_chain(seq))
        except Exception as e:
            known_chains.append(None)
            print(f"  Known {i} failed: {e}")

    # Compute pairwise TM-scores (via RMSD as proxy -- ESM3 Aligner gives RMSD, not TM-score)
    # We'll compute RMSD and sequence identity instead, plus backbone length ratio
    print("Computing pairwise comparisons...")
    results = []
    for i, gc in enumerate(gen_chains):
        if gc is None:
            continue
        max_rmsd_inv = 0  # lower RMSD = more similar
        best_known = -1
        rmsds = []
        for j, kc in enumerate(known_chains):
            if kc is None:
                continue
            try:
                aligner = Aligner(gc, kc)
                rmsd = aligner.rmsd
                rmsds.append(rmsd)
                if best_known == -1 or rmsd < rmsds[best_known]:
                    best_known = len(rmsds) - 1
            except Exception:
                rmsds.append(999.0)

        if rmsds:
            min_rmsd = min(rmsds)
            results.append({
                "gen_idx": i,
                "gen_len": len(generated_seqs[i]),
                "min_rmsd": round(float(min_rmsd), 2),
                "mean_rmsd": round(float(np.mean([r for r in rmsds if r < 900])), 2),
                "n_compared": sum(1 for r in rmsds if r < 900),
            })
            print(f"  Gen {i} (len={len(generated_seqs[i])}): min_RMSD={min_rmsd:.1f}, mean={np.mean([r for r in rmsds if r<900]):.1f}")

    return {"n_generated": len(generated_seqs), "n_known": len(known_seqs), "comparisons": results}


@app.local_entrypoint()
def main():
    # Load generated knotted proteins
    with open("results/guided_gen_combined.json") as f:
        gen_data = json.load(f)
    gen_seqs = [r["sequence"] for r in gen_data["results"]
                if r.get("is_knotted") and r.get("sequence")][:20]

    # Load known knotted proteins
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    known_seqs = [r["Sequence"] for r in ds
                  if r["Tool"] == "Real" and r["Label"] == 1 and len(r["Sequence"]) < 400][:25]

    print(f"Comparing {len(gen_seqs)} generated vs {len(known_seqs)} known knotted proteins")

    t0 = time.time()
    result = compute_tm_scores.remote(gen_seqs, known_seqs)
    wall_time = time.time() - t0

    import numpy as np
    min_rmsds = [r["min_rmsd"] for r in result["comparisons"]]
    mean_rmsds = [r["mean_rmsd"] for r in result["comparisons"]]

    print(f"\n{'='*50}")
    print(f"STRUCTURAL COMPARISON ({wall_time/60:.1f} min)")
    print(f"{'='*50}")
    print(f"Min RMSD to nearest known knotted:")
    print(f"  Mean of min: {np.mean(min_rmsds):.1f} A")
    print(f"  Median of min: {np.median(min_rmsds):.1f} A")
    print(f"  Range: {min(min_rmsds):.1f} - {max(min_rmsds):.1f} A")
    print(f"  Proteins with min_RMSD < 5A: {sum(1 for r in min_rmsds if r < 5)}/{len(min_rmsds)}")

    Path("results").mkdir(exist_ok=True)
    result["wall_time_s"] = round(wall_time, 1)
    result["summary"] = {
        "mean_min_rmsd": round(float(np.mean(min_rmsds)), 2),
        "median_min_rmsd": round(float(np.median(min_rmsds)), 2),
        "n_below_5A": sum(1 for r in min_rmsds if r < 5),
    }
    with open("results/tm_score.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/tm_score.json")
