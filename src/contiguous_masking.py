"""Ablation: contiguous vs random masking.

Reviewer ehvs asked for non-random masking strategies. This compares:
- Random masking (baseline, already tested)
- Contiguous segment masking (mask a single block of consecutive positions)
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-contiguous-mask")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "esm==3.2.3",
        "torch==2.5.1",
        "topoly",
        "huggingface-hub",
        "datasets",
    )
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=30 * MINUTES,
)
def process_one_protein(
    protein_id: str,
    sequence: str,
    masking_percentages: list[int],
    n_trials: int,
    masking_mode: str,
) -> dict:
    """Run masking stability with either 'random' or 'contiguous' masking."""
    import os
    import random
    import tempfile

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    import torch

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    from topoly import alexander

    if len(sequence) > 600:
        return {"protein_id": protein_id, "seq_len": len(sequence), "mode": masking_mode,
                "levels": {}, "total_time_s": 0, "error": "sequence_too_long"}

    results = {"protein_id": protein_id, "seq_len": len(sequence), "mode": masking_mode, "levels": {}}
    t0_total = time.time()

    for mask_pct in masking_percentages:
        level_results = []
        for trial in range(n_trials):
            try:
                random.seed(hash((protein_id, mask_pct, trial, masking_mode)) & 0xFFFFFFFF)
                seq_list = list(sequence)
                n_mask = max(1, int(len(seq_list) * mask_pct / 100))

                if masking_mode == "contiguous":
                    start = random.randint(0, max(0, len(seq_list) - n_mask))
                    indices = list(range(start, min(start + n_mask, len(seq_list))))
                else:
                    indices = random.sample(range(len(seq_list)), min(n_mask, len(seq_list)))

                for i in indices:
                    seq_list[i] = "_"
                masked_seq = "".join(seq_list)

                filled = model.generate(
                    ESMProtein(sequence=masked_seq),
                    GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
                )
                struct_protein = model.generate(
                    ESMProtein(sequence=filled.sequence),
                    GenerationConfig(track="structure", num_steps=8),
                )

                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                    struct_protein.to_pdb(f.name)
                    pdb_path = f.name

                topo = alexander(pdb_path, tries=30)
                total_mass = sum(topo.values())
                unknot_frac = topo.get("0_1", 0) / total_mass if total_mass > 0 else 1.0
                knotted_p = 1.0 - unknot_frac
            except Exception as e:
                knotted_p = -1.0

            level_results.append({"trial": trial, "knotted_p": round(knotted_p, 4)})

        valid = [r["knotted_p"] for r in level_results if r["knotted_p"] >= 0]
        avg = sum(valid) / len(valid) if valid else 0
        results["levels"][mask_pct] = {"avg_knotted_p": round(avg, 4), "trials": level_results}

    results["total_time_s"] = round(time.time() - t0_total, 1)
    return results


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_knotted(limit: int = 50) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    return [{"id": r["ID"], "sequence": r["Sequence"]}
            for r in ds if r["Tool"] == "Real" and r["Label"] == 1][:limit]


@app.local_entrypoint()
def main(n_proteins: int = 50, n_trials: int = 8, levels: str = "25,50,75,85,90"):
    masking_percentages = [int(x) for x in levels.split(",")]

    proteins = load_knotted.remote(limit=n_proteins)
    print(f"Loaded {len(proteins)} proteins")

    # Run both modes in parallel (each protein x mode gets its own GPU)
    all_ids = []
    all_seqs = []
    all_pcts = []
    all_trials = []
    all_modes = []

    for mode in ["random", "contiguous"]:
        for p in proteins:
            all_ids.append(p["id"])
            all_seqs.append(p["sequence"])
            all_pcts.append(masking_percentages)
            all_trials.append(n_trials)
            all_modes.append(mode)

    total_jobs = len(all_ids)
    print(f"Launching {total_jobs} jobs ({len(proteins)} proteins x 2 modes)")
    print(f"  Levels: {masking_percentages}, {n_trials} trials each")

    t0 = time.time()
    results = list(
        process_one_protein.map(
            all_ids, all_seqs, all_pcts, all_trials, all_modes,
            return_exceptions=True,
        )
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"protein_id": all_ids[i], "mode": all_modes[i],
                          "error": str(r), "levels": {}})
        else:
            clean.append(r)
    results = clean
    wall_time = time.time() - t0

    # Compare modes
    import numpy as np
    print(f"\n{'='*70}")
    print(f"CONTIGUOUS vs RANDOM MASKING ({wall_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Level':>8} | {'Random':>10} | {'Contiguous':>12} | {'Diff':>8}")
    print("-" * 45)

    for level in masking_percentages:
        rand_scores = [r["levels"].get(str(level), {}).get("avg_knotted_p", None)
                       for r in results if r.get("mode") == "random" and r.get("levels")]
        cont_scores = [r["levels"].get(str(level), {}).get("avg_knotted_p", None)
                       for r in results if r.get("mode") == "contiguous" and r.get("levels")]
        rand_scores = [s for s in rand_scores if s is not None]
        cont_scores = [s for s in cont_scores if s is not None]
        if rand_scores and cont_scores:
            r_mean = np.mean(rand_scores)
            c_mean = np.mean(cont_scores)
            print(f"{level:>7}% | {r_mean:>10.3f} | {c_mean:>12.3f} | {c_mean-r_mean:>+8.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/contiguous_masking.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "results": results}, f, indent=2)
    print(f"\nSaved to results/contiguous_masking.json")
