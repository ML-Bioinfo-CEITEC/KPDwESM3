"""Sliding window masking: vulnerability profile per protein.

Mask a fixed-size window at each position and check if the knot breaks.
Creates a position-dependent vulnerability map that can be overlaid with
the known knot core position from AlphaKnot.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-sliding-window")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("esm==3.2.3", "torch==2.5.1", "topoly", "huggingface-hub", "datasets")
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=30 * MINUTES,
)
def process_one_protein(
    protein_id: str,
    sequence: str,
    core_start: int,
    core_end: int,
    window_size: int,
    step_size: int,
    n_trials: int,
) -> dict:
    import os, random, tempfile
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    seq_len = len(sequence)
    if seq_len > 500:
        return {"protein_id": protein_id, "error": "too_long"}

    t0 = time.time()
    positions = list(range(0, seq_len - window_size + 1, step_size))
    vulnerability = []

    for pos in positions:
        scores = []
        for trial in range(n_trials):
            try:
                random.seed(hash((protein_id, pos, trial)) & 0xFFFFFFFF)
                seq_list = list(sequence)
                for i in range(pos, min(pos + window_size, seq_len)):
                    seq_list[i] = "_"

                filled = model.generate(
                    ESMProtein(sequence="".join(seq_list)),
                    GenerationConfig(track="sequence", num_steps=8, temperature=1.0))
                struct = model.generate(
                    ESMProtein(sequence=filled.sequence),
                    GenerationConfig(track="structure", num_steps=8))

                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                    struct.to_pdb(f.name)
                    topo = alexander(f.name, tries=30)
                total = sum(topo.values())
                kp = 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0
                scores.append(kp)
            except:
                scores.append(-1)

        valid = [s for s in scores if s >= 0]
        avg = sum(valid) / len(valid) if valid else 0
        vulnerability.append({
            "position": pos,
            "norm_position": round(pos / seq_len, 3),
            "avg_knotted_p": round(avg, 4),
            "in_core": pos >= core_start and pos + window_size <= core_end,
            "overlaps_core": not (pos + window_size < core_start or pos > core_end),
        })

    return {
        "protein_id": protein_id,
        "seq_len": seq_len,
        "core_start": core_start,
        "core_end": core_end,
        "window_size": window_size,
        "vulnerability": vulnerability,
        "total_time_s": round(time.time() - t0, 1),
    }


@app.local_entrypoint()
def main(n_proteins: int = 40, window_size: int = 50, step_size: int = 10, n_trials: int = 4):
    with open("results/alphaknot_data.json") as f:
        alphaknot = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    seq_map = {r["ID"].replace("R_", ""): r["Sequence"]
               for r in ds if r["Tool"] == "Real" and r["Label"] == 1}

    proteins = []
    for uid, ak in alphaknot.items():
        if uid in seq_map and len(seq_map[uid]) <= 400 and ak["core_length"] > 30:
            proteins.append({
                "id": f"R_{uid}", "sequence": seq_map[uid],
                "core_start": ak["core_start"], "core_end": ak["core_end"],
            })
    proteins = proteins[:n_proteins]
    print(f"Sliding window: {len(proteins)} proteins, window={window_size}, step={step_size}")

    t0 = time.time()
    results = list(
        process_one_protein.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            [p["core_start"] for p in proteins],
            [p["core_end"] for p in proteins],
            [window_size] * len(proteins),
            [step_size] * len(proteins),
            [n_trials] * len(proteins),
            return_exceptions=True,
        )
    )

    clean = [r if not isinstance(r, Exception) else
             {"protein_id": proteins[i]["id"], "error": str(r)}
             for i, r in enumerate(results)]
    valid = [r for r in clean if "vulnerability" in r]
    wall_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"SLIDING WINDOW ({wall_time/60:.1f} min, {len(valid)} proteins)")
    print(f"{'='*60}")

    # Aggregate: avg vulnerability by normalized position
    import numpy as np
    norm_bins = np.arange(0, 1.01, 0.05)
    core_scores, noncore_scores = [], []

    for r in valid:
        for v in r["vulnerability"]:
            if v["overlaps_core"]:
                core_scores.append(v["avg_knotted_p"])
            else:
                noncore_scores.append(v["avg_knotted_p"])

    if core_scores and noncore_scores:
        print(f"Avg knot_p when window overlaps core: {np.mean(core_scores):.3f}")
        print(f"Avg knot_p when window outside core:  {np.mean(noncore_scores):.3f}")
        print(f"Difference: {np.mean(noncore_scores) - np.mean(core_scores):.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/sliding_window.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "n_valid": len(valid),
                    "results": valid}, f, indent=2)
    print(f"Saved to results/sliding_window.json")
