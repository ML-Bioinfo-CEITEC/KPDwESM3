"""Targeted masking experiment: mask INSIDE vs OUTSIDE the knot core.

Uses AlphaKnot ground-truth knot core positions to test whether
masking the core region breaks the knot faster than masking outside it.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-targeted-mask")

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
    core_start: int,
    core_end: int,
    masking_fractions: list[float],
    n_trials: int,
    mode: str,
) -> dict:
    """Mask either inside or outside the knot core and check topology."""
    import os
    import random
    import tempfile

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    import torch
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    if len(sequence) > 600:
        return {"protein_id": protein_id, "mode": mode, "error": "too_long", "levels": {}}

    seq_len = len(sequence)
    core_indices = list(range(core_start, min(core_end + 1, seq_len)))
    noncore_indices = [i for i in range(seq_len) if i not in core_indices]

    results = {"protein_id": protein_id, "mode": mode, "seq_len": seq_len,
               "core_start": core_start, "core_end": core_end,
               "core_len": len(core_indices), "noncore_len": len(noncore_indices),
               "levels": {}}

    t0 = time.time()
    for frac in masking_fractions:
        level_results = []
        for trial in range(n_trials):
            try:
                random.seed(hash((protein_id, frac, trial, mode)) & 0xFFFFFFFF)
                seq_list = list(sequence)

                if mode == "core_only":
                    pool = core_indices
                elif mode == "noncore_only":
                    pool = noncore_indices
                else:
                    pool = list(range(seq_len))

                n_mask = max(1, int(len(pool) * frac))
                n_mask = min(n_mask, len(pool))
                indices = random.sample(pool, n_mask)
                for i in indices:
                    seq_list[i] = "_"

                filled = model.generate(
                    ESMProtein(sequence="".join(seq_list)),
                    GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
                )
                struct = model.generate(
                    ESMProtein(sequence=filled.sequence),
                    GenerationConfig(track="structure", num_steps=8),
                )
                with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                    struct.to_pdb(f.name)
                    topo = alexander(f.name, tries=30)
                total = sum(topo.values())
                kp = 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0
            except Exception as e:
                kp = -1.0

            level_results.append(round(kp, 4))

        valid = [v for v in level_results if v >= 0]
        avg = sum(valid) / len(valid) if valid else 0
        n_masked_total = max(1, int(len(pool) * frac)) if mode != "random" else max(1, int(seq_len * frac))
        pct_of_total = round(n_masked_total / seq_len * 100, 1)

        results["levels"][str(frac)] = {
            "avg_knotted_p": round(avg, 4),
            "n_masked": n_masked_total,
            "pct_of_total_seq": pct_of_total,
            "trials": level_results,
        }

    results["total_time_s"] = round(time.time() - t0, 1)
    return results


@app.local_entrypoint()
def main(n_proteins: int = 40):
    # Load AlphaKnot data and dataset
    with open('results/alphaknot_data.json') as f:
        alphaknot = json.load(f)

    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    knotted = {r["ID"].replace("R_", ""): r["Sequence"]
               for r in ds if r["Tool"] == "Real" and r["Label"] == 1}

    # Pick proteins that have AlphaKnot data and reasonable core sizes
    proteins = []
    for uid, ak in alphaknot.items():
        if uid in knotted and ak['core_length'] > 20 and len(knotted[uid]) <= 500:
            proteins.append({
                "id": f"R_{uid}", "sequence": knotted[uid],
                "core_start": ak["core_start"], "core_end": ak["core_end"],
            })
    proteins = proteins[:n_proteins]
    print(f"Using {len(proteins)} proteins with known knot cores")

    fractions = [0.25, 0.5, 0.75, 1.0]
    n_trials = 4

    # Run 3 modes: core_only, noncore_only, random
    all_ids, all_seqs, all_cs, all_ce, all_fracs, all_trials, all_modes = [], [], [], [], [], [], []
    for mode in ["core_only", "noncore_only", "random"]:
        for p in proteins:
            all_ids.append(p["id"])
            all_seqs.append(p["sequence"])
            all_cs.append(p["core_start"])
            all_ce.append(p["core_end"])
            all_fracs.append(fractions)
            all_trials.append(n_trials)
            all_modes.append(mode)

    print(f"Launching {len(all_ids)} jobs (3 modes x {len(proteins)} proteins)")

    t0 = time.time()
    results = list(
        process_one_protein.map(
            all_ids, all_seqs, all_cs, all_ce, all_fracs, all_trials, all_modes,
            return_exceptions=True,
        )
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"protein_id": all_ids[i], "mode": all_modes[i], "error": str(r), "levels": {}})
        else:
            clean.append(r)
    wall_time = time.time() - t0

    # Analyze
    print(f"\n{'='*70}")
    print(f"TARGETED MASKING RESULTS ({wall_time/60:.1f} min)")
    print(f"{'='*70}")

    import numpy as np
    for frac in fractions:
        print(f"\nMasking {frac*100:.0f}% of target region:")
        for mode in ["core_only", "noncore_only", "random"]:
            vals = [r["levels"].get(str(frac), {}).get("avg_knotted_p")
                    for r in clean if r.get("mode") == mode and r.get("levels")]
            vals = [v for v in vals if v is not None]
            pcts = [r["levels"].get(str(frac), {}).get("pct_of_total_seq")
                    for r in clean if r.get("mode") == mode and r.get("levels")]
            pcts = [p for p in pcts if p is not None]
            if vals:
                print(f"  {mode:>15}: knot_p={np.mean(vals):.3f} "
                      f"(masking ~{np.mean(pcts):.0f}% of total seq)")

    Path("results").mkdir(exist_ok=True)
    with open("results/targeted_masking.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "results": clean}, f, indent=2)
    print(f"\nSaved to results/targeted_masking.json")
