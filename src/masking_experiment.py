"""Masking stability experiment: how much sequence must be altered to break a knot?

Parallelized across Modal containers -- each container handles one protein
across all masking levels and trials.
"""

import json
import time
from pathlib import Path

import modal

# --- Shared infra (inlined to avoid cross-module import issues on Modal) ---
app = modal.App("esm3-masking")

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
    topoly_tries: int = 30,
) -> dict:
    """Run masking stability for one protein across all masking levels."""
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
        print(f"SKIP {protein_id}: sequence too long ({len(sequence)} aa), risk of OOM")
        return {"protein_id": protein_id, "seq_len": len(sequence), "levels": {},
                "total_time_s": 0, "error": "sequence_too_long"}

    results = {"protein_id": protein_id, "seq_len": len(sequence), "levels": {}}
    t0_total = time.time()

    for mask_pct in masking_percentages:
        level_results = []
        for trial in range(n_trials):
            try:
                random.seed(hash((protein_id, mask_pct, trial)) & 0xFFFFFFFF)

                seq_list = list(sequence)
                n_mask = max(1, int(len(seq_list) * mask_pct / 100))
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
                    pdb_path = f.name
                    struct_protein.to_pdb(pdb_path)

                topo = alexander(pdb_path, tries=topoly_tries)
                total_mass = sum(topo.values())
                unknot_frac = topo.get("0_1", 0) / total_mass if total_mass > 0 else 1.0
                knotted_p = 1.0 - unknot_frac
            except Exception as e:
                print(f"  Error on {protein_id} mask={mask_pct}% trial={trial}: {e}")
                knotted_p = -1.0
                topo = {"error": str(e)}

            level_results.append({
                "trial": trial,
                "knotted_p": round(knotted_p, 4),
                "topology": {k: round(v, 4) if isinstance(v, float) else v for k, v in topo.items()},
            })

        avg_knotted = sum(r["knotted_p"] for r in level_results) / len(level_results)
        results["levels"][mask_pct] = {
            "trials": level_results,
            "avg_knotted_p": round(avg_knotted, 4),
        }

    results["total_time_s"] = round(time.time() - t0_total, 1)
    return results


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_dataset_proteins(tool: str = "Real", label: int = 1, limit: int | None = None) -> list[dict]:
    """Load proteins from HuggingFace dataset."""
    from datasets import load_dataset

    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [
        {"id": r["ID"], "sequence": r["Sequence"], "label": r["Label"]}
        for r in ds
        if r["Tool"] == tool and r["Label"] == label
    ]
    if limit:
        proteins = proteins[:limit]
    print(f"Loaded {len(proteins)} proteins (Tool={tool}, Label={label})")
    return proteins


@app.local_entrypoint()
def main(
    n_proteins: int = 5,
    n_trials: int = 4,
    levels: str = "25,50,75",
    topoly_tries: int = 30,
):
    masking_percentages = [int(x) for x in levels.split(",")]

    print(f"Config: {n_proteins} proteins, {len(masking_percentages)} levels {masking_percentages}, {n_trials} trials")
    print(f"Total trials: {n_proteins * len(masking_percentages) * n_trials}")

    proteins = load_dataset_proteins.remote(tool="Real", label=1, limit=n_proteins)

    t0 = time.time()
    results = list(
        process_one_protein.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            [masking_percentages] * len(proteins),
            [n_trials] * len(proteins),
            [topoly_tries] * len(proteins),
            return_exceptions=True,
        )
    )

    # Handle any exceptions from individual containers
    clean_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"ERROR on protein {proteins[i]['id']}: {r}")
            clean_results.append({"protein_id": proteins[i]["id"], "seq_len": len(proteins[i]["sequence"]),
                                   "levels": {}, "total_time_s": 0, "error": str(r)})
        else:
            clean_results.append(r)
    results = clean_results
    wall_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS ({wall_time:.1f}s wall time)")
    print(f"{'='*60}")

    for r in results:
        print(f"\n{r['protein_id']} (len={r['seq_len']}, {r['total_time_s']}s GPU):")
        for pct, data in sorted(r["levels"].items(), key=lambda x: int(x[0])):
            print(f"  mask={pct}%: avg_knotted_p={data['avg_knotted_p']:.3f}")

    total_gpu_s = sum(r["total_time_s"] for r in results)
    trials_total = n_proteins * len(masking_percentages) * n_trials
    per_trial = total_gpu_s / trials_total if trials_total > 0 else 0

    print(f"\nTotal GPU time: {total_gpu_s:.0f}s ({total_gpu_s/3600:.2f} GPU-hours)")
    print(f"Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)")
    print(f"Per trial (avg): {per_trial:.2f}s")
    print(f"Parallelism factor: {total_gpu_s/wall_time:.1f}x")

    Path("results").mkdir(exist_ok=True)
    outfile = Path("results") / "masking_smoke.json"
    with open(outfile, "w") as f:
        json.dump({"config": {"n_proteins": n_proteins, "n_trials": n_trials,
                               "masking_percentages": masking_percentages},
                    "wall_time_s": round(wall_time, 1),
                    "total_gpu_s": round(total_gpu_s, 1),
                    "per_trial_s": round(per_trial, 2),
                    "results": results}, f, indent=2)
    print(f"\nSaved to {outfile}")
