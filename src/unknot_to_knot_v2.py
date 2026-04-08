"""Unknotted-to-knotted v2: one protein per GPU, matching original notebook params.

Original notebook (Knotting_unknotted.ipynb) used:
  MASKINGP=5, num_decoding_steps=len(seq)//64, num_samples_per_step=10,
  NTRIES=5, MINIMALKNOTP=0.80, topoly tries=100
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-v2")

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
def transform_one(protein_id: str, sequence: str) -> dict:
    import os
    import random
    import tempfile
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein
    from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    MASKING_PCT = 5
    MAX_ITERS = 5
    MIN_KNOT_SCORE = 0.80
    TOPOLY_TRIES = 100

    def knot_status(protein: ESMProtein) -> float:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            pdb_path = f.name
        topo = alexander(pdb_path, tries=TOPOLY_TRIES)
        total = sum(topo.values())
        if total == 0:
            return 0.0
        return 1.0 - topo.get("0_1", 0) / total

    class KnotScoringFunction(GuidedDecodingScoringFunction):
        def __call__(self, protein: ESMProtein) -> float:
            assert protein.ptm is not None
            return float(knot_status(protein))

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotScoringFunction())

    t0 = time.time()
    current_seq = sequence
    history = []

    for iteration in range(MAX_ITERS):
        try:
            random.seed(hash((protein_id, iteration)) & 0xFFFFFFFF)
            seq_list = list(current_seq)
            n_mask = max(1, int(len(seq_list) * MASKING_PCT / 100))
            indices = random.sample(range(len(seq_list)), n_mask)
            for i in indices:
                seq_list[i] = "_"
            masked_seq = "".join(seq_list)

            num_steps = max(1, len(current_seq) // 64)

            generated = guided.guided_generate(
                protein=ESMProtein(sequence=masked_seq),
                num_decoding_steps=num_steps,
                num_samples_per_step=10,
                verbose=False,
            )

            score = knot_status(generated)
            current_seq = generated.sequence
            history.append({"iter": iteration, "score": round(score, 4)})

            if score >= MIN_KNOT_SCORE:
                return {
                    "protein_id": protein_id, "seq_len": len(sequence),
                    "success": True, "iterations": iteration + 1,
                    "final_score": round(score, 4),
                    "time_s": round(time.time() - t0, 1),
                    "history": history, "error": None,
                }
        except Exception as e:
            history.append({"iter": iteration, "error": str(e)})
            break

    return {
        "protein_id": protein_id, "seq_len": len(sequence),
        "success": False, "iterations": len(history),
        "final_score": round(history[-1].get("score", 0), 4) if history else 0,
        "time_s": round(time.time() - t0, 1),
        "history": history, "error": None,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_unknotted(limit: int = 50) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [
        {"id": r["ID"], "sequence": r["Sequence"]}
        for r in ds
        if r["Tool"] == "Real" and r["Label"] == 0 and len(r["Sequence"]) <= 400
    ][:limit]
    print(f"Loaded {len(proteins)} unknotted proteins (<=400 aa)")
    return proteins


@app.local_entrypoint()
def main(n_proteins: int = 50):
    print(f"Unknotted-to-knotted v2: {n_proteins} proteins, original notebook params")
    print(f"  mask=5%, steps=len//64, samples=10, iters=5, threshold=0.80, topoly=100")
    print(f"  Each protein on own GPU, ~5-10 min each")
    print(f"  With 10 GPU limit: ~{(n_proteins // 10 + 1) * 10} min wall time est.")

    proteins = load_unknotted.remote(limit=n_proteins)

    t0 = time.time()
    results = list(
        transform_one.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            return_exceptions=True,
        )
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"protein_id": proteins[i]["id"], "success": False,
                          "error": str(r), "iterations": 0, "final_score": 0})
        else:
            clean.append(r)
    results = clean
    wall_time = time.time() - t0

    n_success = sum(1 for r in results if r.get("success"))
    n_valid = sum(1 for r in results if r.get("error") is None)

    print(f"\n{'='*60}")
    print(f"UNKNOTTED-TO-KNOTTED v2 ({wall_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Success: {n_success}/{n_valid} ({n_success/n_valid*100:.0f}% of valid)")
    print(f"Total: {len(results)}, Errors: {len(results) - n_valid}")

    for r in results:
        s = "KNOTTED" if r.get("success") else "failed"
        print(f"  {r['protein_id']}: {s} (iter={r.get('iterations',0)}, "
              f"score={r.get('final_score',0):.2f}, {r.get('time_s',0):.0f}s)")

    total_gpu = sum(r.get("time_s", 0) for r in results)
    print(f"\nGPU time: {total_gpu/3600:.2f} hrs, Wall: {wall_time/60:.1f} min")

    Path("results").mkdir(exist_ok=True)
    with open("results/unknot_v2.json", "w") as f:
        json.dump({
            "wall_time_s": round(wall_time, 1),
            "success_rate": round(n_success / n_valid, 4) if n_valid else 0,
            "n_success": n_success, "n_valid": n_valid,
            "results": results,
        }, f, indent=2)
    print(f"Saved to results/unknot_v2.json")
