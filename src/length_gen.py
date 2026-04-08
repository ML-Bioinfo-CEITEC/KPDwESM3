"""Test guided generation success rate at different protein lengths."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-length-gen")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("esm==3.2.3", "torch==2.5.1", "topoly", "huggingface-hub")
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=30 * MINUTES,
)
def generate_one(attempt_id: int, protein_length: int) -> dict:
    import os, tempfile, warnings
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

    def knot_status(protein):
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            topo = alexander(f.name, tries=30)
        total = sum(topo.values())
        return 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0

    class KnotSF(GuidedDecodingScoringFunction):
        def __call__(self, protein):
            assert protein.ptm is not None
            return float(knot_status(protein))

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotSF())

    t0 = time.time()
    try:
        generated = guided.guided_generate(
            protein=ESMProtein(sequence="_" * protein_length),
            num_decoding_steps=protein_length // 32,
            num_samples_per_step=10, verbose=False)
        score = knot_status(generated)
        return {"attempt": attempt_id, "length": protein_length,
                "knot_score": round(score, 4), "is_knotted": score > 0.5,
                "time_s": round(time.time() - t0, 1), "error": None}
    except Exception as e:
        return {"attempt": attempt_id, "length": protein_length,
                "knot_score": 0, "is_knotted": False,
                "time_s": round(time.time() - t0, 1), "error": str(e)}


@app.local_entrypoint()
def main(n_per_length: int = 10):
    lengths = [100, 150, 200, 256, 350, 500]
    total = len(lengths) * n_per_length
    print(f"Testing {len(lengths)} lengths x {n_per_length} attempts = {total} total")

    all_ids, all_lens = [], []
    for length in lengths:
        for i in range(n_per_length):
            all_ids.append(i + length * 1000)
            all_lens.append(length)

    t0 = time.time()
    results = list(generate_one.map(all_ids, all_lens, return_exceptions=True))
    wall_time = time.time() - t0

    clean = [r if not isinstance(r, Exception) else
             {"attempt": all_ids[i], "length": all_lens[i], "error": str(r)}
             for i, r in enumerate(results)]

    import numpy as np
    print(f"\n{'='*50}")
    print(f"LENGTH-DEPENDENT GENERATION ({wall_time/60:.1f} min)")
    print(f"{'='*50}")
    for length in lengths:
        batch = [r for r in clean if r.get("length") == length and r.get("error") is None]
        n_knotted = sum(1 for r in batch if r.get("is_knotted"))
        if batch:
            scores = [r["knot_score"] for r in batch]
            print(f"  len={length:>4}: {n_knotted}/{len(batch)} knotted ({n_knotted/len(batch)*100:.0f}%), "
                  f"mean_score={np.mean(scores):.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/length_gen.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "results": clean}, f, indent=2)
    print(f"\nSaved to results/length_gen.json")
