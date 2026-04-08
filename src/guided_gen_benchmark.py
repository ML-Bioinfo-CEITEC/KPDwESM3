"""Benchmark guided generation of knotted proteins.

Tests de novo generation (fully masked -> knotted protein) using ESM3GuidedDecoding
with the knot scoring function from the paper.
"""

import time
from pathlib import Path

import modal

app = modal.App("esm3-guided-bench")

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
    )
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=60 * MINUTES,
)
def benchmark_guided_generation(
    n_attempts: int = 3,
    protein_length: int = 256,
    num_decoding_steps: int | None = None,
    num_samples_per_step: int = 10,
) -> dict:
    import json
    import os
    import tempfile
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    if num_decoding_steps is None:
        num_decoding_steps = protein_length // 32

    print(f"Config: {n_attempts} attempts, length={protein_length}, "
          f"steps={num_decoding_steps}, samples/step={num_samples_per_step}")

    # Scoring function: write PDB, run topoly, return knotted probability
    def knot_status(protein: ESMProtein) -> float:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            pdb_path = f.name
            protein.to_pdb(pdb_path)
        topo = alexander(pdb_path, tries=30)
        total = sum(topo.values())
        if total == 0:
            return 0.0
        unknot_frac = topo.get("0_1", 0) / total
        return 1.0 - unknot_frac

    class KnotScoringFunction(GuidedDecodingScoringFunction):
        def __call__(self, protein: ESMProtein) -> float:
            assert protein.ptm is not None, "Protein must have pTM scores"
            return float(knot_status(protein))

    guided_decoding = ESM3GuidedDecoding(
        client=model, scoring_function=KnotScoringFunction()
    )

    results = []
    for i in range(n_attempts):
        print(f"\n--- Attempt {i+1}/{n_attempts} ---")
        starting_protein = ESMProtein(sequence="_" * protein_length)

        t0 = time.time()
        try:
            generated = guided_decoding.guided_generate(
                protein=starting_protein,
                num_decoding_steps=num_decoding_steps,
                num_samples_per_step=num_samples_per_step,
            )
            elapsed = time.time() - t0

            score = knot_status(generated)
            print(f"  Time: {elapsed:.1f}s, knot_score: {score:.3f}, "
                  f"seq_len: {len(generated.sequence)}")

            results.append({
                "attempt": i,
                "time_s": round(elapsed, 1),
                "knot_score": round(score, 4),
                "is_knotted": score > 0.5,
                "sequence": generated.sequence,
                "error": None,
            })
        except Exception as e:
            elapsed = time.time() - t0
            print(f"  FAILED after {elapsed:.1f}s: {e}")
            results.append({
                "attempt": i,
                "time_s": round(elapsed, 1),
                "knot_score": 0.0,
                "is_knotted": False,
                "sequence": None,
                "error": str(e),
            })

    n_knotted = sum(1 for r in results if r["is_knotted"])
    avg_time = sum(r["time_s"] for r in results) / len(results)
    success_rate = n_knotted / len(results)

    summary = {
        "config": {
            "n_attempts": n_attempts,
            "protein_length": protein_length,
            "num_decoding_steps": num_decoding_steps,
            "num_samples_per_step": num_samples_per_step,
        },
        "success_rate": round(success_rate, 3),
        "n_knotted": n_knotted,
        "avg_time_per_attempt_s": round(avg_time, 1),
        "results": results,
    }

    print(f"\n{'='*60}")
    print(f"SUMMARY: {n_knotted}/{n_attempts} knotted ({success_rate*100:.0f}%)")
    print(f"Avg time per attempt: {avg_time:.1f}s")
    print(f"{'='*60}")

    return summary


@app.local_entrypoint()
def main(
    n_attempts: int = 3,
    protein_length: int = 256,
    samples_per_step: int = 10,
):
    import json

    result = benchmark_guided_generation.remote(
        n_attempts=n_attempts,
        protein_length=protein_length,
        num_samples_per_step=samples_per_step,
    )

    print(f"\nSuccess rate: {result['success_rate']*100:.0f}%")
    print(f"Avg time: {result['avg_time_per_attempt_s']}s per attempt")

    Path("results").mkdir(exist_ok=True)
    with open("results/guided_gen_bench.json", "w") as f:
        json.dump(result, f, indent=2)
    print("Saved to results/guided_gen_bench.json")
