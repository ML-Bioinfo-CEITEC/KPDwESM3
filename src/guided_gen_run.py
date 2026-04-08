"""Guided generation of knotted proteins -- parallelized across Modal containers."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-guided-gen")

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
    timeout=30 * MINUTES,
)
def generate_one(
    attempt_id: int,
    protein_length: int = 256,
    num_samples_per_step: int = 10,
) -> dict:
    """One guided generation attempt on its own GPU."""
    import os
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

    num_decoding_steps = protein_length // 32

    def knot_status(protein: ESMProtein) -> float:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            pdb_path = f.name
        topo = alexander(pdb_path, tries=30)
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
    try:
        generated = guided.guided_generate(
            protein=ESMProtein(sequence="_" * protein_length),
            num_decoding_steps=num_decoding_steps,
            num_samples_per_step=num_samples_per_step,
        )
        elapsed = time.time() - t0
        score = knot_status(generated)

        # Also get the topology breakdown
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            generated.to_pdb(f.name)
            topo = alexander(f.name, tries=30)

        return {
            "attempt": attempt_id,
            "time_s": round(elapsed, 1),
            "knot_score": round(score, 4),
            "is_knotted": score > 0.5,
            "topology": {k: round(v, 4) for k, v in topo.items()},
            "sequence": generated.sequence,
            "error": None,
        }
    except Exception as e:
        return {
            "attempt": attempt_id,
            "time_s": round(time.time() - t0, 1),
            "knot_score": 0.0,
            "is_knotted": False,
            "topology": {},
            "sequence": None,
            "error": str(e),
        }


@app.local_entrypoint()
def main(n_attempts: int = 30, protein_length: int = 256, samples_per_step: int = 10):
    print(f"Launching {n_attempts} guided generation attempts in parallel...")
    print(f"  protein_length={protein_length}, samples_per_step={samples_per_step}")

    t0 = time.time()
    results = list(
        generate_one.map(
            list(range(n_attempts)),
            [protein_length] * n_attempts,
            [samples_per_step] * n_attempts,
        )
    )
    wall_time = time.time() - t0

    n_knotted = sum(1 for r in results if r["is_knotted"])
    n_errors = sum(1 for r in results if r["error"])
    avg_time = sum(r["time_s"] for r in results) / len(results)
    total_gpu = sum(r["time_s"] for r in results)

    print(f"\n{'='*60}")
    print(f"GUIDED GENERATION RESULTS")
    print(f"{'='*60}")
    print(f"Success: {n_knotted}/{n_attempts} ({n_knotted/n_attempts*100:.0f}%)")
    print(f"Errors: {n_errors}")
    print(f"Avg time/attempt: {avg_time:.0f}s")
    print(f"Total GPU time: {total_gpu:.0f}s ({total_gpu/3600:.2f} GPU-hrs)")
    print(f"Wall time: {wall_time:.0f}s ({wall_time/60:.1f} min)")

    for r in sorted(results, key=lambda x: x["attempt"]):
        status = "KNOTTED" if r["is_knotted"] else "unknotted"
        topo = r.get("topology", {})
        print(f"  #{r['attempt']:2d}: {status} (score={r['knot_score']:.2f}, {r['time_s']}s) {topo}")

    Path("results").mkdir(exist_ok=True)
    outfile = Path("results") / f"guided_gen_{n_attempts}.json"
    with open(outfile, "w") as f:
        json.dump({
            "wall_time_s": round(wall_time, 1),
            "total_gpu_s": round(total_gpu, 1),
            "success_rate": round(n_knotted / n_attempts, 4),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {outfile}")
