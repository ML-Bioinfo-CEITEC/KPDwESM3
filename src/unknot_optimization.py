"""Unknotted-to-knotted parameter optimization.

Tests several strategies:
1. Original params with more iterations (10 instead of 5)
2. Higher masking (10%) with more samples (15)
3. Start from shorter proteins only (<250 aa)
4. Use structure track in guided generation instead of sequence
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-opt")

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
def run_config(
    config_name: str,
    proteins: list[dict],
    masking_pct: int,
    max_iters: int,
    samples_per_step: int,
    decoding_divisor: int,
    min_score: float,
) -> dict:
    import os, random, tempfile, warnings
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
            topo = alexander(f.name, tries=100)
        total = sum(topo.values())
        return 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0

    class KnotSF(GuidedDecodingScoringFunction):
        def __call__(self, protein):
            assert protein.ptm is not None
            return float(knot_status(protein))

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotSF())

    results = []
    for p in proteins:
        t0 = time.time()
        current_seq = p["sequence"]
        success = False
        final_score = 0.0

        for it in range(max_iters):
            try:
                random.seed(hash((p["id"], it, config_name)) & 0xFFFFFFFF)
                seq_list = list(current_seq)
                n_mask = max(1, int(len(seq_list) * masking_pct / 100))
                for i in random.sample(range(len(seq_list)), n_mask):
                    seq_list[i] = "_"

                num_steps = max(1, len(current_seq) // decoding_divisor)
                generated = guided.guided_generate(
                    protein=ESMProtein(sequence="".join(seq_list)),
                    num_decoding_steps=num_steps,
                    num_samples_per_step=samples_per_step,
                    verbose=False,
                )
                score = knot_status(generated)
                current_seq = generated.sequence
                final_score = score

                if score >= min_score:
                    success = True
                    break
            except:
                break

        results.append({
            "id": p["id"], "success": success,
            "final_score": round(final_score, 4),
            "time_s": round(time.time() - t0, 1),
        })
        s = "OK" if success else "fail"
        print(f"  {config_name} | {p['id']}: {s} ({final_score:.2f})")

    n_success = sum(1 for r in results if r["success"])
    return {
        "config": config_name,
        "n_success": n_success,
        "n_total": len(results),
        "rate": round(n_success / len(results), 4) if results else 0,
        "results": results,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_proteins(max_len: int = 250, limit: int = 20) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [{"id": r["ID"], "sequence": r["Sequence"]}
                for r in ds if r["Tool"] == "Real" and r["Label"] == 0
                and len(r["Sequence"]) <= max_len][:limit]
    print(f"Loaded {len(proteins)} unknotted proteins (<={max_len} aa)")
    return proteins


@app.local_entrypoint()
def main():
    configs = [
        # Original notebook style, more iterations
        {"name": "orig_10iter",    "mask": 5,  "iters": 10, "samp": 10, "div": 64, "score": 0.80},
        # Higher masking, more samples
        {"name": "high_mask_15s",  "mask": 10, "iters": 10, "samp": 15, "div": 64, "score": 0.75},
        # Very gentle: 3% mask, many iters
        {"name": "gentle_3pct",    "mask": 3,  "iters": 15, "samp": 10, "div": 64, "score": 0.80},
        # Aggressive: 20% mask, many samples
        {"name": "aggressive_20",  "mask": 20, "iters": 5,  "samp": 15, "div": 100, "score": 0.75},
    ]

    proteins = load_proteins.remote(max_len=250, limit=20)
    print(f"Testing {len(configs)} configs on {len(proteins)} short proteins\n")

    t0 = time.time()
    results = list(
        run_config.map(
            [c["name"] for c in configs],
            [proteins] * len(configs),
            [c["mask"] for c in configs],
            [c["iters"] for c in configs],
            [c["samp"] for c in configs],
            [c["div"] for c in configs],
            [c["score"] for c in configs],
            return_exceptions=True,
        )
    )
    wall_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"UNKNOT-TO-KNOT OPTIMIZATION ({wall_time/60:.1f} min)")
    print(f"{'='*60}")
    for r in results:
        if isinstance(r, Exception):
            print(f"  ERROR: {r}")
        else:
            print(f"  {r['config']:<25} {r['n_success']}/{r['n_total']} = {r['rate']*100:.0f}%")

    Path("results").mkdir(exist_ok=True)
    clean = [r for r in results if not isinstance(r, Exception)]
    with open("results/unknot_optimization.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "configs": clean}, f, indent=2)
    print(f"\nSaved to results/unknot_optimization.json")
