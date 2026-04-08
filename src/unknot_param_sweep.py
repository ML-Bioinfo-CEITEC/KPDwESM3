"""Parameter sweep for unknotted-to-knotted conversion.

Tests different masking %, decoding steps, and samples/step
to find what matches the paper's 31% success rate.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-sweep")

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
    timeout=60 * MINUTES,
)
def run_sweep_config(
    config_name: str,
    proteins: list[dict],
    masking_percent: int,
    decoding_steps_divisor: int,
    samples_per_step: int,
    min_knot_score: float,
    max_iterations: int,
    topoly_tries: int,
) -> dict:
    """Run one parameter config across multiple proteins on one GPU."""
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

    def knot_status(protein: ESMProtein) -> float:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            pdb_path = f.name
        topo = alexander(pdb_path, tries=topoly_tries)
        total = sum(topo.values())
        if total == 0:
            return 0.0
        return 1.0 - topo.get("0_1", 0) / total

    class KnotScoringFunction(GuidedDecodingScoringFunction):
        def __call__(self, protein: ESMProtein) -> float:
            assert protein.ptm is not None
            return float(knot_status(protein))

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotScoringFunction())

    results = []
    for p in proteins:
        if len(p["sequence"]) > 500:
            results.append({"id": p["id"], "success": False, "error": "too_long",
                           "iterations": 0, "final_score": 0})
            continue

        t0 = time.time()
        current_seq = p["sequence"]
        success = False
        final_score = 0.0
        iters_done = 0

        for iteration in range(max_iterations):
            try:
                random.seed(hash((p["id"], iteration)) & 0xFFFFFFFF)
                seq_list = list(current_seq)
                n_mask = max(1, int(len(seq_list) * masking_percent / 100))
                indices = random.sample(range(len(seq_list)), n_mask)
                for i in indices:
                    seq_list[i] = "_"
                masked_seq = "".join(seq_list)

                num_steps = max(1, len(current_seq) // decoding_steps_divisor)

                generated = guided.guided_generate(
                    protein=ESMProtein(sequence=masked_seq),
                    num_decoding_steps=num_steps,
                    num_samples_per_step=samples_per_step,
                    verbose=False,
                )

                score = knot_status(generated)
                current_seq = generated.sequence
                iters_done = iteration + 1
                final_score = score

                if score >= min_knot_score:
                    success = True
                    break
            except Exception as e:
                print(f"  {config_name} {p['id']} iter={iteration}: {e}")
                break

        elapsed = time.time() - t0
        results.append({
            "id": p["id"],
            "success": success,
            "iterations": iters_done,
            "final_score": round(final_score, 4),
            "time_s": round(elapsed, 1),
            "error": None,
        })
        status = "KNOTTED" if success else "failed"
        print(f"  {config_name} | {p['id']}: {status} (iter={iters_done}, score={final_score:.2f}, {elapsed:.0f}s)")

    n_valid = sum(1 for r in results if r.get("error") != "too_long")
    n_success = sum(1 for r in results if r["success"])
    rate = n_success / n_valid if n_valid > 0 else 0

    return {
        "config_name": config_name,
        "masking_percent": masking_percent,
        "decoding_steps_divisor": decoding_steps_divisor,
        "samples_per_step": samples_per_step,
        "min_knot_score": min_knot_score,
        "n_proteins": len(proteins),
        "n_valid": n_valid,
        "n_success": n_success,
        "success_rate": round(rate, 4),
        "results": results,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_unknotted(limit: int = 10) -> list[dict]:
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
def main(n_proteins: int = 10):
    configs = [
        # Original notebook (Knotting_unknotted.ipynb)
        {"name": "orig_5pct_div64_10samp",  "mask": 5,  "div": 64,  "samp": 10, "thresh": 0.80},
        # Copy1
        {"name": "copy1_5pct_div100_10samp", "mask": 5,  "div": 100, "samp": 10, "thresh": 0.80},
        # Copy2
        {"name": "copy2_20pct_div100_5samp", "mask": 20, "div": 100, "samp": 5,  "thresh": 0.75},
        # My original (too aggressive)
        {"name": "mine_20pct_divN8_5samp",   "mask": 20, "div": 50,  "samp": 5,  "thresh": 0.75},
    ]

    proteins = load_unknotted.remote(limit=n_proteins)
    print(f"Testing {len(configs)} configs on {len(proteins)} proteins each\n")

    t0 = time.time()
    results = list(
        run_sweep_config.map(
            [c["name"] for c in configs],
            [proteins] * len(configs),
            [c["mask"] for c in configs],
            [c["div"] for c in configs],
            [c["samp"] for c in configs],
            [c["thresh"] for c in configs],
            [5] * len(configs),  # max_iterations
            [100] * len(configs),  # topoly_tries
            return_exceptions=True,
        )
    )
    wall_time = time.time() - t0

    print(f"\n{'='*70}")
    print(f"PARAMETER SWEEP RESULTS ({wall_time/60:.1f} min)")
    print(f"{'='*70}")
    print(f"{'Config':<35} {'Success':>10} {'Rate':>8}")
    print("-" * 55)

    clean = []
    for r in results:
        if isinstance(r, Exception):
            print(f"ERROR: {r}")
        else:
            clean.append(r)
            print(f"{r['config_name']:<35} {r['n_success']}/{r['n_valid']:>3}    {r['success_rate']*100:>5.1f}%")

    Path("results").mkdir(exist_ok=True)
    with open("results/unknot_param_sweep.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "configs": clean}, f, indent=2)
    print(f"\nSaved to results/unknot_param_sweep.json")
