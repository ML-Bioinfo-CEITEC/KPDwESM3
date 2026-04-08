"""Unknot-to-knot optimization v2: one protein per GPU, multiple configs tried sequentially."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-opt2")

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

CONFIGS = [
    {"name": "gentle_3pct",   "mask": 3,  "iters": 15, "samp": 10, "div": 64},
    {"name": "orig_5pct",     "mask": 5,  "iters": 10, "samp": 10, "div": 64},
    {"name": "medium_10pct",  "mask": 10, "iters": 10, "samp": 15, "div": 64},
    {"name": "aggro_20pct",   "mask": 20, "iters": 5,  "samp": 10, "div": 100},
]


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=20 * MINUTES,
)
def try_all_configs(protein_id: str, sequence: str) -> dict:
    """Try all configs on one protein, return the best result."""
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

    config_results = {}
    for cfg in CONFIGS:
        t0 = time.time()
        current_seq = sequence
        best_score = 0.0
        success = False

        for it in range(cfg["iters"]):
            try:
                random.seed(hash((protein_id, it, cfg["name"])) & 0xFFFFFFFF)
                seq_list = list(current_seq)
                n_mask = max(1, int(len(seq_list) * cfg["mask"] / 100))
                for i in random.sample(range(len(seq_list)), n_mask):
                    seq_list[i] = "_"

                num_steps = max(1, len(current_seq) // cfg["div"])
                generated = guided.guided_generate(
                    protein=ESMProtein(sequence="".join(seq_list)),
                    num_decoding_steps=num_steps,
                    num_samples_per_step=cfg["samp"],
                    verbose=False,
                )
                score = knot_status(generated)
                current_seq = generated.sequence
                best_score = max(best_score, score)

                if score >= 0.80:
                    success = True
                    break
            except:
                break

        config_results[cfg["name"]] = {
            "success": success,
            "best_score": round(best_score, 4),
            "time_s": round(time.time() - t0, 1),
        }

        if success:
            break  # No need to try other configs

    any_success = any(r["success"] for r in config_results.values())
    best_config = max(config_results.items(), key=lambda x: x[1]["best_score"])

    return {
        "protein_id": protein_id,
        "seq_len": len(sequence),
        "any_success": any_success,
        "best_config": best_config[0],
        "best_score": best_config[1]["best_score"],
        "config_results": config_results,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_proteins(max_len: int = 250, limit: int = 50) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    return [{"id": r["ID"], "sequence": r["Sequence"]}
            for r in ds if r["Tool"] == "Real" and r["Label"] == 0
            and len(r["Sequence"]) <= max_len][:limit]


@app.local_entrypoint()
def main(n_proteins: int = 50, max_len: int = 250):
    proteins = load_proteins.remote(max_len=max_len, limit=n_proteins)
    print(f"Testing {len(proteins)} proteins (max {max_len} aa), 4 configs each")
    print(f"Each protein tries configs until one succeeds or all fail")

    t0 = time.time()
    results = list(
        try_all_configs.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            return_exceptions=True,
        )
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"protein_id": proteins[i]["id"], "any_success": False,
                          "error": str(r)})
        else:
            clean.append(r)
    wall_time = time.time() - t0

    n_success = sum(1 for r in clean if r.get("any_success"))
    n_valid = sum(1 for r in clean if "error" not in r)

    print(f"\n{'='*60}")
    print(f"UNKNOT-TO-KNOT OPTIMIZATION ({wall_time/60:.1f} min)")
    print(f"{'='*60}")
    print(f"Success: {n_success}/{n_valid} ({n_success/n_valid*100:.0f}%)")

    # Which configs worked?
    from collections import Counter
    winning_configs = Counter(r.get("best_config") for r in clean if r.get("any_success"))
    print(f"\nWinning configs: {dict(winning_configs)}")

    for r in clean:
        if r.get("any_success"):
            print(f"  {r['protein_id']}: {r['best_config']} (score={r['best_score']:.2f})")

    Path("results").mkdir(exist_ok=True)
    with open("results/unknot_opt_v2.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "n_success": n_success,
                    "n_valid": n_valid, "rate": round(n_success/n_valid, 4) if n_valid else 0,
                    "results": clean}, f, indent=2)
    print(f"\nSaved to results/unknot_opt_v2.json")
