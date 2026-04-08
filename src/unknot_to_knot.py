"""Unknotted-to-knotted transformation experiment.

Start from real unknotted proteins and try to convert them to knotted
using iterative guided generation (mask 5%, guided regenerate, repeat).
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot2knot")

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
def transform_one_protein(
    protein_id: str,
    sequence: str,
    max_iterations: int = 10,
    masking_percent: int = 20,
    min_knot_score: float = 0.75,
    num_samples_per_step: int = 5,
) -> dict:
    """Try to convert one unknotted protein to knotted via iterative guided generation."""
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

    if len(sequence) > 500:
        return {"protein_id": protein_id, "seq_len": len(sequence),
                "success": False, "iterations": 0, "time_s": 0,
                "error": "sequence_too_long", "history": []}

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

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
    current_seq = sequence
    history = []

    for iteration in range(max_iterations):
        try:
            # Mask masking_percent% of the sequence
            import random
            random.seed(hash((protein_id, iteration)) & 0xFFFFFFFF)
            seq_list = list(current_seq)
            n_mask = max(1, int(len(seq_list) * masking_percent / 100))
            indices = random.sample(range(len(seq_list)), n_mask)
            for i in indices:
                seq_list[i] = "_"
            masked_seq = "".join(seq_list)

            num_decoding_steps = max(1, n_mask // 8)

            generated = guided.guided_generate(
                protein=ESMProtein(sequence=masked_seq),
                num_decoding_steps=num_decoding_steps,
                num_samples_per_step=num_samples_per_step,
                verbose=False,
            )

            score = knot_status(generated)
            current_seq = generated.sequence

            history.append({
                "iteration": iteration,
                "knot_score": round(score, 4),
                "n_changes": sum(1 for a, b in zip(sequence, current_seq) if a != b),
            })

            print(f"  {protein_id} iter={iteration}: score={score:.3f}")

            if score >= min_knot_score:
                elapsed = time.time() - t0
                return {
                    "protein_id": protein_id,
                    "seq_len": len(sequence),
                    "success": True,
                    "iterations": iteration + 1,
                    "final_score": round(score, 4),
                    "time_s": round(elapsed, 1),
                    "final_sequence": current_seq,
                    "error": None,
                    "history": history,
                }
        except Exception as e:
            print(f"  {protein_id} iter={iteration}: ERROR {e}")
            history.append({"iteration": iteration, "error": str(e)})

    elapsed = time.time() - t0
    return {
        "protein_id": protein_id,
        "seq_len": len(sequence),
        "success": False,
        "iterations": max_iterations,
        "final_score": round(history[-1].get("knot_score", 0), 4) if history else 0,
        "time_s": round(elapsed, 1),
        "final_sequence": current_seq,
        "error": None,
        "history": history,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_unknotted_proteins(limit: int | None = None) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [
        {"id": r["ID"], "sequence": r["Sequence"], "label": r["Label"]}
        for r in ds
        if r["Tool"] == "Real" and r["Label"] == 0
    ]
    if limit:
        proteins = proteins[:limit]
    print(f"Loaded {len(proteins)} unknotted proteins")
    return proteins


@app.local_entrypoint()
def main(
    n_proteins: int = 10,
    max_iterations: int = 10,
    masking_percent: int = 20,
):
    print(f"Config: {n_proteins} unknotted proteins, max {max_iterations} iterations, "
          f"{masking_percent}% masking per iteration")

    proteins = load_unknotted_proteins.remote(limit=n_proteins)

    t0 = time.time()
    results = list(
        transform_one_protein.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            [max_iterations] * len(proteins),
            [masking_percent] * len(proteins),
            return_exceptions=True,
        )
    )

    clean_results = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"ERROR on {proteins[i]['id']}: {r}")
            clean_results.append({"protein_id": proteins[i]["id"], "success": False,
                                   "error": str(r), "history": []})
        else:
            clean_results.append(r)
    results = clean_results
    wall_time = time.time() - t0

    n_success = sum(1 for r in results if r.get("success"))
    n_errors = sum(1 for r in results if r.get("error"))

    print(f"\n{'='*60}")
    print(f"UNKNOTTED-TO-KNOTTED RESULTS ({wall_time/60:.1f} min wall time)")
    print(f"{'='*60}")
    print(f"Success: {n_success}/{len(results)} ({n_success/len(results)*100:.0f}%)")
    print(f"Errors: {n_errors}")

    for r in results:
        status = "KNOTTED" if r.get("success") else "failed"
        iters = r.get("iterations", "?")
        score = r.get("final_score", 0)
        print(f"  {r['protein_id']}: {status} (iters={iters}, score={score:.2f}, {r.get('time_s', 0):.0f}s)")

    Path("results").mkdir(exist_ok=True)
    outfile = Path("results") / "unknot_to_knot.json"
    with open(outfile, "w") as f:
        json.dump({
            "wall_time_s": round(wall_time, 1),
            "success_rate": round(n_success / len(results), 4) if results else 0,
            "n_success": n_success,
            "n_total": len(results),
            "results": results,
        }, f, indent=2)
    print(f"\nSaved to {outfile}")
