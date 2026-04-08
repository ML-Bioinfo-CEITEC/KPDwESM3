"""Final unknotted-to-knotted run: 100 short proteins, gentle_3pct config, 40 min timeout."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-final")

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
    timeout=40 * MINUTES,
)
def transform_one(protein_id: str, sequence: str) -> dict:
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

    MASK_PCT = 3
    MAX_ITERS = 15
    SAMPLES = 10
    DIV = 64
    MIN_SCORE = 0.80

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

    t0 = time.time()
    current_seq = sequence
    history = []

    for it in range(MAX_ITERS):
        try:
            random.seed(hash((protein_id, it)) & 0xFFFFFFFF)
            seq_list = list(current_seq)
            n_mask = max(1, int(len(seq_list) * MASK_PCT / 100))
            for i in random.sample(range(len(seq_list)), n_mask):
                seq_list[i] = "_"

            num_steps = max(1, len(current_seq) // DIV)
            generated = guided.guided_generate(
                protein=ESMProtein(sequence="".join(seq_list)),
                num_decoding_steps=num_steps,
                num_samples_per_step=SAMPLES,
                verbose=False,
            )
            score = knot_status(generated)
            current_seq = generated.sequence
            history.append({"iter": it, "score": round(score, 4)})

            if score >= MIN_SCORE:
                return {
                    "protein_id": protein_id, "seq_len": len(sequence),
                    "success": True, "iterations": it + 1,
                    "final_score": round(score, 4),
                    "time_s": round(time.time() - t0, 1),
                    "history": history, "error": None,
                }
        except Exception as e:
            history.append({"iter": it, "error": str(e)})
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
def load_proteins(max_len: int = 250, limit: int = 100) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    return [{"id": r["ID"], "sequence": r["Sequence"]}
            for r in ds if r["Tool"] == "Real" and r["Label"] == 0
            and len(r["Sequence"]) <= max_len][:limit]


@app.local_entrypoint()
def main(n_proteins: int = 100):
    proteins = load_proteins.remote(max_len=250, limit=n_proteins)
    print(f"Unknot-to-knot FINAL: {len(proteins)} proteins, gentle_3pct, 40 min timeout")

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
    wall_time = time.time() - t0

    n_success = sum(1 for r in clean if r.get("success"))
    n_errors = sum(1 for r in clean if r.get("error"))
    n_completed = len(clean) - n_errors

    print(f"\n{'='*60}")
    print(f"UNKNOT-TO-KNOT FINAL ({wall_time/60:.0f} min)")
    print(f"{'='*60}")
    print(f"Completed: {n_completed}/{len(clean)} (errors/timeouts: {n_errors})")
    print(f"Success: {n_success}/{n_completed} = {n_success/n_completed*100:.0f}% (of completed)")
    print(f"Success: {n_success}/{len(clean)} = {n_success/len(clean)*100:.0f}% (of all)")

    for r in sorted(clean, key=lambda x: -x.get("final_score", 0))[:15]:
        s = "OK" if r.get("success") else ("ERR" if r.get("error") else "fail")
        print(f"  {r['protein_id']}: {s} score={r.get('final_score',0):.2f} "
              f"iter={r.get('iterations',0)} {r.get('time_s',0):.0f}s")

    Path("results").mkdir(exist_ok=True)
    with open("results/unknot_final.json", "w") as f:
        json.dump({
            "wall_time_s": round(wall_time, 1),
            "n_success": n_success, "n_completed": n_completed, "n_total": len(clean),
            "rate_completed": round(n_success / n_completed, 4) if n_completed else 0,
            "rate_total": round(n_success / len(clean), 4) if clean else 0,
            "results": clean,
        }, f, indent=2)
    print(f"\nSaved to results/unknot_final.json")
