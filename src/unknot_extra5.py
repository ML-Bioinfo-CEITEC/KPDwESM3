"""Run 5 extra unknotted-to-knotted proteins (indices 100-104) as backup."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-extra")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("esm==3.2.3", "torch==2.5.1", "topoly", "huggingface-hub", "datasets")
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60

# Reuse the same transform function from unknot_final
@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=40 * MINUTES,
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
    for it in range(15):
        try:
            random.seed(hash((protein_id, it)) & 0xFFFFFFFF)
            seq_list = list(current_seq)
            n_mask = max(1, int(len(seq_list) * 3 / 100))
            for i in random.sample(range(len(seq_list)), n_mask):
                seq_list[i] = "_"
            generated = guided.guided_generate(
                protein=ESMProtein(sequence="".join(seq_list)),
                num_decoding_steps=max(1, len(current_seq) // 64),
                num_samples_per_step=10, verbose=False)
            score = knot_status(generated)
            current_seq = generated.sequence
            history.append({"iter": it, "score": round(score, 4)})
            if score >= 0.80:
                return {"protein_id": protein_id, "seq_len": len(sequence),
                        "success": True, "iterations": it + 1,
                        "final_score": round(score, 4),
                        "time_s": round(time.time() - t0, 1), "history": history, "error": None}
        except Exception as e:
            history.append({"iter": it, "error": str(e)})
            break
    return {"protein_id": protein_id, "seq_len": len(sequence),
            "success": False, "iterations": len(history),
            "final_score": round(history[-1].get("score", 0), 4) if history else 0,
            "time_s": round(time.time() - t0, 1), "history": history, "error": None}


@app.local_entrypoint()
def main():
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [{"id": r["ID"], "sequence": r["Sequence"]}
                for r in ds if r["Tool"] == "Real" and r["Label"] == 0
                and len(r["Sequence"]) <= 250][100:105]
    print(f"Running 5 extra proteins: {[p['id'] for p in proteins]}")

    t0 = time.time()
    results = list(transform_one.map(
        [p["id"] for p in proteins], [p["sequence"] for p in proteins],
        return_exceptions=True))

    clean = [r if not isinstance(r, Exception) else
             {"protein_id": proteins[i]["id"], "success": False, "error": str(r)}
             for i, r in enumerate(results)]

    n_ok = sum(1 for r in clean if r.get("success"))
    print(f"\nExtra 5: {n_ok}/5 succeeded")
    for r in clean:
        s = "OK" if r.get("success") else "fail"
        print(f"  {r['protein_id']}: {s} score={r.get('final_score',0):.2f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/unknot_extra5.json", "w") as f:
        json.dump({"results": clean}, f, indent=2)
    print("Saved to results/unknot_extra5.json")
