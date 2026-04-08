"""Extract and save ESM3 embeddings for all real proteins (for UMAP visualization)."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-extract-emb")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("esm==3.2.3", "torch==2.5.1", "huggingface-hub", "datasets")
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

MINUTES = 60


@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=30 * MINUTES,
)
def extract_batch(proteins: list[dict]) -> list[dict]:
    import os
    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, LogitsConfig

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    results = []
    for p in proteins:
        try:
            pt = model.encode(ESMProtein(sequence=p["sequence"]))
            logits = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
            emb = logits.embeddings[0][1:-1].mean(dim=0).float().cpu().tolist()
            results.append({"id": p["id"], "label": p["label"], "embedding": emb})
        except Exception as e:
            print(f"Error {p['id']}: {e}")
    return results


@app.local_entrypoint()
def main():
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [{"id": r["ID"], "sequence": r["Sequence"], "label": r["Label"]}
                for r in ds if r["Tool"] == "Real"]
    print(f"Extracting embeddings for {len(proteins)} proteins")

    batch_size = 500
    batches = [proteins[i:i+batch_size] for i in range(0, len(proteins), batch_size)]

    t0 = time.time()
    all_results = list(extract_batch.map(batches, return_exceptions=True))
    elapsed = time.time() - t0

    embeddings = []
    for batch in all_results:
        if isinstance(batch, Exception):
            print(f"Batch error: {batch}")
        else:
            embeddings.extend(batch)

    print(f"Got {len(embeddings)} embeddings in {elapsed:.0f}s")

    Path("results").mkdir(exist_ok=True)
    with open("results/embeddings_all.json", "w") as f:
        json.dump(embeddings, f)
    print("Saved to results/embeddings_all.json")
