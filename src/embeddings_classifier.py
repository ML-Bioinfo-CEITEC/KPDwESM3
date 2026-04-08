"""Extract ESM3 embeddings for all proteins and train a knot classifier."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-embeddings")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"

esm3_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "esm==3.2.3",
        "torch==2.5.1",
        "huggingface-hub",
        "datasets",
        "scikit-learn",
        "numpy",
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
def extract_embeddings_batch(proteins: list[dict]) -> list[dict]:
    """Extract mean-pooled ESM3 embeddings for a batch of proteins."""
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
            protein = ESMProtein(sequence=p["sequence"])
            pt = model.encode(protein)
            logits = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
            emb = logits.embeddings[0]  # [seq_len+2, 1536]
            mean_emb = emb[1:-1].mean(dim=0)  # exclude BOS/EOS, mean pool
            results.append({
                "id": p["id"],
                "label": p["label"],
                "embedding": mean_emb.float().cpu().tolist(),
            })
        except Exception as e:
            print(f"Error on {p['id']}: {e}")

    return results


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_all_proteins() -> list[dict]:
    """Load all Real proteins from dataset."""
    from datasets import load_dataset

    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    proteins = [
        {"id": r["ID"], "sequence": r["Sequence"], "label": r["Label"]}
        for r in ds
        if r["Tool"] == "Real"
    ]
    print(f"Loaded {len(proteins)} Real proteins "
          f"({sum(1 for p in proteins if p['label']==1)} knotted, "
          f"{sum(1 for p in proteins if p['label']==0)} unknotted)")
    return proteins


@app.function(
    image=esm3_image,
    timeout=30 * MINUTES,
)
def train_classifier(embeddings_data: list[dict]) -> dict:
    """Train MLP classifier on embeddings, return metrics."""
    import numpy as np
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    X = np.array([d["embedding"] for d in embeddings_data])
    y = np.array([d["label"] for d in embeddings_data])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"Training: {len(X_train)} samples, Validation: {len(X_val)} samples")
    print(f"Train knotted: {y_train.sum()}, unknotted: {(1-y_train).sum()}")

    clf = MLPClassifier(
        hidden_layer_sizes=(1024, 256),
        activation="relu",
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42,
        verbose=True,
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_val)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred).tolist()
    report = classification_report(y_val, y_pred, output_dict=True)

    print(f"\nValidation accuracy: {acc:.4f}")
    print(f"Confusion matrix:\n{cm}")

    return {
        "accuracy": round(acc, 4),
        "confusion_matrix": cm,
        "classification_report": report,
        "n_train": len(X_train),
        "n_val": len(X_val),
    }


@app.local_entrypoint()
def main():
    print("Loading proteins...")
    proteins = load_all_proteins.remote()
    print(f"Got {len(proteins)} proteins")

    # Split into batches for parallel embedding extraction
    batch_size = 500
    batches = [proteins[i:i+batch_size] for i in range(0, len(proteins), batch_size)]
    print(f"Extracting embeddings in {len(batches)} batches of ~{batch_size}...")

    t0 = time.time()
    all_results = list(extract_embeddings_batch.map(batches))
    embed_time = time.time() - t0

    embeddings_data = [item for batch in all_results for item in batch]
    print(f"Got {len(embeddings_data)} embeddings in {embed_time:.0f}s wall time")

    # Train classifier
    print("\nTraining classifier...")
    t0 = time.time()
    metrics = train_classifier.remote(embeddings_data)
    clf_time = time.time() - t0

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Embedding extraction: {embed_time:.0f}s wall time")
    print(f"Classifier training: {clf_time:.0f}s")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Confusion matrix: {metrics['confusion_matrix']}")

    Path("results").mkdir(exist_ok=True)
    with open("results/embeddings_classifier.json", "w") as f:
        json.dump({
            "embed_wall_time_s": round(embed_time, 1),
            "clf_time_s": round(clf_time, 1),
            "n_proteins": len(embeddings_data),
            "metrics": metrics,
        }, f, indent=2)
    print("\nSaved to results/embeddings_classifier.json")
