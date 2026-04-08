"""Benchmark individual operations on Modal to estimate full run times."""

import modal
from pathlib import Path

app = modal.App("esm3-benchmark")

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


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=30 * 60,
)
def benchmark():
    import json
    import os
    import random
    import tempfile
    import time

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    timings = {}

    # --- 1. Model load ---
    t0 = time.time()
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig, LogitsConfig
    import torch

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    timings["model_load_s"] = round(time.time() - t0, 2)
    print(f"[1] Model load: {timings['model_load_s']}s")

    # --- 2. Load a few real proteins from dataset ---
    t0 = time.time()
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    timings["dataset_load_s"] = round(time.time() - t0, 2)
    print(f"[2] Dataset load: {timings['dataset_load_s']}s")

    real_knotted = [r for r in ds if r["Tool"] == "Real" and r["Label"] == 1]
    real_unknotted = [r for r in ds if r["Tool"] == "Real" and r["Label"] == 0]
    print(f"    Knotted: {len(real_knotted)}, Unknotted: {len(real_unknotted)}")

    # Pick proteins of different lengths for benchmarking
    random.seed(42)
    test_proteins = []
    for target_len in [100, 200, 300, 500]:
        candidates = [p for p in real_knotted if abs(len(p["Sequence"]) - target_len) < 50]
        if candidates:
            p = random.choice(candidates)
            test_proteins.append({"id": p["ID"], "seq": p["Sequence"], "label": p["Label"]})

    print(f"\n    Test proteins: {[(p['id'], len(p['seq'])) for p in test_proteins]}")

    # --- 3. Structure generation (seq -> structure) at different lengths ---
    print("\n[3] Structure generation (seq -> structure):")
    struct_times = {}
    for p in test_proteins:
        seq = p["seq"]
        L = len(seq)
        t0 = time.time()
        protein = ESMProtein(sequence=seq)
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        elapsed = round(time.time() - t0, 2)
        struct_times[L] = elapsed
        print(f"    len={L}: {elapsed}s")
    timings["structure_generation_by_len"] = struct_times

    # --- 4. Topoly on generated structures ---
    print("\n[4] Topoly knot detection:")
    from topoly import alexander
    topoly_times = {}
    topoly_results = {}
    for p in test_proteins:
        seq = p["seq"]
        L = len(seq)
        protein = ESMProtein(sequence=seq)
        protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            pdb_path = f.name
            protein.to_pdb(pdb_path)
        t0 = time.time()
        result = alexander(pdb_path, tries=30)
        elapsed = round(time.time() - t0, 2)
        topoly_times[L] = elapsed
        topoly_results[p["id"]] = str(result)
        total = sum(result.values())
        unknot_frac = result.get("0_1", 0) / total if total > 0 else 1.0
        print(f"    len={L} ({p['id']}): {elapsed}s, result={result}, knotted_p={1-unknot_frac:.2f}")
    timings["topoly_by_len"] = topoly_times
    timings["topoly_results"] = topoly_results

    # --- 5. Sequence generation (masked seq -> filled seq) ---
    print("\n[5] Masked sequence generation:")
    seqgen_times = {}
    for p in test_proteins:
        seq = p["seq"]
        L = len(seq)
        seq_list = list(seq)
        mask_count = max(1, L // 10)
        indices = random.sample(range(L), mask_count)
        for i in indices:
            seq_list[i] = "_"
        masked = "".join(seq_list)
        t0 = time.time()
        filled = model.generate(
            ESMProtein(sequence=masked),
            GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
        )
        elapsed = round(time.time() - t0, 2)
        seqgen_times[L] = elapsed
        print(f"    len={L}, masked {mask_count} pos: {elapsed}s")
    timings["sequence_generation_by_len"] = seqgen_times

    # --- 6. Embedding extraction ---
    print("\n[6] Embedding extraction:")
    embed_times = {}
    for p in test_proteins:
        seq = p["seq"]
        L = len(seq)
        t0 = time.time()
        pt = model.encode(ESMProtein(sequence=seq))
        logits = model.logits(pt, LogitsConfig(sequence=True, return_embeddings=True))
        embed = logits.embeddings
        elapsed = round(time.time() - t0, 2)
        embed_times[L] = elapsed
        print(f"    len={L}: {elapsed}s, shape={embed.shape}")
    timings["embedding_extraction_by_len"] = embed_times

    # --- 7. Full pipeline: mask + seqgen + structgen + topoly (one protein, one trial) ---
    print("\n[7] Full single trial (mask 10% -> seqgen -> structgen -> topoly):")
    p = test_proteins[1]  # ~200 aa
    seq = p["seq"]
    L = len(seq)
    t0_total = time.time()

    seq_list = list(seq)
    mask_count = max(1, L // 10)
    indices = random.sample(range(L), mask_count)
    for i in indices:
        seq_list[i] = "_"
    masked = "".join(seq_list)

    t0 = time.time()
    filled = model.generate(
        ESMProtein(sequence=masked),
        GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
    )
    t_seq = time.time() - t0

    t0 = time.time()
    struct_protein = model.generate(
        ESMProtein(sequence=filled.sequence),
        GenerationConfig(track="structure", num_steps=8),
    )
    t_struct = time.time() - t0

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        pdb_path = f.name
        struct_protein.to_pdb(pdb_path)

    t0 = time.time()
    result = alexander(pdb_path, tries=30)
    t_topoly = time.time() - t0

    t_total = time.time() - t0_total
    print(f"    len={L}: seqgen={t_seq:.2f}s, structgen={t_struct:.2f}s, topoly={t_topoly:.2f}s, TOTAL={t_total:.2f}s")
    timings["full_trial_200aa"] = {
        "seqgen": round(t_seq, 2),
        "structgen": round(t_struct, 2),
        "topoly": round(t_topoly, 2),
        "total": round(t_total, 2),
    }

    # --- Summary ---
    print("\n" + "=" * 60)
    print("TIMING SUMMARY")
    print("=" * 60)
    print(json.dumps(timings, indent=2))

    return timings


@app.local_entrypoint()
def main():
    import json
    result = benchmark.remote()
    print("\n\nFINAL TIMINGS:")
    print(json.dumps(result, indent=2))
