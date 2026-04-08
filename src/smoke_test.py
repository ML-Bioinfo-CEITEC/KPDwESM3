"""Minimal smoke test: load ESM3 on Modal GPU, generate structure, check topology with topoly."""

import modal
from pathlib import Path

app = modal.App("esm3-smoke-test")

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
    .env({"HF_HOME": str(MODELS_PATH)})
)


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=20 * 60,
)
def smoke_test():
    import os
    import tempfile
    import time

    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])

    # 1. Load ESM3
    print("=== Loading ESM3 model ===")
    t0 = time.time()

    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    print(f"Model loaded in {time.time() - t0:.1f}s")

    # 2. Generate structure from a known knotted protein fragment
    # Using a short sequence (~100 aa) for speed
    test_seq = "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQQIAATGFHIK"
    print(f"\n=== Generating structure for sequence (len={len(test_seq)}) ===")
    t0 = time.time()

    protein = ESMProtein(sequence=test_seq)
    protein = model.generate(protein, GenerationConfig(track="structure", num_steps=8))
    print(f"Structure generated in {time.time() - t0:.1f}s")

    # 3. Save to PDB and run topoly
    print("\n=== Running topoly knot detection ===")
    t0 = time.time()

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
        pdb_path = f.name
        protein.to_pdb(pdb_path)

    from topoly import alexander
    result = alexander(pdb_path, tries=30)
    print(f"Topoly result: {result}")
    print(f"Topoly ran in {time.time() - t0:.1f}s")

    # 4. Test masking + sequence generation
    print("\n=== Testing masked sequence generation ===")
    t0 = time.time()

    seq_list = list(test_seq)
    import random
    random.seed(42)
    mask_indices = random.sample(range(len(seq_list)), len(seq_list) // 10)
    for i in mask_indices:
        seq_list[i] = "_"
    masked_seq = "".join(seq_list)
    print(f"Masked {len(mask_indices)} positions ({len(mask_indices)/len(test_seq)*100:.0f}%)")

    masked_protein = ESMProtein(sequence=masked_seq)
    filled_protein = model.generate(
        masked_protein,
        GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
    )
    print(f"Filled sequence: {filled_protein.sequence[:50]}...")
    print(f"Sequence generation in {time.time() - t0:.1f}s")

    # 5. Test embedding extraction
    print("\n=== Testing embedding extraction ===")
    t0 = time.time()

    from esm.sdk.api import LogitsConfig
    protein_for_embed = ESMProtein(sequence=test_seq)
    protein_tensor = model.encode(protein_for_embed)
    logits_output = model.logits(
        protein_tensor,
        LogitsConfig(sequence=True, return_embeddings=True),
    )
    embed = logits_output.embeddings
    print(f"Embedding shape: {embed.shape}")
    print(f"Embedding extraction in {time.time() - t0:.1f}s")

    print("\n=== ALL SMOKE TESTS PASSED ===")
    return {
        "topoly_result": str(result),
        "embedding_shape": list(embed.shape),
        "filled_seq_sample": filled_protein.sequence[:50],
    }


@app.local_entrypoint()
def main():
    result = smoke_test.remote()
    print(f"\nResult: {result}")
