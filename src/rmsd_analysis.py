"""RMSD analysis: how much does 3D structure drift when sequence is masked and regenerated?

For each knotted protein:
1. Generate reference structure from original sequence
2. At each masking level: mask -> regenerate sequence -> predict structure -> RMSD vs reference
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-rmsd")

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
def process_one_protein(protein_id: str, sequence: str, masking_percentages: list[int]) -> dict:
    import os
    import random
    import tempfile

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from esm.utils.structure.aligner import Aligner
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    if len(sequence) > 600:
        return {"protein_id": protein_id, "seq_len": len(sequence),
                "error": "sequence_too_long", "levels": {}}

    t0 = time.time()

    ref_protein = ESMProtein(sequence=sequence)
    ref_protein = model.generate(ref_protein, GenerationConfig(track="structure", num_steps=8))
    ref_chain = ref_protein.to_protein_chain()

    results = {"protein_id": protein_id, "seq_len": len(sequence), "levels": {}}

    for pct in masking_percentages:
        random.seed(hash((protein_id, pct)) & 0xFFFFFFFF)

        try:
            seq_list = list(sequence)
            n_mask = max(1, int(len(seq_list) * pct / 100))
            indices = random.sample(range(len(seq_list)), min(n_mask, len(seq_list)))
            for i in indices:
                seq_list[i] = "_"
            masked_seq = "".join(seq_list)

            filled = model.generate(
                ESMProtein(sequence=masked_seq),
                GenerationConfig(track="sequence", num_steps=8, temperature=1.0),
            )
            struct = model.generate(
                ESMProtein(sequence=filled.sequence),
                GenerationConfig(track="structure", num_steps=8),
            )

            new_chain = struct.to_protein_chain()
            rmsd = Aligner(ref_chain, new_chain).rmsd

            with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
                struct.to_pdb(f.name)
                topo = alexander(f.name, tries=30)
            total = sum(topo.values())
            knotted_p = 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0

            seq_identity = sum(1 for a, b in zip(sequence, filled.sequence) if a == b) / len(sequence)

            results["levels"][pct] = {
                "rmsd": round(float(rmsd), 3),
                "knotted_p": round(knotted_p, 4),
                "seq_identity": round(seq_identity, 4),
                "new_seq": filled.sequence,
            }
        except Exception as e:
            results["levels"][pct] = {"error": str(e)}

    results["total_time_s"] = round(time.time() - t0, 1)
    return results


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def load_knotted(limit: int = 100) -> list[dict]:
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    return [{"id": r["ID"], "sequence": r["Sequence"]}
            for r in ds if r["Tool"] == "Real" and r["Label"] == 1][:limit]


@app.local_entrypoint()
def main(n_proteins: int = 100, levels: str = "5,10,20,30,40,50,60,70,80,85,90,95"):
    masking_pcts = [int(x) for x in levels.split(",")]
    print(f"RMSD analysis: {n_proteins} proteins, {len(masking_pcts)} levels")

    proteins = load_knotted.remote(limit=n_proteins)

    t0 = time.time()
    results = list(
        process_one_protein.map(
            [p["id"] for p in proteins],
            [p["sequence"] for p in proteins],
            [masking_pcts] * len(proteins),
            return_exceptions=True,
        )
    )

    clean = []
    for i, r in enumerate(results):
        if isinstance(r, Exception):
            clean.append({"protein_id": proteins[i]["id"], "error": str(r), "levels": {}})
        else:
            clean.append(r)
    results = clean
    wall_time = time.time() - t0

    valid = [r for r in results if r.get("levels") and not r.get("error")]
    print(f"\n{'='*70}")
    print(f"RMSD ANALYSIS ({wall_time/60:.1f} min, {len(valid)} proteins)")
    print(f"{'='*70}")

    import statistics
    print(f"{'Level':>8} | {'Mean RMSD':>10} | {'Med RMSD':>10} | {'Mean knot_p':>12} | {'Seq identity':>13}")
    print("-" * 60)
    for pct in masking_pcts:
        rmsds = [r["levels"][str(pct)]["rmsd"] for r in valid
                 if str(pct) in r["levels"] and "rmsd" in r["levels"][str(pct)]]
        knots = [r["levels"][str(pct)]["knotted_p"] for r in valid
                 if str(pct) in r["levels"] and "knotted_p" in r["levels"][str(pct)]]
        seqids = [r["levels"][str(pct)]["seq_identity"] for r in valid
                  if str(pct) in r["levels"] and "seq_identity" in r["levels"][str(pct)]]
        if rmsds:
            print(f"{pct:>7}% | {statistics.mean(rmsds):>10.2f} | {statistics.median(rmsds):>10.2f} | "
                  f"{statistics.mean(knots):>12.3f} | {statistics.mean(seqids):>12.3f}")

    Path("results").mkdir(exist_ok=True)
    with open("results/rmsd_analysis.json", "w") as f:
        json.dump({"wall_time_s": round(wall_time, 1), "n_valid": len(valid), "results": results}, f, indent=2)
    print(f"\nSaved to results/rmsd_analysis.json")
