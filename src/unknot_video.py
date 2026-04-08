"""Generate intermediate PDB files from unknotted-to-knotted transformation for video creation.

Picks a protein that successfully converted and re-runs the transformation,
saving PDB files at each iteration.
"""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-unknot-video")

volume = modal.Volume.from_name("esm3-models", create_if_missing=True)
VOLUME_PATH = Path("/vol")
MODELS_PATH = VOLUME_PATH / "models"
VIDEO_PATH = VOLUME_PATH / "video_pdbs"

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
def generate_video_pdbs(
    protein_id: str,
    sequence: str,
    max_iterations: int = 15,
    masking_percent: int = 5,
) -> dict:
    """Run unknotted-to-knotted transformation, saving PDB at each step."""
    import os
    import random
    import tempfile
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    from huggingface_hub import login
    login(token=os.environ["HF_TOKEN"])

    import torch
    from esm.models.esm3 import ESM3
    from esm.sdk.api import ESMProtein, GenerationConfig
    from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction
    from topoly import alexander

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True

    def knot_status(protein: ESMProtein) -> tuple:
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            pdb_path = f.name
        topo = alexander(pdb_path, tries=100)
        total = sum(topo.values())
        kp = 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0
        return kp, topo, pdb_path

    class KnotScoringFunction(GuidedDecodingScoringFunction):
        def __call__(self, protein: ESMProtein) -> float:
            assert protein.ptm is not None
            kp, _, _ = knot_status(protein)
            return float(kp)

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotScoringFunction())

    # Save initial structure
    initial_protein = ESMProtein(sequence=sequence)
    initial_struct = model.generate(initial_protein, GenerationConfig(track="structure", num_steps=8))
    kp_init, topo_init, pdb_init = knot_status(initial_struct)

    frames = [{
        "iteration": 0,
        "knot_score": round(kp_init, 4),
        "topology": {k: round(v, 4) for k, v in topo_init.items()},
        "pdb_data": open(pdb_init).read(),
        "sequence": sequence,
    }]
    print(f"Frame 0: score={kp_init:.3f}, topology={topo_init}")

    current_seq = sequence
    for iteration in range(1, max_iterations + 1):
        try:
            random.seed(hash((protein_id, iteration)) & 0xFFFFFFFF)
            seq_list = list(current_seq)
            n_mask = max(1, int(len(seq_list) * masking_percent / 100))
            indices = random.sample(range(len(seq_list)), n_mask)
            for i in indices:
                seq_list[i] = "_"
            masked_seq = "".join(seq_list)

            num_steps = max(1, len(current_seq) // 64)

            generated = guided.guided_generate(
                protein=ESMProtein(sequence=masked_seq),
                num_decoding_steps=num_steps,
                num_samples_per_step=10,
                verbose=False,
            )

            kp, topo, pdb_path = knot_status(generated)
            current_seq = generated.sequence

            frames.append({
                "iteration": iteration,
                "knot_score": round(kp, 4),
                "topology": {k: round(v, 4) for k, v in topo.items()},
                "pdb_data": open(pdb_path).read(),
                "sequence": current_seq,
            })
            print(f"Frame {iteration}: score={kp:.3f}, topology={topo}")

            if kp >= 0.80:
                print(f"SUCCESS at iteration {iteration}!")
                break
        except Exception as e:
            print(f"Error at iteration {iteration}: {e}")
            break

    return {
        "protein_id": protein_id,
        "n_frames": len(frames),
        "final_score": frames[-1]["knot_score"],
        "success": frames[-1]["knot_score"] >= 0.80,
        "frames": frames,
    }


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=10 * MINUTES,
)
def get_successful_protein() -> dict:
    """Get a protein that we know converts successfully."""
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")

    # Use A0A4V4LCF3 which succeeded in our v2 run at iteration 4
    target_ids = ["A0A4V4LCF3", "A0A0Q5ARJ8", "A0A7Y2BEP9"]
    for r in ds:
        uid = r["ID"].replace("R_", "")
        if uid in target_ids and r["Tool"] == "Real" and r["Label"] == 0:
            return {"id": r["ID"], "sequence": r["Sequence"]}

    # Fallback: just pick a short unknotted one
    for r in ds:
        if r["Tool"] == "Real" and r["Label"] == 0 and len(r["Sequence"]) < 200:
            return {"id": r["ID"], "sequence": r["Sequence"]}


@app.local_entrypoint()
def main():
    protein = get_successful_protein.remote()
    print(f"Using {protein['id']} (len={len(protein['sequence'])})")

    result = generate_video_pdbs.remote(
        protein_id=protein["id"],
        sequence=protein["sequence"],
        max_iterations=15,
        masking_percent=5,
    )

    print(f"\nFrames generated: {result['n_frames']}")
    print(f"Final score: {result['final_score']}")
    print(f"Success: {result['success']}")

    # Save PDB files locally
    Path("results/video_pdbs").mkdir(parents=True, exist_ok=True)
    for frame in result["frames"]:
        pdb_path = Path(f"results/video_pdbs/frame_{frame['iteration']:03d}.pdb")
        with open(pdb_path, "w") as f:
            f.write(frame["pdb_data"])
        print(f"  Saved {pdb_path} (score={frame['knot_score']:.3f})")

    # Save metadata
    meta = {k: v for k, v in result.items() if k != "frames"}
    meta["frame_scores"] = [(f["iteration"], f["knot_score"]) for f in result["frames"]]
    with open("results/video_pdbs/metadata.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved metadata to results/video_pdbs/metadata.json")
