"""Generate videos for multiple successful unknotted-to-knotted conversions."""

import json
import time
from pathlib import Path

import modal

app = modal.App("esm3-multi-video")

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


@app.function(
    image=esm3_image, volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G", timeout=40 * MINUTES,
)
def generate_video(protein_id: str, sequence: str, max_iters: int) -> dict:
    import os, random, tempfile, warnings
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

    def knot_status(protein):
        with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False) as f:
            protein.to_pdb(f.name)
            topo = alexander(f.name, tries=100)
        total = sum(topo.values())
        return 1.0 - topo.get("0_1", 0) / total if total > 0 else 0.0, f.name

    class KnotSF(GuidedDecodingScoringFunction):
        def __call__(self, protein):
            assert protein.ptm is not None
            score, _ = knot_status(protein)
            return float(score)

    guided = ESM3GuidedDecoding(client=model, scoring_function=KnotSF())

    # Frame 0: initial structure
    init_struct = model.generate(ESMProtein(sequence=sequence),
                                  GenerationConfig(track="structure", num_steps=8))
    score0, pdb0 = knot_status(init_struct)
    frames = [{"iter": 0, "score": round(score0, 4), "pdb": open(pdb0).read()}]

    current_seq = sequence
    for it in range(1, max_iters + 1):
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

            score, pdb_path = knot_status(generated)
            current_seq = generated.sequence
            frames.append({"iter": it, "score": round(score, 4), "pdb": open(pdb_path).read()})

            if score >= 0.80:
                break
        except:
            break

    return {"protein_id": protein_id, "n_frames": len(frames),
            "final_score": frames[-1]["score"], "frames": frames}


@app.local_entrypoint()
def main():
    from datasets import load_dataset
    ds = load_dataset("EvaKlimentova/Diffusion-all_knots", split="train")
    seq_map = {r["ID"]: r["Sequence"] for r in ds if r["Tool"] == "Real" and r["Label"] == 0}

    targets = [
        ("R_A0A072V0N7", 1),   # quick converter
        ("R_A0A3L9I5G0", 7),   # medium
        ("R_J9CSJ6", 14),      # slow
    ]

    t0 = time.time()
    results = list(generate_video.map(
        [t[0] for t in targets],
        [seq_map[t[0]] for t in targets],
        [t[1] + 2 for t in targets],  # add buffer iterations
        return_exceptions=True,
    ))

    for i, r in enumerate(results):
        if isinstance(r, Exception):
            print(f"ERROR {targets[i][0]}: {r}")
            continue

        pid = r["protein_id"]
        out_dir = Path(f"results/videos/{pid}")
        out_dir.mkdir(parents=True, exist_ok=True)

        for frame in r["frames"]:
            with open(out_dir / f"frame_{frame['iter']:03d}.pdb", "w") as f:
                f.write(frame["pdb"])

        meta = {"protein_id": pid, "n_frames": r["n_frames"],
                "final_score": r["final_score"],
                "scores": [(f["iter"], f["score"]) for f in r["frames"]]}
        with open(out_dir / "metadata.json", "w") as f:
            json.dump(meta, f, indent=2)

        print(f"{pid}: {r['n_frames']} frames, final={r['final_score']:.2f}")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min")
