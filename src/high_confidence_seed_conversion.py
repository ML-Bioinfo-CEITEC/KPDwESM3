"""Try to induce knots in high-confidence generated seed sequences.

This fallback starts from generated sequences that ESM3 predicts confidently,
then applies small iterative masked edits under a topology + confidence score.
Every edited sequence is independently structure-predicted before QC.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from pathlib import Path

import modal

from overnight_candidate_search import (
    CANONICAL_AA,
    MODELS_PATH,
    VOLUME_PATH,
    _float_list,
    _mean_float,
    _topology_from_protein,
    evaluate_result,
    volume,
)


app = modal.App("esm3-high-confidence-knot-conversion")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "esm==3.2.3",
        "torch==2.5.1",
        "topoly",
        "huggingface-hub",
    )
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
    .add_local_python_source("overnight_candidate_search")
)


def _load_model():
    import os
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)
    from huggingface_hub import login

    login(token=os.environ["HF_TOKEN"])
    import torch
    from esm.models.esm3 import ESM3

    model = ESM3.from_pretrained("esm3_sm_open_v1").to("cuda")
    model = model.half()
    torch.backends.cuda.matmul.allow_tf32 = True
    return model


@app.function(
    image=image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=45 * 60,
    retries=1,
)
def convert_seed(
    job_id: str,
    seed_sequence: str,
    mask_pct: float,
    confidence_weight: float,
    target_trefoil: bool,
    max_iterations: int,
) -> dict:
    from esm.sdk.api import ESMProtein, GenerationConfig
    from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction

    started = time.time()
    result = {
        "job_id": job_id,
        "mask_pct": mask_pct,
        "confidence_weight": confidence_weight,
        "target_trefoil": target_trefoil,
        "seed_sequence": seed_sequence,
        "error": None,
        "history": [],
    }
    try:
        model = _load_model()

        class Scorer(GuidedDecodingScoringFunction):
            def __call__(self, protein: ESMProtein) -> float:
                topology, _ = _topology_from_protein(protein, tries=20)
                total = sum(topology.values())
                knot = 1.0 - topology.get("0_1", 0.0) / total if total else 0.0
                trefoil = topology.get("3_1", 0.0) / total if total else 0.0
                plddt = _mean_float(protein.plddt) or 0.0
                ptm = _mean_float(protein.ptm) or 0.0
                confidence = 0.5 * (plddt + ptm)
                topology_term = (
                    2.0 * trefoil + 0.75 * knot if target_trefoil else 2.5 * knot
                )
                return float(topology_term + confidence_weight * confidence)

        guided = ESM3GuidedDecoding(client=model, scoring_function=Scorer())
        current_sequence = seed_sequence
        best = None

        for iteration in range(max_iterations):
            rng = random.Random(hash((job_id, iteration)) & 0xFFFFFFFF)
            chars = list(current_sequence)
            n_mask = max(1, round(len(chars) * mask_pct / 100))
            for index in rng.sample(range(len(chars)), n_mask):
                chars[index] = "_"
            guided_protein = guided.guided_generate(
                protein=ESMProtein(sequence="".join(chars)),
                num_decoding_steps=max(1, len(chars) // 64),
                num_samples_per_step=10,
                verbose=False,
            )
            current_sequence = guided_protein.sequence or ""
            if not current_sequence or not set(current_sequence).issubset(CANONICAL_AA):
                result["history"].append(
                    {"iteration": iteration, "error": "noncanonical_sequence"}
                )
                break

            # Independent structure prediction of the edited sequence.
            predicted = model.generate(
                ESMProtein(sequence=current_sequence),
                GenerationConfig(track="structure", num_steps=8),
            )
            topology, pdb_text = _topology_from_protein(predicted, tries=50)
            plddt = _mean_float(predicted.plddt)
            ptm = _mean_float(predicted.ptm)
            qc = evaluate_result(
                sequence=current_sequence,
                pdb_text=pdb_text,
                plddt=plddt,
                ptm=ptm,
                topology=topology,
            )
            entry = {
                "iteration": iteration + 1,
                "sequence": current_sequence,
                "topology": topology,
                "plddt_values": _float_list(predicted.plddt),
                "qc": qc,
            }
            result["history"].append(
                {
                    "iteration": iteration + 1,
                    "plddt": qc["plddt"],
                    "ptm": qc["ptm"],
                    "knot_score": qc["knot_score"],
                    "geometry_pass": qc["geometry"]["geometry_pass"],
                    "passes": qc["passes"],
                    "resolved_topology": qc["resolved_topology"],
                }
            )
            if best is None or (
                bool(qc["passes"]),
                qc["knot_score"],
                qc["plddt"] or 0,
            ) > (
                bool(best["qc"]["passes"]),
                best["qc"]["knot_score"],
                best["qc"]["plddt"] or 0,
            ):
                best = {**entry, "pdb_text": pdb_text}
            if qc["passes"]:
                break

        if best:
            result["best"] = best
            result["passes"] = bool(best["qc"]["passes"])
        else:
            result["passes"] = False
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["passes"] = False
    result["time_s"] = round(time.time() - started, 2)
    return result


@app.local_entrypoint()
def main(
    seed_json: str = "results/confidence_guided_n20.json",
    output_json: str = "results/overnight_candidates/seed_conversion.json",
    max_iterations: int = 12,
    repeats: int = 1,
    seed_count: int = 2,
):
    data = json.loads(Path(seed_json).read_text())
    rows = [
        r
        for r in data["results"]
        if not r.get("error")
        and r.get("sequence")
        and (r.get("confidence_fields", {}).get("plddt") or 0) >= 0.40
    ]
    rows.sort(
        key=lambda r: r.get("confidence_fields", {}).get("plddt") or 0,
        reverse=True,
    )
    seeds = rows[:seed_count]
    configs = [
        (1.0, 1.0, True),
        (2.0, 1.0, True),
        (3.0, 1.0, True),
        (1.0, 2.0, True),
        (2.0, 2.0, True),
        (3.0, 2.0, True),
        (2.0, 1.0, False),
        (3.0, 2.0, False),
    ]
    jobs = []
    for repeat in range(repeats):
        for seed_index, seed in enumerate(seeds):
            for config_index, (mask, weight, target) in enumerate(configs):
                jobs.append(
                    (
                        f"rep{repeat}_seed{seed_index}_cfg{config_index}_m{mask}_w{weight}_t{int(target)}",
                        seed["sequence"],
                        mask,
                        weight,
                        target,
                        max_iterations,
                    )
                )
    print(
        f"SEED_CONVERSION_START seeds={len(seeds)} jobs={len(jobs)} "
        f"max_iterations={max_iterations}",
        flush=True,
    )
    results = []
    for item in convert_seed.map(
        *list(zip(*jobs)),
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(item, Exception):
            results.append({"error": f"{type(item).__name__}: {item}", "passes": False})
        else:
            results.append(item)
        passing = sum(bool(r.get("passes")) for r in results)
        best_plddt = max(
            (
                (r.get("best") or {}).get("qc", {}).get("plddt") or 0
                for r in results
                if (r.get("best") or {}).get("qc", {}).get("knot_score", 0) > 0.5
            ),
            default=0,
        )
        print(
            f"SEED_CONVERSION_STATUS completed={len(results)}/{len(jobs)} "
            f"passing={passing} best_knotted_plddt={best_plddt:.4f}",
            flush=True,
        )
    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdb_dir = out.parent / "pdb_seed_conversion"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for result in results:
        result = dict(result)
        best = result.get("best")
        if best and best.get("pdb_text"):
            path = pdb_dir / f"{result.get('job_id','unknown')}.pdb"
            path.write_text(best.pop("pdb_text"))
            best["pdb_path"] = str(path)
        cleaned.append(result)
    summary = {
        "n_jobs": len(jobs),
        "n_completed": len(cleaned),
        "n_errors": sum(bool(r.get("error")) for r in cleaned),
        "n_passing": sum(bool(r.get("passes")) for r in cleaned),
    }
    out.write_text(json.dumps({"summary": summary, "results": cleaned}, indent=2))
    print("SEED_CONVERSION_COMPLETE " + json.dumps(summary), flush=True)
