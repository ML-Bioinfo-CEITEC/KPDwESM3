"""Overnight search for plausible, moderately confident generated knotted proteins.

This is an analysis-only pipeline for reviewer follow-up. It does not edit the
manuscript. It:

1. Re-predicts structures for saved guided-generation sequences.
2. If needed, performs confidence-aware topology-guided generation.
3. Applies topology, confidence, sequence, continuity, and clash QC.
4. Re-predicts provisional candidates to estimate repeatability.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import modal


app = modal.App("esm3-overnight-plausible-knots")
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
    )
    .env({"HF_HOME": str(MODELS_PATH), "TOKENIZERS_PARALLELISM": "false"})
)

CANONICAL_AA = set("ACDEFGHIKLMNPQRSTVWY")


def _mean_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu()
        if hasattr(value, "mean"):
            return float(value.mean().item())
        return float(value)
    except Exception:
        return None


def _float_list(value: Any) -> list[float]:
    if value is None:
        return []
    try:
        if hasattr(value, "detach"):
            value = value.detach().float().cpu()
        if hasattr(value, "flatten"):
            return [float(x) for x in value.flatten().tolist()]
        return [float(x) for x in value]
    except Exception:
        return []


def _distance(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    return math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))


def analyze_pdb_geometry(pdb_text: str) -> dict:
    """Conservative backbone continuity/clash checks from a PDB string."""
    residues: list[dict[str, Any]] = []
    by_key: dict[tuple[str, str, str], dict[str, Any]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("ATOM") or len(line) < 54:
            continue
        chain = line[21].strip() or "A"
        resid = line[22:26].strip()
        icode = line[26].strip()
        key = (chain, resid, icode)
        if key not in by_key:
            entry = {"key": key, "atoms": {}, "bfactors": []}
            by_key[key] = entry
            residues.append(entry)
        atom = line[12:16].strip()
        try:
            xyz = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
            bfactor = float(line[60:66]) if len(line) >= 66 else None
        except ValueError:
            continue
        by_key[key]["atoms"][atom] = xyz
        if bfactor is not None:
            by_key[key]["bfactors"].append(bfactor)

    ca = [(i, r["atoms"]["CA"]) for i, r in enumerate(residues) if "CA" in r["atoms"]]
    ca_gaps: list[float] = []
    cn_gaps: list[float] = []
    for left, right in zip(residues, residues[1:]):
        if left["key"][0] != right["key"][0]:
            continue
        if "CA" in left["atoms"] and "CA" in right["atoms"]:
            ca_gaps.append(_distance(left["atoms"]["CA"], right["atoms"]["CA"]))
        if "C" in left["atoms"] and "N" in right["atoms"]:
            cn_gaps.append(_distance(left["atoms"]["C"], right["atoms"]["N"]))

    nonlocal_ca_clashes = 0
    min_nonlocal_ca = float("inf")
    for pos, (i, xyz_i) in enumerate(ca):
        for j, xyz_j in ca[pos + 1 :]:
            if abs(j - i) < 4:
                continue
            dist = _distance(xyz_i, xyz_j)
            min_nonlocal_ca = min(min_nonlocal_ca, dist)
            if dist < 2.8:
                nonlocal_ca_clashes += 1

    backbone_atoms = []
    for i, residue in enumerate(residues):
        for atom in ("N", "CA", "C", "O"):
            if atom in residue["atoms"]:
                backbone_atoms.append((i, atom, residue["atoms"][atom]))
    severe_backbone_clashes = 0
    min_nonlocal_backbone = float("inf")
    for pos, (i, _atom_i, xyz_i) in enumerate(backbone_atoms):
        for j, _atom_j, xyz_j in backbone_atoms[pos + 1 :]:
            if abs(j - i) <= 1:
                continue
            dist = _distance(xyz_i, xyz_j)
            min_nonlocal_backbone = min(min_nonlocal_backbone, dist)
            if dist < 1.7:
                severe_backbone_clashes += 1

    if ca:
        center = tuple(sum(xyz[k] for _, xyz in ca) / len(ca) for k in range(3))
        radius_gyration = math.sqrt(
            sum(_distance(xyz, center) ** 2 for _, xyz in ca) / len(ca)
        )
        max_extent = max(
            max(xyz[k] for _, xyz in ca) - min(xyz[k] for _, xyz in ca)
            for k in range(3)
        )
    else:
        radius_gyration = None
        max_extent = None

    max_ca_gap = max(ca_gaps) if ca_gaps else None
    max_cn_gap = max(cn_gaps) if cn_gaps else None
    min_ca_gap = min(ca_gaps) if ca_gaps else None
    min_cn_gap = min(cn_gaps) if cn_gaps else None
    geometry_pass = bool(
        len(ca) >= 40
        and max_ca_gap is not None
        and max_ca_gap <= 4.5
        and min_ca_gap is not None
        and min_ca_gap >= 3.0
        and max_cn_gap is not None
        and max_cn_gap <= 2.0
        and min_cn_gap is not None
        and min_cn_gap >= 1.0
        and nonlocal_ca_clashes == 0
        and severe_backbone_clashes == 0
    )
    return {
        "n_residues": len(residues),
        "n_ca": len(ca),
        "max_ca_gap": round(max_ca_gap, 4) if max_ca_gap is not None else None,
        "min_ca_gap": round(min_ca_gap, 4) if min_ca_gap is not None else None,
        "max_cn_gap": round(max_cn_gap, 4) if max_cn_gap is not None else None,
        "min_cn_gap": round(min_cn_gap, 4) if min_cn_gap is not None else None,
        "min_nonlocal_ca": (
            round(min_nonlocal_ca, 4) if math.isfinite(min_nonlocal_ca) else None
        ),
        "nonlocal_ca_clashes_lt_2_8": nonlocal_ca_clashes,
        "min_nonlocal_backbone": (
            round(min_nonlocal_backbone, 4)
            if math.isfinite(min_nonlocal_backbone)
            else None
        ),
        "severe_backbone_clashes_lt_1_7": severe_backbone_clashes,
        "radius_gyration": round(radius_gyration, 4) if radius_gyration else None,
        "max_extent": round(max_extent, 4) if max_extent else None,
        "geometry_pass": geometry_pass,
    }


def evaluate_result(
    *,
    sequence: str,
    pdb_text: str,
    plddt: float | None,
    ptm: float | None,
    topology: dict[str, float],
) -> dict:
    total = sum(float(v) for v in topology.values())
    unknot = float(topology.get("0_1", 0.0))
    knot_score = 1.0 - unknot / total if total > 0 else 0.0
    resolved = {
        k: float(v)
        for k, v in topology.items()
        if k not in {"0_1", "TMC"} and float(v) > 0
    }
    geometry = analyze_pdb_geometry(pdb_text)
    canonical = bool(sequence) and set(sequence).issubset(CANONICAL_AA)
    passes = bool(
        canonical
        and plddt is not None
        and plddt > 0.50
        and knot_score > 0.50
        and geometry["geometry_pass"]
    )
    return {
        "canonical_sequence": canonical,
        "plddt": round(plddt, 6) if plddt is not None else None,
        "ptm": round(ptm, 6) if ptm is not None else None,
        "knot_score": round(knot_score, 6),
        "resolved_topology": resolved,
        "has_resolved_topology": bool(resolved),
        "geometry": geometry,
        "passes": passes,
    }


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


def _topology_from_protein(protein, tries: int = 30) -> tuple[dict[str, float], str]:
    import tempfile
    from topoly import alexander

    pdb_text = protein.to_pdb_string()
    with tempfile.NamedTemporaryFile(suffix=".pdb", mode="w", delete=False) as handle:
        handle.write(pdb_text)
        path = handle.name
    topology = alexander(path, tries=tries)
    return {k: float(v) for k, v in topology.items()}, pdb_text


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=30 * 60,
    retries=1,
)
def repredict_sequence(
    job_id: str,
    source_attempt: int,
    replicate: int,
    sequence: str,
) -> dict:
    from esm.sdk.api import ESMProtein, GenerationConfig

    started = time.time()
    result = {
        "job_id": job_id,
        "stage": "existing_rescan",
        "source_attempt": source_attempt,
        "replicate": replicate,
        "sequence": sequence,
        "error": None,
    }
    try:
        if not sequence or not set(sequence).issubset(CANONICAL_AA):
            raise ValueError("noncanonical_sequence")
        model = _load_model()
        protein = model.generate(
            ESMProtein(sequence=sequence),
            GenerationConfig(track="structure", num_steps=8),
        )
        topology, pdb_text = _topology_from_protein(protein)
        plddt_values = _float_list(protein.plddt)
        plddt = _mean_float(protein.plddt)
        ptm = _mean_float(protein.ptm)
        result.update(
            {
                "topology": topology,
                "plddt_values": plddt_values,
                "pdb_text": pdb_text,
                "qc": evaluate_result(
                    sequence=sequence,
                    pdb_text=pdb_text,
                    plddt=plddt,
                    ptm=ptm,
                    topology=topology,
                ),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["time_s"] = round(time.time() - started, 2)
    return result


@app.function(
    image=esm3_image,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    gpu="A10G",
    timeout=45 * 60,
    retries=1,
)
def confidence_guided_generate(
    job_id: str,
    attempt: int,
    protein_length: int,
    confidence_weight: float,
    target_trefoil: bool,
) -> dict:
    from esm.sdk.api import ESMProtein
    from esm.sdk.experimental import ESM3GuidedDecoding, GuidedDecodingScoringFunction

    started = time.time()
    result = {
        "job_id": job_id,
        "stage": "confidence_generation",
        "attempt": attempt,
        "protein_length": protein_length,
        "confidence_weight": confidence_weight,
        "target_trefoil": target_trefoil,
        "error": None,
    }
    try:
        model = _load_model()

        class ConfidenceTopologyScorer(GuidedDecodingScoringFunction):
            def __call__(self, protein: ESMProtein) -> float:
                topology, _ = _topology_from_protein(protein, tries=20)
                total = sum(topology.values())
                knot = 1.0 - topology.get("0_1", 0.0) / total if total else 0.0
                trefoil = topology.get("3_1", 0.0) / total if total else 0.0
                plddt = _mean_float(protein.plddt) or 0.0
                ptm = _mean_float(protein.ptm) or 0.0
                confidence = 0.5 * (plddt + ptm)
                if target_trefoil:
                    topology_term = 1.5 * trefoil + 0.5 * knot
                else:
                    topology_term = 2.0 * knot
                return float(topology_term + confidence_weight * confidence)

        guided = ESM3GuidedDecoding(
            client=model,
            scoring_function=ConfidenceTopologyScorer(),
        )
        protein = guided.guided_generate(
            protein=ESMProtein(sequence="_" * protein_length),
            num_decoding_steps=max(6, protein_length // 32),
            num_samples_per_step=10,
        )
        sequence = protein.sequence or ""
        topology, pdb_text = _topology_from_protein(protein)
        plddt_values = _float_list(protein.plddt)
        plddt = _mean_float(protein.plddt)
        ptm = _mean_float(protein.ptm)
        result.update(
            {
                "sequence": sequence,
                "topology": topology,
                "plddt_values": plddt_values,
                "pdb_text": pdb_text,
                "qc": evaluate_result(
                    sequence=sequence,
                    pdb_text=pdb_text,
                    plddt=plddt,
                    ptm=ptm,
                    topology=topology,
                ),
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["time_s"] = round(time.time() - started, 2)
    return result


OUTPUT_DIR = Path("results/overnight_candidates")
PDB_DIR = OUTPUT_DIR / "pdb"
STATE_PATH = OUTPUT_DIR / "search_state.json"
LOG_PATH = OUTPUT_DIR / "progress.log"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _log(message: str) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    line = f"{_now()} {message}"
    with LOG_PATH.open("a") as handle:
        handle.write(line + "\n")
    print(line, flush=True)


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(path)


def _without_pdb(result: dict) -> dict:
    clean = dict(result)
    clean.pop("pdb_text", None)
    return clean


def _candidate_key(result: dict) -> str:
    return result.get("sequence", "")


def _candidate_rank(result: dict) -> tuple:
    qc = result.get("qc") or {}
    geometry = qc.get("geometry") or {}
    return (
        bool(qc.get("passes")),
        bool(qc.get("has_resolved_topology")),
        float(qc.get("plddt") or 0),
        float(qc.get("ptm") or 0),
        float(qc.get("knot_score") or 0),
        -int(geometry.get("nonlocal_ca_clashes_lt_2_8") or 0),
    )


def _save_pdb(result: dict, prefix: str) -> str | None:
    pdb_text = result.get("pdb_text")
    if not pdb_text:
        return None
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    safe_job = result.get("job_id", "unknown").replace("/", "_")
    path = PDB_DIR / f"{prefix}_{safe_job}.pdb"
    path.write_text(pdb_text)
    return str(path)


def _update_state(
    *,
    stage: str,
    status: str,
    completed: int,
    total: int,
    results: list[dict],
    started_at: float,
    note: str = "",
) -> None:
    valid = [r for r in results if not r.get("error") and r.get("qc")]
    knotted = [r for r in valid if (r["qc"].get("knot_score") or 0) > 0.5]
    passing_sequences = {
        _candidate_key(r) for r in valid if r["qc"].get("passes")
    }
    best = max(
        ((r["qc"].get("plddt") or 0) for r in knotted),
        default=0,
    )
    state = {
        "updated_at": _now(),
        "stage": stage,
        "status": status,
        "completed": completed,
        "total": total,
        "errors": sum(bool(r.get("error")) for r in results),
        "knotted_results": len(knotted),
        "passing_models": sum(bool(r["qc"].get("passes")) for r in valid),
        "passing_distinct_sequences": len(passing_sequences),
        "best_plddt_among_knotted": round(float(best), 6),
        "elapsed_minutes": round((time.time() - started_at) / 60, 2),
        "note": note,
    }
    _write_json(STATE_PATH, state)
    print(
        "OVERNIGHT_STATUS "
        + json.dumps(state, separators=(",", ":")),
        flush=True,
    )


def _collect_map(
    function,
    args: list[tuple],
    *,
    stage: str,
    all_results: list[dict],
    started_at: float,
) -> list[dict]:
    stage_results: list[dict] = []
    if not args:
        return stage_results
    columns = list(zip(*args))
    iterator = function.map(
        *columns,
        order_outputs=False,
        return_exceptions=True,
    )
    for item in iterator:
        if isinstance(item, Exception):
            result = {"stage": stage, "error": f"{type(item).__name__}: {item}"}
        else:
            result = item
        if result.get("qc") and (
            result["qc"].get("passes")
            or (
                (result["qc"].get("plddt") or 0) >= 0.45
                and (result["qc"].get("knot_score") or 0) > 0.5
            )
        ):
            result["pdb_path"] = _save_pdb(
                result,
                "pass" if result["qc"].get("passes") else "near",
            )
        stage_results.append(result)
        all_results.append(result)
        _update_state(
            stage=stage,
            status="running",
            completed=len(stage_results),
            total=len(args),
            results=all_results,
            started_at=started_at,
        )
    return stage_results


def _distinct_passing(results: list[dict]) -> list[dict]:
    best_by_sequence: dict[str, dict] = {}
    for result in results:
        if not result.get("qc", {}).get("passes"):
            continue
        sequence = _candidate_key(result)
        if not sequence:
            continue
        if sequence not in best_by_sequence or _candidate_rank(result) > _candidate_rank(
            best_by_sequence[sequence]
        ):
            best_by_sequence[sequence] = result
    return sorted(best_by_sequence.values(), key=_candidate_rank, reverse=True)


def _write_outputs(all_results: list[dict], final_candidates: list[dict]) -> None:
    cleaned = [_without_pdb(r) for r in all_results]
    passing = [_without_pdb(r) for r in final_candidates]
    near = [
        _without_pdb(r)
        for r in sorted(
            (
                r
                for r in all_results
                if not r.get("error")
                and r.get("qc")
                and not r["qc"].get("passes")
                and (r["qc"].get("knot_score") or 0) > 0.5
            ),
            key=_candidate_rank,
            reverse=True,
        )[:100]
    ]
    _write_json(OUTPUT_DIR / "all_results.json", cleaned)
    _write_json(OUTPUT_DIR / "candidates.json", passing)
    _write_json(OUTPUT_DIR / "near_misses.json", near)
    with (OUTPUT_DIR / "candidates.fasta").open("w") as handle:
        for rank, result in enumerate(final_candidates, 1):
            qc = result["qc"]
            handle.write(
                f">candidate_{rank}|plddt={qc['plddt']}|ptm={qc['ptm']}|"
                f"knot={qc['knot_score']}|job={result.get('job_id')}\n"
            )
            handle.write(result["sequence"] + "\n")

    lines = [
        "# Overnight Plausible Knot Search",
        "",
        f"Passing distinct candidates: {len(final_candidates)}",
        "",
    ]
    for rank, result in enumerate(final_candidates, 1):
        qc = result["qc"]
        geom = qc["geometry"]
        lines.extend(
            [
                f"## Candidate {rank}",
                f"- job: `{result.get('job_id')}`",
                f"- source stage: {result.get('stage')}",
                f"- mean pLDDT: {qc.get('plddt')}",
                f"- pTM: {qc.get('ptm')}",
                f"- knot score: {qc.get('knot_score')}",
                f"- resolved topology: {qc.get('resolved_topology')}",
                f"- max Cα gap: {geom.get('max_ca_gap')} Å",
                f"- max C–N gap: {geom.get('max_cn_gap')} Å",
                f"- nonlocal Cα clashes: {geom.get('nonlocal_ca_clashes_lt_2_8')}",
                f"- severe backbone clashes: {geom.get('severe_backbone_clashes_lt_1_7')}",
                f"- PDB: `{result.get('pdb_path')}`",
                "",
            ]
        )
    (OUTPUT_DIR / "summary.md").write_text("\n".join(lines))


@app.local_entrypoint()
def main(
    mode: str = "overnight",
    input_json: str = "results/guided_gen_combined.json",
    existing_limit: int | None = None,
    existing_replicates: int = 3,
    target_candidates: int = 5,
    max_guided_attempts: int = 192,
    batch_size: int = 24,
    max_hours: float = 8.0,
):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    started_at = time.time()
    deadline = started_at + max_hours * 3600
    all_results: list[dict] = []

    _log(
        f"search_start mode={mode} target={target_candidates} "
        f"existing_replicates={existing_replicates} max_guided={max_guided_attempts}"
    )

    data = json.loads(Path(input_json).read_text())
    records = [
        r
        for r in data["results"]
        if r.get("is_knotted") and r.get("sequence") and not r.get("error")
    ]
    if existing_limit is not None:
        records = records[:existing_limit]

    if mode == "smoke":
        records = records[:2]
        existing_replicates = 1
        max_guided_attempts = min(max_guided_attempts, 2)
        batch_size = 2
        target_candidates = 1

    existing_jobs = []
    for record in records:
        for replicate in range(existing_replicates):
            existing_jobs.append(
                (
                    f"existing_{record['attempt']}_r{replicate}",
                    int(record["attempt"]),
                    replicate,
                    record["sequence"],
                )
            )
    _log(f"stage_existing_start jobs={len(existing_jobs)}")
    existing_results = _collect_map(
        repredict_sequence,
        existing_jobs,
        stage="existing_rescan",
        all_results=all_results,
        started_at=started_at,
    )
    _log(
        f"stage_existing_done results={len(existing_results)} "
        f"passing_distinct={len(_distinct_passing(all_results))}"
    )

    guided_launched = 0
    round_index = 0
    configs = [
        (200, 0.5, True),
        (256, 0.5, True),
        (200, 1.0, True),
        (256, 1.0, True),
        (256, 0.5, False),
        (300, 1.0, True),
    ]
    while (
        len(_distinct_passing(all_results)) < target_candidates
        and guided_launched < max_guided_attempts
        and time.time() < deadline
    ):
        remaining = max_guided_attempts - guided_launched
        this_batch = min(batch_size, remaining)
        jobs = []
        for offset in range(this_batch):
            length, weight, target_trefoil = configs[
                (guided_launched + offset) % len(configs)
            ]
            attempt = guided_launched + offset
            jobs.append(
                (
                    f"guided_{attempt}_L{length}_w{weight}_t{int(target_trefoil)}",
                    attempt,
                    length,
                    weight,
                    target_trefoil,
                )
            )
        _log(
            f"stage_guided_batch_start round={round_index} jobs={len(jobs)} "
            f"already_passing={len(_distinct_passing(all_results))}"
        )
        batch_results = _collect_map(
            confidence_guided_generate,
            jobs,
            stage="confidence_generation",
            all_results=all_results,
            started_at=started_at,
        )
        guided_launched += len(jobs)
        round_index += 1
        _log(
            f"stage_guided_batch_done results={len(batch_results)} "
            f"guided_launched={guided_launched} "
            f"passing_distinct={len(_distinct_passing(all_results))}"
        )
        _write_outputs(all_results, _distinct_passing(all_results))

    provisional = _distinct_passing(all_results)[:10]
    _log(f"stage_repeatability_start provisional={len(provisional)}")
    repeat_jobs = []
    for candidate_index, candidate in enumerate(provisional):
        for replicate in range(3):
            repeat_jobs.append(
                (
                    f"verify_{candidate_index}_r{replicate}",
                    100000 + candidate_index,
                    replicate,
                    candidate["sequence"],
                )
            )
    repeat_results = _collect_map(
        repredict_sequence,
        repeat_jobs,
        stage="repeatability",
        all_results=all_results,
        started_at=started_at,
    )

    by_sequence: dict[str, list[dict]] = {}
    for result in repeat_results:
        if result.get("sequence"):
            by_sequence.setdefault(result["sequence"], []).append(result)
    for candidate in provisional:
        repeats = by_sequence.get(candidate["sequence"], [])
        valid = [r for r in repeats if not r.get("error") and r.get("qc")]
        candidate["repeatability"] = {
            "n": len(valid),
            "n_pass": sum(bool(r["qc"].get("passes")) for r in valid),
            "n_knotted": sum((r["qc"].get("knot_score") or 0) > 0.5 for r in valid),
            "n_plddt_gt_0_5": sum((r["qc"].get("plddt") or 0) > 0.5 for r in valid),
            "mean_plddt": (
                round(statistics.mean(r["qc"]["plddt"] for r in valid), 6)
                if valid
                else None
            ),
        }

    final_candidates = sorted(
        provisional,
        key=lambda r: (
            r.get("repeatability", {}).get("n_pass", 0),
            r.get("repeatability", {}).get("n_knotted", 0),
            *_candidate_rank(r),
        ),
        reverse=True,
    )[:target_candidates]
    _write_outputs(all_results, final_candidates)

    status = (
        "target_reached"
        if len(final_candidates) >= target_candidates
        else "completed_below_target"
    )
    if time.time() >= deadline:
        status = "time_limit_reached"
    _update_state(
        stage="complete",
        status=status,
        completed=len(all_results),
        total=len(all_results),
        results=all_results,
        started_at=started_at,
        note=f"final_candidates={len(final_candidates)} guided_launched={guided_launched}",
    )
    _log(
        f"search_complete status={status} final_candidates={len(final_candidates)} "
        f"guided_launched={guided_launched}"
    )
