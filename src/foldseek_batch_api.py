"""Rate-limited Foldseek API search and candidate evidence scoring."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import requests


BASE_URL = "https://search.foldseek.com/api"
DATABASES = ["pdb100", "afdb-swissprot", "afdb50"]
KNOT_FAMILY_PATTERNS = {
    "SPOUT/RlmB/TrmH methyltransferase": re.compile(
        r"\b(SPOUT|SpoU|RlmB|TrmH|TrmD|RlmH|RNA.*methyltransferase|"
        r"ribosomal RNA.*methyltransferase)\b",
        re.IGNORECASE,
    ),
    "ubiquitin C-terminal hydrolase": re.compile(
        r"\b(ubiquitin.*hydrolase|deubiquitin|UCHL|UCH-L)\b",
        re.IGNORECASE,
    ),
    "transcarbamylase": re.compile(r"\btranscarbamylase\b", re.IGNORECASE),
}


def submit(path: Path) -> dict:
    with path.open("rb") as handle:
        response = requests.post(
            f"{BASE_URL}/ticket",
            files={"q": (path.name, handle, "chemical/x-pdb")},
            data=[
                ("mode", "3diaa"),
                *[("database[]", database) for database in DATABASES],
            ],
            timeout=120,
        )
    response.raise_for_status()
    return response.json()


def status(ticket_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/ticket/{ticket_id}", timeout=60)
    response.raise_for_status()
    return response.json()


def result(ticket_id: str) -> dict:
    response = requests.get(f"{BASE_URL}/result/{ticket_id}/0", timeout=180)
    response.raise_for_status()
    return response.json()


def flatten_hits(data: dict) -> list[dict]:
    hits = []
    for block in data.get("results", []):
        database = block.get("db")
        for group in block.get("alignments", []):
            for hit in group:
                item = dict(hit)
                item["database"] = database
                qlen = hit.get("qLen") or 0
                dlen = hit.get("dbLen") or 0
                item["query_coverage"] = (
                    (abs(hit["qEndPos"] - hit["qStartPos"]) + 1) / qlen
                    if qlen
                    else 0
                )
                item["target_coverage"] = (
                    (abs(hit["dbEndPos"] - hit["dbStartPos"]) + 1) / dlen
                    if dlen
                    else 0
                )
                hits.append(item)
    return hits


def knot_family(target: str) -> str | None:
    for family, pattern in KNOT_FAMILY_PATTERNS.items():
        if pattern.search(target):
            return family
    return None


def summarize(
    *,
    candidate_data: dict,
    result_dir: Path,
) -> list[dict]:
    summaries = []
    for candidate in candidate_data["candidates"]:
        candidate_id = candidate["id"]
        data = json.loads((result_dir / f"{candidate_id}_result.json").read_text())
        hits = flatten_hits(data)
        eligible = [
            hit
            for hit in hits
            if hit["eval"] < 1e-5
            and hit["query_coverage"] > 0.70
            and hit["seqId"] < 35
        ]
        pdb_eligible = [hit for hit in eligible if hit["database"] == "pdb100"]
        best = min(eligible, key=lambda hit: hit["eval"]) if eligible else None
        best_pdb = min(pdb_eligible, key=lambda hit: hit["eval"]) if pdb_eligible else None
        bonus_hits = [
            {**hit, "knot_family": knot_family(hit["target"])}
            for hit in eligible
            if knot_family(hit["target"])
        ]
        source_plddt = candidate["source_plddt"]
        relaxed_knot = candidate["relaxed_knot_score"]
        geometry_pass = candidate["geometry"]["geometry_pass"]
        core_pass = bool(
            geometry_pass
            and relaxed_knot > 0.9
            and source_plddt > 0.6
            and best
            and best_pdb
        )
        summaries.append(
            {
                "id": candidate_id,
                "source_plddt": source_plddt,
                "source_ptm": candidate["source_ptm"],
                "geometry_pass": geometry_pass,
                "relaxed_knot_score": relaxed_knot,
                "plddt_gt_0_6": source_plddt > 0.6,
                "plddt_gt_0_7_preferred": source_plddt > 0.7,
                "foldseek_eligible_hit": bool(best),
                "experimental_pdb_eligible_hit": bool(best_pdb),
                "core_pass": core_pass,
                "best_hit": best,
                "best_pdb_hit": best_pdb,
                "knot_family_hits": bonus_hits[:20],
                "n_eligible_hits": len(eligible),
                "n_eligible_pdb_hits": len(pdb_eligible),
            }
        )
    return summaries


def write_summary(path: Path, summaries: list[dict]) -> None:
    ranked = sorted(
        summaries,
        key=lambda item: (
            item["core_pass"],
            bool(item["knot_family_hits"]),
            item["plddt_gt_0_7_preferred"],
            item["source_plddt"],
            item["relaxed_knot_score"],
        ),
        reverse=True,
    )
    lines = [
        "# Foldseek evidence across 32 plausible generated knots",
        "",
        "Core criteria: geometry pass; relaxed knot score >0.9; source pLDDT >0.6; "
        "Foldseek E-value <1e-5; query coverage >70%; sequence identity <35%; "
        "and a qualifying experimental PDB100 hit.",
        "",
        "| ID | pLDDT | pTM | Knot | Best E-value | Qcov | SeqID | PDB hit | Knot-family bonus | Core pass |",
        "|---|---:|---:|---:|---:|---:|---:|---|---|---|",
    ]
    for item in ranked:
        best = item["best_hit"]
        family = (
            item["knot_family_hits"][0]["knot_family"]
            if item["knot_family_hits"]
            else ""
        )
        lines.append(
            f"| {item['id']} | {item['source_plddt']:.3f} | "
            f"{item['source_ptm']:.3f} | {item['relaxed_knot_score']:.3f} | "
            f"{best['eval']:.2e} | {best['query_coverage']:.1%} | "
            f"{best['seqId']:.1f}% | {'yes' if item['experimental_pdb_eligible_hit'] else 'no'} | "
            f"{family} | {'YES' if item['core_pass'] else 'no'} |"
            if best
            else f"| {item['id']} | {item['source_plddt']:.3f} | "
            f"{item['source_ptm']:.3f} | {item['relaxed_knot_score']:.3f} | "
            f"--- | --- | --- | no | {family} | no |"
        )
    lines.extend(
        [
            "",
            f"Core-pass candidates: {sum(item['core_pass'] for item in summaries)}/{len(summaries)}",
            f"Candidates with qualifying Foldseek hits: {sum(item['foldseek_eligible_hit'] for item in summaries)}/{len(summaries)}",
            f"Candidates with qualifying experimental PDB hits: {sum(item['experimental_pdb_eligible_hit'] for item in summaries)}/{len(summaries)}",
            f"Candidates with knot-family-associated hits: {sum(bool(item['knot_family_hits']) for item in summaries)}/{len(summaries)}",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--candidate-json",
        default="results/overnight_candidates/expanded_candidates.json",
    )
    parser.add_argument(
        "--candidate-dir",
        default="results/overnight_candidates/expanded_candidates",
    )
    parser.add_argument("--output-dir", default="results/foldseek_top5")
    parser.add_argument("--max-active", type=int, default=5)
    parser.add_argument("--poll-seconds", type=int, default=10)
    parser.add_argument("--max-retries", type=int, default=2)
    args = parser.parse_args()

    candidate_data = json.loads(Path(args.candidate_json).read_text())
    candidate_dir = Path(args.candidate_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    pending = [
        candidate["id"]
        for candidate in candidate_data["candidates"]
        if not (output_dir / f"{candidate['id']}_result.json").exists()
    ]
    active: dict[str, str] = {}
    retries: dict[str, int] = {candidate_id: 0 for candidate_id in pending}
    completed = sum(
        (output_dir / f"{candidate['id']}_result.json").exists()
        for candidate in candidate_data["candidates"]
    )
    total = len(candidate_data["candidates"])

    while pending or active:
        while pending and len(active) < args.max_active:
            candidate_id = pending.pop(0)
            pdb_path = candidate_dir / f"{candidate_id}.pdb"
            try:
                ticket = submit(pdb_path)
                active[candidate_id] = ticket["id"]
                (output_dir / f"{candidate_id}_ticket.json").write_text(
                    json.dumps(ticket, indent=2)
                )
                print(
                    f"FOLDSEEK_SUBMIT id={candidate_id} ticket={ticket['id']} "
                    f"active={len(active)}/{args.max_active}",
                    flush=True,
                )
            except Exception as exc:
                retries[candidate_id] += 1
                is_rate_limit = (
                    isinstance(exc, requests.HTTPError)
                    and exc.response is not None
                    and exc.response.status_code == 429
                )
                if is_rate_limit:
                    pending.insert(0, candidate_id)
                    print(
                        f"FOLDSEEK_RATE_LIMIT id={candidate_id}; cooling down 60s",
                        flush=True,
                    )
                    time.sleep(60)
                    break
                elif retries[candidate_id] <= args.max_retries:
                    pending.append(candidate_id)
                else:
                    print(f"FOLDSEEK_FAILED id={candidate_id} error={exc}", flush=True)

        time.sleep(args.poll_seconds)
        for candidate_id, ticket_id in list(active.items()):
            try:
                state = status(ticket_id)
                (output_dir / f"{candidate_id}_status.json").write_text(
                    json.dumps(state, indent=2)
                )
                if state.get("status") == "COMPLETE":
                    data = result(ticket_id)
                    (output_dir / f"{candidate_id}_result.json").write_text(
                        json.dumps(data, indent=2)
                    )
                    del active[candidate_id]
                    completed += 1
                    print(
                        f"FOLDSEEK_COMPLETE id={candidate_id} completed={completed}/{total} "
                        f"active={len(active)}/{args.max_active}",
                        flush=True,
                    )
                elif state.get("status") in {"ERROR", "UNKNOWN"}:
                    del active[candidate_id]
                    retries[candidate_id] += 1
                    if retries[candidate_id] <= args.max_retries:
                        pending.append(candidate_id)
                    else:
                        print(
                            f"FOLDSEEK_FAILED id={candidate_id} state={state}",
                            flush=True,
                        )
            except Exception as exc:
                print(
                    f"FOLDSEEK_POLL_WARNING id={candidate_id} error={exc}",
                    flush=True,
                )

        state = {
            "updated_at": time.time(),
            "total": total,
            "completed": completed,
            "pending": len(pending),
            "active": len(active),
            "max_active": args.max_active,
        }
        (output_dir / "batch_state.json").write_text(json.dumps(state, indent=2))

    summaries = summarize(candidate_data=candidate_data, result_dir=output_dir)
    (output_dir / "foldseek_all32_summary.json").write_text(
        json.dumps({"candidates": summaries}, indent=2)
    )
    with (output_dir / "foldseek_all32_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "id",
                "source_plddt",
                "source_ptm",
                "geometry_pass",
                "relaxed_knot_score",
                "plddt_gt_0_6",
                "plddt_gt_0_7_preferred",
                "foldseek_eligible_hit",
                "experimental_pdb_eligible_hit",
                "core_pass",
                "n_eligible_hits",
                "n_eligible_pdb_hits",
            ],
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow({key: item[key] for key in writer.fieldnames})
    write_summary(output_dir / "FOLDSEEK_ALL32_SUMMARY.md", summaries)
    print(
        "FOLDSEEK_BATCH_COMPLETE "
        + json.dumps(
            {
                "total": total,
                "core_pass": sum(item["core_pass"] for item in summaries),
                "experimental_pdb": sum(
                    item["experimental_pdb_eligible_hit"] for item in summaries
                ),
                "knot_family": sum(bool(item["knot_family_hits"]) for item in summaries),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
