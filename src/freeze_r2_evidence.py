"""Freeze authoritative R2 metrics for the two strict-pass candidates."""

import json
from pathlib import Path

from topoly import alexander

from overnight_candidate_search import analyze_pdb_geometry


def main():
    root = Path("results/overnight_candidates")
    expanded = {
        row["id"]: row
        for row in json.loads((root / "expanded_candidates.json").read_text())[
            "candidates"
        ]
    }
    foldseek = {
        row["id"]: row
        for row in json.loads(
            Path("results/foldseek_top5/foldseek_all32_summary.json").read_text()
        )["candidates"]
    }
    frozen = []
    for candidate_id in ("candidate_01", "candidate_02"):
        source = expanded[candidate_id]
        fs = foldseek[candidate_id]
        pdb_path = root / "expanded_candidates" / f"{candidate_id}.pdb"
        topology = {k: float(v) for k, v in alexander(str(pdb_path), tries=500).items()}
        total = sum(topology.values())
        knot_score = 1 - topology.get("0_1", 0) / total if total else 0
        frozen.append(
            {
                "id": candidate_id,
                "sequence": source["sequence"],
                "length": source["length"],
                "source_plddt": source["source_plddt"],
                "source_ptm": source["source_ptm"],
                "pre_relaxation_knot_score": source["source_knot_score"],
                "relaxation_restraint_k": source["restraint_k"],
                "relaxation_ca_rmsd_angstrom": source["ca_rmsd_angstrom"],
                "post_relaxation_topology_500": topology,
                "post_relaxation_knot_score_500": knot_score,
                "geometry": analyze_pdb_geometry(pdb_path.read_text()),
                "pdb_path": str(pdb_path),
                "best_foldseek_hit": fs["best_hit"],
                "best_experimental_pdb_hit": fs["best_pdb_hit"],
                "knot_family_hits": fs["knot_family_hits"],
            }
        )
    output = Path("results/r2_frozen_candidate_evidence.json")
    output.write_text(json.dumps({"candidates": frozen}, indent=2))
    for row in frozen:
        print(
            row["id"],
            f"pLDDT={row['source_plddt']:.3f}",
            f"pTM={row['source_ptm']:.3f}",
            f"knot500={row['post_relaxation_knot_score_500']:.3f}",
            f"geometry={row['geometry']['geometry_pass']}",
            row["best_foldseek_hit"]["target"],
        )


if __name__ == "__main__":
    main()
