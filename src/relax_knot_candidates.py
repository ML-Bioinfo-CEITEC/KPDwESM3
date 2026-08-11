"""Restrained OpenMM minimization of high-confidence knotted near misses."""

from __future__ import annotations

import json
from pathlib import Path

import modal

from overnight_candidate_search import analyze_pdb_geometry


app = modal.App("esm3-relax-knot-candidates")

image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install(
        "openmm",
        "topoly",
        "numpy",
        "git+https://github.com/openmm/pdbfixer.git",
    )
    .add_local_python_source("overnight_candidate_search")
)


@app.function(image=image, cpu=4, timeout=30 * 60, retries=1)
def relax_one(
    job_id: str,
    pdb_text: str,
    restraint_k: float,
) -> dict:
    import io
    import tempfile
    import time

    import numpy as np
    from openmm import CustomExternalForce, LangevinMiddleIntegrator
    from openmm import unit
    from openmm.app import ForceField, HBonds, NoCutoff, PDBFile, Simulation
    from pdbfixer import PDBFixer
    from topoly import alexander

    started = time.time()
    result = {"job_id": job_id, "restraint_k": restraint_k, "error": None}
    try:
        fixer = PDBFixer(pdbfile=io.StringIO(pdb_text))
        fixer.findMissingResidues()
        fixer.missingResidues = {}
        fixer.findNonstandardResidues()
        fixer.replaceNonstandardResidues()
        fixer.removeHeterogens(keepWater=False)
        fixer.findMissingAtoms()
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(7.0)

        forcefield = ForceField("amber14-all.xml")
        system = forcefield.createSystem(
            fixer.topology,
            nonbondedMethod=NoCutoff,
            constraints=HBonds,
        )

        # Restrain C-alpha atoms to preserve global topology while allowing
        # local stereochemistry and clashes to relax.
        restraint = CustomExternalForce(
            "0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)"
        )
        restraint.addGlobalParameter(
            "k", restraint_k * unit.kilojoule_per_mole / unit.nanometer**2
        )
        for name in ("x0", "y0", "z0"):
            restraint.addPerParticleParameter(name)
        positions = fixer.positions
        ca_indices = []
        for atom in fixer.topology.atoms():
            if atom.name == "CA":
                pos = positions[atom.index].value_in_unit(unit.nanometer)
                restraint.addParticle(atom.index, [pos.x, pos.y, pos.z])
                ca_indices.append(atom.index)
        system.addForce(restraint)

        integrator = LangevinMiddleIntegrator(
            300 * unit.kelvin,
            1 / unit.picosecond,
            0.002 * unit.picoseconds,
        )
        simulation = Simulation(fixer.topology, system, integrator)
        simulation.context.setPositions(positions)
        initial_state = simulation.context.getState(getEnergy=True)
        initial_energy = initial_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )
        simulation.minimizeEnergy(maxIterations=3000)
        final_state = simulation.context.getState(getPositions=True, getEnergy=True)
        final_energy = final_state.getPotentialEnergy().value_in_unit(
            unit.kilojoule_per_mole
        )

        output = io.StringIO()
        PDBFile.writeFile(
            fixer.topology,
            final_state.getPositions(),
            output,
            keepIds=True,
        )
        relaxed_pdb = output.getvalue()

        with tempfile.NamedTemporaryFile(
            suffix=".pdb", mode="w", delete=False
        ) as handle:
            handle.write(relaxed_pdb)
            path = handle.name
        topology = {k: float(v) for k, v in alexander(path, tries=100).items()}
        total = sum(topology.values())
        knot_score = (
            1.0 - topology.get("0_1", 0.0) / total if total > 0 else 0.0
        )

        original_pdb = PDBFile(io.StringIO(pdb_text))
        original_ca = [
            original_pdb.positions[a.index].value_in_unit(unit.nanometer)
            for a in original_pdb.topology.atoms()
            if a.name == "CA"
        ]
        relaxed_ca = [
            final_state.getPositions()[a.index].value_in_unit(unit.nanometer)
            for a in fixer.topology.atoms()
            if a.name == "CA"
        ]
        n = min(len(original_ca), len(relaxed_ca))
        if n:
            diffs = np.array(
                [
                    [
                        relaxed_ca[i].x - original_ca[i].x,
                        relaxed_ca[i].y - original_ca[i].y,
                        relaxed_ca[i].z - original_ca[i].z,
                    ]
                    for i in range(n)
                ]
            )
            ca_rmsd_angstrom = float(np.sqrt(np.mean(np.sum(diffs**2, axis=1))) * 10)
        else:
            ca_rmsd_angstrom = None

        result.update(
            {
                "initial_energy_kj_mol": initial_energy,
                "final_energy_kj_mol": final_energy,
                "topology": topology,
                "knot_score": knot_score,
                "geometry": analyze_pdb_geometry(relaxed_pdb),
                "ca_rmsd_angstrom": ca_rmsd_angstrom,
                "relaxed_pdb": relaxed_pdb,
            }
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["time_s"] = round(time.time() - started, 2)
    return result


@app.local_entrypoint()
def main(
    output_json: str = "results/overnight_candidates/relaxation.json",
    pool: bool = False,
    seed_pool: bool = False,
    seed_json: str = "results/overnight_candidates/seed_conversion.json",
    max_candidates: int = 30,
    pool_strengths: str = "200",
):
    root = Path("results/overnight_candidates")
    if seed_pool:
        seed_data = json.loads(Path(seed_json).read_text())
        sources = [
            Path(result["best"]["pdb_path"])
            for result in seed_data["results"]
            if result.get("best", {}).get("pdb_path")
            and (result["best"].get("qc", {}).get("plddt") or 0) > 0.5
            and (result["best"].get("qc", {}).get("knot_score") or 0) > 0.5
            and Path(result["best"]["pdb_path"]).exists()
        ][:max_candidates]
        strengths = [float(value) for value in pool_strengths.split(",") if value]
    elif pool:
        rows = json.loads((root / "all_results.json").read_text())
        best_by_sequence = {}
        for row in rows:
            qc = row.get("qc") or {}
            path = row.get("pdb_path")
            if (
                row.get("error")
                or not path
                or not qc.get("canonical_sequence")
                or (qc.get("plddt") or 0) <= 0.5
                or (qc.get("knot_score") or 0) <= 0.5
            ):
                continue
            sequence = row.get("sequence")
            if not sequence:
                continue
            if sequence not in best_by_sequence or qc["plddt"] > best_by_sequence[sequence]["qc"]["plddt"]:
                best_by_sequence[sequence] = row
        ranked = sorted(
            best_by_sequence.values(),
            key=lambda row: row["qc"]["plddt"],
            reverse=True,
        )[:max_candidates]
        sources = [Path(row["pdb_path"]) for row in ranked if Path(row["pdb_path"]).exists()]
        strengths = [float(value) for value in pool_strengths.split(",") if value]
    else:
        preferred = [
            root / "pdb_seed_conversion/seed0_cfg3_m1.0_w2.0_t1.pdb",
            root / "pdb/near_guided_105_L256_w1.0_t1.pdb",
            root / "pdb/near_guided_71_L300_w1.0_t1.pdb",
            root / "pdb/near_existing_42_r0.pdb",
        ]
        sources = [path for path in preferred if path.exists()]
        strengths = [10.0, 50.0, 200.0]
    jobs = [
        (
            f"{source.stem}_k{strength}",
            source.read_text(),
            strength,
        )
        for source in sources
        for strength in strengths
    ]
    print(f"RELAX_START sources={len(sources)} jobs={len(jobs)}", flush=True)
    results = []
    for item in relax_one.map(
        *list(zip(*jobs)),
        order_outputs=False,
        return_exceptions=True,
    ):
        if isinstance(item, Exception):
            result = {"error": f"{type(item).__name__}: {item}"}
        else:
            result = item
        results.append(result)
        passing = sum(
            not r.get("error")
            and r.get("knot_score", 0) > 0.5
            and r.get("geometry", {}).get("geometry_pass")
            for r in results
        )
        print(
            f"RELAX_STATUS completed={len(results)}/{len(jobs)} passing={passing}",
            flush=True,
        )

    out = Path(output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    pdb_dir = out.parent / f"pdb_{out.stem}"
    pdb_dir.mkdir(parents=True, exist_ok=True)
    cleaned = []
    for result in results:
        result = dict(result)
        pdb_text = result.pop("relaxed_pdb", None)
        if pdb_text:
            path = pdb_dir / f"{result.get('job_id','unknown')}.pdb"
            path.write_text(pdb_text)
            result["pdb_path"] = str(path)
        cleaned.append(result)
    summary = {
        "n_jobs": len(jobs),
        "n_completed": len(cleaned),
        "n_errors": sum(bool(r.get("error")) for r in cleaned),
        "n_passing_geometry_and_knot": sum(
            not r.get("error")
            and r.get("knot_score", 0) > 0.5
            and r.get("geometry", {}).get("geometry_pass")
            for r in cleaned
        ),
    }
    out.write_text(json.dumps({"summary": summary, "results": cleaned}, indent=2))
    print("RELAX_COMPLETE " + json.dumps(summary), flush=True)
