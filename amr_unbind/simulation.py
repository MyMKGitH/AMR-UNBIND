from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import math
import time

import numpy as np
import pandas as pd
from openff.toolkit.topology import Molecule
from openmm import app, unit
import openmm as mm
from openmmforcefields.generators import SystemGenerator
from openmmforcefields.generators import SMIRNOFFTemplateGenerator

from .config import SimulationConfig
from .errors import SimulationError, SystemConstructionError
from .platform import PlatformSelection, choose_platform
from .progress import Heartbeat, MINIMIZED_FRACTION, NPT_FRACTION, NVT_FRACTION, SMD_FRACTION, SYSTEM_BUILT_FRACTION, safe_call


@dataclass(frozen=True)
class SystemBuild:
    topology: app.Topology
    positions: unit.Quantity
    system: mm.System
    ligand_indices: tuple[int, ...]
    protein_backbone_indices: tuple[int, ...]
    n_protein_atoms: int
    n_ligand_atoms: int
    n_total_atoms: int


def _residue_is_protein(residue) -> bool:
    return residue.name.upper() in {
        "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS",
        "ILE", "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
    }


def _add_backbone_restraints(system: mm.System, positions_nm: np.ndarray, indices: list[int], k: float) -> None:
    if not indices or k <= 0:
        return
    force = mm.CustomExternalForce("0.5*k_protein*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
    force.addGlobalParameter("k_protein", float(k))
    force.addPerParticleParameter("x0")
    force.addPerParticleParameter("y0")
    force.addPerParticleParameter("z0")
    for idx in indices:
        p = positions_nm[idx]
        force.addParticle(int(idx), [float(p[0]), float(p[1]), float(p[2])])
    system.addForce(force)


def _copy_system(system: mm.System) -> mm.System:
    return mm.XmlSerializer.deserialize(mm.XmlSerializer.serialize(system))


def build_system(prepared_pdb: Path, ligand_sdf: Path, config: SimulationConfig) -> SystemBuild:
    try:
        pdb = app.PDBFile(str(prepared_pdb))
        off_mol = Molecule.from_file(str(ligand_sdf), allow_undefined_stereo=True)
        off_mol.assign_partial_charges(partial_charge_method="gasteiger")
        if off_mol.n_conformers < 1:
            raise SystemConstructionError("Ligand file contains no 3D conformer.")

        modeller = app.Modeller(pdb.topology, pdb.positions)
        protein_atom_count = modeller.topology.getNumAtoms()
        ligand_topology = off_mol.to_topology().to_openmm()
        ligand_positions = off_mol.conformers[0].to_openmm()
        modeller.add(ligand_topology, ligand_positions)
        after_ligand_count = modeller.topology.getNumAtoms()
        ligand_indices = tuple(range(protein_atom_count, after_ligand_count))
        ligand_atom_objects = list(modeller.topology.atoms())[protein_atom_count:after_ligand_count]
        ligand_residue = ligand_atom_objects[0].residue if ligand_atom_objects else None
        if ligand_residue is not None:
            ligand_residue.name = "LIG"

        # Capture ligand indices before solvent is appended. This avoids the
        # original bug where "last N atoms" referred to water/ions.
        solvation_ff = app.ForceField(config.forcefield_protein, config.forcefield_water)
        smirnoff = SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield=config.small_molecule_forcefield)
        solvation_ff.registerTemplateGenerator(smirnoff.generator)
        modeller.addSolvent(
            solvation_ff,
            model="tip3p",
            padding=config.solvation_padding_nm * unit.nanometer,
            neutralize=True,
            ionicStrength=config.ionic_strength_molar * unit.molar,
        )

        if len(ligand_indices) != off_mol.n_atoms:
            raise SystemConstructionError(
                f"Ligand atom count mismatch: expected {off_mol.n_atoms}, found {len(ligand_indices)}."
            )

        backbone = []
        for atom in list(modeller.topology.atoms())[:protein_atom_count]:
            if _residue_is_protein(atom.residue) and atom.name in {"N", "CA", "C", "O"}:
                backbone.append(atom.index)

        system_generator = SystemGenerator(
            forcefields=[config.forcefield_protein, config.forcefield_water],
            small_molecule_forcefield=config.small_molecule_forcefield,
            molecules=[off_mol],
            forcefield_kwargs={
                "constraints": app.HBonds,
                "rigidWater": True,
                "removeCMMotion": True,
                "ewaldErrorTolerance": 1e-5,
            },
            periodic_forcefield_kwargs={
                "nonbondedMethod": app.PME,
                "nonbondedCutoff": 1.0 * unit.nanometer,
            },
            nonperiodic_forcefield_kwargs={
                "nonbondedMethod": app.NoCutoff,
            },
        )
        system = system_generator.create_system(modeller.topology)
        if system.getNumParticles() != modeller.topology.getNumAtoms():
            raise SystemConstructionError(
                f"System/topology particle mismatch: {system.getNumParticles()} vs {modeller.topology.getNumAtoms()}"
            )

        pos_nm = np.asarray(modeller.positions.value_in_unit(unit.nanometer), dtype=float)
        _add_backbone_restraints(
            system,
            pos_nm,
            backbone,
            config.protein_restraint_k_kj_mol_nm2,
        )
        return SystemBuild(
            topology=modeller.topology,
            positions=modeller.positions,
            system=system,
            ligand_indices=ligand_indices,
            protein_backbone_indices=tuple(backbone),
            n_protein_atoms=protein_atom_count,
            n_ligand_atoms=len(ligand_indices),
            n_total_atoms=system.getNumParticles(),
        )
    except SystemConstructionError:
        raise
    except Exception as exc:
        raise SystemConstructionError(f"OpenMM system construction failed: {exc}") from exc


def _new_simulation(topology, system, integrator, selection: PlatformSelection):
    platform = mm.Platform.getPlatformByName(selection.name)
    return app.Simulation(topology, system, integrator, platform, selection.properties)


def _ligand_com(positions_nm: np.ndarray, system: mm.System, ligand_indices: tuple[int, ...]) -> np.ndarray:
    masses = np.array(
        [system.getParticleMass(i).value_in_unit(unit.dalton) for i in ligand_indices], dtype=float
    )
    coords = positions_nm[np.asarray(ligand_indices, dtype=int)]
    total_mass = float(masses.sum())
    if not np.isfinite(total_mass) or total_mass <= 0:
        raise SimulationError("Ligand atom masses are invalid or sum to zero.")
    return (coords * masses[:, None]).sum(axis=0) / total_mass


def _add_com_projection_force(
    system: mm.System,
    protein_indices: tuple[int, ...],
    ligand_indices: tuple[int, ...],
    direction: np.ndarray,
    initial_target: float,
    k: float,
) -> mm.CustomCentroidBondForce:
    expr = "0.5*k*((x2-x1)*nx + (y2-y1)*ny + (z2-z1)*nz - target)^2"
    force = mm.CustomCentroidBondForce(2, expr)
    force.addGroup(list(protein_indices))
    force.addGroup(list(ligand_indices))
    force.addBond([0, 1])
    force.setUsesPeriodicBoundaryConditions(True)

    force.addGlobalParameter("k", float(k))
    force.addGlobalParameter("nx", float(direction[0]))
    force.addGlobalParameter("ny", float(direction[1]))
    force.addGlobalParameter("nz", float(direction[2]))
    force.addGlobalParameter("target", float(initial_target))
    force.addEnergyParameterDerivative("target")
    return force


def _write_positions(path: Path, topology: app.Topology, positions) -> None:
    with path.open("w", encoding="utf-8") as handle:
        app.PDBFile.writeFile(topology, positions, handle, keepIds=True)


def _validate_state(simulation: app.Simulation, max_abs_nm: float = 100.0) -> tuple[float, float]:
    state = simulation.context.getState(getPositions=True, getEnergy=True)
    positions_nm = state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    if not np.isfinite(positions_nm).all() or float(np.max(np.abs(positions_nm))) > max_abs_nm:
        raise SimulationError("Simulation produced invalid or explosively large coordinates.")
    potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
    if not math.isfinite(float(potential)):
        raise SimulationError("Simulation produced a non-finite potential energy.")
    return float(potential), float(np.max(np.abs(positions_nm)))


def _integrator(config: SimulationConfig, seed: int) -> mm.LangevinMiddleIntegrator:
    integrator = mm.LangevinMiddleIntegrator(
        config.temperature_kelvin * unit.kelvin,
        config.friction_per_ps / unit.picosecond,
        config.timestep_fs * unit.femtoseconds,
    )
    integrator.setRandomNumberSeed(int(seed))
    return integrator


def _effective_report_interval(total_steps: int, requested: int) -> int:
    """Choose a reporting interval that records the final SMD step too."""
    if total_steps < 1 or requested < 1:
        raise ValueError("total_steps and requested report interval must be positive")
    upper = min(total_steps, requested)
    for interval in range(upper, 0, -1):
        if total_steps % interval == 0:
            return interval
    return 1


def _run_chunked_equilibration(
    sim: app.Simulation,
    total_steps: int,
    *,
    stage: str,
    start_fraction: float,
    end_fraction: float,
    progress_callback=None,
    min_chunks: int = 10,
    max_chunks: int = 40,
) -> None:
    """Run `total_steps` of dynamics in smaller `sim.step(...)` chunks,
    reporting real (not heartbeat-only) progress between chunks.

    NVT and NPT equilibration used to each be a single `sim.step(N)` call
    -- fine for a fast GPU, but on a slow or CPU-only machine (the
    situation this whole progress overhaul was prompted by) that single
    call can run for a long time with nothing reported until it finishes,
    identical from the outside to being hung. Chunking gives real,
    monotonically increasing progress through each phase, the same way
    the SMD phase already reports progress step by step. `min_chunks`/
    `max_chunks` bound the chunk count so a very short equilibration
    (few hundred steps) doesn't pay per-chunk Python/OpenMM call overhead
    for no visible benefit, and a very long one still reports often enough
    to be useful without reporting on literally every step.
    """
    if total_steps <= 0:
        return
    chunk_count = max(1, min(max_chunks, max(min_chunks, total_steps // 200)))
    chunk_count = min(chunk_count, total_steps)
    base, remainder = divmod(total_steps, chunk_count)
    completed = 0
    for i in range(chunk_count):
        this_chunk = base + (1 if i < remainder else 0)
        if this_chunk <= 0:
            continue
        sim.step(this_chunk)
        completed += this_chunk
        if progress_callback:
            frac = start_fraction + (end_fraction - start_fraction) * (completed / total_steps)
            safe_call(progress_callback, stage, frac, f"{completed}/{total_steps} steps")


def _minimization_retry_schedule(config: SimulationConfig) -> list[tuple[int, float]]:
    """(maxIterations, tolerance_kj_mol_nm) pairs to try, in order, for the
    pre-velocity clash-relief minimization. Each subsequent attempt uses a
    larger iteration budget and a looser force tolerance than the last --
    appropriate when the previous attempt hit its iteration limit before
    converging, which is the common failure mode for a system with severe
    initial clashes (a badly docked pose, or solvent packed too tightly)
    rather than a mild one.
    """
    max_iters = config.minimization_iterations if config.minimization_iterations > 0 else 2500
    return [(max_iters, 10.0), (max(max_iters * 4, 10000), 50.0)]


def _minimize_with_retry(min_sim: app.Simulation, config: SimulationConfig) -> None:
    """Runs the pre-velocity clash-relief minimization, retrying once with a
    much larger budget if the result isn't left in a finite, sane state.

    A single fixed-budget minimization can still leave a severely clashing
    system (a poorly docked pose, or solvent packed too tightly against the
    protein/ligand) only partially relaxed. Handing that structure to
    setVelocitiesToTemperature() and the integrator is what turns a bad
    starting structure into the opaque, low-level "Particle coordinate is
    NaN" crash deep inside OpenMM -- with no indication of *why* it
    happened. Validating right here, before velocities are ever assigned,
    means that failure mode either gets resolved by the retry, or turns
    into a clear, actionable error instead of a silent (or cryptic)
    downstream crash. OpenMM itself can raise its own exception type (not
    SimulationError) when a minimization step produces a non-finite state,
    so any exception here -- ours or OpenMM's -- is treated as "try again,"
    not "necessarily done."
    """
    last_exc: Exception | None = None
    for iters, tolerance in _minimization_retry_schedule(config):
        try:
            min_sim.minimizeEnergy(
                maxIterations=iters,
                tolerance=tolerance * unit.kilojoule / (unit.mole * unit.nanometer),
            )
            _validate_state(min_sim)
            return
        except Exception as exc:  # noqa: BLE001 -- see docstring: any failure here means retry.
            last_exc = exc
    raise SimulationError(
        "Initial energy minimization could not resolve severe atomic overlap, "
        "even after retrying with a much larger iteration budget and looser "
        "tolerance. This is almost always a badly docked ligand pose (try "
        "automatic docking again, or a different --docking-center-angstrom) "
        "or solvation padding too tight for the system (try increasing "
        "solvation_padding_nm in SimulationConfig) -- not something more "
        "minimization iterations alone can fix."
    ) from last_exc


def run_simulation(
    build: SystemBuild,
    geometry_vector: np.ndarray,
    config: SimulationConfig,
    run_dir: Path,
    progress_callback=None,
) -> pd.DataFrame:
    config.validate()
    vector = np.asarray(geometry_vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-12:
        raise SimulationError("Invalid pulling direction.")
    vector /= norm

    selection = choose_platform(config.preferred_platform, config.cuda_device_index)
    (run_dir / "compute_metadata.json").write_text(
        json.dumps(
            {
                "platform": selection.name,
                "platform_properties": selection.properties,
                "platform_reason": selection.reason,
                "pull_steps": config.pull_steps,
                "pull_time_ps": config.pull_time_ps,
                "direction": vector.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    try:
        # 1) Minimize on a copy of the base system.
        min_system = _copy_system(build.system)
        min_sim = _new_simulation(build.topology, min_system, _integrator(config, config.seed), selection)
        min_sim.context.setPositions(build.positions)

        # Minimize coordinates FIRST to remove atomic overlap before setting
        # velocities, with a validated retry if the first pass doesn't
        # actually leave the system in a finite state -- see
        # _minimize_with_retry's docstring. Minimization has no
        # intermediate progress of its own (OpenMM's minimizeEnergy() does
        # not expose a per-iteration callback), so a Heartbeat -- not
        # chunking -- is what keeps this from looking silent on a slow
        # machine or a severely-clashing structure that needs the retry's
        # larger iteration budget.
        with Heartbeat(progress_callback, "minimized", SYSTEM_BUILT_FRACTION, note="energy minimization"):
            _minimize_with_retry(min_sim, config)
        min_sim.context.setVelocitiesToTemperature(config.temperature_kelvin * unit.kelvin, config.seed)
        min_sim.minimizeEnergy(maxIterations=config.minimization_iterations)
        min_state = min_sim.context.getState(getPositions=True, getVelocities=True)
        min_energy, _ = _validate_state(min_sim)
        if progress_callback:
            safe_call(progress_callback, "minimized", MINIMIZED_FRACTION, "complete")

        # 2) NVT equilibration.
        nvt_system = _copy_system(build.system)
        nvt_sim = _new_simulation(build.topology, nvt_system, _integrator(config, config.seed + 1), selection)
        nvt_sim.context.setPositions(min_state.getPositions())
        nvt_sim.context.setVelocities(min_state.getVelocities())
        nvt_steps = int(round(config.nvt_equilibration_ps * 1000.0 / config.timestep_fs))
        _run_chunked_equilibration(
            nvt_sim,
            nvt_steps,
            stage="nvt",
            start_fraction=MINIMIZED_FRACTION,
            end_fraction=NVT_FRACTION,
            progress_callback=progress_callback,
        )
        nvt_state = nvt_sim.context.getState(getPositions=True, getVelocities=True)
        nvt_energy, _ = _validate_state(nvt_sim)
        if progress_callback:
            safe_call(progress_callback, "nvt", NVT_FRACTION, "complete")

        # 3) NPT equilibration.
        npt_system = _copy_system(build.system)
        barostat = mm.MonteCarloBarostat(
            config.pressure_atm * unit.atmosphere,
            config.temperature_kelvin * unit.kelvin,
            25,
        )
        barostat.setRandomNumberSeed(config.seed + 20)
        npt_system.addForce(barostat)
        npt_sim = _new_simulation(build.topology, npt_system, _integrator(config, config.seed + 2), selection)
        npt_sim.context.setPositions(nvt_state.getPositions())
        npt_sim.context.setVelocities(nvt_state.getVelocities())
        npt_steps = int(round(config.npt_equilibration_ps * 1000.0 / config.timestep_fs))
        _run_chunked_equilibration(
            npt_sim,
            npt_steps,
            stage="npt",
            start_fraction=NVT_FRACTION,
            end_fraction=NPT_FRACTION,
            progress_callback=progress_callback,
        )
        npt_state = npt_sim.context.getState(getPositions=True, getVelocities=True)
        npt_energy, _ = _validate_state(npt_sim)
        if progress_callback:
            safe_call(progress_callback, "npt", NPT_FRACTION, "complete")
    except Exception as exc:
        raise SimulationError(f"Minimization/equilibration failed: {exc}") from exc

    # 4) SMD: no barostat, to keep the moving-restraint protocol distinct from
    # pressure control. Protein-backbone restraints remain active.
    smd_system = _copy_system(build.system)
    initial_positions_nm = np.asarray(
        npt_state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
    )
    ligand_masses = np.array(
        [smd_system.getParticleMass(i).value_in_unit(unit.dalton) for i in build.ligand_indices],
        dtype=float,
    )
    prot_com = _ligand_com(initial_positions_nm, smd_system, build.protein_backbone_indices)
    lig_com = _ligand_com(initial_positions_nm, smd_system, build.ligand_indices)
    rel_com = lig_com - prot_com
    initial_target = float(np.dot(rel_com, vector))

    smd_force = _add_com_projection_force(
        smd_system,
        build.protein_backbone_indices,
        build.ligand_indices,
        vector,
        initial_target,
        config.steering_k_kj_mol_nm2,
    )
    smd_system.addForce(smd_force)
    smd_sim = _new_simulation(build.topology, smd_system, _integrator(config, config.seed + 3), selection)
    smd_sim.context.setPeriodicBoxVectors(*npt_state.getPeriodicBoxVectors())
    smd_sim.context.setPositions(npt_state.getPositions())
    smd_sim.context.setVelocities(npt_state.getVelocities())

    total_steps = config.pull_steps
    report_interval = _effective_report_interval(total_steps, config.report_interval_steps)
    _write_positions(run_dir / "equilibrated_start.pdb", build.topology, npt_state.getPositions())
    dcd_path = run_dir / "smd_trajectory.dcd"
    smd_sim.reporters.append(
        app.DCDReporter(str(dcd_path), report_interval, enforcePeriodicBox=False)
    )
    smd_sim.reporters.append(
        app.StateDataReporter(
            str(run_dir / "state_data.csv"),
            report_interval,
            step=True,
            time=True,
            potentialEnergy=True,
            kineticEnergy=True,
            temperature=True,
            volume=True,
            density=True,
            speed=True,
            totalSteps=total_steps,
        )
    )

    rows: list[dict[str, float | int]] = []
    cumulative_work = 0.0
    previous_force = 0.0
    previous_target = initial_target
    dt_ps = config.timestep_fs / 1000.0
    t_start = time.time()

    try:
        for step in range(1, total_steps + 1):
            time_ps = step * dt_ps
            target = initial_target + config.pull_velocity_nm_per_ps * time_ps
            smd_sim.context.setParameter("target", float(target))
            smd_sim.step(1)

            # Integrate protocol work at full integrator resolution (every
            # step), not only at report_interval. The moving-restraint work
            # is W = integral of dE/d(target) d(target); sampling it only at
            # the (much coarser) reporting cadence would silently coarsen the
            # trapezoidal quadrature and bias the reported work. Reading the
            # analytic parameter derivative avoids the cost of a full
            # position readout on every step.
            deriv_state = smd_sim.context.getState(getParameterDerivatives=True)
            force = float(deriv_state.getEnergyParameterDerivatives()["target"])
            if not math.isfinite(force):
                raise SimulationError(f"Non-finite steering force encountered at step {step}.")
            cumulative_work += 0.5 * (previous_force + force) * (target - previous_target)
            previous_force = force
            previous_target = target

            should_record = step == 1 or step % report_interval == 0 or step == total_steps
            if should_record:
                state = smd_sim.context.getState(
                    getPositions=True, getEnergy=True, enforcePeriodicBox=True
                )
                positions_nm = np.asarray(
                    state.getPositions(asNumpy=True).value_in_unit(unit.nanometer), dtype=float
                )
                prot_com = _ligand_com(positions_nm, smd_system, build.protein_backbone_indices)
                lig_com = _ligand_com(positions_nm, smd_system, build.ligand_indices)

                # Minimum image convention relative to protein backbone
                box_vectors = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(unit.nanometer)
                box_l = np.diagonal(box_vectors)
                rel_com = lig_com - prot_com
                rel_com -= np.round(rel_com / box_l) * box_l

                coordinate = float(np.dot(rel_com, vector))
                displacement = coordinate - initial_target
                potential = state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole)
                kinetic = state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)
                rows.append(
                    {
                        "Step": step,
                        "Time (ps)": step * dt_ps,
                        "Target Coordinate (nm)": target,
                        "Reaction Coordinate (nm)": coordinate,
                        "Displacement (nm)": displacement,
                        "Steering Force (kJ/mol/nm)": force,
                        "Cumulative Work (kJ/mol)": cumulative_work,
                        "Potential Energy (kJ/mol)": float(potential),
                        "Kinetic Energy (kJ/mol)": float(kinetic),
                    }
                )
                if progress_callback:
                    safe_call(
                        progress_callback,
                        "smd",
                        NPT_FRACTION + (SMD_FRACTION - NPT_FRACTION) * step / total_steps,
                        f"{step}/{total_steps} steps, {time.time() - t_start:.0f}s elapsed",
                    )

            if step % config.checkpoint_interval_steps == 0:
                smd_sim.saveCheckpoint(str(run_dir / "smd_checkpoint.chk"))
                smd_sim.saveState(str(run_dir / "smd_state.xml"))

            if step % max(report_interval, 100) == 0 or step == total_steps:
                _validate_state(smd_sim)
    except Exception as exc:
        raise SimulationError(f"SMD failed at step {step}/{total_steps}: {exc}") from exc

    if not rows:
        raise SimulationError("SMD produced no analysis samples.")

    result = pd.DataFrame(rows)
    result.to_csv(run_dir / "smd_analysis.csv", index=False, float_format="%.8g")
    smd_sim.saveState(str(run_dir / "final_state.xml"))
    smd_sim.saveCheckpoint(str(run_dir / "final_checkpoint.chk"))
    final_positions = smd_sim.context.getState(getPositions=True).getPositions()
    _write_positions(run_dir / "final_structure.pdb", build.topology, final_positions)

    summary = {
        "minimum_energy_kj_mol": min_energy,
        "nvt_final_energy_kj_mol": nvt_energy,
        "npt_final_energy_kj_mol": npt_energy,
        "final_reaction_coordinate_nm": float(result.iloc[-1]["Reaction Coordinate (nm)"]),
        "requested_pull_distance_nm": config.pull_distance_nm,
        "achieved_displacement_nm": float(result.iloc[-1]["Displacement (nm)"]),
        "final_force_kj_mol_nm": float(result.iloc[-1]["Steering Force (kJ/mol/nm)"]),
        "cumulative_work_kj_mol": float(result.iloc[-1]["Cumulative Work (kJ/mol)"]),
        "trajectory_report_interval_steps": report_interval,
        "trajectory_frames": int(total_steps // report_interval),
        "elapsed_wall_seconds": time.time() - t_start,
    }
    (run_dir / "simulation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return result
