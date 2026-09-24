from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
from Bio.PDB import PDBParser

from .errors import InputValidationError


@dataclass(frozen=True)
class PullGeometry:
    vector: np.ndarray
    distance_nm: float
    core_point_angstrom: np.ndarray
    exit_point_angstrom: np.ndarray


def _heavy_atom_centroid(residue) -> np.ndarray:
    coords = []
    for atom in residue.get_atoms():
        element = (getattr(atom, "element", "") or "").strip().upper()
        atom_name = (getattr(atom, "name", "") or "").strip().upper()
        if element == "H" or atom_name.startswith("H"):
            continue
        coords.append(atom.get_coord())
    if not coords:
        raise InputValidationError(f"Residue {residue.get_full_id()} has no heavy atoms.")
    return np.mean(np.asarray(coords, dtype=float), axis=0)


def count_protein_heavy_atoms_near_point(
    pdb_path: Path,
    point_angstrom,
    radius_angstrom: float,
) -> int:
    """Count real protein (``ATOM``, not ``HETATM``/water) heavy atoms
    within ``radius_angstrom`` of ``point_angstrom`` in the structure at
    ``pdb_path``.

    This answers a direct, structure-agnostic question -- "is there
    actually protein mass here, or would a docking search at this point
    find nothing to dock against" -- that a pure distance-from-one-
    reference-point check cannot: whether a given distance from
    ``core_point_angstrom`` is "too far" depends entirely on how large
    that particular protein is and where its pocket happens to sit
    relative to the core/exit residues, so no single threshold on that
    distance can reliably separate "a real, if distant, pocket on this
    protein" from "a coordinate that happens to be a similar distance
    away but has no relationship to this protein at all" (e.g. reused
    from a different structure's coordinate frame). Checking for real
    nearby atoms sidesteps that -- it's true or false regardless of the
    protein's size or shape.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = structure[0]
    point = np.asarray(point_angstrom, dtype=float)
    count = 0
    for chain in model:
        for residue in chain:
            if residue.id[0] != " ":
                continue  # skip HETATM/water -- only real protein residues count
            for atom in residue:
                element = (getattr(atom, "element", "") or "").strip().upper()
                name = (getattr(atom, "name", "") or "").strip().upper()
                if element == "H" or name.startswith("H"):
                    continue
                if np.linalg.norm(atom.get_coord() - point) <= radius_angstrom:
                    count += 1
    return count


def calculate_pull_geometry(
    pdb_path: Path,
    core_chain: str,
    core_residue: int,
    exit_chain: str,
    exit_residue: int,
) -> PullGeometry:
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("target", str(pdb_path))
    model = structure[0]
    try:
        core = model[core_chain][core_residue]
        exit_res = model[exit_chain][exit_residue]
    except KeyError as exc:
        raise InputValidationError(f"Could not resolve selected residue/chain: {exc}") from exc

    core_point = _heavy_atom_centroid(core)
    exit_point = _heavy_atom_centroid(exit_res)
    vector_ang = exit_point - core_point
    distance_ang = float(np.linalg.norm(vector_ang))
    if not np.isfinite(distance_ang) or distance_ang < 1e-6:
        raise InputValidationError("Core and exit residues define a zero or invalid direction vector.")

    vector = vector_ang / distance_ang
    return PullGeometry(
        vector=vector.astype(float),
        distance_nm=distance_ang / 10.0,
        core_point_angstrom=core_point,
        exit_point_angstrom=exit_point,
    )
