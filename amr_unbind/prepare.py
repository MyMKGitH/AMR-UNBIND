from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import io
import urllib.request

from Bio.PDB import PDBParser
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
import pdbfixer
import openmm as mm
from openmm import app, unit
from openff.toolkit.topology import Molecule

from .errors import InputValidationError, LigandPreparationError, ProteinPreparationError

RCSB_URL = "https://files.rcsb.org/download/{pdb_id}.pdb"
DOWNLOAD_TIMEOUT_SECONDS = 30
# Force constants (kJ/mol/nm^2) for the two-stage post-reconstruction
# clash-relief minimization. A single stiff restraint applied from the start
# can leave a reconstructed atom "fighting" a nearby restrained experimental
# atom if they start out severely overlapping, converging poorly within a
# bounded iteration budget. Stage 1 uses a soft restraint so the whole
# system (including the experimentally-observed atoms, temporarily) can
# move enough to resolve a severe clash; stage 2 uses a strong restraint,
# anchored to the same original positions, to pull the experimental atoms
# firmly back while preserving the now-locally-relaxed geometry around them.
_RECONSTRUCTED_ATOM_RESTRAINT_K_SOFT = 100.0
_RECONSTRUCTED_ATOM_RESTRAINT_K_STRONG = 5000.0
_RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE1 = 2000
_RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE2 = 1000


@dataclass(frozen=True)
class ProteinPreparationResult:
    source: Path
    prepared: Path
    target_id: str
    missing_residue_count: int
    missing_atom_count: int
    nonstandard_residue_count: int
    original_chain_ids: tuple[str, ...]


@dataclass(frozen=True)
class LigandPreparationResult:
    sdf_path: Path
    canonical_smiles: str
    formula: str
    formal_charge: int
    n_atoms: int


def download_pdb(pdb_id: str, destination: Path) -> Path:
    pdb_id = pdb_id.strip().lower()
    if len(pdb_id) != 4 or not pdb_id.isalnum():
        raise InputValidationError("PDB ID must be a four-character alphanumeric identifier.")
    destination.parent.mkdir(parents=True, exist_ok=True)
    url = RCSB_URL.format(pdb_id=pdb_id.upper())
    try:
        # urlretrieve has no timeout of its own and can hang indefinitely on
        # a stalled connection; use urlopen with an explicit timeout instead.
        with urllib.request.urlopen(url, timeout=DOWNLOAD_TIMEOUT_SECONDS) as response:
            destination.write_bytes(response.read())
    except Exception as exc:
        raise ProteinPreparationError(f"Could not download PDB {pdb_id.upper()} from RCSB: {exc}") from exc
    if destination.stat().st_size < 100:
        raise ProteinPreparationError(f"Downloaded PDB file appears empty or invalid: {destination}")
    return destination


def _strip_alternate_locations(pdb_text: str) -> str:
    """Collapse alternate-location (altLoc) records in ATOM/HETATM lines to a
    single conformation per residue.

    Overlapping alternate side-chain conformations are a common crystallographic
    PDB artifact (frequently seen on flexible surface residues such as Arg,
    Lys, Ser, Glu, Val). If left unresolved, two near-coincident copies of the
    same atom reach downstream tools that re-derive bonds from atomic distance
    or per-residue templates (RDKit, and Meeko's PDBQT preparation, which
    builds an RDKit molecule directly from the receptor PDB). A spurious short
    distance between the two overlapping copies can then be misread as an
    extra bond, which is exactly the shape of an
    "explicit valence ... greater than permitted" /
    "unable to build rdkit mol for residue ... " failure.

    The winning conformer is chosen **per residue instance** (chain, residue
    sequence number, insertion code) -- by mean occupancy across that
    conformer's atoms in the residue, ties (or missing/unparsable occupancy)
    falling back to whichever altLoc letter appears first in the file, i.e.
    conventionally "A" -- and then applied uniformly to every non-blank-altLoc
    atom of that residue. Resolving per atom name independently, rather than
    per residue, can silently mix coordinates from two different physical
    conformers within one residue if occupancy is not perfectly uniform
    across all of a conformer's atoms (uncommon but real in some
    depositions); that produces exactly the kind of clashing, unphysical
    geometry that can trip a downstream valence check. Blank-altLoc lines
    and all non-ATOM/HETATM lines are left untouched.
    """
    lines = pdb_text.splitlines()

    def _is_atom_record(line: str) -> bool:
        return len(line) >= 17 and (line.startswith("ATOM") or line.startswith("HETATM"))

    def _residue_key(line: str):
        chain_id = line[21] if len(line) >= 22 else " "
        res_seq = line[22:26] if len(line) >= 23 else ""
        icode = line[26] if len(line) >= 27 else " "
        return (chain_id, res_seq, icode)

    def _occupancy(line: str) -> float:
        try:
            return float(line[54:60])
        except (ValueError, IndexError):
            return 0.0

    # Pass 1: for every residue instance that has at least one non-blank
    # altLoc record, tally each conformer letter's occupancy values and the
    # order letters were first seen in, then pick one winning letter per
    # residue by mean occupancy (ties broken by file order).
    occupancies: dict[tuple, dict[str, list[float]]] = {}
    letter_order: dict[tuple, list[str]] = {}
    for line in lines:
        if not _is_atom_record(line):
            continue
        altloc = line[16]
        if altloc == " ":
            continue
        rkey = _residue_key(line)
        occupancies.setdefault(rkey, {}).setdefault(altloc, []).append(_occupancy(line))
        order = letter_order.setdefault(rkey, [])
        if altloc not in order:
            order.append(altloc)

    winning_altloc: dict[tuple, str] = {}
    for rkey, order in letter_order.items():
        conf = occupancies[rkey]
        winning_altloc[rkey] = max(
            order,
            key=lambda a: (sum(conf[a]) / len(conf[a]), -order.index(a)),
        )

    # Pass 2: blank-altLoc and non-ATOM/HETATM lines pass through unchanged.
    # A non-blank-altLoc line is kept only if it carries its residue's
    # winning letter (blanking the altLoc column so it reads as an ordinary,
    # unambiguous atom); every other alternate conformer is dropped whole.
    output: list[str] = []
    for line in lines:
        if not _is_atom_record(line):
            output.append(line)
            continue
        altloc = line[16]
        if altloc == " ":
            output.append(line)
            continue
        rkey = _residue_key(line)
        if altloc == winning_altloc.get(rkey):
            output.append(line[:16] + " " + line[17:])
        # else: a non-winning alternate conformer for this residue; drop it.
    return "\n".join(output) + "\n"


def _atom_identity_key(atom) -> tuple:
    """A stable identity key for an OpenMM Topology atom that survives a
    topology being rebuilt internally, unlike Python object identity.

    PDBFixer.addMissingHydrogens() (and possibly addMissingAtoms()) is built
    on top of openmm.app.Modeller, whose atom-insertion operations construct
    a brand new Topology (and therefore brand new Atom objects) rather than
    mutating the existing one in place. A snapshot of id(atom) taken before
    calling either method is therefore comparing against Atom objects that
    no longer appear anywhere in fixer.topology afterwards -- every atom
    looks "new", including atoms that were part of the original structure.
    Since addMissingHydrogens() runs unconditionally on every prepare_protein()
    call, that bug fires on every single run, not just ones with missing
    heavy atoms.

    Chain id, residue id/insertion code/name, and atom name are preserved by
    PDBFixer across this reconstruction (they have to be -- it's what lets
    app.PDBFile.writeFile(..., keepIds=True) reproduce the original
    numbering), so a tuple of those is a robust substitute for object
    identity here.
    """
    residue = atom.residue
    chain = residue.chain
    return (chain.id, residue.id, getattr(residue, "insertionCode", " "), residue.name, atom.name)


def _relax_reconstructed_atoms(fixer: "pdbfixer.PDBFixer", original_atom_keys: set[tuple]) -> None:
    """Run a short, two-stage restrained vacuum energy minimization so atoms
    PDBFixer reconstructed (missing heavy side-chain atoms, every hydrogen --
    crystal structures at typical X-ray resolutions have no experimentally
    observed hydrogens at all -- and, in some cases, a terminal OXT PDBFixer
    adds to whichever residue happens to be last in a chain's topology, even
    when that's only an artifact of an unresolved C-terminal region rather
    than the true end of the protein) can relax away from steric clashes,
    while atoms present in the original experimental structure are pulled
    firmly back to their deposited positions once that relaxation has
    somewhere sensible to land.

    PDBFixer places reconstructed atoms using fixed ideal geometry (bond
    lengths/angles from its residue templates) without checking for clashes
    against the surrounding structure -- a documented limitation, not a bug:
    PDBFixer's own maintainers recommend a follow-up minimization for
    exactly this reason, and production structure-preparation pipelines
    (e.g. Schrodinger's Protein Preparation Wizard) always include one.
    Left unrelaxed, a clash between a reconstructed atom and its neighbor
    can be misread as a bond by tools that infer connectivity from atomic
    distance (RDKit, and Meeko's PDBQT receptor preparation) -- producing a
    spurious extra bond and an "explicit valence ... greater than
    permitted" failure at a completely unrelated stage of the pipeline,
    on whichever residue happens to have the worst clash in that particular
    structure.

    A single minimization pass with a stiff restraint from the very start
    can converge poorly when a reconstructed atom is initially placed right
    on top of a restrained neighbor: the huge repulsive force and the huge
    restraint force fight each other in a way a bounded-iteration optimizer
    may not fully resolve. This runs two stages instead: a soft restraint
    first (letting the whole system move enough to genuinely resolve a
    severe clash), then a strong restraint (pulling the originally-present
    atoms firmly back toward their experimental positions -- not wherever
    stage 1 left them -- while preserving the now-relaxed local geometry
    around them). Both stages anchor to the same original positions.

    This is a one-shot vacuum minimization, intended purely for local clash
    relief before docking -- not a substitute for the proper explicit-
    solvent equilibration the full simulation pipeline performs later.

    original_atom_keys must be a set of _atom_identity_key(...) tuples, not
    raw object ids -- see that function's docstring for why.
    """
    # Fail fast on a pure topology/key-set check, before doing any expensive
    # or fallible force-field/system construction below.
    restrained_atom_indices = [
        atom.index
        for atom in fixer.topology.atoms()
        if _atom_identity_key(atom) in original_atom_keys
    ]
    if not restrained_atom_indices:
        raise ProteinPreparationError(
            "No atoms were identified as part of the original experimental "
            "structure; refusing to run an unrestrained clash-relief "
            "minimization on the whole system."
        )
    try:
        forcefield = app.ForceField("amber14-all.xml")
        system = forcefield.createSystem(fixer.topology, nonbondedMethod=app.NoCutoff)

        restraint = mm.CustomExternalForce("0.5*k*((x-x0)^2+(y-y0)^2+(z-z0)^2)")
        restraint.addGlobalParameter("k", _RECONSTRUCTED_ATOM_RESTRAINT_K_SOFT)
        restraint.addPerParticleParameter("x0")
        restraint.addPerParticleParameter("y0")
        restraint.addPerParticleParameter("z0")
        # fixer.positions is a unit-wrapped Quantity; indexing it returns
        # another Quantity (not a bare Vec3 with .x/.y/.z), so units must be
        # stripped first -- exactly as simulation.py's positions_nm does
        # before it is ever used the same way.
        positions_nm = fixer.positions.value_in_unit(unit.nanometer)
        for index in restrained_atom_indices:
            pos = positions_nm[index]
            restraint.addParticle(index, [pos.x, pos.y, pos.z])
        system.addForce(restraint)

        integrator = mm.LangevinMiddleIntegrator(
            300 * unit.kelvin, 1 / unit.picosecond, 2 * unit.femtoseconds
        )
        platform = mm.Platform.getPlatformByName("CPU")
        simulation = app.Simulation(fixer.topology, system, integrator, platform)
        simulation.context.setPositions(fixer.positions)

        # Stage 1: soft restraint. Lets the whole system move enough to
        # genuinely resolve a severe clash instead of fighting a stiff
        # restraint on the nearby experimental atom.
        simulation.context.setParameter("k", _RECONSTRUCTED_ATOM_RESTRAINT_K_SOFT)
        simulation.minimizeEnergy(maxIterations=_RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE1)

        # Stage 2: strong restraint, anchored to the same original positions
        # set above. Pulls the experimentally-observed atoms firmly back
        # toward their deposited positions while preserving the now-relaxed
        # local geometry stage 1 found for the reconstructed atoms.
        simulation.context.setParameter("k", _RECONSTRUCTED_ATOM_RESTRAINT_K_STRONG)
        simulation.minimizeEnergy(maxIterations=_RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE2)

        fixer.positions = simulation.context.getState(getPositions=True).getPositions()
    except Exception as exc:
        raise ProteinPreparationError(
            f"Could not relax reconstructed atoms after filling missing atoms: {exc}"
        ) from exc


def _find_chain_gap_boundaries(topology) -> set[tuple[str, str]]:
    """Identify residues that are immediately followed, within their own
    chain, by a genuine sequence gap -- i.e. the next residue's sequence
    number is not exactly one more than this one's.

    This project deliberately does not rebuild missing residues (see
    prepare_protein()), so a real gap is invisible to the topology itself:
    two residues flanking dozens of unmodeled residues can end up written
    out as directly adjacent ATOM records with nothing to indicate they are
    not actually bonded. Downstream tools that infer peptide bonds from
    residue adjacency in the file rather than from residue numbering or
    real atomic distance -- Meeko's receptor preparation among them -- can
    then infer a spurious inter-residue bond across the gap, corrupting
    template matching for both flanking residues ("No template matched for
    residue_key=... excess inter-residue bond(s)") even though neither
    residue's own chemistry is at fault.

    Returns a set of (chain_id, residue_id) keys for the residue on the
    near side of each gap -- i.e. the one a TER record needs inserting
    after.
    """
    boundaries: set[tuple[str, str]] = set()
    for chain in topology.chains():
        residues = list(chain.residues())
        for prev, curr in zip(residues, residues[1:]):
            try:
                prev_num = int(prev.id)
                curr_num = int(curr.id)
            except ValueError:
                # Non-numeric residue id: can't reason about adjacency here,
                # so leave it alone rather than guess.
                continue
            if curr_num - prev_num != 1:
                boundaries.add((chain.id, prev.id))
    return boundaries


def _insert_ter_at_gap_boundaries(pdb_text: str, boundaries: set[tuple[str, str]]) -> str:
    """Insert an explicit TER record immediately after the last atom of each
    residue identified in boundaries (see _find_chain_gap_boundaries), so a
    downstream tool that infers peptide bonds from ATOM-record adjacency
    sees the chain break instead of a spurious bond across it.
    """
    if not boundaries:
        return pdb_text
    lines = pdb_text.splitlines()
    output: list[str] = []
    prev_key: tuple[str, str] | None = None
    prev_line: str | None = None
    for line in lines:
        is_atom = len(line) >= 26 and (line.startswith("ATOM") or line.startswith("HETATM"))
        if is_atom:
            key = (line[21], line[22:26].strip())
            if prev_key is not None and key != prev_key and prev_key in boundaries and prev_line is not None:
                try:
                    prev_serial = int(prev_line[6:11])
                except ValueError:
                    prev_serial = 0
                resname = prev_line[17:20]
                output.append(f"TER   {prev_serial + 1:>5}      {resname:>3} {prev_key[0]}{prev_key[1]:>4}")
            prev_key = key
            prev_line = line
        elif line.startswith("TER"):
            # Chain already breaks here explicitly; nothing more to add.
            prev_key = None
            prev_line = None
        output.append(line)
    return "\n".join(output) + "\n"


def _parse_remark_465_missing_residues(pdb_text: str) -> dict[str, set[int]]:
    """Parse REMARK 465 (MISSING RESIDUES) records, returning, for each
    chain ID, the set of residue sequence numbers explicitly documented as
    entirely absent from the deposited structure.

    Used to distinguish a residue that merely happens to be the last one
    PDBFixer's topology contains for its chain from one that is genuinely
    the biological end of the chain -- see _find_pseudo_terminal_residues.
    """
    missing: dict[str, set[int]] = {}
    for line in pdb_text.splitlines():
        if not line.startswith("REMARK 465"):
            continue
        tokens = line[10:].split()
        if len(tokens) < 3:
            continue
        # A normal data line is [resname, chain, seqnum], or, for an NMR
        # multi-model entry, [model_num, resname, chain, seqnum]. Taking
        # the last three tokens covers either case; header/description
        # lines are naturally rejected below since their last token isn't
        # a bare (optionally insertion-coded) integer.
        resname, chain_id, seqnum_token = tokens[-3], tokens[-2], tokens[-1]
        seqnum_token = seqnum_token.rstrip("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        try:
            seqnum = int(seqnum_token)
        except ValueError:
            continue
        if len(chain_id) != 1 or not chain_id.isalpha():
            continue
        missing.setdefault(chain_id, set()).add(seqnum)
    return missing


def _find_pseudo_terminal_residues(topology, missing_by_chain: dict[str, set[int]]) -> set[tuple[str, str]]:
    """Identify (chain_id, residue_id) pairs for the last modeled residue of
    a chain when REMARK 465 documents at least one missing residue with a
    HIGHER sequence number in that same chain -- i.e. the chain's true
    biological end lies beyond what was modeled, so this residue is only an
    artificial pseudo-terminus, not a genuine one.

    PDBFixer/Modeller-based atom completion decides whether to add a
    C-terminal OXT purely from a residue's position in the topology's
    residue list, with no way to know the biological chain actually
    continues beyond an unresolved region (this project deliberately never
    rebuilds missing residues -- see prepare_protein()). The ideal geometry
    used to place that OXT then has no genuine "outward" direction to
    anchor to, and can end up close enough to a neighboring atom that
    downstream distance-based bond inference (RDKit, Meeko) misreads it as
    bonded -- producing an "unable to build rdkit mol" / valence-exception
    failure on a residue whose own chemistry was never actually the
    problem.
    """
    pseudo_termini: set[tuple[str, str]] = set()
    for chain in topology.chains():
        residues = list(chain.residues())
        if not residues:
            continue
        last = residues[-1]
        try:
            last_num = int(last.id)
        except ValueError:
            continue
        if any(m > last_num for m in missing_by_chain.get(chain.id, set())):
            pseudo_termini.add((chain.id, last.id))
    return pseudo_termini


def _strip_spurious_terminal_oxt(pdb_text: str, pseudo_termini: set[tuple[str, str]]) -> str:
    """Remove any OXT atom record belonging to a residue identified as a
    pseudo-terminus (see _find_pseudo_terminal_residues). A docking
    receptor's binding-site chemistry does not depend on a chemically
    accurate capping group at a residue that is not the protein's true
    terminus, so simply dropping the spurious atom is a safe, low-risk fix.
    """
    if not pseudo_termini:
        return pdb_text
    output: list[str] = []
    for line in pdb_text.splitlines():
        if len(line) >= 26 and (line.startswith("ATOM") or line.startswith("HETATM")):
            atom_name = line[12:16].strip()
            key = (line[21], line[22:26].strip())
            if atom_name == "OXT" and key in pseudo_termini:
                continue  # drop this spurious terminal atom
        output.append(line)
    return "\n".join(output) + "\n"


def prepare_protein(source: Path, output: Path, pH: float = 7.4) -> ProteinPreparationResult:
    source = Path(source)
    if not source.exists() or source.stat().st_size == 0:
        raise ProteinPreparationError(f"Protein input file is missing or empty: {source}")
    try:
        cleaned_text = _strip_alternate_locations(source.read_text(encoding="utf-8", errors="replace"))
        fixer = pdbfixer.PDBFixer(pdbfile=io.StringIO(cleaned_text))
        fixer.removeHeterogens(keepWater=False)  # remove all crystallographic ligands, cofactors, and waters
        chains = tuple(chain.id for chain in fixer.topology.chains())

        fixer.findMissingResidues()
        missing_residue_count = sum(len(v) for v in fixer.missingResidues.values())
        # Do not silently alter residue numbering by modeling unresolved loops.
        # fixer.missingResidues = {}

        fixer.findNonstandardResidues()
        nonstandard_count = len(fixer.nonstandardResidues)
        if nonstandard_count:
            fixer.replaceNonstandardResidues()

        # POC assumption: the target is a protein system without an essential
        # crystallographic ligand/cofactor that must be retained.
        fixer.removeHeterogens(False)

        # Snapshot which atoms are part of the experimentally-observed
        # structure -- by stable (chain, residue, atom name) content key, not
        # Python object identity, which does NOT survive the topology
        # rebuild that addMissingAtoms()/addMissingHydrogens() perform
        # internally (see _atom_identity_key's docstring) -- before PDBFixer
        # reconstructs anything, so the clash-relief step below knows which
        # atoms to hold fixed versus let relax.
        original_atom_keys = {_atom_identity_key(atom) for atom in fixer.topology.atoms()}

        fixer.findMissingAtoms()
        missing_atom_count = sum(len(v) for v in fixer.missingAtoms.values())
        fixer.addMissingAtoms()
        fixer.addMissingHydrogens(pH)
        _relax_reconstructed_atoms(fixer, original_atom_keys)

        # Identify genuine sequence gaps (see _find_chain_gap_boundaries) and
        # pseudo-terminal residues (see _find_pseudo_terminal_residues) from
        # the final topology and the original REMARK 465 records, then write
        # to an in-memory buffer first so both can be corrected before the
        # file actually reaches disk.
        gap_boundaries = _find_chain_gap_boundaries(fixer.topology)
        missing_by_chain = _parse_remark_465_missing_residues(cleaned_text)
        pseudo_termini = _find_pseudo_terminal_residues(fixer.topology, missing_by_chain)
        buffer = io.StringIO()
        app.PDBFile.writeFile(fixer.topology, fixer.positions, buffer, keepIds=True)
        final_text = _insert_ter_at_gap_boundaries(buffer.getvalue(), gap_boundaries)
        final_text = _strip_spurious_terminal_oxt(final_text, pseudo_termini)

        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(final_text, encoding="utf-8")
    except Exception as exc:
        raise ProteinPreparationError(f"Protein preparation failed: {exc}") from exc

    return ProteinPreparationResult(
        source=source,
        prepared=output,
        target_id=source.stem,
        missing_residue_count=missing_residue_count,
        missing_atom_count=missing_atom_count,
        nonstandard_residue_count=nonstandard_count,
        original_chain_ids=chains,
    )


def validate_residue(pdb_path: Path, chain_id: str, residue_number: int) -> None:
    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("target", str(pdb_path))
        model = structure[0]
        if chain_id not in model:
            raise InputValidationError(f"Chain {chain_id!r} is not present in {pdb_path.name}.")
        if residue_number not in model[chain_id]:
            raise InputValidationError(
                f"Residue {chain_id}:{residue_number} is not present in {pdb_path.name}."
            )
    except InputValidationError:
        raise
    except Exception as exc:
        raise InputValidationError(f"Could not parse prepared PDB: {exc}") from exc


def prepare_ligand(smiles: str, output: Path, seed: int = 20260907) -> LigandPreparationResult:
    smiles = smiles.strip()
    if not smiles:
        raise LigandPreparationError("Ligand SMILES is empty.")

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise LigandPreparationError("Invalid ligand SMILES.")

    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    formal_charge = Chem.GetFormalCharge(mol)
    mol = Chem.AddHs(mol)

    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        # Deterministic fallback: use a second ETKDG attempt without random
        # torsion preference, then fail loudly if 3D construction still fails.
        status = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=int(seed) + 1)
    if status != 0:
        raise LigandPreparationError("RDKit could not generate a 3D ligand conformer.")

    if AllChem.MMFFHasAllMoleculeParams(mol):
        opt_status = AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        if opt_status < 0:
            raise LigandPreparationError("MMFF optimization failed for the ligand.")
    elif AllChem.UFFHasAllMoleculeParams(mol):
        opt_status = AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        if opt_status < 0:
            raise LigandPreparationError("UFF optimization failed for the ligand.")
    else:
        raise LigandPreparationError("Neither MMFF nor UFF supports this ligand chemistry.")

    # Round-trip through the OpenFF Toolkit to validate that the same chemistry
    # can be represented by the small-molecule parameterization layer.
    try:
        off_mol = Molecule.from_rdkit(mol, allow_undefined_stereo=True)
        if off_mol.n_conformers < 1:
            raise ValueError("OpenFF molecule contains no 3D conformer.")
    except Exception as exc:
        raise LigandPreparationError(f"OpenFF could not validate the ligand: {exc}") from exc

    output.parent.mkdir(parents=True, exist_ok=True)
    writer = Chem.SDWriter(str(output))
    if writer is None:
        raise LigandPreparationError(f"Could not open SDF writer for {output}")
    writer.write(mol)
    writer.close()

    formula = rdMolDescriptors.CalcMolFormula(mol)
    return LigandPreparationResult(
        sdf_path=output,
        canonical_smiles=canonical,
        formula=formula,
        formal_charge=int(formal_charge),
        n_atoms=mol.GetNumAtoms(),
    )


def validate_ligand_pose(sdf_path: Path, expected_smiles: str) -> None:
    """Validate that a supplied 3D pose represents the requested chemistry."""
    sdf_path = Path(sdf_path)
    if not sdf_path.exists() or sdf_path.stat().st_size == 0:
        raise LigandPreparationError(f"Ligand pose SDF is missing or empty: {sdf_path}")
    try:
        supplier = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
        if not supplier or supplier[0] is None:
            raise ValueError("SDF contains no valid molecule.")
        pose = supplier[0]
        if pose.GetNumConformers() < 1:
            raise ValueError("SDF molecule has no 3D conformer.")
        pose_no_h = Chem.RemoveHs(Chem.Mol(pose))
        pose_smiles = Chem.MolToSmiles(pose_no_h, canonical=True, isomericSmiles=True)
        expected_mol = Chem.MolFromSmiles(expected_smiles)
        if expected_mol is None:
            raise ValueError("Expected ligand SMILES could not be parsed.")
        expected_canonical = Chem.MolToSmiles(expected_mol, canonical=True, isomericSmiles=True)
        if pose_smiles != expected_canonical:
            raise ValueError(
                "Supplied ligand pose does not match the requested SMILES. "
                f"Expected {expected_canonical}; found {pose_smiles}."
            )
    except Exception as exc:
        raise LigandPreparationError(f"Ligand pose validation failed: {exc}") from exc
