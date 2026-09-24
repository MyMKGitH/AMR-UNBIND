from pathlib import Path

import numpy as np
import pandas as pd

from amr_unbind.analysis import load_analysis, qc_metrics
from amr_unbind.chimerax import ChimeraXBridge
from amr_unbind.docking import _meeko_command
from amr_unbind.geometry import _heavy_atom_centroid
from amr_unbind.ml import trajectory_features


class Atom:
    def __init__(self, coord, element=None, name=""):
        self._coord = np.asarray(coord, dtype=float)
        self.element = element
        self.name = name

    def get_coord(self):
        return self._coord


class Residue:
    def __init__(self, atoms):
        self._atoms = atoms

    def get_atoms(self):
        return iter(self._atoms)


def _write_analysis(path: Path, include_target: bool = True, constant_disp: bool = False):
    disp = [0.1, 0.1, 0.1] if constant_disp else [0.0, 0.1, 0.2]
    data = {
        "Time (ps)": [1.0, 2.0, 3.0],
        "Reaction Coordinate (nm)": disp,
        "Displacement (nm)": disp,
        "Steering Force (kJ/mol/nm)": [10.0, 20.0, 30.0],
        "Cumulative Work (kJ/mol)": [0.0, 1.5, 4.0],
    }
    if include_target:
        data["Target Coordinate (nm)"] = [0.01, 0.02, 0.03]
    pd.DataFrame(data).to_csv(path, index=False)


def test_analysis_requires_target_coordinate(tmp_path):
    path = tmp_path / "missing.csv"
    _write_analysis(path, include_target=False)
    try:
        load_analysis(path)
    except ValueError as exc:
        assert "Target Coordinate (nm)" in str(exc)
    else:
        raise AssertionError("Missing target column was not rejected")


def test_qc_metrics_accepts_valid_analysis(tmp_path):
    path = tmp_path / "valid.csv"
    _write_analysis(path)
    metrics = qc_metrics(load_analysis(path))
    assert metrics["target_coordinate_monotonic"] is True


def test_ml_slope_is_safe_for_constant_displacement(tmp_path):
    path = tmp_path / "constant.csv"
    _write_analysis(path, constant_disp=True)
    features = trajectory_features(path)
    assert features["force_slope_kj_mol_nm2"] == 0.0


def test_geometry_filters_hydrogen_when_element_is_blank():
    residue = Residue([
        Atom([0.0, 0.0, 0.0], element=None, name="H1"),
        Atom([2.0, 0.0, 0.0], element=" C ", name="CA"),
    ])
    centroid = _heavy_atom_centroid(residue)
    assert np.allclose(centroid, [2.0, 0.0, 0.0])


def test_chimerax_movie_command_contains_playback_and_wait(tmp_path):
    bridge = ChimeraXBridge()
    structure = tmp_path / "a.pdb"
    traj = tmp_path / "a.dcd"
    structure.write_text("x")
    traj.write_bytes(b"x")
    captured = {}

    def fake_execute(command, timeout=20.0):
        captured["command"] = command
        (tmp_path / "movie.mp4").write_bytes(b"mp4")
        return "ok"

    bridge.execute = fake_execute
    bridge.render_movie(structure, traj, tmp_path / "movie.mp4", n_frames=12)
    command = captured["command"]
    assert "movie record supersample 2" in command
    assert "coordset #2 1,-1" in command
    assert "wait 12" in command
    assert "movie encode" in command


def test_meeko_commands_use_active_python():
    assert _meeko_command("meeko.cli.mk_prepare_ligand", "-h")[1:3] == ["-m", "meeko.cli.mk_prepare_ligand"]


def test_simulation_defines_total_steps_before_reporter():
    source = Path("amr_unbind/simulation.py").read_text()
    assert source.index("total_steps = config.pull_steps") < source.index("totalSteps=total_steps")


def test_simulation_transfers_npt_box_vectors_before_smd_positions():
    source = Path("amr_unbind/simulation.py").read_text()
    assert "smd_sim.context.setPeriodicBoxVectors(*npt_state.getPeriodicBoxVectors())" in source



def test_vina_candidate_order_and_environment(monkeypatch, tmp_path):
    from amr_unbind import docking

    fake = tmp_path / "vina.exe"
    fake.write_text("stub")
    monkeypatch.setenv("AMR_UNBIND_VINA", str(fake))
    candidates = docking._executable_candidates(tmp_path)
    assert candidates[0] == fake


def test_docking_input_validation_rejects_invalid_box():
    from amr_unbind.docking import _validate_docking_inputs

    try:
        _validate_docking_inputs((0.0, 0.0, 0.0), (0.0, 20.0, 20.0), 8)
    except Exception as exc:
        assert "box size" in str(exc).lower()
    else:
        raise AssertionError("Invalid docking box was accepted")


def test_vina_score_parser(tmp_path):
    from amr_unbind.docking import _parse_vina_score

    path = tmp_path / "pose.pdbqt"
    path.write_text("REMARK VINA RESULT:      -7.321      0.000      0.000\nATOM      1  C\n")
    assert _parse_vina_score(path) == -7.321


def test_vina_score_parser_missing_returns_none(tmp_path):
    from amr_unbind.docking import _parse_vina_score

    path = tmp_path / "pose.pdbqt"
    path.write_text("ATOM      1  C\n")
    assert _parse_vina_score(path) is None


def test_systemgenerator_uses_periodic_kwargs_not_generic_kwargs():
    source = Path("amr_unbind/simulation.py").read_text()
    generic_start = source.index("forcefield_kwargs={")
    periodic_start = source.index("periodic_forcefield_kwargs={")
    generic = source[generic_start:periodic_start]
    assert '"nonbondedMethod": app.PME' not in generic
    assert '"nonbondedMethod": app.PME' in source[periodic_start:]



def test_python_vina_failure_falls_back_to_executable(monkeypatch, tmp_path):
    import sys
    import types
    from amr_unbind import docking

    receptor = tmp_path / "rec.pdb"
    ligand = tmp_path / "lig.sdf"
    receptor.write_text("ATOM\n")
    ligand.write_text("ligand\n")
    out = tmp_path / "dock"
    out.mkdir()
    rec_pdbqt = out / "receptor.pdbqt"
    lig_pdbqt = out / "ligand.pdbqt"
    rec_pdbqt.write_text("REMARK receptor\n")
    lig_pdbqt.write_text("REMARK ligand\n")

    monkeypatch.setitem(sys.modules, "vina", types.ModuleType("vina"))
    monkeypatch.setattr(
        docking, "_prepare_pdbqt_inputs", lambda *args, **kwargs: (rec_pdbqt, lig_pdbqt, None)
    )
    monkeypatch.setattr(docking, "find_vina_executable", lambda *args, **kwargs: tmp_path / "vina.exe")
    monkeypatch.setattr(
        docking,
        "_dock_with_python",
        lambda *args, **kwargs: (_ for _ in ()).throw(docking.DockingError("python backend failed")),
    )
    monkeypatch.setattr(docking, "_dock_with_executable", lambda *args, **kwargs: -8.25)
    pose = out / "docked_pose.pdbqt"
    pose.write_text("REMARK VINA RESULT: -8.250 0 0\nATOM\n")
    def fake_executable(*args, **kwargs):
        pose.write_text("REMARK VINA RESULT: -8.250 0 0\nATOM\n")
        return -8.25
    monkeypatch.setattr(docking, "_dock_with_executable", fake_executable)
    monkeypatch.setattr(docking, "_export_pose_to_sdf", lambda *args, **kwargs: Path(args[1]).write_text("sdf"))

    result = docking.automatic_vina_docking(
        receptor,
        ligand,
        out,
        center_angstrom=(1.0, 2.0, 3.0),
        project_root=tmp_path,
    )
    assert result.method == "AutoDock Vina (executable)"
    assert result.score_kcal_mol == -8.25
    assert result.executable == str(tmp_path / "vina.exe")



def test_analysis_rejects_empty_csv(tmp_path):
    path = tmp_path / "empty.csv"
    path.write_text("Time (ps),Reaction Coordinate (nm),Target Coordinate (nm),Displacement (nm),Steering Force (kJ/mol/nm),Cumulative Work (kJ/mol)\n")
    try:
        load_analysis(path)
    except ValueError as exc:
        assert "no trajectory samples" in str(exc).lower()
    else:
        raise AssertionError("Empty analysis CSV was accepted")


# --- v1.0.3 regression tests -------------------------------------------------


def test_config_rejects_pull_distance_beyond_solvation_padding():
    from amr_unbind.config import SimulationConfig

    try:
        SimulationConfig(pull_distance_nm=1.5, solvation_padding_nm=1.0).validate()
    except ValueError as exc:
        assert "solvation_padding_nm" in str(exc)
    else:
        raise AssertionError("Pull distance exceeding solvation padding was accepted")


def test_config_default_and_example_configs_still_validate():
    import json

    from amr_unbind.config import SimulationConfig

    SimulationConfig().validate()
    example = Path(__file__).parents[1] / "examples" / "config_pilot.json"
    SimulationConfig.from_dict(json.loads(example.read_text(encoding="utf-8")))


def test_run_label_sanitization_neutralizes_unsafe_characters(tmp_path):
    from amr_unbind.io import _sanitize_run_label, create_run_dir

    assert _sanitize_run_label("3F2R") == "3f2r"
    unsafe = _sanitize_run_label("../../etc/passwd")
    assert "/" not in unsafe and ".." not in unsafe.strip(".")
    assert _sanitize_run_label("   ") == "target"

    run_dir = create_run_dir(tmp_path, "../../weird name!!")
    assert run_dir.exists()
    assert run_dir.is_relative_to(tmp_path)


def test_cli_platform_default_does_not_clobber_config_file(monkeypatch, tmp_path):
    import json

    from amr_unbind import cli
    from amr_unbind.config import SimulationConfig
    from amr_unbind.pipeline import AMRUnbindPipeline

    captured = {}

    def fake_run(self, **kwargs):
        captured["config"] = kwargs["config"]
        return tmp_path

    monkeypatch.setattr(AMRUnbindPipeline, "run", fake_run)

    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text(json.dumps(SimulationConfig(preferred_platform="cuda").to_dict()))

    rc = cli.main([
        "--pdb-id", "3f2r", "--smiles", "CCO",
        "--core-chain", "A", "--core-residue", "1",
        "--exit-chain", "A", "--exit-residue", "2",
        "--config", str(cfg_path),
    ])
    assert rc == 0
    assert captured["config"].preferred_platform == "cuda"

    rc = cli.main([
        "--pdb-id", "3f2r", "--smiles", "CCO",
        "--core-chain", "A", "--core-residue", "1",
        "--exit-chain", "A", "--exit-residue", "2",
        "--config", str(cfg_path), "--platform", "cpu",
    ])
    assert rc == 0
    assert captured["config"].preferred_platform == "cpu"


def test_cli_reports_invalid_config_cleanly_instead_of_crashing(capsys):
    from amr_unbind import cli

    rc = cli.main([
        "--pdb-id", "3f2r", "--smiles", "CCO",
        "--core-chain", "A", "--core-residue", "1",
        "--exit-chain", "A", "--exit-residue", "2",
        "--seed", "-5",
    ])
    assert rc == 1
    assert "seed" in capsys.readouterr().err.lower()


def test_pipeline_converts_invalid_config_valueerror_to_input_validation_error(tmp_path):
    from amr_unbind.config import SimulationConfig
    from amr_unbind.errors import InputValidationError
    from amr_unbind.pipeline import AMRUnbindPipeline

    bad_config = SimulationConfig(pull_distance_nm=5.0, solvation_padding_nm=1.0)
    pipeline = AMRUnbindPipeline(tmp_path)
    try:
        pipeline.run(
            pdb_id="3f2r",
            smiles="CCO",
            core_chain="A",
            core_residue=1,
            exit_chain="A",
            exit_residue=2,
            config=bad_config,
        )
    except InputValidationError:
        pass
    else:
        raise AssertionError("Invalid config should raise InputValidationError, not run silently")


# --- v1.0.4 regression tests -------------------------------------------------


def test_strip_alternate_locations_resolves_per_residue_not_per_atom():
    """Regression test for a real field failure (v1.0.5): a second altLoc
    crash on a different residue (GLU A:81) after the v1.0.4 fix suggested a
    gap. This reproduces the actual gap: if occupancy is not perfectly
    uniform across all of one conformer's atoms in a residue, resolving the
    winning altLoc independently per atom name can mix coordinates from two
    different physical conformers within one residue (e.g. keep CB from
    conformer A but CG from conformer B) -- a "Frankenstein" residue with
    unphysical geometry that can trip a downstream valence check exactly
    like the one reported. The fix resolves one winning conformer per
    residue instance, by mean occupancy across that conformer's atoms.
    """
    from amr_unbind.prepare import _strip_alternate_locations

    # CB: A=0.60/B=0.40 (A wins if resolved per-atom). CG: A=0.30/B=0.70 (B
    # wins if resolved per-atom). A naive per-atom resolver keeps CB-A +
    # CG-B, mixing two different conformers. Residue-level mean occupancy
    # (A: 0.45, B: 0.55) must pick B consistently for the whole residue.
    pdb = (
        "ATOM      1  N   GLU A  81      10.000  10.000  10.000  1.00 20.00           N\n"
        "ATOM      2  CA  GLU A  81      11.000  10.000  10.000  1.00 20.00           C\n"
        "ATOM      3  CB AGLU A  81      12.000  10.000  10.000  0.60 20.00           C\n"
        "ATOM      4  CB BGLU A  81      12.100  10.050  10.050  0.40 20.00           C\n"
        "ATOM      5  CG AGLU A  81      13.000  10.000  10.000  0.30 20.00           C\n"
        "ATOM      6  CG BGLU A  81      13.100  10.050  10.050  0.70 20.00           C\n"
    )
    out_lines = _strip_alternate_locations(pdb).splitlines()
    kept_cb = [l for l in out_lines if l[12:16].strip() == "CB"]
    kept_cg = [l for l in out_lines if l[12:16].strip() == "CG"]
    assert len(kept_cb) == 1 and len(kept_cg) == 1
    # Both survivors must come from the SAME conformer (B: serials 4 and 6),
    # never a mix of A and B.
    assert kept_cb[0][6:11].strip() == "4"
    assert kept_cg[0][6:11].strip() == "6"


def test_strip_alternate_locations_insertion_code_distinguishes_residues():
    from amr_unbind.prepare import _strip_alternate_locations

    def make_line(serial, name, altloc, resname, chain, resseq, icode, occ):
        return (
            f"ATOM  {serial:>5} {name:<4}{altloc}{resname:<3} {chain}{resseq:>4}{icode}   "
            f"{10.0:8.3f}{10.0:8.3f}{10.0:8.3f}{occ:6.2f}{20.0:6.2f}          C"
        )

    # Two DIFFERENT residues sharing residue number 81, distinguished only by
    # insertion code (blank vs "X"). Each has its own independent altLoc
    # pair; they must never be merged into one resolution group.
    lines = [
        make_line(1, " CB", "A", "GLU", "A", 81, " ", 0.60),
        make_line(2, " CB", "B", "GLU", "A", 81, " ", 0.40),
        make_line(3, " CB", "A", "GLU", "A", 81, "X", 0.10),
        make_line(4, " CB", "B", "GLU", "A", 81, "X", 0.90),
    ]
    out = _strip_alternate_locations("\n".join(lines) + "\n").splitlines()
    assert len(out) == 2
    kept_serials = sorted(l[6:11].strip() for l in out)
    assert kept_serials == ["1", "4"]  # residue 81 -> A; residue 81X -> B, independently


def test_strip_alternate_locations_keeps_highest_occupancy_conformer():
    from amr_unbind.prepare import _strip_alternate_locations

    pdb = (
        "ATOM      1  N   ARG A  89      10.000  10.000  10.000  1.00 20.00           N\n"
        "ATOM      2  CA  ARG A  89      11.000  10.000  10.000  1.00 20.00           C\n"
        "ATOM      3  CB AARG A  89      12.000  10.000  10.000  0.60 20.00           C\n"
        "ATOM      4  CB BARG A  89      12.100  10.050  10.050  0.40 20.00           C\n"
        "ATOM      5  CG AARG A  89      13.000  10.000  10.000  0.60 20.00           C\n"
        "ATOM      6  CG BARG A  89      13.100  10.050  10.050  0.40 20.00           C\n"
        "ATOM      7  O   ARG A  89      11.500  11.000  10.000  1.00 20.00           O\n"
        "HETATM    8  O   HOH B 501      20.000  20.000  20.000  1.00 30.00           O\n"
        "TER\n"
        "END\n"
    )
    out_lines = _strip_alternate_locations(pdb).splitlines()

    # 10 input lines, 2 losing altLoc records (CB-B, CG-B) dropped -> 8 remain.
    assert len(out_lines) == 8
    assert "TER" in out_lines and "END" in out_lines

    kept_cb = [l for l in out_lines if l[12:16].strip() == "CB"]
    kept_cg = [l for l in out_lines if l[12:16].strip() == "CG"]
    assert len(kept_cb) == 1 and kept_cb[0][6:11].strip() == "3"  # higher-occupancy A
    assert len(kept_cg) == 1 and kept_cg[0][6:11].strip() == "5"  # higher-occupancy A
    assert kept_cb[0][16] == " " and kept_cg[0][16] == " "  # altLoc column blanked
    assert any("HOH" in l for l in out_lines)  # unrelated records untouched


def test_strip_alternate_locations_ties_keep_first_listed():
    from amr_unbind.prepare import _strip_alternate_locations

    pdb = (
        "ATOM      1  OG ASER A  10      1.000   1.000   1.000  0.50 20.00           O\n"
        "ATOM      2  OG BSER A  10      1.100   1.050   1.050  0.50 20.00           O\n"
    )
    out = _strip_alternate_locations(pdb).splitlines()
    assert len(out) == 1 and out[0][6:11].strip() == "1"


def test_strip_alternate_locations_noop_on_clean_pdb():
    from amr_unbind.prepare import _strip_alternate_locations

    clean_pdb = "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n"
    assert _strip_alternate_locations(clean_pdb) == clean_pdb


def test_prepare_protein_feeds_pdbfixer_a_stream_not_a_filename(monkeypatch, tmp_path):
    """Regression test for a real field failure: Meeko's receptor preparation
    choked on an altLoc-induced valence error even though prepare_protein()'s
    output was already the receptor path passed to docking. The missing piece
    was that prepare_protein() itself never resolved altLocs before handing
    the structure to PDBFixer. This checks the wiring: PDBFixer must be
    constructed from the altloc-stripped text (via pdbfile=...), not from the
    raw source filename directly."""
    from amr_unbind import prepare as prepare_module

    pdb_with_altloc = (
        "ATOM      1  N   ARG A  89      10.000  10.000  10.000  1.00 20.00           N\n"
        "ATOM      2  CB AARG A  89      12.000  10.000  10.000  0.60 20.00           C\n"
        "ATOM      3  CB BARG A  89      12.100  10.050  10.050  0.40 20.00           C\n"
        "END\n"
    )
    source = tmp_path / "raw.pdb"
    source.write_text(pdb_with_altloc)

    captured = {}

    class FakeTopology:
        def chains(self):
            return []

        def atoms(self):
            return []

    class FakeFixer:
        def __init__(self, filename=None, pdbfile=None):
            captured["filename"] = filename
            captured["pdbfile"] = pdbfile
            if pdbfile is not None:
                captured["pdbfile_text"] = pdbfile.read()
            self.topology = FakeTopology()
            self.positions = []
            self.missingResidues = {}
            self.missingAtoms = {}
            self.nonstandardResidues = []

        def findMissingResidues(self): pass
        def findNonstandardResidues(self): pass
        def replaceNonstandardResidues(self): pass
        def removeHeterogens(self, keepWater): pass
        def findMissingAtoms(self): pass
        def addMissingAtoms(self): pass
        def addMissingHydrogens(self, pH): pass

    monkeypatch.setattr(prepare_module.pdbfixer, "PDBFixer", FakeFixer)
    monkeypatch.setattr(prepare_module.app.PDBFile, "writeFile", staticmethod(lambda *a, **k: None))
    # This test is only about the PDBFixer(pdbfile=...) wiring; the
    # post-reconstruction relaxation step has its own dedicated tests below
    # and needs a real Quantity-like fixer.positions to run against.
    monkeypatch.setattr(prepare_module, "_relax_reconstructed_atoms", lambda fixer, ids: None)

    prepare_module.prepare_protein(source, tmp_path / "prepared.pdb")

    assert captured["filename"] is None
    assert captured["pdbfile"] is not None
    # Both CB altLoc lines went in raw; after our own stripping, PDBFixer
    # should only ever see one CB line, with its altLoc column blank.
    cb_lines = [l for l in captured["pdbfile_text"].splitlines() if l[12:16].strip() == "CB"]
    assert len(cb_lines) == 1
    assert cb_lines[0][16] == " "


# --- v1.0.6 regression tests -------------------------------------------------


def test_atom_identity_key_survives_topology_rebuild(monkeypatch):
    """Regression test for the actual bug behind a real field failure: the
    v1.0.6 fix tracked "original" atoms by Python object identity (id()),
    captured before addMissingAtoms()/addMissingHydrogens() ran. But
    PDBFixer.addMissingHydrogens() is built on openmm.app.Modeller, whose
    atom-insertion operations construct a brand new Topology (and therefore
    brand new Atom objects) rather than mutating the existing one in place.
    Since every atom object is recreated, NONE of the pre-captured ids ever
    matched again -- on every single run, not just ones with missing heavy
    atoms, since hydrogens are reconstructed unconditionally. This resulted
    in "No atoms were identified as part of the original experimental
    structure" on every automatic-docking run.

    _atom_identity_key() must return the SAME key for two different Python
    objects that represent the same (chain, residue, atom name), simulating
    exactly this kind of topology rebuild.
    """
    from amr_unbind.prepare import _atom_identity_key

    class FakeChain:
        def __init__(self, cid):
            self.id = cid

    class FakeResidue:
        def __init__(self, chain, rid, name, icode=" "):
            self.chain = chain
            self.id = rid
            self.name = name
            self.insertionCode = icode

    class FakeAtom:
        def __init__(self, residue, name):
            self.residue = residue
            self.name = name

    chain_before = FakeChain("A")
    residue_before = FakeResidue(chain_before, "81", "GLU")
    atom_before = FakeAtom(residue_before, "CB")

    # Simulate a topology rebuild: entirely new Chain/Residue/Atom objects,
    # same logical content.
    chain_after = FakeChain("A")
    residue_after = FakeResidue(chain_after, "81", "GLU")
    atom_after = FakeAtom(residue_after, "CB")

    assert atom_before is not atom_after  # different Python objects...
    assert id(atom_before) != id(atom_after)  # ...so id()-based tracking fails here
    assert _atom_identity_key(atom_before) == _atom_identity_key(atom_after)  # key-based tracking must not

    # A genuinely different atom (e.g. a newly reconstructed one) must get a
    # different key.
    newly_reconstructed = FakeAtom(residue_after, "CG")
    assert _atom_identity_key(newly_reconstructed) != _atom_identity_key(atom_before)


def test_relax_reconstructed_atoms_restrains_only_original_atoms(monkeypatch):
    """Regression test for a real field failure: automatic docking kept
    failing with an AtomValenceException on a *different* residue each time
    (ARG A:89, then GLU A:81, then SER A:121 -- all three explicitly listed
    in the real target's own REMARK 470 MISSING ATOM records). The root
    cause was never altLocs: PDBFixer reconstructs missing side-chain atoms
    (and every hydrogen -- crystal structures have none) using fixed ideal
    geometry with no clash checking against the rest of the structure, and
    an unrelaxed clash can be misread as a bond by Meeko's distance-based
    connectivity inference. This checks that the fix -- a restrained vacuum
    minimization after reconstruction -- restrains exactly the atoms that
    existed before reconstruction (identified by stable content key, not
    object identity -- see test_atom_identity_key_survives_topology_rebuild
    for why) and leaves newly reconstructed atoms free to move."""
    from amr_unbind import prepare as prepare_module

    class FakeVec3:
        def __init__(self, x, y, z):
            self.x, self.y, self.z = x, y, z

    class FakeQuantity:
        def __init__(self, value):
            self._value = value

        def __getitem__(self, i):
            return FakeQuantity(self._value[i])

        def value_in_unit(self, target_unit):
            def strip(v):
                return [strip(x) for x in v] if isinstance(v, list) else v
            return strip(self._value)

        def __getattr__(self, name):
            # A real Quantity has no .x/.y/.z; accessing it without first
            # stripping units must fail, exactly like the bug this guards.
            raise AttributeError(name)

    class FakeChain:
        def __init__(self, cid):
            self.id = cid

    class FakeResidue:
        def __init__(self, chain, rid, name):
            self.chain = chain
            self.id = rid
            self.name = name
            self.insertionCode = " "

    class FakeAtom:
        def __init__(self, index, residue, name):
            self.index = index
            self.residue = residue
            self.name = name

    class FakeTopology:
        def __init__(self, atoms):
            self._atoms = atoms

        def atoms(self):
            return list(self._atoms)

    class FakeState:
        def __init__(self, positions):
            self._positions = positions

        def getPositions(self):
            return self._positions

    class FakeContext:
        def __init__(self):
            self.param_calls = []

        def setPositions(self, positions):
            self.positions = positions

        def setParameter(self, name, value):
            self.param_calls.append((name, value))

        def getState(self, getPositions=False):
            return FakeState(self.positions)

    class FakeSimulation:
        def __init__(self, topology, system, integrator, platform):
            self.context = FakeContext()
            self.minimize_calls = []

        def minimizeEnergy(self, maxIterations=0):
            self.minimize_calls.append(maxIterations)

    class FakeSystem:
        def addForce(self, f):
            pass

    class FakeForceField:
        def __init__(self, *a, **k):
            pass

        def createSystem(self, topology, nonbondedMethod=None):
            return FakeSystem()

    class FakeCustomExternalForce:
        def __init__(self, expr):
            self.particles = []

        def addGlobalParameter(self, *a): pass
        def addPerParticleParameter(self, *a): pass

        def addParticle(self, index, params):
            self.particles.append((index, params))

    captured = {}

    class InstrumentedCEF(FakeCustomExternalForce):
        def __init__(self, expr):
            super().__init__(expr)
            captured["force"] = self

    class InstrumentedSimulation(FakeSimulation):
        def __init__(self, topology, system, integrator, platform):
            super().__init__(topology, system, integrator, platform)
            captured["simulation"] = self

    class FakeMM:
        CustomExternalForce = InstrumentedCEF

        class Platform:
            @staticmethod
            def getPlatformByName(name):
                return object()

        class LangevinMiddleIntegrator:
            def __init__(self, *a, **k): pass

    class FakeApp:
        NoCutoff = "NoCutoff"
        ForceField = FakeForceField
        Simulation = InstrumentedSimulation

    class UnitConst:
        def __rmul__(self, other): return other
        def __rtruediv__(self, other): return other

    class FakeUnit:
        nanometer = "nanometer"
        kelvin = UnitConst()
        picosecond = UnitConst()
        femtoseconds = UnitConst()

    monkeypatch.setattr(prepare_module, "mm", FakeMM)
    monkeypatch.setattr(prepare_module, "app", FakeApp)
    monkeypatch.setattr(prepare_module, "unit", FakeUnit)

    # Build the "before reconstruction" snapshot from one set of Chain/
    # Residue/Atom objects, then hand _relax_reconstructed_atoms a fixer
    # whose topology uses ENTIRELY DIFFERENT (but same-content) objects for
    # the first three atoms, plus three more genuinely-new atoms -- exactly
    # simulating a Modeller-style topology rebuild.
    chain_before = FakeChain("A")
    residue_before = FakeResidue(chain_before, "81", "GLU")
    original_atoms_before = [FakeAtom(i, residue_before, name) for i, name in enumerate(["N", "CA", "CB"])]
    original_atom_keys = {prepare_module._atom_identity_key(a) for a in original_atoms_before}

    chain_after = FakeChain("A")
    residue_after = FakeResidue(chain_after, "81", "GLU")
    rebuilt_original_atoms = [FakeAtom(i, residue_after, name) for i, name in enumerate(["N", "CA", "CB"])]
    reconstructed_atoms = [FakeAtom(i + 3, residue_after, name) for i, name in enumerate(["CG", "CD", "OE1"])]
    all_atoms = rebuilt_original_atoms + reconstructed_atoms

    class FakeFixer:
        pass

    fixer = FakeFixer()
    fixer.topology = FakeTopology(all_atoms)
    fixer.positions = FakeQuantity([FakeVec3(float(i), float(i) * 2, float(i) * 3) for i in range(6)])

    prepare_module._relax_reconstructed_atoms(fixer, original_atom_keys)

    force = captured["force"]
    assert len(force.particles) == 3
    assert sorted(p[0] for p in force.particles) == [0, 1, 2]
    restrained_positions = {p[0]: p[1] for p in force.particles}
    assert restrained_positions[0] == [0.0, 0.0, 0.0]
    assert restrained_positions[1] == [1.0, 2.0, 3.0]
    assert fixer.positions is not None

    # Two-stage annealed restraint: soft first (letting the system move
    # enough to resolve a severe clash), then strong (pulling the original
    # atoms firmly back to their real positions).
    sim = captured["simulation"]
    assert sim.minimize_calls == [
        prepare_module._RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE1,
        prepare_module._RECONSTRUCTED_ATOM_MINIMIZATION_ITERATIONS_STAGE2,
    ]
    assert sim.context.param_calls == [
        ("k", prepare_module._RECONSTRUCTED_ATOM_RESTRAINT_K_SOFT),
        ("k", prepare_module._RECONSTRUCTED_ATOM_RESTRAINT_K_STRONG),
    ]


def test_relax_reconstructed_atoms_refuses_to_run_fully_unrestrained(monkeypatch):
    from amr_unbind import prepare as prepare_module
    from amr_unbind.errors import ProteinPreparationError

    class FakeChain:
        id = "A"

    class FakeResidue:
        chain = FakeChain()
        id = "1"
        name = "ALA"
        insertionCode = " "

    class FakeAtom:
        def __init__(self, index):
            self.index = index
            self.residue = FakeResidue()
            self.name = "N"

    class FakeTopology:
        def __init__(self, n):
            self._atoms = [FakeAtom(i) for i in range(n)]

        def atoms(self):
            return list(self._atoms)

    class FakeQuantity:
        def value_in_unit(self, target_unit):
            return []

    class FakeFixer:
        pass

    fixer = FakeFixer()
    fixer.topology = FakeTopology(3)
    fixer.positions = FakeQuantity()

    try:
        prepare_module._relax_reconstructed_atoms(fixer, set())
    except ProteinPreparationError as exc:
        assert "No atoms were identified" in str(exc)
    else:
        raise AssertionError("Expected ProteinPreparationError when no atoms are restrained")


# --- v1.0.9 regression tests -------------------------------------------------


def test_find_chain_gap_boundaries_reproduces_real_field_failure():
    """Regression test for a real field failure: automatic docking failed
    with DockingError: Meeko receptor preparation failed: No template
    matched for residue_key='B:103' ... and residue_key='B:123' ... matched
    with excess inter-residue bond(s): B:123. The actual PDB (3F2R) has
    chain B residues 104-122 entirely unresolved (REMARK 465), so TRP B:103
    and MET B:123 -- 20 residues apart in the real sequence -- end up
    written out as directly adjacent ATOM records with nothing to mark the
    gap between them. Meeko infers a peptide bond from that adjacency,
    corrupting template matching for both residues even though neither's
    own chemistry is at fault."""
    from amr_unbind.prepare import _find_chain_gap_boundaries

    class FakeResidue:
        def __init__(self, id_str):
            self.id = id_str

    class FakeChain:
        def __init__(self, chain_id, residue_ids):
            self.id = chain_id
            self._residues = [FakeResidue(r) for r in residue_ids]

        def residues(self):
            return list(self._residues)

    class FakeTopology:
        def __init__(self, chains):
            self._chains = chains

        def chains(self):
            return list(self._chains)

    # Chain B: ...101, 102, 103, [104-122 missing], 123, 124, 125...
    chain_b = FakeChain("B", ["101", "102", "103", "123", "124", "125"])
    boundaries = _find_chain_gap_boundaries(FakeTopology([chain_b]))
    assert boundaries == {("B", "103")}


def test_find_chain_gap_boundaries_handles_multiple_chains_independently():
    from amr_unbind.prepare import _find_chain_gap_boundaries

    class FakeResidue:
        def __init__(self, id_str):
            self.id = id_str

    class FakeChain:
        def __init__(self, chain_id, residue_ids):
            self.id = chain_id
            self._residues = [FakeResidue(r) for r in residue_ids]

        def residues(self):
            return list(self._residues)

    class FakeTopology:
        def __init__(self, chains):
            self._chains = chains

        def chains(self):
            return list(self._chains)

    chain_a = FakeChain("A", ["80", "81", "82", "150", "151"])  # one gap: 82 -> 150
    chain_b = FakeChain("B", ["90", "91", "101", "104"])  # two gaps: 91->101, 101->104
    boundaries = _find_chain_gap_boundaries(FakeTopology([chain_a, chain_b]))
    assert boundaries == {("A", "82"), ("B", "91"), ("B", "101")}

    clean_chain = FakeChain("A", ["1", "2", "3", "4"])
    assert _find_chain_gap_boundaries(FakeTopology([clean_chain])) == set()


def test_insert_ter_at_gap_boundaries_reproduces_real_field_failure():
    from amr_unbind.prepare import _insert_ter_at_gap_boundaries

    pdb = (
        "ATOM   1000  N   TRP B 103     -20.803   4.975  33.842  1.00 41.39           N\n"
        "ATOM   1001  CA  TRP B 103     -19.664   4.098  33.771  1.00 38.03           C\n"
        "ATOM   1002  C   TRP B 103     -18.595   4.424  34.771  1.00 37.89           C\n"
        "ATOM   1003  O   TRP B 103     -17.950   3.507  35.294  1.00 40.71           O\n"
        "ATOM   1004  N   MET B 123     -10.277 -10.908  20.106  1.00 62.30           N\n"
        "ATOM   1005  CA  MET B 123     -11.467 -11.312  20.813  1.00 59.48           C\n"
        "ATOM   1006  C   MET B 123     -12.339 -10.083  20.929  1.00 51.68           C\n"
        "ATOM   1007  O   MET B 123     -11.846  -8.965  21.091  1.00 48.06           O\n"
        "ATOM   1008  N   LEU B 124     -13.636 -10.289  20.820  1.00 44.06           N\n"
        "END\n"
    )
    out_lines = _insert_ter_at_gap_boundaries(pdb, {("B", "103")}).splitlines()

    ter_lines = [i for i, l in enumerate(out_lines) if l.startswith("TER")]
    assert len(ter_lines) == 1  # not inserted at the LEU 124 boundary (no gap there)
    ter_idx = ter_lines[0]
    assert "TRP B 103" in out_lines[ter_idx - 1]
    assert "MET B 123" in out_lines[ter_idx + 1]
    assert out_lines[ter_idx][21] == "B" and out_lines[ter_idx][22:26].strip() == "103"


def test_insert_ter_at_gap_boundaries_noop_without_boundaries():
    from amr_unbind.prepare import _insert_ter_at_gap_boundaries

    pdb = "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n"
    assert _insert_ter_at_gap_boundaries(pdb, set()) == pdb


def test_insert_ter_at_gap_boundaries_respects_existing_ter_records():
    from amr_unbind.prepare import _insert_ter_at_gap_boundaries

    pdb = (
        "ATOM      1  N   ALA A   1      0.000   0.000   0.000  1.00 20.00           N\n"
        "ATOM      2  N   GLY A   2      1.000   0.000   0.000  1.00 20.00           N\n"
        "TER\n"
        "ATOM      3  N   ALA B   1      2.000   0.000   0.000  1.00 20.00           N\n"
    )
    out = _insert_ter_at_gap_boundaries(pdb, {("A", "1")})
    assert out.count("TER") == 2  # the inserted one plus the pre-existing chain-break one


# --- v1.0.10 regression tests ------------------------------------------------


def test_parse_remark_465_reproduces_real_field_structure():
    """Regression test for a real field failure: automatic docking failed
    with RuntimeError: unable to build rdkit mol for residue LEU
    corresponding to key A:455 -- the same residue and error, recurring
    after v1.0.8's minimization-robustness fix, meaning that fix did not
    address the actual cause. LEU A:455 has no REMARK 470 entry (nothing
    about it is missing) but is the last residue PDBFixer's topology
    contains for chain A, because 3F2R's own REMARK 465 documents GLY A:456
    and VAL A:457 as entirely unresolved. This reproduces that exact header
    shape, including the descriptive text lines that must not be
    misparsed as data."""
    from amr_unbind.prepare import _parse_remark_465_missing_residues

    remark_block = (
        "REMARK 465                                                                      \n"
        "REMARK 465 MISSING RESIDUES                                                     \n"
        "REMARK 465 THE FOLLOWING RESIDUES WERE NOT LOCATED IN THE                       \n"
        "REMARK 465 EXPERIMENT. (M=MODEL NUMBER; RES=RESIDUE NAME; C=CHAIN               \n"
        "REMARK 465 IDENTIFIER; SSSEQ=SEQUENCE NUMBER; I=INSERTION CODE.)                \n"
        "REMARK 465                                                                      \n"
        "REMARK 465   M RES C SSSEQI                                                     \n"
        "REMARK 465     GLY A    57                                                      \n"
        "REMARK 465     SER A    58                                                      \n"
        "REMARK 465     ASN A   388                                                      \n"
        "REMARK 465     GLY A   456                                                      \n"
        "REMARK 465     VAL A   457                                                      \n"
        "REMARK 465     GLY B    57                                                      \n"
    )
    missing = _parse_remark_465_missing_residues(remark_block)
    assert missing["A"] == {57, 58, 388, 456, 457}
    assert missing["B"] == {57}


def test_find_pseudo_terminal_residues_flags_leu_455_not_genuine_termini():
    from amr_unbind.prepare import _find_pseudo_terminal_residues

    class FakeResidue:
        def __init__(self, id_str):
            self.id = id_str

    class FakeChain:
        def __init__(self, chain_id, residue_ids):
            self.id = chain_id
            self._residues = [FakeResidue(r) for r in residue_ids]

        def residues(self):
            return list(self._residues)

    class FakeTopology:
        def __init__(self, chains):
            self._chains = chains

        def chains(self):
            return list(self._chains)

    # Chain A's last modeled residue is 455; REMARK 465 documents 456/457
    # as real residues beyond it -> pseudo-terminus.
    chain_a = FakeChain("A", ["453", "454", "455"])
    missing_by_chain = {"A": {456, 457}}
    pseudo = _find_pseudo_terminal_residues(FakeTopology([chain_a]), missing_by_chain)
    assert pseudo == {("A", "455")}

    # A chain with nothing documented beyond its last residue is a genuine
    # terminus and must not be flagged.
    chain_clean = FakeChain("C", ["1", "2", "3"])
    assert _find_pseudo_terminal_residues(FakeTopology([chain_clean]), {}) == set()


def test_strip_spurious_terminal_oxt_reproduces_real_field_failure():
    from amr_unbind.prepare import _strip_spurious_terminal_oxt

    pdb = (
        "ATOM   3000  N   LEU A 455     -10.000  20.000  15.000  1.00 40.00           N\n"
        "ATOM   3001  CA  LEU A 455      -9.000  20.500  15.500  1.00 40.00           C\n"
        "ATOM   3002  C   LEU A 455      -8.000  19.800  16.200  1.00 40.00           C\n"
        "ATOM   3003  O   LEU A 455      -7.500  18.700  16.000  1.00 40.00           O\n"
        "ATOM   3004  CB  LEU A 455      -9.500  21.900  16.000  1.00 40.00           C\n"
        "ATOM   3005  CG  LEU A 455     -10.200  22.800  15.000  1.00 40.00           C\n"
        "ATOM   3006  CD1 LEU A 455     -11.500  22.200  14.500  1.00 40.00           C\n"
        "ATOM   3007  CD2 LEU A 455      -9.300  23.200  13.900  1.00 40.00           C\n"
        "ATOM   3008  OXT LEU A 455      -7.800  19.850  17.450  1.00 40.00           O\n"
        "END\n"
    )
    out_lines = _strip_spurious_terminal_oxt(pdb, {("A", "455")}).splitlines()
    assert not any("OXT" in l for l in out_lines)
    assert sum(1 for l in out_lines if "LEU A 455" in l) == 8  # the other 8 atoms kept
    assert "END" in out_lines

    # No-op when there are no pseudo-termini.
    assert _strip_spurious_terminal_oxt(pdb, set()) == pdb

    # A genuine terminus's OXT (different residue key) must be preserved.
    out_genuine = _strip_spurious_terminal_oxt(pdb, {("A", "999")})
    assert any("OXT" in l for l in out_genuine.splitlines())


# --- v1.0.11 regression tests ------------------------------------------------


def test_resolve_docking_center_angstrom_prefers_override_over_computed():
    """Regression test: the Streamlit app's "Docking Center Coordinates"
    Center X/Y/Z inputs were rendered but never actually reached
    AMRUnbindPipeline.run() -- automatic docking always used the
    geometry-derived core point, silently ignoring whatever the user typed
    in. _resolve_docking_center_angstrom() is the function run() now calls
    to pick the docking-box center; this confirms an explicit override
    takes precedence over the computed core point."""
    from amr_unbind.pipeline import _resolve_docking_center_angstrom

    computed = np.array([1.0, 2.0, 3.0])
    override = (27.23, 11.23, 17.52)
    result = _resolve_docking_center_angstrom(override, computed)
    assert result == (27.23, 11.23, 17.52)


def test_resolve_docking_center_angstrom_falls_back_to_computed_when_none():
    """With no override supplied, behavior must match every prior version:
    the geometry-derived core point is used unchanged."""
    from amr_unbind.pipeline import _resolve_docking_center_angstrom

    computed = np.array([1.0, 2.0, 3.0])
    result = _resolve_docking_center_angstrom(None, computed)
    assert result == (1.0, 2.0, 3.0)


def test_resolve_docking_center_angstrom_rejects_invalid_override():
    from amr_unbind.errors import InputValidationError
    from amr_unbind.pipeline import _resolve_docking_center_angstrom

    computed = np.array([1.0, 2.0, 3.0])
    for bad in [(1.0, 2.0), (1.0, 2.0, float("nan")), (1.0, 2.0, float("inf"))]:
        try:
            _resolve_docking_center_angstrom(bad, computed)
        except InputValidationError:
            pass
        else:
            raise AssertionError(f"Expected InputValidationError for override={bad}")


# --- v1.1.1 regression tests --------------------------------------------------


def test_minimization_retry_schedule_uses_configured_then_larger_looser_attempt():
    """Regression test for a real field failure reported after v1.1.0: even
    with the v1.1.0 fix (minimize before assigning velocities), a
    severely-clashing starting structure -- a badly docked pose, or solvent
    packed too tightly -- could still leave the single, fixed-budget
    minimization pass only partially converged. Handing that structure to
    setVelocitiesToTemperature() and the integrator is what produced the
    opaque, low-level "Particle coordinate is NaN" crash reported from a
    Colab run, with no indication of why. _minimization_retry_schedule()
    must offer a second, more aggressive attempt (larger iteration budget,
    looser force tolerance) rather than only ever trying once."""
    from amr_unbind.simulation import _minimization_retry_schedule
    from amr_unbind.config import SimulationConfig

    schedule = _minimization_retry_schedule(SimulationConfig())
    assert len(schedule) == 2
    (iters1, tol1), (iters2, tol2) = schedule
    assert iters1 == SimulationConfig().minimization_iterations
    assert iters2 > iters1
    assert tol2 > tol1


def test_minimization_retry_schedule_falls_back_when_iterations_unset():
    from amr_unbind.simulation import _minimization_retry_schedule
    from amr_unbind.config import SimulationConfig

    schedule = _minimization_retry_schedule(SimulationConfig(minimization_iterations=0))
    assert schedule[0][0] == 2500  # documented fallback for an unset budget


def test_minimize_with_retry_succeeds_on_first_attempt_without_retrying():
    """When the first minimization pass already leaves a finite, sane
    structure, no second attempt should run -- the retry exists for the
    severe-clash case, not as a mandatory double minimization."""
    import numpy as np

    from amr_unbind.simulation import _minimize_with_retry
    from amr_unbind.config import SimulationConfig

    class _FakeState:
        def __init__(self, positions_nm, potential):
            self._positions, self._potential = positions_nm, potential

        def getPositions(self, asNumpy=True):
            arr = self._positions

            class _Q:
                def value_in_unit(self, u):
                    return arr

            return _Q()

        def getPotentialEnergy(self):
            val = self._potential

            class _E:
                def value_in_unit(self, u):
                    return val

            return _E()

    class _FakeMinSim:
        def __init__(self):
            self.calls = []
            self.context = self
            self._state = _FakeState(np.array([[1.0, 2.0, 3.0]]), -500.0)

        def getState(self, getPositions=True, getEnergy=True):
            return self._state

        def minimizeEnergy(self, maxIterations, tolerance=None):
            self.calls.append((maxIterations, tolerance))

    sim = _FakeMinSim()
    _minimize_with_retry(sim, SimulationConfig())
    assert len(sim.calls) == 1


def test_minimize_with_retry_retries_once_then_raises_actionable_error():
    """The failure mode this guards against: a structure so badly clashing
    that minimization never resolves it, even with the larger retry budget.
    This must surface as a clear SimulationError pointing at the real,
    actionable causes (a bad docked pose, or insufficient solvation
    padding) -- not the low-level, undiagnosable OpenMM exception that
    caused it, though that exception must still be chained via __cause__
    for anyone who needs the underlying traceback."""
    import numpy as np

    from amr_unbind.simulation import _minimize_with_retry
    from amr_unbind.config import SimulationConfig
    from amr_unbind.errors import SimulationError

    class _FakeState:
        def __init__(self, positions_nm, potential):
            self._positions, self._potential = positions_nm, potential

        def getPositions(self, asNumpy=True):
            arr = self._positions

            class _Q:
                def value_in_unit(self, u):
                    return arr

            return _Q()

        def getPotentialEnergy(self):
            val = self._potential

            class _E:
                def value_in_unit(self, u):
                    return val

            return _E()

    class _FakeMinSim:
        """minimizeEnergy() always "succeeds" but never actually resolves
        the NaN it started with -- simulating a severely clashing system a
        larger iteration budget alone cannot fix."""

        def __init__(self):
            self.calls = []
            self.context = self
            self._state = _FakeState(np.array([[np.nan, 0.0, 0.0]]), float("nan"))

        def getState(self, getPositions=True, getEnergy=True):
            return self._state

        def minimizeEnergy(self, maxIterations, tolerance=None):
            self.calls.append((maxIterations, tolerance))

    sim = _FakeMinSim()
    try:
        _minimize_with_retry(sim, SimulationConfig())
        raise AssertionError("Expected SimulationError after both attempts failed")
    except SimulationError as exc:
        assert len(sim.calls) == 2  # both attempts in the schedule were tried
        assert "docked" in str(exc).lower()
        assert "solvation_padding_nm" in str(exc)
        assert exc.__cause__ is not None


# --- v1.3.0 regression tests --------------------------------------------------


def test_meeko_bad_residue_flag_detects_new_flag_name(monkeypatch):
    """Regression test for a real failure: PDB 1STP (streptavidin/biotin --
    a partially-disordered binding loop is genuinely unresolved in this
    structure, a normal and common situation for experimental structures)
    made Meeko's receptor preparation raise "No template matched for
    residue_key='A:151' ... matched with excess inter-residue bond(s)" and
    abort the whole run. Meeko's own documented answer to exactly this
    class of failure is a flag that drops the offending residue instead of
    hard-failing -- but it renamed that flag (--allow_bad_res ->
    --delete_bad_res) at some point, so amr_unbind must detect which name
    the installed version actually accepts rather than hardcode a guess."""
    import subprocess as subprocess_module

    from amr_unbind import docking

    class FakeResult:
        stdout = "-d, --delete_bad_res\n(Flag) Ignore residues with missing atoms instead of raising an error.\n"
        stderr = ""

    monkeypatch.setattr(subprocess_module, "run", lambda *a, **k: FakeResult())
    docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None
    try:
        assert docking._meeko_bad_residue_flag() == "--delete_bad_res"
    finally:
        docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None


def test_meeko_bad_residue_flag_falls_back_to_old_name(monkeypatch):
    import subprocess as subprocess_module

    from amr_unbind import docking

    class FakeResult:
        stdout = "-a, --allow_bad_res\n(Flag) Ignore residues with missing atoms instead of raising an error.\n"
        stderr = ""

    monkeypatch.setattr(subprocess_module, "run", lambda *a, **k: FakeResult())
    docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None
    try:
        assert docking._meeko_bad_residue_flag() == "--allow_bad_res"
    finally:
        docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None


def test_meeko_bad_residue_flag_falls_back_when_help_unavailable(monkeypatch):
    """If meeko isn't installed, or --help itself fails for any reason,
    detection must not raise -- it should fall back to the older flag name
    rather than crash receptor preparation before it even starts."""
    import subprocess as subprocess_module

    from amr_unbind import docking

    def _raise(*a, **k):
        raise FileNotFoundError("no such module")

    monkeypatch.setattr(subprocess_module, "run", _raise)
    docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None
    try:
        assert docking._meeko_bad_residue_flag() == "--allow_bad_res"
    finally:
        docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None


def test_meeko_bad_residue_flag_is_cached_after_first_check(monkeypatch):
    import subprocess as subprocess_module

    from amr_unbind import docking

    calls = []

    class FakeResult:
        stdout = "--delete_bad_res"
        stderr = ""

    def _run(*a, **k):
        calls.append(1)
        return FakeResult()

    monkeypatch.setattr(subprocess_module, "run", _run)
    docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None
    try:
        docking._meeko_bad_residue_flag()
        docking._meeko_bad_residue_flag()
        docking._meeko_bad_residue_flag()
        assert len(calls) == 1
    finally:
        docking._MEEKO_BAD_RESIDUE_FLAG_CACHE = None


def test_write_command_log_captures_stdout_and_stderr(tmp_path):
    """The bad-residue flag makes Meeko drop a problematic residue instead
    of failing outright -- which must never be silent. Meeko's full
    stdout/stderr (which names exactly which residue(s), if any, were
    dropped) is always written to disk so that exclusion is visible even
    though it no longer stops the run."""
    from amr_unbind.docking import _write_command_log

    log_path = tmp_path / "sub" / "test.log"
    _write_command_log(log_path, ["echo", "hello"], "stdout line\n", "stderr line\n")
    content = log_path.read_text()
    assert "echo hello" in content
    assert "stdout line" in content
    assert "stderr line" in content


def test_write_command_log_never_raises_on_write_failure():
    """A logging convenience must never itself turn into a run failure."""
    from pathlib import Path

    from amr_unbind.docking import _write_command_log

    # A path that cannot exist as a writable directory.
    bad_path = Path("/this/path/does/not/exist/and/cannot/be/created\x00") if False else Path(
        "/proc/1/this-cannot-be-created/test.log"
    )
    _write_command_log(bad_path, ["cmd"], "out", "err")  # must not raise


def test_automatic_vina_docking_passes_progress_callback_and_receptor_log_through(monkeypatch, tmp_path):
    """Regression test: automatic_vina_docking() must thread progress_callback
    through to _prepare_pdbqt_inputs/_dock_with_python/_dock_with_executable
    (so Heartbeat reporting during the actual slow calls works end to end,
    not just in the individual functions' own unit tests), and must carry
    the receptor-prep log path through into the returned DockingResult."""
    import sys
    import types
    from amr_unbind import docking

    receptor = tmp_path / "rec.pdb"
    ligand = tmp_path / "lig.sdf"
    receptor.write_text("ATOM\n")
    ligand.write_text("ligand\n")
    out = tmp_path / "dock"
    out.mkdir()
    rec_pdbqt = out / "receptor.pdbqt"
    lig_pdbqt = out / "ligand.pdbqt"
    rec_pdbqt.write_text("REMARK receptor\n")
    lig_pdbqt.write_text("REMARK ligand\n")
    log_path = out / "meeko_receptor_prep.log"
    log_path.write_text("mock meeko output\n")

    received_callbacks = []

    def fake_prepare(receptor_pdb, ligand_sdf, output_dir, progress_callback=None):
        received_callbacks.append(("prepare", progress_callback))
        return rec_pdbqt, lig_pdbqt, log_path

    def fake_dock_python(*args, progress_callback=None, **kwargs):
        received_callbacks.append(("dock", progress_callback))
        return -7.5

    monkeypatch.setitem(sys.modules, "vina", types.ModuleType("vina"))
    monkeypatch.setattr(docking, "_prepare_pdbqt_inputs", fake_prepare)
    monkeypatch.setattr(docking, "_dock_with_python", fake_dock_python)
    monkeypatch.setattr(docking, "_export_pose_to_sdf", lambda *a, **k: Path(a[1]).write_text("sdf"))
    pose = out / "docked_pose.pdbqt"
    pose.write_text("REMARK VINA RESULT: -7.500 0 0\nATOM\n")

    sentinel_callback = lambda stage, fraction, detail="": None  # noqa: E731

    result = docking.automatic_vina_docking(
        receptor,
        ligand,
        out,
        center_angstrom=(1.0, 2.0, 3.0),
        project_root=tmp_path,
        progress_callback=sentinel_callback,
    )

    assert result.receptor_prep_log == log_path
    # Both the prep step and the actual docking call must have received the
    # same callback that was passed in -- not None, not a different one.
    assert ("prepare", sentinel_callback) in received_callbacks
    assert ("dock", sentinel_callback) in received_callbacks


# --- v1.3.1 regression tests --------------------------------------------------


def _write_synthetic_protein_pdb(path):
    """A small, hand-constructed (NOT real structure data) PDB file: a
    tight cluster of protein heavy atoms near (12, 10, 10), plus one
    water molecule far away that must never count as "protein" for this
    check's purposes. Used only to exercise the counting/filtering logic
    in amr_unbind.geometry.count_protein_heavy_atoms_near_point against
    real parsed coordinates, not as a stand-in for any real structure."""
    path.write_text(
        "HEADER    SYNTHETIC TEST STRUCTURE, NOT REAL DATA\n"
        "ATOM      1  N   ALA A   1      10.000  10.000  10.000  1.00  0.00           N\n"
        "ATOM      2  CA  ALA A   1      11.000  10.000  10.000  1.00  0.00           C\n"
        "ATOM      3  C   ALA A   1      12.000  10.000  10.000  1.00  0.00           C\n"
        "ATOM      4  O   ALA A   1      12.500  11.000  10.000  1.00  0.00           O\n"
        "ATOM      5  CB  ALA A   1      10.500  11.000  10.500  1.00  0.00           C\n"
        "ATOM      6  N   ALA A   2      13.000  10.500  10.000  1.00  0.00           N\n"
        "ATOM      7  CA  ALA A   2      14.000  10.500  10.000  1.00  0.00           C\n"
        "ATOM      8  C   ALA A   2      14.500  11.500  10.000  1.00  0.00           C\n"
        "ATOM      9  O   ALA A   2      15.000  11.700  10.500  1.00  0.00           O\n"
        "ATOM     10  CB  ALA A   2      14.200   9.500   9.500  1.00  0.00           C\n"
        "ATOM     11  N   ALA A   3      11.500  10.800  10.200  1.00  0.00           N\n"
        "ATOM     12  CA  ALA A   3      12.200  10.100   9.800  1.00  0.00           C\n"
        "HETATM   13  O   HOH A 100      50.000  50.000  50.000  1.00  0.00           O\n"
        "END\n",
        encoding="utf-8",
    )


def test_count_protein_heavy_atoms_near_point_excludes_water_and_respects_radius(tmp_path):
    from amr_unbind.geometry import count_protein_heavy_atoms_near_point

    pdb = tmp_path / "synthetic.pdb"
    _write_synthetic_protein_pdb(pdb)

    assert count_protein_heavy_atoms_near_point(pdb, (12.5, 10.5, 10.0), radius_angstrom=5.0) == 12
    assert count_protein_heavy_atoms_near_point(pdb, (100.0, 100.0, 100.0), radius_angstrom=5.0) == 0
    # The water molecule must never count, however close the point is to it.
    assert count_protein_heavy_atoms_near_point(pdb, (50.0, 50.0, 50.0), radius_angstrom=2.0) == 0
    assert count_protein_heavy_atoms_near_point(pdb, (11.0, 10.0, 10.0), radius_angstrom=0.5) == 1


def test_docking_center_on_real_protein_mass_is_accepted(tmp_path):
    import numpy as np

    from amr_unbind.pipeline import _validate_docking_center_is_near_protein

    pdb = tmp_path / "synthetic.pdb"
    _write_synthetic_protein_pdb(pdb)
    core_point = np.array([12.0, 10.0, 10.0])

    _validate_docking_center_is_near_protein(
        (12.5, 10.5, 10.0),
        core_point,
        is_override=True,
        prepared_pdb_path=pdb,
        docking_box_angstrom=22.0,
    )  # must not raise


def test_docking_center_empty_of_protein_atoms_is_rejected_even_within_old_distance_threshold(tmp_path):
    """Regression test for a real reported failure: PDB 1STP, automatic
    docking, with the docking-center override left at the app's own
    example defaults (27.23, 11.23, 17.52 -- PDB 3F2R's biotin pocket).
    Checked directly against 1STP's own real deposited coordinates: those
    coordinates were only ~36 A from 1STP's own core point -- comfortably
    within what the first version of this check (a pure distance
    threshold of 100 A) would have accepted, yet nowhere near the real
    binding pocket. No fixed "too far" distance can reliably catch that,
    because 36 A is not an implausible distance for a real pocket in the
    abstract; only checking for actual nearby protein atoms can. This
    test reproduces that exact shape of gap against synthetic data: a
    center 36 A from the core point, with zero real atoms there."""
    import numpy as np

    from amr_unbind.errors import InputValidationError
    from amr_unbind.pipeline import _validate_docking_center_is_near_protein

    pdb = tmp_path / "synthetic.pdb"
    _write_synthetic_protein_pdb(pdb)
    core_point = np.array([12.0, 10.0, 10.0])
    far_but_within_old_threshold = (12.0 + 36.0, 10.0, 10.0)

    try:
        _validate_docking_center_is_near_protein(
            far_but_within_old_threshold,
            core_point,
            is_override=True,
            prepared_pdb_path=pdb,
            docking_box_angstrom=22.0,
        )
        raise AssertionError("Expected InputValidationError")
    except InputValidationError as exc:
        assert "real protein heavy atom" in str(exc)
        assert "3F2R" in str(exc)


def test_docking_center_egregiously_far_is_rejected_by_cheap_prefilter_without_parsing(tmp_path):
    """The cheap distance-from-core-point pre-filter should still catch an
    egregiously wrong center (hundreds of Angstrom off) without needing to
    parse the structure at all -- verified here by pointing at a PDB path
    that does not exist, confirming the function never gets far enough to
    try reading it."""
    import numpy as np

    from amr_unbind.errors import InputValidationError
    from amr_unbind.pipeline import _validate_docking_center_is_near_protein

    core_point = np.array([10.0, 10.0, 10.0])
    nonexistent_pdb = tmp_path / "does_not_exist.pdb"

    try:
        _validate_docking_center_is_near_protein(
            (1000.0, 1000.0, 1000.0),
            core_point,
            is_override=False,
            prepared_pdb_path=nonexistent_pdb,
            docking_box_angstrom=22.0,
        )
        raise AssertionError("Expected InputValidationError")
    except InputValidationError as exc:
        assert "almost certainly not on this protein at all" in str(exc)
        assert "auto-computed" in str(exc)
        assert "3F2R" not in str(exc)

