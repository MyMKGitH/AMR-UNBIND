from pathlib import Path


def test_app_does_not_monkeypatch_openmm_or_vina():
    text = Path("app.py").read_text()
    assert "createSystem =" not in text
    assert "automatic_vina_docking = patched" not in text
    assert "_LAST_LIGAND_PATH" not in text
    assert "_LAST_SMILES" not in text


def test_docking_has_no_manual_pdbqt_fallback_or_pdb_parser_conversion():
    text = Path("amr_unbind/docking.py").read_text()
    assert "mol_to_pdbqt_fallback" not in text
    assert "MolFromPDBFile" not in text
    assert "clean_file_for_vina" not in text


def test_executable_backend_uses_argument_list_not_shell():
    text = Path("amr_unbind/docking.py").read_text()
    assert "shell=True" not in text
    assert "--center_x" in text
    assert "--out" in text
