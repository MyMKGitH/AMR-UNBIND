"""Tests for amr_unbind/cloud.py.

These cover everything that does not require a Modal account, network
access, or a Colab runtime: CloudRunRequest's input validation (which
mirrors AMRUnbindPipeline.run()'s own validation, so a request that would
be rejected locally is rejected before anything is dispatched), job-dict
serialization, and the generated Colab notebook's structure and embedded
content. Actually dispatching to Modal, and actually running a generated
notebook in Colab, are outside what this test suite (or this project's CI)
can exercise -- see VERIFICATION.md.
"""

import json

import pytest

from amr_unbind.cloud import (
    CloudExecutionError,
    CloudRunRequest,
    MODAL_VALID_GPU_TYPES,
    generate_colab_notebook,
    modal_available,
)
from amr_unbind.config import SimulationConfig


def _base_kwargs(**overrides):
    kwargs = dict(
        smiles="CC[N+](C)(C)CCOP(=O)(O)O",
        core_chain="A",
        core_residue=122,
        exit_chain="A",
        exit_residue=205,
        config=SimulationConfig(),
        pdb_id="3f2r",
        automatic_docking=True,
    )
    kwargs.update(overrides)
    return kwargs


def test_modal_available_returns_bool():
    assert isinstance(modal_available(), bool)


def test_valid_request_with_pdb_id_and_automatic_docking():
    req = CloudRunRequest(**_base_kwargs())
    assert req.pdb_id == "3f2r"
    assert req.pdb_bytes is None


def test_valid_request_with_uploaded_pdb_and_supplied_pose():
    req = CloudRunRequest(
        **_base_kwargs(
            pdb_id=None,
            pdb_bytes=b"HEADER\nEND\n",
            pdb_filename="mine.pdb",
            automatic_docking=False,
            pose_sdf_bytes=b"mock\n$$$$\n",
            pose_sdf_filename="mine.sdf",
        )
    )
    assert req.pdb_bytes == b"HEADER\nEND\n"
    assert req.pose_sdf_bytes == b"mock\n$$$$\n"


@pytest.mark.parametrize(
    "overrides",
    [
        {"smiles": ""},
        {"smiles": "   "},
        {"pdb_id": None},  # neither pdb_id nor pdb_bytes
        {"pdb_bytes": b"x", "pdb_filename": "x.pdb"},  # both pdb_id (from base) and pdb_bytes
        {"automatic_docking": False},  # no pose_sdf_bytes either
        {"pose_sdf_bytes": b"x"},  # both automatic_docking (from base) and a supplied pose
        {"docking_box_angstrom": 0.0},
        {"docking_box_angstrom": -5.0},
        {"docking_exhaustiveness": 0},
        {"docking_center_angstrom": (1.0, 2.0)},
    ],
)
def test_invalid_requests_are_rejected(overrides):
    with pytest.raises(CloudExecutionError):
        CloudRunRequest(**_base_kwargs(**overrides))


def test_to_job_dict_is_plain_and_round_trippable():
    req = CloudRunRequest(**_base_kwargs(docking_center_angstrom=(27.23, 11.23, 17.52)))
    job = req.to_job_dict()
    # Must be JSON-serializable except for the (already bytes-or-None) file
    # fields -- this is what actually crosses the wire to Modal / gets
    # embedded in the notebook, so anything non-serializable here is a bug.
    job_without_bytes = {k: v for k, v in job.items() if k not in ("pdb_bytes", "pose_sdf_bytes")}
    json.dumps(job_without_bytes)  # raises if anything isn't serializable
    assert job["docking_center_angstrom"] == [27.23, 11.23, 17.52]
    assert job["config"] == SimulationConfig().to_dict()


def test_colab_notebook_is_valid_nbformat_json():
    req = CloudRunRequest(**_base_kwargs())
    nb = json.loads(generate_colab_notebook(req))
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) > 5
    for cell in nb["cells"]:
        assert cell["cell_type"] in ("code", "markdown")
        assert isinstance(cell["source"], list)
        assert all(isinstance(line, str) for line in cell["source"])


def test_colab_notebook_embeds_pdb_id_path_inputs():
    req = CloudRunRequest(
        **_base_kwargs(docking_center_angstrom=(27.23, 11.23, 17.52))
    )
    nb = json.loads(generate_colab_notebook(req))
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert 'PDB_ID = "3f2r"' in source
    assert "PDB_PATH = None" in source
    assert "AUTOMATIC_DOCKING = True" in source
    assert "DOCKING_CENTER_ANGSTROM = [27.23, 11.23, 17.52]" in source
    # Must call the real pipeline, not a reimplementation.
    assert "from amr_unbind.pipeline import AMRUnbindPipeline" in source
    assert "pipeline.run(" in source


def test_colab_notebook_embeds_uploaded_pdb_and_pose_text():
    req = CloudRunRequest(
        **_base_kwargs(
            pdb_id=None,
            pdb_bytes=b"HEADER    TEST\nEND\n",
            pdb_filename="mytarget.pdb",
            automatic_docking=False,
            pose_sdf_bytes=b"mock sdf\n$$$$\n",
            pose_sdf_filename="mypose.sdf",
        )
    )
    nb = json.loads(generate_colab_notebook(req))
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert "PDB_ID = None" in source
    assert "mytarget.pdb" in source
    assert "HEADER    TEST" in source
    assert "mypose.sdf" in source
    assert "mock sdf" in source


def test_colab_notebook_without_repo_zip_url_uses_manual_upload():
    req = CloudRunRequest(**_base_kwargs())
    nb = json.loads(generate_colab_notebook(req, repo_zip_url=None))
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert "files.upload()" in source


def test_colab_notebook_with_repo_zip_url_downloads_it():
    req = CloudRunRequest(**_base_kwargs())
    nb = json.loads(generate_colab_notebook(req, repo_zip_url="https://example.com/amr-unbind.zip"))
    source = "".join("".join(c["source"]) for c in nb["cells"])
    assert "https://example.com/amr-unbind.zip" in source
    assert "files.upload()" not in source


def test_modal_valid_gpu_types_is_nonempty_tuple_of_strings():
    assert isinstance(MODAL_VALID_GPU_TYPES, tuple)
    assert len(MODAL_VALID_GPU_TYPES) > 0
    assert all(isinstance(g, str) for g in MODAL_VALID_GPU_TYPES)
