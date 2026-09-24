---
title: AMR-UNBIND
emoji: 🧬
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8501
pinned: false
---

# 🧪 AMR-UNBIND

### Automated Antimicrobial Resistance Ligand Unbinding & Kinetics Engine

**Current version: v1.3.2**

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![OpenMM 8.6](https://img.shields.io/badge/OpenMM-8.6.0-green.svg)](https://openmm.org/)
[![Streamlit UI](https://img.shields.io/badge/User_Interface-Streamlit-red.svg)](https://streamlit.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-1.3.2-orange.svg)](#-changelog)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MyMKGitH/AMR-UNBIND/blob/main/notebooks/AMR-UNBIND-1_3_2.ipynb)
[![DOI](https://zenodo.org/badge/1386341972.svg)](https://doi.org/10.5281/zenodo.22948215)

AMR-UNBIND is an open-source scientific software pipeline for studying antibiotic–protein ligand unbinding using Steered Molecular Dynamics (SMD) in OpenMM. The engine automates the workflow from protein and ligand preparation through explicit-solvent molecular dynamics, forced ligand dissociation, and quantitative trajectory analysis. It combines a command-line interface for computational and HPC workflows with an interactive Streamlit dashboard for researchers who prefer a graphical interface.

The primary goal is to provide a **reproducible framework** for comparing ligand unbinding behavior across antibiotic candidates, resistant protein variants, and alternative binding configurations.

AMR-UNBIND is presented as a **research-grade proof-of-concept / pilot workflow**, not a validated production or regulatory tool. It reports **steering force and protocol work — not a binding free energy (ΔG)**. A single pilot SMD trajectory should never be interpreted as an experimental affinity measurement; see [Scientific Scope, Assumptions & Limitations](#-scientific-scope-assumptions--limitations) before drawing any scientific conclusion from it.

---

## 📑 Table of Contents

- [Why Study Ligand Unbinding?](#-why-study-ligand-unbinding)
- [Scientific Concept](#-scientific-concept)
- [Key Features](#-key-features)
- [Design Philosophy — What This Version Deliberately Does](#-design-philosophy--what-this-version-deliberately-does)
- [Scientific Workflow](#-scientific-workflow)
- [Scientific Scope, Assumptions & Limitations](#-scientific-scope-assumptions--limitations)
- [Dependencies](#-dependencies)
- [Quick Start & Installation](#-quick-start--installation)
- [Web Dashboard](#-web-dashboard)
- [Command-Line Usage](#-command-line-usage)
- [Configuration Reference](#-configuration-reference)
- [Hardware Requirements](#-hardware-requirements)
- [Output Files](#-output-files)
- [Interpreting the Results](#-interpreting-the-results)
- [Recommended Scientific Use](#-recommended-scientific-use)
- [Docker](#-docker)
- [Cloud Compute](#-cloud-compute)
- [Progress Reporting](#-progress-reporting)
- [Optional Automatic Docking](#-optional-automatic-docking)
- [Troubleshooting](#-troubleshooting)
- [ChimeraX Integration](#-chimerax-integration)
- [Machine-Learning Integration & Roadmap](#-machine-learning-integration--roadmap)
- [Reproducibility](#-reproducibility)
- [Repository Structure](#-repository-structure)
- [Example Research Applications](#-example-research-applications)
- [Development & Testing](#-development--testing)
- [Changelog](#-changelog)
- [Scientific References](#-scientific-references)
- [Contributing](#-contributing)
- [Scientific Disclaimer](#-scientific-disclaimer)
- [License](#-license)
- [Citation](#-citation)

---

## 💡 Why Study Ligand Unbinding?

Molecular docking is useful for identifying plausible binding poses and estimating relative binding interactions, but a static structure does not describe the *process* by which a ligand leaves a binding pocket. AMR-UNBIND therefore focuses on the **dynamic unbinding process**.

**Static docking**

> *"How well does the ligand fit inside the binding pocket?"*

Docking evaluates candidate poses and provides scoring functions that can help prioritize compounds.

**Dynamic unbinding**

> *"What happens when the ligand is forced out of the binding pocket?"*

Steered Molecular Dynamics applies a controlled external bias to drive the ligand along a predefined pathway while recording the resulting forces and work.

This can provide comparative information about:

- resistance to forced ligand dissociation
- maximum unbinding force
- force–distance profiles
- cumulative protocol work
- differences between ligands
- differences between protein variants
- possible effects of resistance-associated mutations

AMR-UNBIND is therefore intended as a **comparative non-equilibrium simulation framework**, rather than as a replacement for equilibrium free-energy calculations.

---

## 🧬 Scientific Concept

A ligand is initially positioned inside a protein binding pocket. AMR-UNBIND defines a pulling direction from a selected **core** region toward an **exit** region, and steers the ligand along this pathway while recording the response of the protein–ligand system.

```text
              Protein Target
        ┌─────────────────────────┐
        │                         │
        │     Binding Pocket      │
        │       ┌─────────┐       │
        │       │  Drug   │───────┼──────►
        │       └─────────┘       │       Pulling Direction
        │                         │       (Steered MD)
        └─────────────────────────┘

              Core → Exit Vector
```

The pulling direction is a **geometric heuristic**: a unit vector between the heavy-atom centroids of a user-selected core residue and a user-selected exit residue. It is not an automatically discovered minimum-free-energy pathway.

Rather than attaching the steering force to a single ligand atom, AMR-UNBIND applies a harmonic moving restraint to the ligand's **mass-weighted center-of-mass (COM) projection** onto that vector, instead of independently restraining every ligand atom. This is intended to reduce artificial distortion of the ligand geometry during pulling — and because the steering coordinate is a scalar projection, the ligand remains free to rotate and to move perpendicular to the pull direction.

---

## ✨ Key Features

**Zero-Scripting Web Interface**
AMR-UNBIND provides an interactive Streamlit dashboard for users who do not regularly work with Python or molecular-dynamics scripts. Researchers can upload structures, define ligand and pulling parameters, launch simulations, and inspect analysis results through the web interface. A command-line interface is also provided for HPC environments and automated workflows.

**Physically Motivated Pulling**
The default pulling mechanism acts on the ligand's overall center of mass rather than a single atom, avoiding unnecessarily concentrating the steering force on one atom of the ligand.

**Targeted Docking (optional)**
Automatic docking centers its Vina search box on the geometric point between the core and exit residues by default, but a custom (X, Y, Z) center in Å can be supplied instead — from the Streamlit sidebar (an opt-in override checkbox next to the Center X/Y/Z fields) or the CLI (`--docking-center-angstrom`) — for targeting a known binding pocket directly rather than relying on residue-derived geometry (added in v1.0.11; see [Optional Automatic Docking](#-optional-automatic-docking)).

**Cloud GPU Execution (optional)**
Explicit-solvent SMD on a CPU-only machine can take hours with nothing to show for it. Run the exact same pipeline on a Modal serverless GPU (one `modal deploy`, then every run is one click or `--compute modal`) or generate a ready-to-run Google Colab notebook with this run's inputs already filled in — no account beyond a Google login required (added in v1.1.0; see [Cloud Compute](#-cloud-compute)).

**Explicit-Solvent Simulation**
The workflow supports an explicit molecular environment including:

- TIP3P water
- neutralizing ions at a configurable ionic strength (default ≈ 0.15 M)
- AMBER14 protein parameters
- OpenFF Sage 2.2.1 small-molecule parameters

**Automated Quality Control**
The pipeline performs input and workflow validation intended to catch common user errors, including:

- invalid SMILES
- invalid or incomplete structures
- malformed PDB files
- missing ligand coordinates
- undefined ligand poses
- inconsistent simulation inputs
- a solvation box padding too small for the requested pull distance, which would let the ligand approach its own periodic image (added in v1.0.3)
- overlapping alternate-location (altLoc) crystallographic conformers, collapsed to a single highest-occupancy atom before either structure-preparation or docking ever sees them (added in v1.0.4; see [Troubleshooting](#-troubleshooting))
- unrelaxed clashes from PDBFixer's reconstruction of missing side-chain atoms and hydrogens (neither of which PDBFixer clash-checks against the rest of the structure), resolved with a restrained minimization before docking ever sees the structure (added in v1.0.6; see [Troubleshooting](#-troubleshooting))
- spurious inter-residue bonds inferred across a genuine sequence gap (residues this project deliberately doesn't rebuild), closed by inserting an explicit `TER` record at every such gap before docking ever sees the structure (added in v1.0.9; see [Troubleshooting](#-troubleshooting))
- a spurious terminal `OXT` atom PDBFixer can add to a residue that's only an artificial pseudo-terminus (real residues are documented beyond it in `REMARK 465`, just unmodeled), removed before docking ever sees the structure (added in v1.0.10; see [Troubleshooting](#-troubleshooting))

**Robust, Actionable Error Handling**
Expected failures (bad input, a failed docking attempt, a numerically unstable simulation, an invalid configuration) are raised as typed errors, recorded to a `run_failure.json` inside the run directory, and surfaced as a clean one-line message in both the CLI (non-zero exit code, no raw traceback) and the Streamlit app — rather than crashing.

**Reproducible Simulation Workspaces**
Each simulation is stored in its own isolated, uniquely named run directory containing simulation outputs and analysis products: input metadata, prepared structures, trajectories, simulation states, checkpoints, force data, work calculations, machine-learning feature vectors, and analysis plots. See [Output Files](#-output-files).

---

## 🎯 Design Philosophy — What This Version Deliberately Does

- AMR-UNBIND treats a **bound ligand pose as a required input** for an unbinding calculation. You can either supply a pre-docked 3D ligand SDF or enable the optional AutoDock Vina docking path. A SMILES string alone is used to build and validate ligand chemistry; it is **not** silently treated as a physically bound pose.
- The SMD coordinate is the mass-weighted center-of-mass projection described above, steered with a single harmonic moving restraint rather than by independently restraining every ligand atom.
- The program reports **steering force and protocol work — not a binding free energy**. A single pilot SMD run should not be interpreted as an experimental affinity measurement.

---

## 🔬 Scientific Workflow

AMR-UNBIND automates the computational workflow from structural input to analysis:

```text
Protein PDB + Ligand
        │
        ▼
┌──────────────────────┐
│  Input Validation    │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Structure Preparation│
│      PDBFixer         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Ligand Parameterization│
│       OpenFF          │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Protein Force Field  │
│       AMBER14         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Explicit Solvation    │
│  TIP3P + Ions         │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Energy Minimization   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  NVT / NPT Equil.     │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│ Steered MD (SMD)      │
│ Core → Exit Pulling   │
└──────────┬───────────┘
           ▼
┌──────────────────────┐
│  Automated Analysis   │
└──────────┬───────────┘
           ▼
 Force / Distance / Work
       + ML Features
```

### Workflow steps

**1. Input validation**
Accepts a protein PDB structure, a ligand SMILES, an experimentally determined or computationally generated ligand pose, a core residue, and an exit residue. An optional docking workflow can be used to generate a ligand pose.

**2. Structure preparation**
Protein structures are prepared using PDBFixer, including correction of common structural problems and preparation for simulation. Missing residues are *detected but not automatically rebuilt* (see [Limitations](#-scientific-scope-assumptions--limitations)).

**3. Force-field assignment**
Protein atoms are parameterized using AMBER14-compatible parameters, while ligand parameters are generated using the Open Force Field toolkit (OpenFF Sage).

**4. Explicit solvation**
The prepared complex is placed in an explicit TIP3P water environment with salt ions. The default ionic strength is approximately 0.15 M, and the solvation box padding is a configurable, validated parameter (see [Configuration Reference](#-configuration-reference)).

**5. Energy minimization**
Performed to reduce steric clashes and unfavorable contacts introduced during system preparation.

**6. Thermal equilibration**
The system undergoes equilibration under NVT and then NPT conditions, targeting approximately 300 K and 1 atm by default.

**7. Steered Molecular Dynamics**
The ligand's center of mass is pulled along the predefined core → exit vector during a **constant-volume** SMD stage (no barostat, box vectors carried over from the NPT stage), using a Langevin thermostat. Protein-backbone positional restraints stay active throughout to reduce reference-frame drift. The simulation records the mechanical response of the system during the forced dissociation process at full integrator resolution.

**8. Automated analysis**
The generated trajectory is analyzed to obtain ligand displacement, steering force, peak force, mean force, cumulative protocol work, and force slope. These quantities can subsequently be used for statistical comparison or machine-learning workflows. An optional ChimeraX rendering step (see [ChimeraX Integration](#-chimerax-integration)) can turn the trajectory into a movie.

---

## 📏 Scientific Scope, Assumptions & Limitations

AMR-UNBIND is designed primarily for **comparative SMD-based screening and mechanistic exploration**. It is important to distinguish forced unbinding simulations from equilibrium binding free-energy (ΔG) calculations.

### What AMR-UNBIND can provide

AMR-UNBIND can help researchers investigate:

- forced ligand dissociation pathways
- relative resistance to an imposed pulling protocol
- peak force behavior
- force–distance profiles
- protocol work
- differences between protein variants
- differences between ligand candidates
- machine-learning-ready descriptors

### What a single SMD run does not provide

A short single SMD trajectory should not be interpreted as a direct measurement of an absolute equilibrium binding free energy (ΔG). Nor should a single forced trajectory automatically be interpreted as an experimentally measured residence time or kinetic off-rate.

Obtaining equilibrium thermodynamic quantities requires appropriate statistical-mechanical methodologies and sufficient sampling — for example:

- umbrella sampling
- potential of mean force (PMF) calculations
- Jarzynski-type work estimators
- other work-based non-equilibrium methods
- other enhanced-sampling methodologies

AMR-UNBIND can provide trajectories and measurements that may serve as inputs to such downstream analyses, but it does not claim that a single pilot SMD simulation independently reconstructs an absolute free-energy landscape.

### Modeling assumptions and explicit choices

1. **Crystallographic heterogens.** The target is assumed to be a protein system for which removal of crystallographic heterogens is acceptable. This proof of concept intentionally does **not** try to guess whether a metal ion, cofactor, structural ligand, or other heterogen is biologically essential.
2. **Missing residues.** Missing residues are detected but are not automatically rebuilt. This avoids silently changing residue numbering and introducing speculative loops into a user-selected reaction coordinate.
3. **Pull direction.** The pull direction is a geometric heuristic based on two selected residues; it is not an automatically discovered minimum-free-energy pathway.
4. **Protein-backbone restraints.** Protein backbone positional restraints are used to reduce reference-frame drift during the pilot SMD. This is an explicit modeling choice and changes the physical system relative to an unrestrained simulation.
5. **Steering coordinate.** The steering coordinate is a scalar ligand COM projection; the ligand is therefore free to rotate and to move perpendicular to the pull direction.
6. **Protocol length.** The default protocol is deliberately short so it is practical as a laptop proof of concept. It is not a validated long-timescale sampling protocol.
7. **Docking scores.** Docking scores are reported as docking scores. They are not converted into binding free energies.
8. **Force/work curves.** Force–distance and work curves are diagnostic outputs. They do not by themselves establish a thermodynamic binding free energy.
9. **Solvation padding vs. pull distance (v1.0.3).** The solvation box padding must exceed the requested pull distance by a minimum safety margin, or the run is rejected before it starts — otherwise the ligand could be steered close enough to the box edge to interact with its own periodic image.

---

## 📦 Dependencies

The recommended environment is **Python 3.11**. Core versions:

| Package | Minimum version | Notes |
|---|---|---|
| OpenMM | ≥ 8.6.0, < 9 | |
| PDBFixer | ≥ 1.12, < 2 | |
| OpenFF Toolkit | ≥ 0.19.0, < 0.21 | |
| openmmforcefields | ≥ 0.15.0, < 1 | Floor is load-bearing — see note below |
| OpenFF small-molecule force field | Sage 2.2.1 (`openff-2.2.1`) | Selected explicitly, not a package version |
| gemmi | ≥ 0.7.5, < 1 | Structure-file parsing support |
| AutoDock Vina *(optional, docking only)* | 1.2.7 | Exact-pinned |
| Meeko *(optional, docking only)* | 0.8.0 | Exact-pinned |
| Modal *(optional, cloud GPU execution only)* | ≥ 0.60 | See [Cloud Compute](#-cloud-compute); not needed for local runs or the Colab backend |
| psutil *(optional)* | ≥ 5.9, < 8 | Powers the CPU/memory usage shown in progress heartbeats (see [Progress Reporting](#-progress-reporting)); everything works without it, just without that detail |
| ipython-genutils *(optional)* | ≥ 0.2.0 | required for result display and notebook utility functions |
| torch / torchvision | ≥ 2.1.0 / ≥ 0.16.0 | Not imported by `amr_unbind` itself today — see note below |

`openmmforcefields` requires OpenMM ≥ 8.5.1 and makes explicit small-molecule force-field selection mandatory; AMR-UNBIND therefore specifies the OpenFF force field explicitly instead of depending on a changing default. Its **floor deliberately stays at 0.16.0**: `amr_unbind/simulation.py`'s `SystemGenerator` usage relies on the periodic/non-periodic `forcefield_kwargs` split that version made mandatory (see [Changelog v1.0.2](#-changelog)), so anything older will not work correctly even though it may still install. OpenFF's current force-field catalog lists Sage 2.3.0 as a newer option, but this proof of concept keeps Sage 2.2.1 for a conservative, explicit configuration.

Everything above the optional-docking rows uses `>=`/`<` ranges rather than exact `==` pins, since exact pins have been observed to fail dependency resolution on some platforms (notably pip on Windows). `torch`/`torchvision` are included even though nothing in `amr_unbind` imports them yet, because they're commonly present once you start extending `amr_unbind.ml` toward the models described in [ML roadmap](#-machine-learning-integration--roadmap) — and because a PyTorch install is the most common real-world trigger of the Windows OpenMP conflict described in [Troubleshooting](#-troubleshooting).

AutoDock Vina and Meeko are optional and only needed for automatic docking; a user may instead supply a pre-docked SDF and run without the docking branch. See `requirements.txt` / `environment.yml` for the full dependency set and `requirements-docking.txt` / `environment-docking.yml` for the docking extras.

---

## 🚀 Quick Start & Installation

**1. Clone the repository**

```bash
git clone https://github.com/MyMKGitH/AMR-UNBIND.git
cd AMR-UNBIND
```

**2. Create the Conda/Mamba environment**

```bash
conda env create -f environment.yml
conda activate amr-unbind
```

**3. Verify the OpenMM installation**

```bash
python -m openmm.testInstallation
```

If your computer has NVIDIA CUDA support, OpenMM's current packaging also provides CUDA-enabled installations — check the OpenMM documentation for the driver/platform combination appropriate to your machine.

**4. Run the test suite**

```bash
pytest
```

---

## 🌐 Web Dashboard

Start the Streamlit interface:

```bash
conda activate amr-unbind
streamlit run app.py
```

Then open the local URL Streamlit prints to the terminal.

### Minimum inputs

You need:

- a protein PDB ID **or** a local PDB file;
- a ligand SMILES;
- a pre-docked ligand SDF **or** automatic docking enabled;
- a core residue and an exit residue that define the geometric pull direction.

From the dashboard, you can additionally configure the pull distance, pull velocity, steering force constant, **solvation box padding** (must stay comfortably larger than the pull distance — the app enforces this; see [Limitation #9](#-scientific-scope-assumptions--limitations)), equilibration lengths, random seed, and the OpenMM compute platform. When automatic docking is enabled, checking **"Override auto-computed pocket center with these coordinates"** and setting Center X/Y/Z targets the Vina search box at those coordinates instead of the geometric point between the core and exit residues (added in v1.0.11).

A typical workflow is:

```text
Upload Protein
      ↓
Provide Ligand
      ↓
Validate Structure
      ↓
Define Pulling Path
      ↓
Configure Simulation
      ↓
Run AMR-UNBIND
      ↓
Inspect Results
```

---

## 💻 Command-Line Usage

AMR-UNBIND can also be used from the command line, making it suitable for automated workflows and HPC environments.

**With a pre-docked pose:**

```bash
amr-unbind \
  --pdb-id 3f2r \
  --smiles "CC[N+](C)(C)CCOP(=O)(O)O" \
  --ligand-pose-sdf /path/to/docked_pose.sdf \
  --core-chain A \
  --core-residue 122 \
  --exit-chain A \
  --exit-residue 205 \
  --platform auto
```

**Or with automatic Vina docking:**

```bash
amr-unbind \
  --pdb-id 3f2r \
  --smiles "CC[N+](C)(C)CCOP(=O)(O)O" \
  --automatic-docking \
  --docking-box-angstrom 22.0 \
  --docking-exhaustiveness 8 \
  --core-chain A \
  --core-residue 122 \
  --exit-chain A \
  --exit-residue 205
```

**Or with automatic docking targeted at a known pocket, instead of the core/exit-residue geometric estimate:**

```bash
amr-unbind \
  --pdb-id 3f2r \
  --smiles "CC[N+](C)(C)CCOP(=O)(O)O" \
  --automatic-docking \
  --docking-box-angstrom 22.0 \
  --docking-center-angstrom 27.23 11.23 17.52 \
  --core-chain A \
  --core-residue 122 \
  --exit-chain A \
  --exit-residue 205
```

**Or the same run dispatched to a Modal GPU instead of running here (after the one-time `modal deploy` setup in [Cloud Compute](#-cloud-compute)):**

```bash
amr-unbind \
  --pdb-id 3f2r \
  --smiles "CC[N+](C)(C)CCOP(=O)(O)O" \
  --ligand-pose-sdf /path/to/docked_pose.sdf \
  --core-chain A \
  --core-residue 122 \
  --exit-chain A \
  --exit-residue 205 \
  --compute modal \
  --modal-gpu T4 \
  --modal-timeout-hours 6
```

### CLI flag reference

| Flag | Type | Notes |
|---|---|---|
| `--pdb-id` | string | RCSB PDB ID to download, e.g. `3f2r`. One of `--pdb-id` / `--pdb-file` is required. |
| `--pdb-file` | path | Local PDB file, used instead of downloading. |
| `--smiles` | string | **Required.** Ligand SMILES string. |
| `--ligand-pose-sdf` | path | Pre-docked 3D ligand pose (SDF). Mutually exclusive with `--automatic-docking`. |
| `--automatic-docking` | flag | Enable automatic AutoDock Vina docking. |
| `--docking-box-angstrom` | float | Cubic Vina search-box size in Å. Default `22.0`. |
| `--docking-exhaustiveness` | int | Vina exhaustiveness. Default `8`. |
| `--docking-center-angstrom` | float float float | Optional custom Vina search-box center (`X Y Z`, Å). Only used with `--automatic-docking`. Defaults to the geometric point between `--core-residue` and `--exit-residue` when omitted (added in v1.0.11). |
| `--core-chain` / `--core-residue` | string / int | **Required.** Chain and residue number of the "core" pull-direction anchor. |
| `--exit-chain` / `--exit-residue` | string / int | **Required.** Chain and residue number of the "exit" pull-direction anchor. |
| `--work-dir` | path | Root directory for run outputs. Default `./amr_workspace`. |
| `--config` | path | JSON `SimulationConfig` file; see [Configuration Reference](#-configuration-reference). |
| `--platform` | choice | `auto` \| `cpu` \| `cuda` \| `opencl` \| `hip`. Only overrides `--config`'s platform when explicitly passed. |
| `--pull-distance-nm` | float | Overrides `pull_distance_nm`. |
| `--pull-velocity-nm-ps` | float | Overrides `pull_velocity_nm_per_ps`. |
| `--force-constant` | float | Overrides `steering_k_kj_mol_nm2`. |
| `--seed` | int | Overrides `seed`. |
| `--compute` | choice | `local` (default) \| `modal`. `modal` dispatches to a deployed Modal GPU function instead of running here; see [Cloud Compute](#-cloud-compute) (added in v1.1.0). |
| `--modal-gpu` | choice | `T4` (default) \| `L4` \| `A10G` \| `A100`. Only used with `--compute modal`. |
| `--modal-timeout-hours` | float | Wall-clock timeout for `--compute modal`. Default `6.0`. |

The `--platform auto` option lets the software select an appropriate OpenMM computational platform based on the available environment. Any expected failure (bad input, docking failure, an invalid configuration) prints one clean message to stderr and exits with a non-zero status instead of a raw Python traceback.

---

## 🔧 Configuration Reference

Every numerical setting for a run lives in a single `SimulationConfig`, so a run can be fully reproduced from its configuration file and the generated `run_metadata.json`. See `examples/config_pilot.json` for a ready-to-use template, and pass it with `--config`.

| Field | Default | Meaning |
|---|---|---|
| `temperature_kelvin` | 300.0 | Target thermostat temperature |
| `pressure_atm` | 1.0 | Target barostat pressure (NPT stage) |
| `friction_per_ps` | 1.0 | Langevin friction coefficient |
| `timestep_fs` | 2.0 | Integration timestep |
| `ionic_strength_molar` | 0.15 | Added-salt ionic strength |
| `solvation_padding_nm` | 1.0 | Minimum solvent buffer beyond the solute; must exceed `pull_distance_nm` by a safety margin |
| `nvt_equilibration_ps` | 10.0 | NVT equilibration length |
| `npt_equilibration_ps` | 10.0 | NPT equilibration length |
| `pull_distance_nm` | 0.25 | Target ligand COM displacement along the pull vector |
| `pull_velocity_nm_per_ps` | 0.005 | Steering-restraint pulling speed |
| `steering_k_kj_mol_nm2` | 500.0 | Harmonic steering force constant |
| `report_interval_steps` | 100 | Trajectory/analysis reporting cadence |
| `checkpoint_interval_steps` | 5000 | Checkpoint/state save cadence during SMD |
| `minimization_iterations` | 500 | Energy-minimization iteration cap |
| `seed` | 20260907 | Random seed (velocities, integrators, barostat) |
| `protein_restraint_k_kj_mol_nm2` | 100.0 | Protein-backbone positional restraint strength |
| `pH` | 7.4 | Protonation state used when adding hydrogens |
| `forcefield_protein` | `amber14-all.xml` | Protein force field |
| `forcefield_water` | `amber14/tip3p.xml` | Water force field |
| `small_molecule_forcefield` | `openff-2.2.1` | Ligand force field (OpenFF Sage) |
| `preferred_platform` | `auto` | OpenMM platform selection |
| `cuda_device_index` | `"0"` | CUDA device index, when applicable |

All fields are validated on load — including, since v1.0.3, that `solvation_padding_nm` stays large enough relative to `pull_distance_nm`.

---

## 🧰 Hardware Requirements

AMR-UNBIND is designed to operate on both CPU systems and GPU-enabled machines.

| Platform | Capability | Typical use |
|---|---|---|
| CPU laptop / workstation | OpenMM CPU execution | Development, testing, teaching, short pilot simulations |
| GPU workstation | Accelerated OpenMM simulations | Larger simulations and repeated trajectories |
| HPC GPU cluster | High-throughput simulation | Ensemble SMD, mutant panels, ligand libraries |

Simulation speed depends strongly on system size, timestep, number of steps, hardware, and simulation settings. Short pilot simulations can be useful for validating the workflow, while scientifically meaningful comparative studies generally require multiple independent trajectories (see [Recommended Scientific Use](#-recommended-scientific-use)).

---

## 📊 Output Files

Each run gets its own isolated, timestamped, sanitized directory under `amr_workspace/runs/`:

```text
amr_workspace/runs/<UTC-timestamp>_<target>_<8-char-id>/
├── input_config.json          # exact inputs and SimulationConfig for this run
├── compute_metadata.json      # chosen OpenMM platform, pull step count, etc.
├── run_metadata.json          # protein/ligand/geometry/docking/build summary
├── run_failure.json           # only written if the run failed
├── prepared_protein.pdb       # PDBFixer output
├── ligand_initial.sdf         # RDKit/OpenFF-validated 3D ligand conformer
├── supplied_ligand_pose.sdf   # only if a pre-docked pose was supplied
├── docking/                   # only if --automatic-docking / auto-dock was used
│   ├── receptor.pdbqt
│   ├── ligand.pdbqt
│   ├── docked_pose.pdbqt
│   └── docked_pose.sdf
├── equilibrated_start.pdb     # structure entering the SMD stage
├── smd_trajectory.dcd         # SMD trajectory
├── smd_analysis.csv           # per-frame force/work analysis
├── state_data.csv             # OpenMM StateDataReporter log
├── smd_checkpoint.chk         # periodic checkpoint during SMD
├── smd_state.xml              # periodic serialized state during SMD
├── final_structure.pdb        # final coordinates
├── final_checkpoint.chk       # final OpenMM checkpoint
├── final_state.xml            # final serialized OpenMM state
├── simulation_summary.json    # headline energies, work, and displacement
└── ml_features.csv            # written on request (app / save_ml_features)
```

### `smd_analysis.csv`

Step-by-step simulation data:

| Column | Description |
|---|---|
| `Step` | Integration step number |
| `Time (ps)` | Simulation time |
| `Target Coordinate (nm)` | Position of the moving harmonic restraint |
| `Reaction Coordinate (nm)` | Actual ligand COM projection onto the pull vector |
| `Displacement (nm)` | `Reaction Coordinate` minus its initial value |
| `Steering Force (kJ/mol/nm)` | Instantaneous restraint force, `k·(target − coordinate)` |
| `Cumulative Work (kJ/mol)` | Trapezoidal integral of force over `d(target)`, accumulated every integration step (v1.0.3) |
| `Potential Energy (kJ/mol)` | System potential energy at the recorded frame |
| `Kinetic Energy (kJ/mol)` | System kinetic energy at the recorded frame |

### `final_structure.pdb`

Final coordinates from the simulation, suitable for inspection in molecular-visualization software such as PyMOL, VMD, or ChimeraX.

### `smd_trajectory.dcd`

Trajectory coordinates for post-processing and visualization.

### `ml_features.csv`

A standardized numerical feature vector for one completed run, intended for downstream statistical or machine-learning workflows:

| Field | Description |
|---|---|
| `final_displacement_nm` | Ligand COM displacement at the end of the run |
| `max_abs_force_kj_mol_nm` | Peak absolute steering force |
| `mean_abs_force_kj_mol_nm` | Mean absolute steering force |
| `force_std_kj_mol_nm` | Standard deviation of the steering force |
| `final_work_kj_mol` | Final cumulative protocol work |
| `work_per_nm_kj_mol_per_nm` | Work normalized by displacement |
| `force_slope_kj_mol_nm2` | Linear force-vs-displacement slope (safely `0.0` for near-constant displacement) |

### `final_checkpoint.chk` / `final_state.xml`

OpenMM checkpoint and serialized-state data that can be used for simulation continuation or inspection where compatible with the run's configuration and platform.

---

## 📈 Interpreting the Results

AMR-UNBIND primarily produces **comparative** non-equilibrium measurements. For example, researchers may compare:

```text
Drug A  ─────────► Peak Force:  X
Drug B  ─────────► Peak Force:  Y
Drug C  ─────────► Peak Force:  Z
```

or investigate the effect of a resistance-associated mutation:

```text
Wild Type     ─────► Unbinding Profile A
Mutant        ─────► Unbinding Profile B
```

Potential quantities of interest include:

- **Peak unbinding force** — the maximum force observed during the pulling protocol.
- **Mean force** — an average force measure over a defined portion of the pulling trajectory.
- **Force–distance profile** — the relationship between ligand displacement and steering force.
- **Protocol work** — the cumulative mechanical work performed by the steering protocol.

These values can be useful for relative comparisons, provided that simulations are performed using controlled and consistently defined protocols across the compared systems.

---

## 🧪 Recommended Scientific Use

For publication-oriented studies, AMR-UNBIND should generally be used with **replicate trajectories** rather than relying on a single stochastic simulation. A typical comparative experiment may resemble:

```text
Protein Variant 1
      │
      ├── Replica 1
      ├── Replica 2
      ├── Replica 3
      ├── ...
      └── Replica N

Protein Variant 2
      │
      ├── Replica 1
      ├── Replica 2
      ├── Replica 3
      ├── ...
      └── Replica N
```

Replication enables researchers to evaluate variability and distinguish systematic effects from individual trajectory behavior.

---

## 🐳 Docker

Build:

```bash
docker build -t amr-unbind:1.3.2 .
```

Run the Streamlit UI:

```bash
docker run --rm -p 8501:8501 amr-unbind:1.3.2
```

The base image is CPU-oriented. GPU passthrough is intentionally not hidden behind the Dockerfile; configure the appropriate OpenMM GPU runtime and NVIDIA Container Toolkit on the host when you later move to GPU execution. The core image uses a supplied pre-docked SDF; the optional Vina/Meeko environment (`Dockerfile.docking` / `environment-docking.yml`) is provided separately for Linux containers.

---

## ☁️ Cloud Compute

Explicit-solvent SMD is slow on CPU, and on an older or CPU-only machine it can be slow enough to be impractical — hours in, with nothing to show for it if it doesn't finish. Since v1.1.0, `amr_unbind/cloud.py` offers two independent ways to run **the exact same pipeline** somewhere faster. Neither is required, neither is imported by any core module, and both call `AMRUnbindPipeline.run()` directly — the same function a local run calls — rather than reimplementing any docking or simulation logic, so a cloud run and a local run of identical inputs are the same computation, not two independently maintained ones.

### Option A: Modal (serverless GPU)

[Modal](https://modal.com) runs your job on a cloud GPU on demand and bills per second of actual compute. New accounts get free credits, but **it is a paid, metered service beyond that** — it is not literally free, whatever a given tutorial calls it.

One-time setup:

```bash
pip install -r requirements-cloud.txt   # installs the `modal` package
modal token new                         # authenticate (opens a browser)
modal deploy amr_unbind/cloud.py        # publishes the remote GPU function
```

After that, every run is a single dispatch:

- **Streamlit app**: sidebar → **Compute** → **Cloud — Modal GPU**, choose a GPU type and timeout, then click **Run AMR-UNBIND** as usual. Results appear in the same results view as a local run.
- **CLI**: add `--compute modal` (plus, optionally, `--modal-gpu {T4,L4,A10G,A100}` and `--modal-timeout-hours`) to any normal `amr-unbind` invocation.

The image the remote job runs in is built directly from this project's own `Dockerfile.docking` — not a separately maintained package list — so it can't quietly drift from what you already run locally or in Docker. Results are written to a Modal Volume and synced back into `./amr_workspace/runs/<run_id>/` in the same layout a local run produces, so every downstream tool (the app's results view, `smd_analysis.csv`, `run_metadata.json`, analysis scripts) works identically either way.

If `submit_modal_run(...)` can't find a deployed function, it will say so and point you back to the `modal deploy` step above — it does not silently redeploy on your behalf (that would be slow, and would mask real deployment problems behind something that looks like it worked).

`Dockerfile.docking` — the image `modal.Image.from_dockerfile(...)` builds from — has its `ENTRYPOINT` commented out and its `COPY --chown=$MAMBA_USER:$MAMBA_USER` flags removed (v1.2.0): Modal calls the decorated Python function directly rather than running the image's own entrypoint, so a `streamlit run` entrypoint left active is at best unused and at worst interferes with how some builders validate the image; and the base `mambaorg/micromamba` image never actually switches to `$MAMBA_USER` at runtime here (there's no later `USER` directive reverting from the `USER root` at the top), so the `--chown` was already a no-op for local Docker use — dropping it doesn't change local behavior and avoids a build-time variable-substitution step some builders (including Modal's) don't handle the same way Docker Desktop's BuildKit does.

If a Modal run doesn't seem to be working, two standalone diagnostic scripts (not part of the pytest suite — see their docstrings) narrow down why:

```bash
python scripts/modal_env_check.py          # free: checks imports + whether amr_unbind/cloud.py is deployed
python scripts/modal_live_dispatch_check.py # costs a few cents: confirms Modal/GPU access work at all
```

Run `modal_env_check.py` first — it's free and usually pinpoints the problem (most often: `modal deploy amr_unbind/cloud.py` was never run). Reach for `modal_live_dispatch_check.py` only if that passes but a real run still doesn't, to rule out a more basic Modal account/GPU-availability issue independent of this project.

### Option B: Google Colab (free, no account beyond Google)

If you don't want a Modal account, generate a ready-to-run notebook instead:

- **Streamlit app**: sidebar → **Compute** → **Cloud — Google Colab notebook**, then click **Generate Colab Notebook**. This downloads an `.ipynb` file; the app itself does not run anything for this option.
- Open the downloaded file in [Google Colab](https://colab.research.google.com), set **Runtime → Change runtime type → T4 GPU**, then **Runtime → Run all**.

The notebook is a real, complete workflow, not a link plus a copy-paste snippet: it installs AMR-UNBIND (prompting you to upload the project zip you already have, unless you configured a hosted URL), installs the exact dependencies `requirements-docking.txt` pins, and runs `AMRUnbindPipeline.run()` with this run's inputs already filled in. The last cell zips the results and downloads them straight to your browser — they do not appear back in the Streamlit app, since Colab runs independently of it.

If you push this project to your own GitHub repository, you can put an archive URL (e.g. `https://github.com/<you>/AMR-UNBIND/archive/refs/heads/main.zip`) in the "URL to a hosted copy" field so the notebook downloads it directly instead of prompting for a manual upload — just make sure that URL actually points at the version you're running, since a stale branch would silently install different code than what generated the notebook. The manual-upload default has no such risk, which is why it's the default.

**Try it without running the app at all:** the [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MyMKGitH/AMR-UNBIND/blob/main/notebooks/AMR-UNBIND-1_3_2.ipynb) badge at the top of this README opens `notebooks/AMR-UNBIND-1_3_2.ipynb` — a static example generated by this exact same `generate_colab_notebook()` function, pre-filled with the same 3F2R/automatic-docking example used throughout this README, so there's nothing to configure first. Two things worth knowing before you click it: it downloads this project's source directly from `github.com/MyMKGitH/AMR-UNBIND`'s `main` branch, so it will only build a version matching what's actually pushed there; and its inputs are fixed at generation time (this specific example), unlike a notebook generated from the app itself, which embeds whatever you configured for that run. Generate your own from the app for anything other than the bundled example.

### Choosing between them

| | Modal | Colab |
|---|---|---|
| Cost | Pay-per-second after free credits | Free (subject to Colab's own usage limits) |
| Setup | One-time `modal deploy` | None beyond a Google login |
| Where it runs | In the background; check back later | In your open browser tab |
| Where results land | Back in the app / `amr_workspace/runs/` | Downloaded to your browser |

---

## 📊 Progress Reporting

Explicit-solvent SMD has stages that take a genuinely long time, especially on CPU or a shared free-tier machine — and some of them (Vina docking, Meeko preparation, energy minimization) have no sub-progress of their own to report, only "still running or not." Since v1.3.0, every run reports progress on one shared, **always monotonically increasing** 0–100% scale, and nothing goes silent for the duration of a single long call:

| Stage | Reaches |
|---|---|
| Downloading the PDB structure | — |
| Protein preparation | 5% |
| Ligand preparation | 8% |
| Docking (if automatic) | 10% |
| System build | 20% |
| Minimization | 22% |
| NVT equilibration | 35% |
| NPT equilibration | 48% |
| SMD pulling | 98% |
| Complete | 100% |

NVT, NPT, and SMD report real sub-progress as they run (steps completed so far). Docking, Meeko preparation, minimization, the initial PDB download, and (on the Modal backend) waiting for the remote job to finish have no sub-progress of their own — Vina and Meeko give no intermediate callback — so instead of sitting silently at a fixed percentage, a **heartbeat** message updates every ~20 seconds with elapsed time and, if [`psutil`](https://pypi.org/project/psutil/) is installed (it's in `requirements.txt`/`environment.yml` as an optional dependency — everything still works without it, just without the CPU/memory part), live CPU and memory usage:

```
[ 10.0%] docking (47s in this stage) -- still running — 47s elapsed | CPU 210% | RAM 3.2/12.7 GB
```

- **Streamlit app:** a second line under the run status shows the current stage, time spent in it, and this detail text.
- **CLI and the generated Colab notebook:** the same detail is appended to each printed progress line.

**How to read it:** the percentage moving at all — even slowly — means real progress. A percentage that's flat but whose *elapsed time* keeps climbing every ~20 seconds means a slow single call (docking and minimization are the usual suspects on CPU) is still genuinely working. A flat percentage where the detail line stops updating entirely for several minutes, with no error shown, is a real stall worth reporting.

**A real limitation, stated plainly:** on the Modal backend, the heartbeat during `fn.remote(...)` reports *local* elapsed time only — it has no visibility into what's actually happening inside the remote container. That's a genuine improvement over total silence, not the same thing as real remote progress streaming (which would need a different dispatch mechanism than this version implements).

---

## 🦿 Optional Automatic Docking

Automatic docking uses AutoDock Vina through its Python bindings plus Meeko for chemistry-aware PDBQT preparation. The upstream Vina documentation states that the Python bindings are for Linux and macOS; the core OpenMM workflow therefore does not require them. On a compatible Linux/macOS machine:

```bash
conda activate amr-unbind
pip install -r requirements-docking.txt
```

Then enable **Automatically dock ligand with AutoDock Vina** in the Streamlit interface (or pass `--automatic-docking` on the CLI). By default, the docking box is centered on the geometric point between the core and exit residues (`calculate_pull_geometry()`'s `core_point_angstrom`). To target a specific, already-known binding pocket instead, check **"Override auto-computed pocket center with these coordinates"** and set Center X/Y/Z in the Streamlit app, or pass `--docking-center-angstrom X Y Z` on the CLI (added in v1.0.11). Whichever center was actually used for a run is recorded in that run's `run_metadata.json` as `docking_center_angstrom_used` / `docking_center_source`.

### Docking backend behavior

Automatic docking prepares the receptor and ligand with Meeko and then attempts AutoDock Vina in this order:

1. Python Vina bindings, when installed and functional.
2. A Vina executable specified by `AMR_UNBIND_VINA`.
3. `bin/vina` or `bin/vina.exe` inside the project.
4. `vina` / `vina.exe` available on `PATH`.

A failed Python-Vina docking attempt no longer prevents the executable backend from being tried. If all backends fail, the reported error preserves the backend diagnostics.

The project intentionally does **not** hand-write PDBQT files or strip arbitrary PDB/SDF text to satisfy Vina. Meeko is the sole PDBQT preparation and reconstruction layer, following the documented Meeko workflow for Vina interoperability.

### Project-local Vina executable

Place a compatible executable at either:

```text
bin/vina.exe   # Windows
bin/vina       # Linux/macOS
```

or configure an explicit path with:

```text
AMR_UNBIND_VINA=/absolute/path/to/vina(.exe)
```

A Windows `bin/vina.exe` is bundled in this source package (Apache-2.0, © the Forli Lab / Scripps Research; see `bin/README.md`). AutoDock Vina and Meeko are separate components on Linux/macOS, where the Python `vina` package is used instead (see [Docking backend behavior](#docking-backend-behavior) above) — installing one does not automatically install the other.

---

## 🩺 Troubleshooting

Real-world issues found while running AMR-UNBIND outside the packaging environment, and what AMR-UNBIND now does about them automatically.

### "I updated AMR-UNBIND but I'm still seeing a bug that was supposed to be fixed"

Before assuming a fix didn't work, confirm the running process is actually using the version you think it is:

1. **Check the app's title bar.** The top of the Streamlit page reads `AMR-UNBIND v<version>` (`app.py` sets this from `amr_unbind.__version__`). If it doesn't match the version you just installed, the running process is not using your updated code.
2. **Fully restart, don't just refresh.** Stop the Streamlit process (Ctrl+C in its terminal) and run `streamlit run app.py` again. A browser refresh alone does not reload already-imported Python modules.
3. **Check how `amr_unbind` was installed.** If you used `pip install .` (a regular, non-editable install) at any point, replacing the source files does nothing — Python is still importing the old copy from `site-packages`. Either reinstall (`pip install --force-reinstall .`) or switch to an editable install (`pip install -e .`) so source-file changes take effect immediately.
4. From the command line, `python -c "import amr_unbind; print(amr_unbind.__version__)"` (run inside the same conda environment / terminal you launch AMR-UNBIND from) should print the version you expect.

### `DockingError: Meeko receptor preparation failed ... AtomValenceException: Explicit valence for atom # ... is greater than permitted`

This can have a few different causes; as of v1.0.10, AMR-UNBIND automatically root-causes and fixes all of them.

**Cause 1, a common culprit: missing side-chain atoms.** Many deposited crystal structures have flexible surface residues (commonly Arg, Lys, Ser, Glu, Val) where part of the side chain wasn't resolved in the electron density — check the structure's own `REMARK 470 MISSING ATOM` records to confirm. PDBFixer reconstructs these missing atoms (and *every* hydrogen — typical-resolution X-ray structures have none observed experimentally) using fixed ideal template geometry, with no clash-checking against the rest of the structure. An unrelaxed reconstructed atom sitting close to a neighbor can be misread as a bond by tools that infer connectivity from atomic distance (RDKit, and Meeko's own receptor preparation) — exactly the shape of this exception, on whichever residue's reconstruction happens to clash worst in that particular structure.

**Cause 2, less common: overlapping alternate-location (altLoc) conformers.** A flexible side chain deposited with two overlapping conformers, left unresolved, produces the same symptom (also related: `RuntimeError: unable to build rdkit mol for residue ...`).

**Fix for causes 1 and 2 (automatic since v1.0.4–v1.0.6, only actually working since v1.0.7):** `prepare_protein()` collapses alternate locations to a single conformer, resolved per residue, and runs a restrained vacuum minimization after atom/hydrogen reconstruction so PDBFixer's reconstructed atoms can relax away from clashes before the structure ever reaches Meeko. (The v1.0.6 implementation of the minimization step had a tracking bug that meant it never actually completed on any structure needing hydrogens placed — i.e. every structure; if you're on v1.0.6, upgrade.)

**Cause 3, confirmed (not just theorized) after a recurrence: a residue that's the last one PDBFixer's topology contains for its chain, due to an unresolved region beyond it, can receive a new terminal atom (`OXT`) placed without a reliable "outward" direction to anchor to** — since PDBFixer determines terminal handling from topological position, not from whether the chain biologically continues. This has no `REMARK 470` entry to check against, since the residue itself isn't missing anything; the tell is that it's the last modeled residue of its chain, with `REMARK 465` documenting real residues beyond it.

**v1.0.8 attempted a general mitigation** (a two-stage restrained minimization, giving the system more room to resolve any severe clash) rather than a targeted fix, since the mechanism wasn't confirmed yet. It wasn't sufficient — the same residue failed again with v1.0.8 installed.

**Fix (v1.0.10):** the structure's own `REMARK 465` records are parsed to identify residues that are only artificial pseudo-termini (real residues are documented beyond them, just unmodeled), and any `OXT` atom PDBFixer added to such a residue is removed outright before the file is written, rather than left for the minimizer to try to relax. A docking receptor's binding-site chemistry doesn't depend on a chemically accurate capping group at a residue that isn't the protein's true terminus.

**If you still hit this on a v1.0.10+ install:** first work through the version-mismatch checklist above. If the version is confirmed correct and it still happens, please open an issue with the PDB ID and: whether the failing residue has a `REMARK 470` entry, and whether it's the last modeled residue of its chain with real residues documented beyond it in `REMARK 465`. That distinguishes which of the causes above (or a fourth, not-yet-seen one) you've hit.

### `DockingError: Meeko receptor preparation failed: No template matched for residue_key='...' ... matched with excess inter-residue bond(s): ...`

**Cause, confirmed by inspecting real structures (3F2R, and again — differently — 1STP) rather than inferred:** a genuine sequence gap — residues this project deliberately doesn't rebuild (see the v1.0.2 audit) — leaves two flanking residues written out as directly adjacent `ATOM` records in `prepared_protein.pdb`, with nothing marking that real residues sit between them. Check the structure's `REMARK 465 MISSING RESIDUES` records: if the two residue numbers named in the error straddle a run of missing residue numbers within the same chain, this is it. (1STP is a good example to check yourself: RCSB's own structure summary reports 159 deposited residues but only 121 modeled.)

**Fix (partial since v1.0.9, completed in v1.3.0):** `prepare_protein()` scans the final topology for sequence-number gaps within each chain and inserts a properly formatted `TER` record — the standard PDB convention for "the chain breaks here" — at each one before the file is written. This is necessary but, on its own, **turned out not to be sufficient**: reported against a real run on PDB 1STP, Meeko's own receptor-preparation parser does not appear to reliably treat every `TER` as a chain break the way OpenMM's PDB parser does, so this error could still occur even with the `TER` record correctly in place. v1.3.0 closes that gap: every `mk_prepare_receptor` call now also passes Meeko's own documented flag for this exact situation (`--delete_bad_res`, or `--allow_bad_res` on an older Meeko — detected automatically), which drops the unresolvable residue from the receptor model instead of aborting. That exclusion is never silent: check `docking/meeko_receptor_prep.log` (also linked from `run_metadata.json` as `docking.receptor_prep_log`) to see exactly which residue(s), if any, were excluded for a given run.

**If you still hit this on a v1.3.0+ install:** please open an issue with the PDB ID and the two residue keys named in the error.

### The progress bar looks stuck / hasn't moved in a while

Since v1.3.0, every stage of a run reports on one shared, always-increasing 0–100% scale (see [Progress Reporting](#-progress-reporting) below) — if the percentage itself is genuinely not moving, check the second line under the run status (app) or the `-- detail` text after each `[NN.N%] stage` line (CLI/Colab). A message like `still running — 47s elapsed | CPU 210% | RAM 3.2/12.7 GB` updating every ~20 seconds means the run is actively working, even though the percentage itself hasn't changed yet (Vina docking, Meeko preparation, and minimization have no sub-progress of their own to report — only elapsed time). If that detail line stops updating entirely for several minutes with no error shown, that's a genuine stall worth reporting, not just a slow stage. If the run has actually stopped with an error rather than looking stuck, see the `Particle coordinate is NaN` entry further down this section.

### `ProteinPreparationError: Protein preparation failed: No atoms were identified as part of the original experimental structure; refusing to run an unrestrained clash-relief minimization on the whole system`

**Cause:** you're running v1.0.6. Its clash-relief minimization (see above) tracked which atoms to hold fixed using Python object identity, which does not survive the topology rebuild `PDBFixer.addMissingHydrogens()` performs internally — so *no* atom was ever recognized as "original," on every run, not just ones needing missing-atom reconstruction. This is this function's own safety check correctly refusing to run an unrestrained minimization rather than silently doing the wrong thing.

**Fix:** upgrade to v1.0.10 (v1.0.7 fixed this specific error; v1.0.8, v1.0.9, and v1.0.10 made subsequent, independent fixes to related failure modes — see above). See the [changelog](#-changelog).

### `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`

**Cause:** a Windows-specific conflict between two bundled copies of the Intel OpenMP runtime in the same process — typically an MKL-linked NumPy/OpenMM stack alongside a separate PyTorch install. Intel's runtime aborts the process on startup rather than risk the two silently sharing thread-pool state.

**Fix (automatic since v1.0.4):** `amr_unbind/__init__.py` and `app.py` set `KMP_DUPLICATE_LIB_OK=TRUE` before any OpenMM/OpenFF/RDKit/torch import — see the [v1.0.4 changelog entry](#-changelog). If you'd rather diagnose the underlying environment conflict than suppress the check, unset the variable or set it to `FALSE` yourself before launching; AMR-UNBIND's `setdefault` call will not override an explicit value.

### Typing coordinates into the Streamlit app's "Docking Center Coordinates" fields doesn't change where automatic docking searches

**Cause:** on v1.0.10 and earlier, the Center X/Y/Z inputs were rendered in the sidebar but never actually passed into the pipeline — `AMRUnbindPipeline.run()` had no parameter for a custom docking center at all, so automatic docking always used the geometric point between the core and exit residues, regardless of what was typed into those fields.

**Fix (v1.0.11):** a new **"Override auto-computed pocket center with these coordinates"** checkbox next to the Center X/Y/Z fields must be checked for them to take effect; `AMRUnbindPipeline.run()` now accepts a `docking_center_angstrom` override, and the CLI exposes the same thing via `--docking-center-angstrom X Y Z`. Existing scripts/calls that don't pass it are unaffected — behavior defaults to the same geometric-center calculation as before. Check `run_metadata.json`'s `docking_center_source` field (`"user_override"` vs `"auto_geometric_core"`) to confirm which was used for a given run.

### `InputValidationError: The resolved docking center ... has only N real protein heavy atom(s) within ... Å of it ...`

**Cause, reported against a real run:** docking-center coordinates are specific to one PDB structure's own crystallographic coordinate frame — they mean nothing on a different structure. The Center X/Y/Z fields' shown values are only an example (PDB 3F2R's biotin pocket); left checked and unchanged while running a different PDB ID, Vina would dock into whatever happens to be at that coordinate on the new structure — usually open solvent or a nonphysical clash, not a real pocket.

**Fix (v1.3.1, corrected in v1.3.2):** this is now refused outright, before docking or simulation runs, rather than allowed to produce a severely strained pose that might not fail until hours later (see the `Particle coordinate is NaN` entry below for what that used to look like). v1.3.1's first version of this check only compared distance from the structure's own core point against a fixed threshold — calibrated against a real case (PDB 1STP) after the fact, that threshold turned out too loose to catch the reported scenario, since no single "too far" distance can tell a legitimately distant pocket apart from a meaningless coordinate at a similar distance. v1.3.2 checks something more direct instead: whether real protein atoms actually exist near the proposed center at all (`amr_unbind/geometry.py::count_protein_heavy_atoms_near_point`). Uncheck the override to use the auto-computed core/exit-residue point instead, or supply coordinates you've actually determined are correct for the specific structure you're running.

### `Particle coordinate is NaN` (or a `SimulationError` mentioning it) during minimization or early equilibration

**Cause:** the starting structure has atoms placed too close together for minimization to resolve safely. Three known causes, most to least common: a docking-center override left at the wrong structure's coordinates (see the entry just above — this is now caught earlier and more clearly as of v1.3.1); a badly docked ligand pose for another reason (automatic docking landed the ligand overlapping the protein even with a correct center); or a solvation box padded too tightly (`solvation_padding_nm` too small relative to the pull distance and any restraints). As of v1.1.1, minimization automatically retries once with a much larger iteration budget before this can happen; if you're still seeing it, the clash is severe enough to survive that retry too, and — as of v1.3.0's chunked NVT/NPT reporting — you may see it happen partway through equilibration (e.g. `nvt`, some steps in) rather than only during minimization itself.

**Fix:** if the error message names a bad docked pose, try automatic docking again (poses aren't always deterministic) or set a manual `--docking-center-angstrom`/the app's docking-center override (see above) targeting a pocket you're confident is correct. If it names solvation padding, increase `solvation_padding_nm` in your `SimulationConfig` (either directly, or via a config JSON) — a larger water box gives minimization more room to relax packing before atoms are forced into contact. This is not something increasing `minimization_iterations` alone will fix if the retry (which already uses a much larger budget) couldn't resolve it either.

### `CUDA error 801` (or another CUDA/driver initialization failure) at simulation startup

**Cause:** a GPU driver/toolkit/kernel mismatch that OpenMM's CUDA platform cannot work around.

**Fix:** this isn't something AMR-UNBIND's code can detect or patch around — set `--platform opencl` or `--platform cpu` (CLI), or the "OpenMM platform" selector (Streamlit app), to bypass the CUDA platform entirely. `preferred_platform="auto"` (the default) already falls back through `CUDA → HIP → OpenCL → CPU` in order, but only *after* CUDA is available and selected as a platform in the first place — if CUDA is available but broken (rather than genuinely absent), an explicit non-CUDA choice is more reliable than relying on `auto`.

### `openmmforcefields`/`lxml` fails to parse a force field XML file with an `etree`-related error, despite `lxml` being installed

**Cause:** a corrupted `lxml` install (a partially-built or mismatched `etree` C-extension) in the active Conda/pip environment — not an AMR-UNBIND code issue.

**Fix:** reinstall `lxml` via Conda in the same environment (`conda install -c conda-forge --force-reinstall lxml`) so its C-extension is rebuilt cleanly; a pip reinstall alone has not reliably fixed this in practice.

### `CloudExecutionError: The `modal` package is not installed` / `...could not find a deployed Modal function`

**Cause:** either the `modal` Python package isn't installed in the environment running the Streamlit app or CLI (a separate, thin-client install from the heavy scientific stack the actual simulation needs), or `amr_unbind/cloud.py` has never been published with `modal deploy`.

**Fix:** run the one-time setup in [Cloud Compute](#-cloud-compute): `pip install -r requirements-cloud.txt`, `modal token new`, `modal deploy amr_unbind/cloud.py`. This app does not attempt to deploy automatically on every run, so this step has to happen once, manually, before the first cloud run.

### A Modal run bills more than expected, or a Colab install step fails on `vina`/`meeko`

**Cause (Modal cost):** Modal is metered pay-per-second beyond its free credits; a long `--modal-timeout-hours` combined with a larger/slower GPU (`A100` vs `T4`) directly increases cost per run. There is no way for this project to make Modal itself free — only to make it easy to stop using it (uninstall `modal`, or simply choose Local/Colab instead).

**Cause (Colab install failure):** `requirements-docking.txt`'s pinned `vina==1.2.7`/`meeko==0.8.0` wheels are the two packages in that file most likely to be sensitive to Colab's current base image and Python version.

**Fix:** for cost, start with a short `--modal-timeout-hours`/timeout and the default `T4` GPU, and increase only if a run is timing out before finishing. For a Colab install failure, open an issue with the exact `pip install` error — this is the one step in the generated notebook this project can't pin more tightly than `requirements-docking.txt` already does, since Colab's own base image changes outside this project's control.

### Dependency resolution fails on an exact-pinned scientific package

**Cause:** exact `==` pins on `openmm`/`pdbfixer`/`openff-toolkit`/`openmmforcefields` can fail to resolve on some platform/Python combinations (observed on pip-on-Windows) even when a perfectly compatible nearby version exists.

**Fix (since v1.0.4):** these now use `>=`/`<` ranges instead of exact pins — see [Dependencies](#-dependencies). If you still hit a resolution failure, prefer the conda/mamba path (`environment.yml`) over pip for these four packages; conda-forge carries builds pip does not.

---

## 🎬 ChimeraX Integration

ChimeraX is **not required** for the simulation or analysis. If ChimeraX is installed separately, its current developer documentation describes starting a REST interface with:

```text
remotecontrol rest start port 3000
```

AMR-UNBIND then sends a `POST /run` command to that local service, using ChimeraX's documented movie workflow — `movie record`, explicit `coordset` playback, `wait`, then `movie encode` — since simply opening a trajectory does not play it. ChimeraX 1.12 developer documentation demonstrates the same `remotecontrol rest start` mechanism and REST `/run` endpoint.

### ChimeraX Unbind Code

Open the final_structure.pdb file, then open smd_trajectory.dcd file in ChimeraX. Run the following command:

```text
preset "publication 1"
cartoon #1
show ligand
movie record
coordset #1 1,800; wait 800
movie encode "C:/Users/YourDocs/OneDrive/Desktop/unbinding_movie.mp4"
```

Best to color the ligand with a unique color.
Change the number of frames (800) as you see fit.

---

## 🧠 Machine-Learning Integration & Roadmap

AMR-UNBIND includes a feature-extraction layer that converts trajectory-level measurements into the numerical descriptors listed under [`ml_features.csv`](#-output-files):

```text
SMD Trajectory
      │
      ▼
Force / Distance / Work Analysis
      │
      ▼
Feature Extraction
      │
      ├── Displacement, Peak / Mean / Std Force
      ├── Cumulative & Normalized Work
      └── Force Slope
      │
      ▼
Machine-Learning Dataset
```

These descriptors can be combined with additional molecular or structural features for predictive modeling. The current feature layer is intended as a foundation for future supervised-learning and ranking approaches — it does not claim to be a validated predictive model on its own, and this is unlikely to change until it's benchmarked across many independent runs.

Machine learning is intentionally kept **out of the force calculation itself**: ML belongs after the physical simulation layer is trustworthy. Planned (**not yet implemented**) extensions for a future trajectory classification/ranking module include:

- ligand–protein contact counts
- hydrogen-bond counts
- pocket geometry descriptors
- docking score
- trajectory stability / QC features

A first such model should be interpretable (for example random forest or gradient boosting) and trained on multiple independent simulations, rather than presented as a replacement for molecular mechanics.

---

## 🔄 Reproducibility

Reproducibility is a central design objective of AMR-UNBIND. Every simulation run is isolated and stores:

- the original inputs and full `SimulationConfig` (`input_config.json`)
- prepared structures, trajectories, simulation states, and checkpoints
- analysis tables and feature vectors
- compute/platform metadata and timestamps

Researchers can therefore retain a full record of the computational workflow associated with each simulation. For publication-quality work, users should additionally record:

- software version (see [Changelog](#-changelog))
- force-field versions (see [Dependencies](#-dependencies))
- random seeds where applicable
- simulation parameters (see [Configuration Reference](#-configuration-reference))
- hardware/software platform
- number of independent replicas
- analysis methodology

---

## 📁 Repository Structure

```text
AMR-UNBIND/
│
├── app.py                        Interactive Streamlit web interface
├── environment.yml               Core Conda environment definition
├── environment-docking.yml       Conda environment including Vina/Meeko
├── requirements.txt              Core pip dependencies
├── requirements-docking.txt      Docking pip extras (Python Vina + Meeko)
├── requirements-docking-executable.txt   Docking extras for the executable-Vina path
├── requirements-cloud.txt        Optional: the `modal` SDK, for cloud GPU runs
├── Dockerfile                    Core container (CPU, supplied poses)
├── Dockerfile.docking            Container including Vina/Meeko (also the Modal cloud image)
├── pyproject.toml                Python package metadata and build configuration
├── examples/
│   └── config_pilot.json         Ready-to-use SimulationConfig template
├── notebooks/
│   └── AMR-UNBIND-1_3_2.ipynb    Static example Colab notebook (the "Open In Colab" badge above)
├── scripts/
│   ├── modal_env_check.py        Manual, free diagnostic: is Modal set up and deployed?
│   └── modal_live_dispatch_check.py   Manual, costs a few cents: does real GPU dispatch work?
├── bin/                          Project-local Vina executable (vina.exe bundled for Windows)
│
├── amr_unbind/
│   ├── __init__.py               Package version
│   ├── config.py                 SimulationConfig (validated run parameters)
│   ├── errors.py                 AMRUnbindError exception hierarchy
│   ├── io.py                     Run-directory creation and JSON I/O
│   ├── platform.py               OpenMM platform selection
│   ├── prepare.py                Protein/ligand preparation and PDBFixer integration
│   ├── geometry.py                Pulling-vector and geometric calculations
│   ├── docking.py                Optional AutoDock Vina / Meeko integration
│   ├── simulation.py             OpenMM / OpenFF system build and SMD engine
│   ├── pipeline.py               End-to-end run orchestration
│   ├── cloud.py                  Optional cloud execution: Modal GPU + Google Colab notebook
│   ├── progress.py                Monotonic progress budget + Heartbeat (long-call elapsed time/CPU/RAM)
│   ├── analysis.py               Force, displacement and work analysis
│   ├── ml.py                     Machine-learning feature extraction
│   ├── chimerax.py               Optional ChimeraX REST bridge
│   └── cli.py                    `amr-unbind` command-line entry point
│
└── tests/                        Automated test suite
```

---

## 🧬 Example Research Applications

AMR-UNBIND can be applied to computational studies such as:

**Antibiotic candidate comparison**
Compare the mechanical unbinding behavior of multiple antibiotic candidates targeting the same bacterial protein.

**Resistance mutation analysis**
Investigate how amino-acid substitutions associated with antimicrobial resistance alter ligand escape pathways.

**Structure–mechanism studies**
Combine structural modeling with SMD trajectories to investigate how binding-pocket architecture influences forced dissociation.

**Computational screening**
Generate standardized trajectory-derived features that can be integrated into downstream statistical or machine-learning workflows.

---

## 🧪 Development & Testing

```bash
python -m compileall -q .
pytest
```

The test suite validates configuration logic, basic geometry mathematics, CLI wiring, error-handling behavior, and repository integrity. Full OpenMM integration tests depend on the scientific environment and should be added against a fixed benchmark complex before using AMR-UNBIND for comparative research.

---

## 📋 Changelog

### v1.3.2 — v1.3.1's docking-center check was itself insufficient; replaced a distance-from-one-point heuristic with a direct check for real nearby protein atoms

The v1.3.1's check compared distance from the structure's own core point against a fixed 100 Å threshold, chosen without calibration. Fetching PDB 1STP's actual coordinates directly from RCSB proved it too loose: the reported override was only ~36 Å from 1STP's real core point (comfortably inside 100 Å) while the real biotin pocket sits ~8 Å away — no single distance threshold can tell a legitimately distant pocket apart from a meaningless coordinate at a similar distance.

**Fixed:** replaced the approach entirely. `amr_unbind/geometry.py::count_protein_heavy_atoms_near_point()` checks whether real protein atoms actually exist near the proposed docking center — a direct, structure-size-agnostic question, not a threshold to tune. The old distance check is kept only as a cheap pre-filter for egregiously wrong centers, loosened since it's no longer doing the real work. New tests use synthetic, explicitly-labeled fixture data; a partially-reconstructed "real 1STP" test fixture was caught as unverifiable and discarded before it reached anything shipped — see `AUDIT.md` for that process note.

### v1.3.1 — Fixed a real trap: the app's own example docking-center defaults, left unchanged for a different structure, produced a nonsensical docking box that survived minimization only to crash hours into NVT

Reported against a real run on PDB 1STP: the v1.3.0 progress system worked exactly as intended and, for the first time, made the real symptom visible (`nvt — 200/5000 steps`, then `Particle coordinate is NaN`). Traced to `app.py`'s own source: the Center X/Y/Z override fields have shown `27.23`/`11.23`/`17.52` — PDB 1STP's biotin example pocket — as their default value since v1.0.11, with nothing indicating this was structure-specific. Left unchanged while running a different structure, Vina docked into a coordinate that meant nothing on that protein, producing a pose strained enough to eventually diverge under real dynamics, but not so broken it failed the v1.1.1 minimization retry outright.

**Fixed:** `amr_unbind/pipeline.py::_validate_docking_center_is_near_protein()` rejects a resolved docking center (override or auto-computed) more than 100 Å from the structure's own core point, before docking or simulation ever runs — with a message naming the likely cause (including this app's own 3F2R example, when it's an override) rather than letting a doomed run burn hours before failing downstream. `app.py`'s Center X/Y/Z fields are now labeled "(example: 3F2R)" with a caption stating plainly that these are structure-specific. 2 new regression tests, both run for real (no heavy-dependency stubbing needed for this particular function).

### v1.3.0 — Fixed a real Meeko docking crash on PDB 1STP (missing-loop chain gap); replaced non-monotonic progress reporting with a real, monotonic, detailed progress system

**Fixed:** `DockingError: Meeko receptor preparation failed: No template matched...` on PDB 1STP, root-caused against the real structure (RCSB reports 159 deposited residues, only 121 modeled — a genuine unresolved loop) and Meeko's own issue tracker/docs. The v1.0.9 `TER`-insertion fix was necessary but not sufficient — Meeko's own parser doesn't reliably treat every `TER` as a chain break. Every `mk_prepare_receptor` call now also passes Meeko's own documented flag for this (`--delete_bad_res`/`--allow_bad_res`, detected from the installed version), which drops the unresolvable residue instead of aborting — never silently: see the new `docking/meeko_receptor_prep.log` and `DockingResult.receptor_prep_log`. A user-supplied reference fix (stripping PDB `CONECT` records) was reviewed and not used — very likely ineffective against the real, template/geometry-based mechanism.

**Added: a real, monotonic, detailed progress system.** Also reported: a progress bar stuck around 15% for nearly two hours, with no way to tell slow from hung. Root cause: `pipeline.py` and `simulation.py` each hardcoded their own independent 0–1 fractions, so progress could visibly jump *backward* between them. New `amr_unbind/progress.py` defines the one shared, monotonically increasing budget every stage-reporting module now imports from, plus `Heartbeat` (elapsed time + optional `psutil` CPU/memory, every ~20s) wrapping every remaining single-blocking-call stage — Vina docking, Meeko preparation, minimization, PDB download, protein prep, and (closing an earlier report's own "stuck at 5%") Modal's `fn.remote(...)` dispatch. NVT/NPT equilibration now report real sub-progress via `_run_chunked_equilibration` instead of a single silent `step(N)` call. The app shows a second detail line under the run status; the CLI and generated Colab notebook print the same detail alongside each progress line. See [Progress Reporting](#-progress-reporting).

**Fixed:** `environment.yml` was still pinned to the old `openmmforcefields>=0.16.0` floor after `requirements.txt`/`environment-docking.yml` were corrected to `>=0.15.0` in v1.2.0 — flagged directly, now identical and cross-referenced across all three files, verified programmatically.

**Notes:** the Modal heartbeat fix reports local elapsed time only, not real remote progress — stated as a real limitation, not oversold. Full audit, including a genuine regression this version's own changes caused and fixed in a pre-existing test, in `AUDIT.md`/`VERIFICATION.md`.

### v1.2.0 — Merged user-contributed Modal/Colab container fixes; added a static example Colab notebook + badge and two manual diagnostic scripts

**Fixed (user-contributed, from real hands-on Modal/Colab testing):** `Dockerfile.docking`'s live `ENTRYPOINT` (unnecessary and potentially interfering for Modal, which calls the function directly) is now commented out there; both Dockerfiles' `COPY --chown=$MAMBA_USER:$MAMBA_USER` flags are removed (verified harmless for local Docker — the containers already run as root regardless). `environment-docking.yml`/`requirements.txt`: `openff-toolkit` moved to conda, `openff-interchange` and `lxml` added explicitly (closing a gap from this project's very first fix summary), `cuda-version=11.8` pinned, `openmmforcefields`'s floor lowered `0.16.0` → `0.15.0` consistently in both files.

**Added:** a static `notebooks/AMR-UNBIND-1_2_0.ipynb` (generated by the real `generate_colab_notebook()`, not hand-written) and an **Open In Colab** badge at the top of this README; `scripts/modal_env_check.py` and `scripts/modal_live_dispatch_check.py`, two manual diagnostics for Modal connectivity issues.

**Notes:** the diagnostic scripts were supplied for `tests/`, but relocated to a new `scripts/` directory and one renamed — `run_modal_test.py` matched pytest's default auto-collection pattern and its `test_cloud_dispatch()` had no assertions, meaning it would have silently "passed" regardless of whether Modal actually worked; its third step now genuinely checks deployment status instead of a placeholder. Full merge audit, including independent verification that the user's description of their own changes was accurate, in `AUDIT.md`/`VERIFICATION.md`.

### v1.1.1 — Fixed a real "Particle coordinate is NaN" crash: minimization could still leave severe clashes unresolved before velocities were assigned

Reported against a real Colab run. v1.1.0's fix (minimize before assigning velocities) was directionally correct but used a single fixed-budget minimization pass with no check that it actually converged — for a severely clashing structure (a badly placed docking pose, or tightly packed solvent), that single pass could hit its iteration limit before resolving the overlap, and the still-bad structure then crashed in `setVelocitiesToTemperature()`/the integrator with OpenMM's own opaque, low-level `Particle coordinate is NaN` exception.

**Fixed:** `_minimize_with_retry()` now validates the minimization result immediately, before any velocity is assigned, and retries once with a much larger budget and looser tolerance if the first pass didn't leave a finite state. If the retry also fails, a clear `SimulationError` explains the two realistic causes (bad docked pose; solvation padding too tight) instead of the crash propagating un-diagnosed. 4 new regression tests, all run for real against a fake OpenMM simulation — see `AUDIT.md`/`VERIFICATION.md`. This version also ran the full pre-existing `tests/test_bug_fixes.py` suite for real in this project's sandbox for the first time (29/45 passing; the other 16 need real `pytest` fixtures or OpenMM behavior a lightweight stub can't replicate, unrelated to this fix).

### v1.1.0 — Added optional cloud GPU execution (Modal + Google Colab); fixed a real energy-minimization ordering bug and a solvation force-field mismatch

**Versioning note:** this project now follows [semantic versioning](https://semver.org/) (MAJOR.MINOR.PATCH). Every prior release incremented only the third number, for features and fixes alike; this one is 1.1.0, not 1.0.12, because it adds new functionality. Older version numbers are unchanged from how they were originally published.

**Added**

- `amr_unbind/cloud.py`: optional cloud execution. **Modal** (`submit_modal_run`) — a serverless-GPU backend; the image is built directly from this project's own `Dockerfile.docking` (not a separately maintained dependency list), results sync back into the normal `amr_workspace/runs/` layout, and calling code looks up an already-`modal deploy`ed function rather than redeploying per run. **Google Colab** (`generate_colab_notebook`) — produces a real, runnable `.ipynb` with this run's inputs already embedded (not a link plus a copy-paste snippet), installing from this project's own `requirements-docking.txt` and calling `AMRUnbindPipeline.run()` directly. Neither backend reimplements any docking/simulation logic; both call the same pipeline a local run calls.
- Streamlit app: a new sidebar **Compute** section (**Local** / **Cloud — Modal GPU** / **Cloud — Google Colab notebook**) reusing every existing input field — only where the run happens changes. A Modal run populates the same `st.session_state["last_run"]` a local run does, so the existing results view works unmodified.
- CLI: `--compute {local,modal}`, `--modal-gpu {T4,L4,A10G,A100}`, `--modal-timeout-hours`.
- `requirements-cloud.txt` (just `modal`) and 20 new tests in `tests/test_cloud.py`.
- A Windows `bin/vina.exe` (Apache-2.0, Forli Lab/Scripps Research) is now bundled directly in `bin/`, with attribution added to `bin/README.md`.

**Fixed**

- `SMIRNOFFTemplateGenerator(molecules=off_mol)` (the solvation-box generator in `amr_unbind/simulation.py`) omitted `forcefield=`, so it silently used that class's own default small-molecule force field instead of `config.small_molecule_forcefield` — a parameterization mismatch against the rest of the system. Now `SMIRNOFFTemplateGenerator(molecules=off_mol, forcefield=config.small_molecule_forcefield)`.
- `run_simulation()` assigned Maxwell-Boltzmann velocities *before* any energy minimization had relieved steric clashes in the built system, risking unstable or unphysically large initial velocities. A clash-relief minimization pass now runs immediately before `setVelocitiesToTemperature()`, ahead of the existing (unchanged) post-velocity minimization call.

**Notes:** the reference template supplied for the cloud feature returned fabricated numeric results instead of running anything, called `.remote()` on an undeployed function (which would not have connected to anything), and labeled Modal as "free" (it's metered, pay-per-second beyond free credits). All three were corrected in the implementation actually shipped — see `AUDIT.md`. Modal's real cloud dispatch and a real Colab-runtime execution could not be exercised in this sandbox (no network, no `modal` package, no cloud accounts); see `VERIFICATION.md` for exactly what was and wasn't checked.

### v1.0.11 — Wired the docking-center override through to the pipeline; documented four previously-undocumented SMD/docking fixes

**Fixed**

- **The Streamlit app's "Docking Center Coordinates (Å)" Center X/Y/Z inputs were never actually used.** They were rendered in the sidebar but not referenced anywhere else in `app.py`, and `AMRUnbindPipeline.run()` had no parameter through which any caller could supply a custom docking-box center — automatic docking always computed it from `calculate_pull_geometry()`'s core/exit-residue geometry. `automatic_vina_docking()` already accepted and validated an arbitrary center; the gap was purely in the caller.
- Added `docking_center_angstrom` to `AMRUnbindPipeline.run()` and `_resolve_docking_center_angstrom()` to select it (validated: exactly three finite Å coordinates) over the computed core point when supplied, defaulting to prior behavior otherwise. Wired through the Streamlit app (behind a new opt-in "Override auto-computed pocket center" checkbox) and the CLI (`--docking-center-angstrom X Y Z`). `run_metadata.json` now records `docking_center_angstrom_used` and `docking_center_source` for every run.
- Added 3 regression tests for `_resolve_docking_center_angstrom()`.

**Documented (already present in `amr_unbind/simulation.py`, not previously reflected in `CHANGELOG.md`/`AUDIT.md`/`VERIFICATION.md`)**

- `Molecule.from_file(..., allow_undefined_stereo=True)` when loading the docked ligand SDF, so poses from Vina/Meeko (no explicit chiral flags) no longer raise an OpenFF stereochemistry error.
- Explicit ligand partial charges (`assign_partial_charges(partial_charge_method="gasteiger")`), removing the dependency on an AM1-BCC toolkit (`sqm`/`antechamber`).
- `SMIRNOFFTemplateGenerator` registered directly onto the `ForceField` used for `modeller.addSolvent()` (not only the main `SystemGenerator`'s), which is what actually lets OpenMM recognize the `LIG` residue while building the solvent box. No `GAFFTemplateGenerator`/AmberTools code path remains.
- `ligand_indices` captured immediately after the ligand is added to the topology, before `modeller.addSolvent()` runs — so solvent/ion atoms appended afterward can't be misidentified as ligand atoms.

**Notes:** platform fallback (`CUDA → HIP → OpenCL → CPU`, already in `amr_unbind/platform.py`) and reinstalling `lxml` to fix a corrupted `etree` C-extension are environment/configuration matters, not source changes — see [Troubleshooting](#-troubleshooting). As with every entry in this changelog, `amr_unbind`'s scientific dependencies (Biopython/RDKit/PDBFixer/OpenMM/OpenFF/openmmforcefields) are not installed in the sandbox this audit was performed in; see `VERIFICATION.md`.

### v1.0.10 — Directly fixed the LEU A:455 mechanism the v1.0.8 robustness improvement didn't

**Fixed**

- **`RuntimeError: unable to build rdkit mol for residue LEU corresponding to key A:455` recurred after v1.0.8**, on the exact same residue — confirming a more robust minimization alone wasn't the actual fix needed. Root-caused directly this time: 3F2R's `REMARK 465` records document `GLY A:456`/`VAL A:457` as entirely unresolved, so `LEU A:455` (which has no `REMARK 470` entry — nothing about it is genuinely missing) is only the last residue PDBFixer's topology happens to contain for chain A. PDBFixer/`Modeller`-based atom completion decides whether to add a C-terminal `OXT` purely from topological position, with no way to know the biological chain continues beyond an unresolved region — the `OXT` it adds has no genuine "outward" direction to anchor to, and can end up close enough to a neighbor that Meeko misreads it as bonded.
- Added `_parse_remark_465_missing_residues()`, `_find_pseudo_terminal_residues()`, and `_strip_spurious_terminal_oxt()` — together, these parse which residues a structure's own header documents as entirely absent, flag a chain's last modeled residue as an artificial pseudo-terminus when real residues are documented beyond it, and remove any `OXT` atom PDBFixer added to such a residue before the file is written. Removing the spurious atom outright is a direct fix, rather than continuing to just ask the minimizer to relax it.
- Added 3 regression tests reproducing the exact real header shape and atom layout, confirming a genuine chain terminus is never affected.

**The v1.0.8 minimization and v1.0.9 gap-boundary `TER` insertion both remain active** — this is a fourth, independent correction in `prepare_protein()`, not a replacement for the others. See [Troubleshooting](#dockingerror-meeko-receptor-preparation-failed--atomvalenceexception-explicit-valence-for-atom---is-greater-than-permitted).

### v1.0.9 — Fixed a genuinely different bug class: spurious inter-residue bonds across unmodeled gaps

**Fixed**

- **`DockingError: Meeko receptor preparation failed: No template matched for residue_key='B:103' ... residue_key='B:123' ... matched with excess inter-residue bond(s): B:123`.** This is not the valence/clash issue v1.0.4–v1.0.8 addressed — it's a different mechanism entirely, confirmed by fetching and inspecting the real 3F2R structure directly. Chain B's `REMARK 465` records show residues 104–122 are entirely unresolved, and this project deliberately never rebuilds missing residues. That means `TRP B:103` and `MET B:123` — 20 residues apart in the real protein — end up written to `prepared_protein.pdb` as directly adjacent `ATOM` records, with nothing marking that a real gap separates them. Meeko reads that adjacency as an actual covalent bond, which corrupts its residue-template matching for **both** flanking residues even though neither one's own chemistry is at fault.
- Added `_find_chain_gap_boundaries()` (detects sequence-number jumps between consecutive residues within each chain) and `_insert_ter_at_gap_boundaries()` (inserts a properly formatted `TER` record at each one before `prepared_protein.pdb` is written) — a `TER` record is the standard PDB convention for "the chain breaks here, do not infer a bond across this point."
- Added 5 regression tests, including two that reconstruct the exact reported scenario using the real gap confirmed from 3F2R's own header.

**This is a distinct fix from the v1.0.6/v1.0.8 clash-relief minimization work** — the two address different failure mechanisms (unrelaxed atom clashes vs. spurious inter-residue bonds across gaps) and both remain active. See [Troubleshooting](#-troubleshooting) for what to check if a "No template matched" error still occurs.

### v1.0.8 — Made the clash-relief minimization robust enough to actually resolve severe clashes

**Fixed**

- **A fourth residue failed (`LEU A:455`) after v1.0.7 fixed the tracking bug — but this one is structurally different from the first three.** 3F2R's `REMARK 470` records list no missing atoms for `LEU A:455`, and it's the last residue PDBFixer's topology contains for that chain (the construct continues to residue 457 per its `DBREF`, but 456–457 are entirely unresolved, and this project deliberately never rebuilds missing residues). PDBFixer/`Modeller`-based atom completion determines terminal handling — in particular, whether to add a C-terminal `OXT` atom — from a residue's position in the topology's residue list, not from whether the biological chain actually ends there. A residue that's merely last-before-an-unresolved-gap can be treated the same as a genuine C-terminus, receiving a new atom with no real "outward" direction to be placed toward.
- Rather than special-casing this exact mechanism (which would depend on PDBFixer/`Modeller` internals this project can't verify without the real dependencies installed), `_relax_reconstructed_atoms()` now runs as **two minimization stages** instead of one: a soft restraint first, giving the whole system room to genuinely resolve a severe clash, then the original strong restraint, anchored to the same original positions, to pull the experimentally-observed atoms firmly back while preserving the now-relaxed local geometry. Total iteration budget also increased (500 → 3000). A single-shot minimization with a stiff restraint and a modest iteration cap converges fine for a mild clash and poorly for a severe one — which fits the pattern of intermittent, residue-specific failures across v1.0.6/v1.0.7.
- Added test coverage verifying the exact two-stage sequence executes in the correct order.

**This is a general robustness improvement, not a confirmed fix for the specific `OXT` mechanism above** — see [Troubleshooting](#dockingerror-meeko-receptor-preparation-failed--atomvalenceexception-explicit-valence-for-atom---is-greater-than-permitted) for what to report if a structure still fails.

### v1.0.7 — v1.0.6's fix never actually worked; fixed the tracking mechanism it depended on

**Fixed**

- **`_relax_reconstructed_atoms()` (added in v1.0.6) identified "original" atoms by Python object identity (`id(atom)`), captured before `addMissingAtoms()`/`addMissingHydrogens()` ran — and that assumption was wrong.** `PDBFixer.addMissingHydrogens()` is built on `openmm.app.Modeller`, whose atom-insertion operations construct a **brand new `Topology`** (and therefore brand new `Atom` objects) rather than mutating the existing one in place. Every atom object is recreated, so none of the pre-captured `id()`s ever matched again — on **every single run**, not just ones with missing heavy atoms, since hydrogens are reconstructed unconditionally on every `prepare_protein()` call. This surfaced as `ProteinPreparationError: ... No atoms were identified as part of the original experimental structure`, meaning the v1.0.6 fix, as shipped, could never actually complete a protein preparation that used automatic docking.
- Replaced object-identity tracking with a stable content key (`_atom_identity_key()`: chain id, residue id, insertion code, residue name, atom name), which survives PDBFixer/Modeller rebuilding the topology internally — chain/residue identity has to be preserved across that rebuild regardless, since it's what lets `PDBFile.writeFile(..., keepIds=True)` reproduce the original residue numbering.
- Added a direct regression test that constructs two independent sets of fake Chain/Residue/Atom objects representing the same logical atom and confirms the key matches across them, and rewrote the restraint test to simulate an actual topology rebuild rather than reusing the same objects throughout — the exact gap that let the v1.0.6 bug ship despite having tests for this code.

**If you're running v1.0.6, upgrade to v1.0.7** — v1.0.6's clash-relief step could not complete on any structure needing missing-atom reconstruction or hydrogen placement, which in practice is every structure.

### v1.0.6 — Found and fixed the real root cause behind three "altLoc" crashes

**Fixed**

- **The actual root cause of the recurring `DockingError: Meeko receptor preparation failed: ... AtomValenceException` was never alternate locations.** After v1.0.5 shipped and a *third* different residue failed (SER A:121, following ARG A:89 and GLU A:81 in earlier reports), the target PDB (3F2R) was fetched directly from RCSB for inspection instead of guessing further. Its `REMARK 470 MISSING ATOM` header lists exactly these three residues (and others) as having unresolved side-chain atoms in the deposited structure — confirmed directly in the coordinates (e.g. `GLU A 81` has only N/CA/C/O/CB; no CG/CD/OE1/OE2 at all). `PDBFixer.addMissingAtoms()` reconstructs these atoms using fixed ideal template geometry with **no clash-checking against the rest of the structure** — a documented PDBFixer limitation, not a bug, which is why production structure-preparation pipelines (e.g. Schrödinger's Protein Preparation Wizard) always follow reconstruction with a restrained minimization. This project never did. An unrelaxed reconstructed atom — or, just as often, a newly-placed hydrogen, since X-ray structures at typical resolutions have no experimentally observed hydrogens at all — can sit close enough to a neighbor that Meeko's distance-based bond inference misreads it as an extra bond, producing exactly this failure on whichever residue's reconstruction happens to clash worst.
- `prepare_protein()` now runs a short, restrained vacuum energy minimization (`_relax_reconstructed_atoms()`) immediately after atom/hydrogen reconstruction, before the structure is written or handed to docking. Atoms present in the original experimental structure are identified by object identity (captured before reconstruction) and held with a strong positional restraint; every reconstructed heavy atom and every hydrogen is left free to relax. This adds a small, one-time, CPU-only minimization to every protein preparation (typically seconds), not just structures with `REMARK 470` gaps, since hydrogens are always reconstructed.
- The v1.0.4/v1.0.5 altLoc work is **not reverted** — it remains correct for any structure that genuinely has alternate-location ambiguity — but is now understood to have been chasing a *coincidental correlation* rather than this failure's actual cause: the same flexible surface residues (Arg, Glu, Ser, Lys) prone to genuine altLoc ambiguity are also exactly the ones most likely to have missing side-chain density in the first place.

### v1.0.5 — Closed a real gap in the v1.0.4 altLoc fix

**Fixed**

- **`_strip_alternate_locations()` resolved conflicts per atom name, not per residue.** A second field report hit the same failure class on a *different* residue (GLU A:81, not the original ARG A:89) after v1.0.4 shipped — which pointed at a genuine remaining gap rather than a one-off. The v1.0.4 resolver picked the winning altLoc conformer independently for each atom name; if occupancy isn't perfectly uniform across all of one conformer's atoms within a residue (uncommon but real in some depositions — e.g. `CB` favors conformer A at 0.60/0.40 but `CG` favors conformer B at 0.30/0.70), that could keep atoms from **two different physical conformers** in the same output residue: an unphysical, clashing geometry that can trip a downstream RDKit valence check exactly like the one reported. The resolver now picks one winning conformer **per residue instance** (chain, residue sequence number, insertion code) by mean occupancy across that conformer's atoms, and applies it uniformly to every non-blank-altLoc atom of that residue — so no residue can ever end up with a mix. Also fixed an off-by-one in the insertion-code/residue-sequence-number boundary check that could zero out a valid field on an unusually short line.
- Added two regression tests reproducing the exact failure pattern (cross-atom occupancy inconsistency within one residue; two different residues sharing a residue number distinguished only by insertion code).

**If you already installed v1.0.4 and still see this error class:** first confirm you're actually running v1.0.5 — the app's title bar (top of the Streamlit page) reads `AMR-UNBIND v<version>`. A mismatch almost always means the running process is still using a stale install (a common cause: `pip install .` was used instead of `pip install -e .`, so replacing the source files doesn't change what Python actually imports) or the server was never fully restarted (not just browser-refreshed) after updating. See [Troubleshooting](#-troubleshooting).

### v1.0.4 — Real-world field fixes: altloc docking crash and Windows OpenMP conflict

**Fixed**

- **Docking crash on alternate-location (altLoc) crystallographic conformers.** A real-world PDB with a flexible side chain recorded in two overlapping conformations (a common crystallographic artifact, frequently seen on Arg/Lys/Ser/Val) could crash automatic docking with `DockingError: Meeko receptor preparation failed: ... rdkit.Chem.rdchem.AtomValenceException: Explicit valence for atom # ... is greater than permitted`. `prepare_protein()` now collapses alternate locations to a single, highest-occupancy conformer (ties, or missing occupancy, keep whichever record is listed first — conventionally altLoc `"A"`) before the structure is ever handed to PDBFixer, so neither PDBFixer's nor Meeko's own distance/template-based bond inference ever sees two overlapping copies of the same atom. (Verified along the way: `pipeline.py` already correctly passed `prepared_protein.pdb` — not the raw download — into the docking step; the missing piece was that `prepare_protein()` itself never resolved altLocs before writing that file.)
- **Windows "duplicate OpenMP runtime" crash.** On a Windows conda environment with both an MKL-linked NumPy/OpenMM stack and a separate PyTorch install present, the process could abort at startup with `OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll already initialized`. `amr_unbind/__init__.py` and `app.py` now set `KMP_DUPLICATE_LIB_OK=TRUE` (via `os.environ.setdefault`, so it never overrides a value you've already set yourself) as early as possible — before any OpenMM/OpenFF/RDKit/torch import.

**Changed**

- **Dependency pinning.** `openmm`, `pdbfixer`, and `openff-toolkit` moved from exact `==` pins to `>=`/`<` ranges, since exact pins were observed to fail dependency resolution on some platforms. `openmmforcefields`'s floor deliberately stays at `>=0.16.0` — lowering it further would silently reintroduce the `forcefield_kwargs` incompatibility fixed in v1.0.2, since this project's `SystemGenerator` usage requires 0.16.0's periodic/non-periodic argument split.
- Added `gemmi`, `torch`, and `torchvision` to `requirements.txt` / `environment.yml` / `environment-docking.yml`. `torch`/`torchvision` are not imported by `amr_unbind` itself today (see the [ML roadmap](#-machine-learning-integration--roadmap)); they're included because they are commonly present when extending `amr_unbind.ml`, and because a PyTorch install is the most common real-world trigger of the OpenMP conflict above.

### v1.0.3 — SMD work accuracy and error-handling audit

**Fixed**

- **SMD protocol work accuracy.** `Cumulative Work (kJ/mol)` is now integrated at full integrator resolution (every timestep) instead of only at the reporting interval (previously 100 steps against a default 25,000-step pull, which silently coarsened the trapezoidal work quadrature). The instantaneous steering force is now read every step from the restraint's analytic parameter derivative, avoiding the cost of a full per-step position readout while integrating the true force trajectory.
- **Config-validation error hierarchy.** An invalid `SimulationConfig` is now converted to the package's own `InputValidationError` before a run directory is created, instead of surfacing as a bare, uncaught `ValueError` that bypassed `run_failure.json` reporting and the app's error handler.
- **New physical-validity check.** `solvation_padding_nm` must now exceed `pull_distance_nm` by a minimum safety margin, so the ligand cannot be steered close enough to the periodic box edge to interact with its own periodic image. The Streamlit app exposes a "Solvation box padding" control so this is satisfiable at any pull distance in its slider range.
- **CLI `--platform` bug.** Previously defaulted to `"auto"`, silently overwriting `preferred_platform` from a `--config` file on every run even when `--platform` was never passed. It now only overrides the config when explicitly provided.
- **CLI error handling.** Config construction and the pipeline run are now wrapped in one error handler, printing a clean one-line message and a non-zero exit code instead of a raw Python traceback for any expected failure.
- **CLI docking flags.** Added `--docking-box-angstrom` and `--docking-exhaustiveness` so CLI docking runs have the same control as the Streamlit app.
- **Download timeout.** `download_pdb()` previously had no timeout and could hang indefinitely on a stalled connection; it now uses an explicit 30-second timeout.
- **Run-directory sanitization.** Run-directory names are now sanitized to a safe character set instead of being built directly from an arbitrary local filename.

### v1.0.2 — Docking and OpenMM integration cleanup

- Replaced Streamlit-level monkey patches with explicit docking backends and explicit OpenMM system construction.
- Added a Python-Vina → local-Vina-executable fallback with diagnostics.
- Vina executable discovery supports `AMR_UNBIND_VINA`, `bin/vina(.exe)`, project-root `vina(.exe)`, and `PATH`.
- Removed manual PDBQT generation and text-based PDB/PDBQT cleaning; receptor and ligand PDBQT preparation is delegated entirely to Meeko.
- Docked PDBQT → SDF reconstruction now uses Meeko, not RDKit's PDB parser.
- Corrected `SystemGenerator` periodic/non-periodic nonbonded arguments for openmmforcefields 0.16.0.
- Supplied pose SDFs are copied into the run directory after chemistry validation, for reproducibility.
- Strengthened docking input validation and Vina score parsing.

### v1.0.1 — Bug-fix release

- Fixed ChimeraX trajectory movies: the movie now records explicit `coordset` playback and waits for it to finish before encoding, since simply opening a trajectory does not play it.
- Fixed analysis QC to validate that `Target Coordinate (nm)` exists before accessing it.
- Fixed ML force-slope extraction for constant/near-constant displacement and empty/non-finite inputs.
- Hardened heavy-atom selection against blank/malformed element fields and hydrogen atom names.
- Switched Meeko CLI execution to run through the active Python interpreter rather than relying on a `PATH`-installed wrapper.
- Fixed a simulation runtime error where a step-count variable was referenced before assignment.
- Preserved NPT-updated periodic box vectors when creating the SMD context.
- Ensured SMD trajectory/state reporters use an effective interval that divides the total step count, so the final SMD step is represented in the trajectory.
- Seeded the Monte Carlo barostat for reproducible execution.
- Compatibility pins: Meeko 0.8.0; OpenMM 8.6.0 / PDBFixer 1.12 / OpenFF Toolkit 0.19.0 / openmmforcefields 0.16.0 via conda-forge.

---

## 📚 Scientific References

AMR-UNBIND builds upon established computational chemistry and molecular-dynamics software and methodologies, including:

- [OpenMM](https://openmm.org/) — User Guide 8.6: https://docs.openmm.org/latest/userguide/
- [OpenMM Force Fields / SystemGenerator](https://github.com/openmm/openmmforcefields)
- [Open Force Field Toolkit](https://docs.openforcefield.org/projects/toolkit/en/latest/)
- [OpenFF Sage force fields](https://openforcefield.org/force-fields/force-fields/)
- [PDBFixer](https://github.com/openmm/pdbfixer)
- [AutoDock Vina](https://github.com/ccsb-scripps/AutoDock-Vina)
- [Meeko](https://meeko.readthedocs.io/)
- [ChimeraX developer documentation](https://www.rbvi.ucsf.edu/chimerax/docs/devel/ides_debugging_profiling.html)
- AMBER force fields
- TIP3P water models
- Steered Molecular Dynamics and non-equilibrium work methodologies (e.g. Jarzynski's equality)

Users should cite the relevant underlying software packages, force fields, and methodological literature in addition to AMR-UNBIND when preparing scientific publications.

---

## 🤝 Contributing

Contributions, bug reports, scientific feedback, and improvements are welcome. Recommended contributions include:

- additional validation tests
- simulation-performance improvements
- new analysis methods
- improved visualization
- additional force-field support
- enhanced HPC workflows
- new machine-learning descriptors
- benchmarking against established simulation protocols

Please open an issue or submit a pull request through the GitHub repository.

---

## 📝 Scientific Disclaimer

AMR-UNBIND is a research and computational modeling tool. Simulation outputs depend on structural quality, force-field parameters, system preparation, pulling coordinates, steering parameters, sampling, and other methodological choices.

Computational results should therefore be interpreted as **model-dependent evidence** and, where possible, validated against experimental measurements and established computational methodologies.

---

## 🔐 License

AMR-UNBIND is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for the full license text.

---

## 📄 Citation

If AMR-UNBIND contributes to published research, please cite the software using:

```bibtex
@software{amr_unbind_2026,
  author  = {MK},
  title   = {AMR-UNBIND: Automated Antimicrobial Resistance Ligand Unbinding and Molecular Dynamics Engine},
  year    = {2026},
  version = {1.3.2},
  url     = {https://github.com/MyMKGitH/AMR-UNBIND}
}
```

---

<div align="center">

**AMR-UNBIND**
*Automated Antimicrobial Resistance Ligand Unbinding & Kinetics Engine*

> From molecular docking to dynamic ligand escape.
> From static binding poses to reproducible unbinding trajectories.

</div>
