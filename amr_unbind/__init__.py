"""AMR-UNBIND: Antimicrobial Resistance Ligand Unbinding workflow."""

import os

# On Windows conda environments it's common to end up with more than one
# bundled copy of the Intel OpenMP runtime (libiomp5md.dll) once both a
# PyTorch install and an MKL-linked NumPy/OpenMM stack are present in the
# same interpreter. Intel's OpenMP runtime aborts the process on startup
# ("OMP: Error #15: Initializing libiomp5md.dll, but found libiomp5md.dll
# already initialized") rather than risk two runtimes silently sharing
# thread-pool state. This is set as early as possible -- before any
# submodule of this package (and therefore before OpenMM/OpenFF/RDKit/torch)
# is imported -- since the check only matters at first load. It is the
# workaround Intel, Anaconda, and PyTorch's own docs recommend for this
# exact conflict; the (small, largely theoretical) tradeoff is that a truly
# broken environment will keep running instead of aborting loudly.
# `setdefault` is used so an explicit value the user has already set (e.g.
# to deliberately re-enable the check once the underlying conflict is fixed)
# is never overridden.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

__version__ = "1.3.2"
