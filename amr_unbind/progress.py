from __future__ import annotations

"""Progress reporting utilities shared across the pipeline.

Two problems this module exists to fix, both reported against a real run
(explicit-solvent SMD on PDB 1STP, run from Colab):

1. **Non-monotonic progress.** Every stage of the pipeline (protein prep,
   ligand prep, docking, system build, minimization, NVT, NPT, SMD) used
   to report its own, independently-chosen fraction, with no shared
   budget -- so the reported progress could visibly jump backward (e.g.
   "system-built" at 15%, then "minimized" at 2%) as control moved
   between amr_unbind/pipeline.py and amr_unbind/simulation.py. A person
   watching a progress bar do that has no way to tell real progress from
   the bar simply being wrong, and "stuck around 15%" is exactly what a
   backward jump into the low single digits, followed by a slow climb
   back up through equilibration, would look like from a quick glance.
   The fix for that is in pipeline.py/simulation.py/docking.py directly
   (a single, hand-maintained, monotonically increasing fraction budget
   across the whole run) -- this module doesn't own that budget, but the
   Heartbeat class below assumes whatever fraction it's given already
   fits into it.

2. **Total silence during a single long blocking call.** Vina docking,
   Meeko receptor/ligand preparation, and (before this version) an entire
   NVT or NPT equilibration phase are each one blocking call with no
   hook to report progress *during* them. A slow one (a heavily-loaded
   free Colab CPU can make Vina docking take a very long time) looks
   identical, from the outside, to a genuinely hung one: the last message
   on screen doesn't change either way. `Heartbeat` re-invokes the
   progress callback every `interval_s` seconds for the duration of a
   `with` block, reporting elapsed time (and CPU/memory usage, if
   `psutil` is installed) so "still working" and "no longer moving" are
   visibly different.
"""

import threading
import time
from typing import Callable, Optional

# A progress callback is called as callback(stage, fraction, detail).
# `detail` is always passed; every callback implementation in this project
# accepts it (with a default so older/external callers that only take two
# positional arguments still work -- see _safe_call below for the runtime
# fallback that covers a caller this project doesn't control).
ProgressCallback = Callable[..., None]


def format_elapsed(seconds: float) -> str:
    """A short, human-readable duration: "42s", "3m 05s", "1h 12m"."""
    total = max(0, int(seconds))
    if total < 60:
        return f"{total}s"
    minutes, secs = divmod(total, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def resource_usage_snapshot() -> Optional[str]:
    """A short "CPU 210% | RAM 3.2/12.7 GB" string for this process and
    machine, or None if `psutil` isn't installed. Purely informational --
    never raises, and its absence never blocks anything else in this
    module. `psutil` is optional (see requirements.txt) specifically so
    this stays true on any install that doesn't have it.
    """
    try:
        import psutil
    except ImportError:
        return None
    try:
        process = psutil.Process()
        cpu_percent = process.cpu_percent(interval=0.1)
        mem = psutil.virtual_memory()
        used_gb = mem.used / (1024**3)
        total_gb = mem.total / (1024**3)
        return f"CPU {cpu_percent:.0f}% | RAM {used_gb:.1f}/{total_gb:.1f} GB"
    except Exception:
        return None


def safe_call(callback: Optional[ProgressCallback], stage: str, fraction: float, detail: str = "") -> None:
    """Invoke callback(stage, fraction, detail), tolerating both a
    callback that only accepts (stage, fraction) -- for any external
    caller this project doesn't control -- and any exception the callback
    itself raises. A progress report must never be the thing that breaks
    a run.
    """
    if callback is None:
        return
    try:
        callback(stage, fraction, detail)
    except TypeError:
        try:
            callback(stage, fraction)
        except Exception:
            pass
    except Exception:
        pass


# Old private name, kept as an alias: progress.py's own Heartbeat class (and
# any external code that imported the old name) both keep working.
_safe_call = safe_call


# The full pipeline's progress budget, in stage order -- the single source
# of truth every module that reports progress (pipeline.py, docking.py,
# simulation.py) imports its boundaries from, rather than each hardcoding
# its own numbers. Scattered, independently-chosen fractions across files
# is exactly what caused reported progress to visibly jump backward before
# this table existed (e.g. "system-built" at 15%, then "minimized" at 2%
# immediately after, because simulation.py's stages used their own
# unrelated 0-1 scale) -- see this file's module docstring.
#
# Each constant is where that stage's raw work reaches 100% complete; a
# stage's own progress reports climb from the previous constant up to its
# own. Keep these three files' stage order and this list in sync:
# pipeline.py (protein-prepared, ligand-prepared, docking, system-built),
# simulation.py (minimized, nvt, npt, smd), and this comment.
PROTEIN_PREPARED_FRACTION = 0.05
LIGAND_PREPARED_FRACTION = 0.08
DOCKING_FRACTION = 0.10  # flat during docking's heartbeat -- Vina/Meeko report no sub-progress of their own
SYSTEM_BUILT_FRACTION = 0.20
MINIMIZED_FRACTION = 0.22
NVT_FRACTION = 0.35
NPT_FRACTION = 0.48
SMD_FRACTION = 0.98
COMPLETE_FRACTION = 1.0


class Heartbeat:
    """Context manager: while the ``with`` block runs, re-invokes
    ``progress_callback(stage, fraction, detail)`` every ``interval_s``
    seconds with an updated elapsed-time (and CPU/memory, if available)
    message, so a single long blocking call never leaves the UI showing
    the exact same message for its whole duration.

    ``fraction`` stays fixed at whatever value is passed in -- a
    heartbeat reports that a stage is *still going*, not how far through
    it is, since the wrapped call (Vina docking, Meeko preparation, an
    unchunked minimization) has no intermediate progress of its own to
    report. Runs in a daemon thread; safe to use even if
    ``progress_callback`` is ``None`` (a no-op then) or raises.
    """

    def __init__(
        self,
        progress_callback: Optional[ProgressCallback],
        stage: str,
        fraction: float,
        *,
        interval_s: float = 20.0,
        note: str = "",
    ) -> None:
        self._progress_callback = progress_callback
        self._stage = stage
        self._fraction = fraction
        self._interval_s = interval_s
        self._note = note
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._start_time = 0.0

    def _tick(self) -> None:
        elapsed = time.time() - self._start_time
        parts = [f"still running \u2014 {format_elapsed(elapsed)} elapsed"]
        usage = resource_usage_snapshot()
        if usage:
            parts.append(usage)
        if self._note:
            parts.append(self._note)
        _safe_call(self._progress_callback, self._stage, self._fraction, " | ".join(parts))

    def _run(self) -> None:
        while not self._stop.wait(self._interval_s):
            self._tick()

    def __enter__(self) -> "Heartbeat":
        self._start_time = time.time()
        if self._progress_callback is not None:
            # An immediate tick means the UI shows "still running -- 0s
            # elapsed" right away, rather than nothing until the first
            # interval passes -- useful confirmation this stage actually
            # started using the heartbeat, not just silence again.
            _safe_call(self._progress_callback, self._stage, self._fraction, self._note or "starting")
            self._thread = threading.Thread(target=self._run, daemon=True)
            self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
