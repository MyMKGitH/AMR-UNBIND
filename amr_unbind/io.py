from __future__ import annotations

from pathlib import Path
from uuid import uuid4
import json
import re
from datetime import datetime, timezone

# target is a PDB ID (validated as 4 alphanumeric characters elsewhere) or,
# for a locally supplied PDB file, an arbitrary filename stem. Sanitize it
# before using it as a run-directory path component so an unusual filename
# cannot produce a broken or unexpected directory name.
_UNSAFE_RUN_LABEL_CHARS = re.compile(r"[^a-z0-9._-]+")


def _sanitize_run_label(target: str) -> str:
    label = _UNSAFE_RUN_LABEL_CHARS.sub("-", target.strip().lower()).strip("-._")
    return label or "target"


def create_run_dir(root: Path, target: str) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{stamp}_{_sanitize_run_label(target)}_{uuid4().hex[:8]}"
    path = root / "runs" / run_id
    path.mkdir(parents=True, exist_ok=False)
    return path


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, default=str), encoding="utf-8")
