from __future__ import annotations

from pathlib import Path
import urllib.parse
import urllib.request

from .errors import ChimeraXError


class ChimeraXBridge:
    """Optional bridge to a running ChimeraX REST server."""

    def __init__(self, port: int = 3000, host: str = "127.0.0.1"):
        self.base_url = f"http://{host}:{port}"

    def execute(self, command: str, timeout: float = 20.0) -> str:
        data = urllib.parse.urlencode({"command": command}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/run",
            data=data,
            method="POST",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as response:
                text = response.read().decode("utf-8", errors="replace")
                if response.status >= 400:
                    raise ChimeraXError(
                        f"ChimeraX REST server returned HTTP {response.status}: {text}"
                    )
                return text
        except ChimeraXError:
            raise
        except Exception as exc:
            raise ChimeraXError(
                f"Could not reach ChimeraX REST server at {self.base_url}: {exc}"
            ) from exc

    def render_movie(
        self,
        structure: Path,
        trajectory: Path,
        output_mp4: Path,
        n_frames: int | None = None,
        timeout: float = 180.0,
    ) -> str:
        """Render a multi-frame trajectory as an MP4 using ChimeraX.

        ChimeraX's movie recorder captures graphics frames, so the trajectory
        must be explicitly played with ``coordset`` after ``movie record``.
        ``wait`` ensures playback completes before encoding begins.
        """
        structure = Path(structure)
        trajectory = Path(trajectory)
        output_mp4 = Path(output_mp4)
        if not structure.exists():
            raise ChimeraXError(f"Structure file not found: {structure}")
        if not trajectory.exists():
            raise ChimeraXError(f"Trajectory file not found: {trajectory}")

        def q(path: Path) -> str:
            return '"' + str(path.resolve()).replace('"', '\\"') + '"'

        output_mp4.parent.mkdir(parents=True, exist_ok=True)
        commands = [
            "close session",
            f"open {q(structure)}",
            f"open {q(trajectory)}",
            "cartoon",
            "show atoms",
            "movie record supersample 2",
            "coordset #2 1,-1",
        ]
        if n_frames is not None:
            if int(n_frames) < 1:
                raise ChimeraXError("n_frames must be positive when supplied.")
            commands.append(f"wait {int(n_frames)}")
        else:
            # Conservative fallback for callers that do not know frame count.
            # Preferred callers pass the actual DCD frame count from run metadata.
            commands.append("wait 1000")
        commands.append(f"movie encode {q(output_mp4)} quality medium")

        result = self.execute("; ".join(commands), timeout=timeout)
        if not output_mp4.exists():
            raise ChimeraXError(
                f"ChimeraX returned successfully but did not create the movie: {output_mp4}"
            )
        if output_mp4.stat().st_size == 0:
            raise ChimeraXError(f"ChimeraX created an empty movie file: {output_mp4}")
        return result
