"""File-backed logging infrastructure for containerized execution pipelines.

GPU-accelerated inference stages generate voluminous console output—RFdiffusion
emits progress per denoising step per trajectory, while ColabFold and Boltz-2
log per-recycle metrics. Streaming raw stdout/stderr directly into an interactive
notebook clutters cell outputs, inflates `.ipynb` file sizes, and is lost upon
kernel restarts.

To ensure reproducibility and clean execution records, raw container streams are
redirected to disk, while the notebook receives only high-level status updates:
the container image invoked, execution runtime, exit codes, and log file paths.

Execution logging is routed to two distinct file sinks:

1. The Campaign Event Log (`<log_dir>/pipeline.log`):
   - A persistent, append-only timeline managed via the Python standard `logging`
     module.
   - Captures structured event records across sessions, preserving instance-wide
     execution history across kernel restarts.

2. Per-Run Execution Transcripts (`<log_dir>/<timestamp>-<image_tag>.log`):
   - Verbatim byte-for-byte captures of stdout and stderr for an individual container run.
   - Formatted with a header recording the exact `docker run` argument vector (argv),
     environment variables, and volume mounts for debugging and auditing.

Architectural Decoupling:
    This module operates independently of `pipeline.docker`. The `docker.run` wrapper
    invokes `runlog` functionality, maintaining strict separation of concerns and
    allowing isolated unit testing without requiring a running Docker daemon.

Notebook Usage Example:

    from pipeline import runlog

    # Initialize log directory routing from campaign config
    runlog.configure(REPO_ROOT / CONFIG["paths"]["logs"])

    # Subsequent container invocations via docker.run are automatically recorded.
    # On execution failure, inspect the trailing transcript output:
    runlog.tail(n=50)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

#: Env override, checked before the configured directory. Set it to redirect
#: logs on a shared instance without editing config.
LOG_DIR_ENV = "BINDER_LOG_DIR"

#: Used when neither the env var nor configure() has set a directory. Relative,
#: so it lands under whatever the process's cwd is -- the repo root, for both
#: the notebook and the dry-run harness.
DEFAULT_LOG_DIR = Path("results/logs")

#: The campaign timeline, inside the log directory.
CAMPAIGN_LOG = "pipeline.log"

#: Name of the package logger. Every module logs through a child of this.
LOGGER_NAME = "binder"

#: How many transcript lines each RunLog keeps in memory for an error tail.
#: ContainerError quotes at most 40; the extra headroom costs nothing and
#: covers a traceback that outruns its own last frame.
TAIL_BUFFER = 200

# Module state. _log_dir is None until configure() or the first log_dir() call
# resolves it, so importing this module creates no directories -- importing
# pipeline must stay free of side effects for the tests.
_log_dir: Path | None = None
_configured = False
_last_transcript: Path | None = None


def log_dir() -> Path:
    """Directory holding the campaign log and every run transcript.

    Resolution order: `$BINDER_LOG_DIR`, then whatever `configure()` was given,
    then `DEFAULT_LOG_DIR`. The env var wins over `configure()` on purpose --
    it is the lever available when the notebook cannot be edited.
    """
    env = os.environ.get(LOG_DIR_ENV)
    if env:
        return Path(env).expanduser()
    if _log_dir is not None:
        return _log_dir
    return DEFAULT_LOG_DIR


def configure(
    directory: str | Path | None = None,
    *,
    console: bool = True,
    level: int = logging.INFO,
    force: bool = False,
) -> Path:
    """Point logging at `directory` and attach handlers. Returns the directory.

    Idempotent: calling it twice does not double every line, which re-running a
    notebook cell otherwise would. Pass `force=True` to move the log directory
    mid-session.

    `console=False` silences even the one-line-per-event stream, for a
    SnakeMake run whose own driver already reports progress.
    """
    global _log_dir, _configured

    if directory is not None:
        _log_dir = Path(directory).expanduser()

    target = log_dir()
    target.mkdir(parents=True, exist_ok=True)

    log = logging.getLogger(LOGGER_NAME)
    log.setLevel(level)
    # Events belong to this logger's handlers only. Without this, a notebook
    # that has called logging.basicConfig() (or any library that has) gets
    # every line a second time from the root handler.
    log.propagate = False

    if _configured and not force:
        return target

    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()

    # The campaign log is appended, never truncated: a kernel restart mid
    # campaign must not erase what already ran.
    file_handler = logging.FileHandler(target / CAMPAIGN_LOG, encoding="utf-8")
    file_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s  %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
    )
    log.addHandler(file_handler)

    if console:
        # Bare message, no timestamp or level: this is the stream a human
        # watches in a notebook cell, and it should read like the progress
        # prints it replaces. stdout rather than the default stderr so the
        # ordering against ordinary print() output is preserved.
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        log.addHandler(console_handler)

    _configured = True
    return target


def logger(name: str | None = None) -> logging.Logger:
    """The package logger, or a named child of it.

    Safe to call before `configure()`: a logger with no handlers emits nothing
    but the stdlib's own last-resort warning, and `docker.run` configures
    lazily anyway.
    """
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


def _slug(image: str) -> str:
    """`binder-boltz2:latest` -> `binder-boltz2`. Filesystem-safe, tag dropped.

    The tag is always `latest` by convention and carries no information, while
    `:` is awkward in a filename on every platform.
    """
    stem = image.split(":", 1)[0].rsplit("/", 1)[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", stem).strip("-") or "container"


def transcript_path(image: str, label: str | None = None) -> Path:
    """An unused path for one run's transcript, inside `log_dir()`.

    Second resolution is not enough -- the ProteinMPNN helper scripts run back
    to back and would collide -- so a counter is appended when needed rather
    than silently overwriting the earlier run's transcript.
    """
    directory = log_dir()
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    parts = [stamp, _slug(image)] + ([_slug(label)] if label else [])
    base = "-".join(parts)

    candidate = directory / f"{base}.log"
    counter = 2
    while candidate.exists():
        candidate = directory / f"{base}-{counter}.log"
        counter += 1
    return candidate


def last_transcript() -> Path | None:
    """Path of the most recent transcript opened in this process, if any."""
    return _last_transcript


def tail(lines: int = 40, path: str | Path | None = None) -> str:
    """Return (and print) the last `lines` of a transcript.

    Defaults to the most recent run, which is what a failed stage wants. Reads
    the whole file: transcripts are tens of MB at worst, and seeking backwards
    for a line boundary is not worth the bug surface.
    """
    target = Path(path) if path is not None else _last_transcript
    if target is None:
        text = "no transcript yet in this session"
    elif not Path(target).exists():
        text = f"transcript missing: {target}"
    else:
        text = "".join(Path(target).read_text(errors="replace").splitlines(True)[-lines:])
    print(text)
    return text


@dataclass
class RunLog:
    """One container run's transcript file, plus an in-memory tail.

    Used as a context manager so the file handle closes on an exception as well
    as on success; the transcript of a crashed run is the one that matters.
    """

    image: str
    path: Path
    command: str = ""
    echo: bool = False
    _handle: object | None = field(default=None, repr=False)
    _buffer: deque = field(default_factory=lambda: deque(maxlen=TAIL_BUFFER), repr=False)
    _started: float = field(default=0.0, repr=False)
    lines_written: int = 0

    def __enter__(self) -> RunLog:
        global _last_transcript
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", encoding="utf-8")
        self._started = time.monotonic()
        _last_transcript = self.path
        # The header makes the transcript self-contained: the exact argv is
        # what a reader needs to reproduce the run, and a transcript that has
        # been copied out of the log directory has nothing else to go on.
        self._handle.write(f"# image   : {self.image}\n")
        self._handle.write(f"# started : {datetime.now().isoformat(timespec='seconds')}\n")
        if self.command:
            self._handle.write(f"# command : {self.command}\n")
        self._handle.write("#" + "-" * 70 + "\n")
        self._handle.flush()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._handle is not None:
            self._handle.write(
                f"#{'-' * 70}\n# elapsed : {self.elapsed:.1f}s"
                f"  ({self.lines_written} lines)\n"
            )
            self._handle.close()
            self._handle = None
        return False  # never swallow ContainerError

    @property
    def elapsed(self) -> float:
        return time.monotonic() - self._started if self._started else 0.0

    def write(self, line: str) -> None:
        """Append one line of container output.

        Flushed per line, not per buffer: a stage that hangs or is killed by the
        OOM reaper must still leave the lines it had already produced on disk,
        which is exactly the case a buffered writer loses.
        """
        self._buffer.append(line)
        self.lines_written += 1
        if self._handle is not None:
            self._handle.write(line)
            self._handle.flush()
        if self.echo:
            sys.stdout.write(line)
            sys.stdout.flush()

    def tail(self, lines: int = 40) -> str:
        """The last `lines` of output held in memory, for a ContainerError."""
        return "".join(list(self._buffer)[-lines:])
