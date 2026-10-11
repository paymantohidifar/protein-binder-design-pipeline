"""One wrapper for every containerised stage.

Each tool runs in its own image, which includes GPU passthrough, 
the shared weight cache, host UID mapping, streamed logs, and a 
non-zero exit that raises.
"""

from __future__ import annotations

import os
import shlex
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path


class ContainerError(RuntimeError):
    """A container exited non-zero.

    Captures container execution failures, printing the exact image name, return code,
    executed command, and the final tail lines of stdout/stderr for immediate debugging.
    """

    def __init__(self, image: str, returncode: int, command: str, tail: str):
        self.image = image
        self.returncode = returncode
        self.tail = tail
        super().__init__(
            f"{image} exited {returncode}\n  command: {command}\n  last output:\n{tail}"
        )


def cache_dir() -> Path:
    """Shared host cache for model weights.

    Determines where large downloaded neural network weights are stored
    on the host system. Defaults to `~/.cache/binder-pipeline` on the user's home partition
    unless overridden by the `BINDER_CACHE` environment variable.
    """
    return Path(os.environ.get("BINDER_CACHE", Path.home() / ".cache/binder-pipeline"))


@dataclass
class Mount:
    """Represents a volume mount bind between host system and container filesystem."""

    host: Path
    container: str
    read_only: bool = False

    def as_flag(self) -> str:
        """Formats the mount point as a standard Docker `-v` argument string.

        Resolves host path to absolute representation and appends `:ro` if read-only.
        Example output: `/home/user/project/data:/work:ro`
        """
        mode = ":ro" if self.read_only else ""
        return f"{self.host.resolve()}:{self.container}{mode}"


@dataclass
class ContainerRun:
    """Configures the execution specification for launching a Docker container stage."""

    image: str
    command: list[str]
    mounts: list[Mount] = field(default_factory=list)
    workdir: str | None = None
    gpus: bool = True
    env: dict[str, str] = field(default_factory=dict)
    map_user: bool = True
    shm_size: str = "8g"
    entrypoint: str | None = None

    def argv(self) -> list[str]:
        """Constructs the full `docker run` command array for subprocess execution.

        Configures runtime flags:
        - `--rm`: Automatically cleans up container filesystem on exit
        - `--gpus all`: Enables GPU device passthrough via NVIDIA Container Toolkit
        - `--user UID:GID`: Maps container user/group permissions to current host user so
          written output files stay owned by host user, avoiding root-owned file locks
        - `--shm-size 8g`: Expands shared memory space (default 64MB breaks PyTorch dataloaders)
        - `-v /cache`: Binds host weight cache to container `/cache` path
        - Environment overrides: Redirects HuggingFace (`HF_HOME`) and PyTorch (`TORCH_HOME`)
        """
        argv = ["docker", "run", "--rm"]
        if self.gpus:
            argv += ["--gpus", "all"]
        if self.map_user:
            argv += ["--user", f"{os.getuid()}:{os.getgid()}"]
        # PyTorch dataloaders segfault on the 64 MB default /dev/shm.
        argv += ["--shm-size", self.shm_size]

        # Automatically ensure host cache folder exists and bind it to /cache in the container
        cache = cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        argv += ["-v", Mount(cache, "/cache").as_flag()]

        # Process custom user mounts
        for mount in self.mounts:
            argv += ["-v", mount.as_flag()]

        # Set environment variables for weight redirection
        for key, value in {
            "HF_HOME": "/cache/hf",
            "TORCH_HOME": "/cache/torch",
            **self.env,
        }.items():
            argv += ["-e", f"{key}={value}"]

        if self.workdir:
            argv += ["-w", self.workdir]
        if self.entrypoint:
            argv += ["--entrypoint", self.entrypoint]

        # Append image tag and execution commands to the argument list
        return argv + [self.image] + self.command

    def as_shell(self) -> str:
        """Returns the full CLI command formatted as a shell-safe string for printing/logging."""
        return shlex.join(self.argv())


def run(
    spec: ContainerRun,
    dry_run: bool = False,
    stream: bool = True,
    tail_lines: int = 40,
) -> str:
    """Execute a container, returning its combined output.

    With dry_run=True the command is printed and not executed -- which is how
    the notebook is validated on a machine with no GPU.
    Streams logs line-by-line in real time to standard stdout while accumulating full terminal history.
    If the container exits non-zero, raises `ContainerError` containing error log tail.
    """
    command = spec.as_shell()
    if dry_run:
        print(f"[dry-run] {command}")
        return ""

    print(f"[run] {spec.image}: {shlex.join(spec.command)}", flush=True)

    # Launch container process with unbuffered line streaming
    # Note: while subprocess.run() wait for a command to finish before returning 
    # control to the script, Popen() opens a process asynchronously in the 
    # background and gives user a process object (proc) to interact with while it runs.
    proc = subprocess.Popen(
        spec.argv(),            # Array of command arguments: ["docker", "run", ...]
        stdout=subprocess.PIPE, # Intercept standard output
        stderr=subprocess.STDOUT,# Redirect standard error into standard output
        text=True,              # Decode output streams as UTF-8 text strings (not bytes)
        bufsize=1,              # Line-buffered output for immediate real-time streaming
        )
    lines: list[str] = []
    assert proc.stdout is not None

    # Stream container logs live to host stdout while saving lines to array
    for line in proc.stdout:
        lines.append(line)
        if stream:
            sys.stdout.write(line)
            sys.stdout.flush()

    returncode = proc.wait()

    output = "".join(lines)

    # Raise exception if container failed, formatting tail logs
    if returncode != 0:
        raise ContainerError(
            spec.image, returncode, command, "".join(lines[-tail_lines:])
        )
    return output


def gpu_available() -> bool:
    """Checks whether host has a working NVIDIA GPU driver installation via `nvidia-smi`."""
    try:
        subprocess.run(["nvidia-smi"], capture_output=True, check=True)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False


def check_gpu_compatibility(image: str) -> dict:
    """Verify the image's CUDA build actually supports the host GPU.

    The failure this exists to catch: RFdiffusion's official container pins
    PyTorch 1.12 / CUDA 11.x, which has no sm_90 kernels, so it cannot run on an
    H100. PyTorch reports this as "no kernel image is available for execution"
    only once a tensor reaches the GPU -- which can be minutes into a design
    run. Allocating one tensor up front turns that into an immediate, legible
    failure.
    """
    # Probe script: checks compute capability and forces a dummy tensor allocation onto GPU memory
    probe = (
        "import torch;"
        "cap=torch.cuda.get_device_capability();"
        "torch.zeros(1).cuda();"
        "print(f'OK {torch.cuda.get_device_name(0)} sm_{cap[0]}{cap[1]} "
        "torch={torch.__version__} cuda={torch.version.cuda}')"
    )
    # entrypoint="python" is essential: without it these args land as hydra
    # arguments to run_inference.py and the probe never executes.
    spec = ContainerRun(
        image=image,
        command=["-c", probe],
        entrypoint="python",
        map_user=False,  # Run probe as container user to avoid permission conflicts on probe
    )
    try:
        output = run(spec, stream=False)
    except ContainerError as exc:
        return {"image": image, "ok": False, "detail": exc.tail.strip()}
    return {
        "image": image,
        "ok": True,
        "detail": output.strip().splitlines()[-1],
    }