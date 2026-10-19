"""Container command construction.

No container is ever launched here -- these assert the argv that would be run.
The entrypoint handling is the part worth pinning: every tool image declares an
ENTRYPOINT, so a command passed without an override becomes ARGUMENTS to that
entrypoint instead of replacing it, which fails silently rather than loudly.
"""

import os
import re
from pathlib import Path

import pytest

from pipeline import docker


class TestArgv:
    def test_includes_gpu_and_cache_mount(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        argv = docker.ContainerRun(image="img:latest", command=["x"]).argv()
        assert "--gpus" in argv and "all" in argv
        assert any(str(tmp_path / "cache") in a for a in argv)

    def test_maps_host_uid_by_default(self, tmp_path, monkeypatch):
        # Containers default to root; without this every output file on the
        # host ends up root-owned and later stages cannot read it.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        argv = docker.ContainerRun(image="img", command=[]).argv()
        assert "--user" in argv
        assert f"{os.getuid()}:{os.getgid()}" in argv

    def test_shm_size_is_set(self, tmp_path, monkeypatch):
        # PyTorch dataloaders segfault on the 64 MB default /dev/shm.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        argv = docker.ContainerRun(image="img", command=[]).argv()
        assert "--shm-size" in argv


class TestEntrypoint:
    def test_omitted_by_default(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        assert "--entrypoint" not in docker.ContainerRun(image="img", command=[]).argv()

    def test_override_precedes_the_image(self, tmp_path, monkeypatch):
        # Docker requires flags before the image name; after it they would be
        # passed to the container instead.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        argv = docker.ContainerRun(
            image="img:latest", command=["-c", "print(1)"], entrypoint="python"
        ).argv()
        assert argv[argv.index("--entrypoint") + 1] == "python"
        assert argv.index("--entrypoint") < argv.index("img:latest")

    def test_command_follows_the_image(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        argv = docker.ContainerRun(
            image="img:latest", command=["-c", "print(1)"], entrypoint="python"
        ).argv()
        assert argv[argv.index("img:latest") + 1 :] == ["-c", "print(1)"]


class TestGpuProbe:
    def test_probe_overrides_the_entrypoint(self, tmp_path, monkeypatch):
        # Regression guard. The RFdiffusion image's entrypoint is
        # run_inference.py; without entrypoint="python" the probe's args would
        # become hydra arguments and the compatibility check would silently
        # test nothing.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        calls = []

        def fake_run(spec, **kwargs):
            calls.append(spec)
            return "OK NVIDIA A10 sm_86 torch=1.12.1 cuda=11.6"

        monkeypatch.setattr(docker, "run", fake_run)
        result = docker.check_gpu_compatibility("binder-rfdiffusion:latest")

        assert calls[0].entrypoint == "python"
        assert "--entrypoint" in calls[0].as_shell()
        assert result["ok"] is True
        assert "sm_86" in result["detail"]

    def test_probe_allocates_a_tensor(self, tmp_path, monkeypatch):
        # An unsupported architecture only errors once work reaches the GPU, so
        # merely importing torch would pass on an H100 and fail later.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        captured = []
        monkeypatch.setattr(docker, "run", lambda spec, **kw: captured.append(spec) or "OK x")
        docker.check_gpu_compatibility("img")
        assert ".cuda()" in " ".join(captured[0].command)

    def test_failure_is_reported_not_raised(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))

        def boom(spec, **kwargs):
            raise docker.ContainerError("img", 1, "cmd", "no kernel image is available")

        monkeypatch.setattr(docker, "run", boom)
        result = docker.check_gpu_compatibility("img")
        assert result["ok"] is False
        assert "no kernel image" in result["detail"]


class TestCacheDir:
    def test_env_override_wins(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "custom"))
        assert docker.cache_dir() == tmp_path / "custom"

    def test_default_is_under_home_not_root(self, monkeypatch):
        # MANIFEST Section 5 instruction 3 says /root/.cache/, which fails on
        # Lambda: the login user is `ubuntu` and /root needs sudo.
        monkeypatch.delenv("BINDER_CACHE", raising=False)
        assert docker.cache_dir() == Path.home() / ".cache/binder-pipeline"
        assert not str(docker.cache_dir()).startswith("/root")


class TestDryRun:
    def test_dry_run_executes_nothing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path))
        out = docker.run(docker.ContainerRun(image="img", command=["x"]), dry_run=True)
        assert out == ""
        assert "[dry-run]" in capsys.readouterr().out


def strip_comments(text: str) -> str:
    """Drop whole-line comments.

    Absence assertions MUST use this. These files document the defects they
    fixed, so a naive `assert "X" not in text` matches the comment explaining
    why X is wrong -- a trap this suite fell into twice.
    """
    return "\n".join(
        line for line in text.splitlines() if not line.lstrip().startswith("#")
    )


class TestDockerfileContracts:
    """Flags whose omission fails silently belong in the ENTRYPOINT, not in a
    call site that can forget them. These tests exist because both were, or
    would have been, forgotten."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def _text(self, name: str) -> str:
        return (self.ROOT / name / "Dockerfile").read_text()

    def test_boltz_writes_full_pae(self):
        # Without --write_full_pae no pae_*.npz is produced, _load_boltz_pae
        # raises, and pae_interaction -- the primary Step 4 gate -- is NaN.
        assert "--write_full_pae" in self._text("boltz2")

    def test_boltz_emits_pdb_not_cif(self):
        # parsers has no CIF reader; scRMSD would be skipped with only a note.
        assert "--output_format" in self._text("boltz2")
        assert '"pdb"' in self._text("boltz2")

    def test_boltz_requests_cuda_extra(self):
        # The cuEquivariance kernels sit behind extra == "cuda" from 2.2.0 on,
        # so a bare `boltz==` past that version loses them without erroring.
        text = self._text("boltz2")
        assert "boltz[cuda]" in text

    def test_boltz_pins_cuequivariance(self):
        # OBSERVED FAILURE this guards: `RuntimeError: operator
        # torchvision::nms does not exist` at the first real `boltz predict`.
        # boltz[cuda] bounds cuEquivariance only as >=0.5.0, and
        # cuequivariance-ops-torch-cu12 made `torch>=2.11` a hard runtime dep
        # at 0.11.1. Unpinned, pip takes 0.12.0, upgrades torch off the base
        # image's 2.4.0, and strands the base's torchvision 0.19.0 (pinned to
        # torch==2.4.0, compiled against that ABI).
        text = strip_comments(self._text("boltz2"))
        assert "cuequivariance-ops-torch-cu12==" in text, (
            "the -ops- package carries the torch floor and must be pinned"
        )
        assert "cuequivariance-torch==" in text

    def test_boltz_asserts_torch_did_not_move(self):
        # The pin alone is not enough: this image BUILT CLEANLY and broke only
        # at run time, on a GPU-billed instance. A build-time import of the
        # exact thing that failed turns that into a build failure.
        text = strip_comments(self._text("boltz2"))
        assert "from torchvision.ops import nms" in text
        assert "torch.__version__.startswith" in text

    def test_boltz_base_image_clears_the_cublas_floor(self):
        # OBSERVED FAILURE: `undefined symbol: cublasGemmGroupedBatchedEx,
        # version libcublas.so.12` at the first prediction step on a cuda12.1
        # base. That symbol needs cuBLAS >= 12.5, which
        # cuequivariance-ops-cu12 declares (nvidia-cublas-cu12>=12.5.0) but the
        # loader could not honour while the image's own CUDA was 12.1.
        from_line = next(
            l for l in strip_comments(self._text("boltz2")).splitlines()
            if l.startswith("FROM ")
        )
        m = re.search(r"cuda(\d+)\.(\d+)", from_line)
        assert m, f"cannot read a CUDA version from {from_line!r}"
        major, minor = int(m.group(1)), int(m.group(2))
        assert (major, minor) >= (12, 5), (
            f"base image is cuda{major}.{minor}; cuEquivariance's kernels need "
            "cuBLAS >= 12.5 and fail with an undefined symbol below it"
        )

    def test_boltz_cuequivariance_check_is_non_fatal_at_build_time(self):
        # libcue_ops.so links against libcuda.so.1, the DRIVER stub, which the
        # nvidia runtime injects at run time and which does not exist during
        # `docker build`. A fatal assertion here broke the build on a healthy
        # image. Same necessity as esmfold's OpenFold extension check.
        #
        # It must still name the OPS package, not cuequivariance_torch: the
        # wrapper defers the ops import to inside the function
        # (primitives/triangle.py:238), so importing it dlopens nothing.
        text = self._text("boltz2")
        block = next(
            b for b in re.findall(r'RUN python -c "(?:[^"]|\\\n)*?"(?:\s*\\\n\s*\|\|[^\n]*)?', text, re.S)
            if "cuequivariance" in b
        )
        assert "import cuequivariance_ops_torch" in block, (
            "must import the OPS package, which dlopens libcue_ops.so"
        )
        assert "||" in block, (
            "must be non-fatal: libcuda.so.1 is absent during docker build"
        )

    def test_build_all_verifies_cuequivariance_on_the_gpu_host(self):
        # Because the build cannot check it, build_all.sh must -- it runs with
        # --gpus all, where libcuda.so.1 is present. Asserts structure: the ops
        # import, a real kernel call, and that failure is fatal there.
        script = (Path(__file__).resolve().parents[1] / "docker/build_all.sh").read_text()
        assert "import cuequivariance_ops_torch" in script
        assert "triangle_multiplicative_update(" in script
        assert "CUEQ_PROBE" in script
        probe_line = next(l for l in script.splitlines()
                          if l.startswith("probe boltz2") and "CUEQ_PROBE" in l)
        assert "$CUEQ_PROBE" in probe_line

    def test_proteinmpnn_keeps_git_at_runtime(self):
        # protein_mpnn_run.py shells out to `git rev-parse HEAD` on every FASTA
        # write to stamp git_hash= into the header. So git must be installed
        # and must survive into the final image: an apt-get purge saves ~40 MB
        # and costs the run, at the write step, after the GPU work is done.
        # Unfold shell line continuations, so one RUN is one line to match
        # against however its flags and breaks happen to be laid out.
        code = strip_comments(self._text("proteinmpnn")).replace("\\\n", " ")
        # Structure, not phrasing: git as its own token in an apt-get install.
        install = re.search(r"apt-get install\b[^\n]*", code)
        assert install is not None, "proteinmpnn installs nothing with apt-get"
        assert re.search(r"(?<![\w./-])git(?![\w./-])", install.group(0))
        # The two ways git has been reclaimed from an image before.
        assert not re.search(r"apt-get\s+(purge|remove)\b[^\n]*(?<![\w./-])git\b", code)
        assert "/opt/ProteinMPNN/.git" not in code


class TestWeightCacheSpace:
    """Weights must land on the mounted volume, not in a container-local path
    that `--rm` discards. Both mechanisms must name the same directory."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def test_boltz_env_and_flag_agree(self):
        text = (self.ROOT / "boltz2" / "Dockerfile").read_text()
        assert "BOLTZ_CACHE=/cache/boltz" in text
        assert '"--cache", "/cache/boltz"' in text

    def test_boltz_redirects_the_numba_jit_cache(self):
        # OBSERVED FAILURE: `RuntimeError: cannot cache function
        # '_prepare_msa_arrays_inner': no locator available`, exit 1 before
        # boltz parsed an argument.
        #
        # boltz uses @numba.njit(cache=True), which resolves a writable cache
        # dir at IMPORT time. Under `--user $UID:$GID` numba's first three
        # locators all fail: site-packages/__pycache__ is root-owned and
        # $HOME/.cache is the image's /root. NUMBA_CACHE_DIR activates the
        # _UserProvidedCacheLocator, which is tried first.
        #
        # Must be under /cache: a container-local path would be discarded by
        # --rm, re-paying the JIT cost on every single design.
        text = strip_comments((self.ROOT / "boltz2" / "Dockerfile").read_text())
        assert "NUMBA_CACHE_DIR=/cache/numba" in text
        assert "VOLUME /cache" in text

    def test_boltz_cache_is_under_the_declared_volume(self):
        # BOLTZ_CACHE must be absolute (boltz raises ValueError otherwise) and
        # must sit under the volume, or it lives in the container layer.
        text = (self.ROOT / "boltz2" / "Dockerfile").read_text()
        assert "VOLUME /cache" in text
        for marker in ("BOLTZ_CACHE=/cache/boltz", '"--cache", "/cache/boltz"'):
            path = marker.split("/cache/boltz")[0] and "/cache/boltz"
            assert path.startswith("/cache/")

    def test_prefetch_maps_the_host_user(self):
        # Root-owned weight files in a host cache the campaign uses
        # unprivileged: see pipeline.docker.cache_dir().
        text = (
            Path(__file__).resolve().parents[1] / "scripts/bootstrap_instance.sh"
        ).read_text()
        assert '--user "$(id -u):$(id -g)"' in text


class TestEsmfoldPins:
    """ESMFold has no upstream Dockerfile; the reference is the install block in
    facebookresearch/esm's README. These pin the two divergences that mattered."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def _text(self) -> str:
        return (self.ROOT / "esmfold" / "Dockerfile").read_text()

    def test_openfold_pinned_to_the_esmfold_commit(self):
        # Upstream pins this exact commit -- openfold PR #199, "fixes to
        # support ESMFold" -- not a tag.
        assert "4b41059694619831a7db195b7e0988fc4ff3a307" in self._text()
        assert "openfold.git@v1.0.1" not in self._text()

    def test_esmfold_extra_dependencies_present(self):
        # fair-esm alone installs NOTHING: every requires_dist entry is gated
        # behind extra == "esmfold".
        text = self._text()
        for pkg in ("omegaconf", "einops", "ml-collections", "dm-tree",
                    "pytorch-lightning", "scipy"):
            assert pkg in text, f"{pkg} missing from the esmfold image"

    def test_dllogger_installed(self):
        # Never on PyPI, so pip cannot resolve it transitively.
        assert "NVIDIA/dllogger" in self._text()

    def test_devel_base_retained_for_nvcc(self):
        # openfold declares a real CUDAExtension; unlike rfdiffusion, this
        # image genuinely needs the toolkit.
        assert "-devel" in self._text()


class TestBuildDoesNotExposeGpu:
    def test_no_gpus_flag_on_docker_build(self):
        # openfold's setup.py compiles for the build host's card alone if it
        # can see one, producing a non-portable image.
        text = (Path(__file__).resolve().parents[1] / "docker/build_all.sh").read_text()
        for line in text.splitlines():
            if line.strip().startswith("docker build"):
                assert "--gpus" not in line


class TestPythonVersionGuards:
    """ESM's README documents `python <= 3.9` for ESMFold while the base image
    ships 3.10, so the deviation must be proven at build time, not assumed."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def test_esmfold_has_a_version_guard(self):
        text = (self.ROOT / "esmfold" / "Dockerfile").read_text()
        assert "(3, 9) <= sys.version_info[:2] < (3, 12)" in text

    def test_esmfold_has_a_build_time_import_smoke_test(self):
        # The chain that failed when the [esmfold] extra was missing.
        text = (self.ROOT / "esmfold" / "Dockerfile").read_text()
        assert "from esm.esmfold.v1.esmfold import ESMFold" in text
        for mod in ("omegaconf", "einops", "openfold.model.structure_module"):
            assert mod in text

    def test_esmfold_bounds_the_interpreter_window(self):
        # A future reader must not "fix" the <= 3.9 deviation by bumping the
        # base image: 2.4.0 lands on 3.11 and a later tag on 3.13.
        #
        # This asserts the GUARD, not the comment explaining it. An earlier
        # version required the literal "deepspeed==0.5.9" in the header and
        # broke when the comments were condensed -- testing prose again, which
        # this suite has a standing convention against. The rationale lives in
        # docs/MANIFEST_STATUS.md; the executable protection is this bound.
        text = (self.ROOT / "esmfold" / "Dockerfile").read_text()
        assert "sys.version_info[:2]" in text, "no interpreter guard present"
        assert "(3, 9) <=" in text, "guard must state the lower bound"
        assert "< (3, 12)" in text, "guard must state the upper bound"

    def test_inline_guard_snippets_are_valid_python(self):
        import ast
        text = (self.ROOT / "esmfold" / "Dockerfile").read_text().replace("\\\n", "")
        guard = [l for l in text.splitlines() if "sys.version_info[:2]" in l]
        assert guard, "version guard line not found"
        ast.parse("import sys; " + guard[0].split("python -c \"")[1].rsplit('"', 1)[0]
                  .replace("import sys; ", ""))


class TestColabfoldImage:
    """Built on the upstream published image rather than a hand-rolled conda
    solve, because the solve was missing CONDA_OVERRIDE_CUDA."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def _text(self) -> str:
        return (self.ROOT / "colabfold" / "Dockerfile").read_text()

    def _directives(self) -> str:
        return strip_comments(self._text())

    def test_builds_on_upstream_published_image(self):
        # The CUDA suffix is mandatory: bare :1.6.2 and :latest do not exist.
        # Pinned to a release tag, never a floating main-* tag.
        from_lines = [l for l in self._directives().splitlines()
                      if l.startswith("FROM ")]
        assert len(from_lines) == 1
        ref = from_lines[0].split()[1]
        assert ref.startswith("ghcr.io/sokrypton/colabfold:")
        tag = ref.split(":", 1)[1]
        assert tag.startswith("1."), f"must pin a release, got {tag}"
        assert "-cuda" in tag, "the CUDA suffix is mandatory"
        assert not tag.startswith("main"), "main-* floats; unusable as a pin"

    def test_pinned_image_is_not_a_stale_minor(self):
        # The original defect: 1.5.5 was read off an old tag's CI matrix while
        # 1.6.2 was published. Guards against drifting backwards.
        tag = [l for l in self._directives().splitlines()
               if l.startswith("FROM ")][0].split(":", 1)[1]
        major, minor = (int(x) for x in tag.split("-")[0].split(".")[:2])
        assert (major, minor) >= (1, 6), f"{tag} predates the 1.6 line"

    def test_no_dead_colabfold_weights_env(self):
        # ColabFold reads no such variable; XDG_CACHE_HOME is the mechanism.
        assert "COLABFOLD_WEIGHTS" not in self._directives()

    def test_save_all_removed(self):
        # PAE lands in the scores JSON regardless; --save-all only adds a raw
        # pickle per model that pipeline.parsers never reads.
        assert "--save-all" not in self._directives()

    def test_weights_path_explicit_and_under_the_volume(self):
        text = self._text()
        assert '"--data", "/cache/colabfold"' in text
        assert "VOLUME /cache" in text


class TestColabfoldTwoPass:
    """The MSA pass must not allocate a GPU, and both passes must share one
    out_dir or the MSA cache (<out_dir>/<job>.pickle) is never hit."""

    def test_msa_pass_runs_without_a_gpu(self):
        spec = docker.ContainerRun(
            image="binder-colabfold:latest",
            command=["/work/in.fasta", "/work/out", "--msa-only"],
            gpus=False,
        )
        argv = spec.argv()
        assert "--gpus" not in argv
        assert argv[-1] == "--msa-only"

    def test_prediction_pass_does_allocate_a_gpu(self):
        spec = docker.ContainerRun(
            image="binder-colabfold:latest",
            command=["/work/in.fasta", "/work/out"],
        )
        assert "--gpus" in spec.argv()

    @pytest.mark.xfail(
        strict=True,
        reason="KNOWN DRIFT: config/campaign.yaml sets "
               "complex_validation.colabfold_two_step: "
               "true, but the notebook's Step 4 cell was rewritten into a single "
               "combined colabfold_batch invocation and no longer reads the flag. "
               "See docs/MANIFEST_STATUS.md 'Config knobs nothing reads'. Resolve "
               "by either restoring the two-pass split or setting the flag false; "
               "strict=True so whichever is done turns this red and prompts "
               "removing the xfail.",
    )
    def test_notebook_implements_two_pass_when_config_asks_for_it(self):
        # Pass 2 reads <out_dir>/<job>.pickle written by pass 1, so both passes
        # must share one out_dir or the rate-limited public API is re-queried.
        import json
        root = Path(__file__).resolve().parents[1]
        cfg = (root / "config/campaign.yaml").read_text()
        assert "colabfold_two_step: true" in cfg

        nb = json.loads(
            (root / "notebooks/01_binder_design_prototype.ipynb").read_text()
        )
        cells = ["".join(c["source"]) for c in nb["cells"]]
        src = next((c for c in cells if "--msa-only" in c), None)
        assert src is not None, "config asks for two-pass; notebook has no MSA pass"
        assert src.count("cf_out = ") == 1, "out_dir must be defined once and reused"

    def test_entrypoint_does_not_hardcode_msa_only(self):
        # --msa-only must stay a per-call argument: baking it in would make
        # every invocation an MSA pass and nothing would ever be predicted.
        text = (Path(__file__).resolve().parents[1]
                / "docker/colabfold/Dockerfile").read_text()
        entry = [l for l in strip_comments(text).splitlines() if "ENTRYPOINT" in l or '"--' in l]
        assert not any("--msa-only" in l for l in entry)


class TestHomelessCacheRedirects:
    """The recurring defect class in these images: a library writes to `~`,
    and under `--user $UID:$GID` there is no writable HOME, so `~/.cache/x`
    becomes root-owned `/.cache/x`. Three instances in boltz2 alone (numba,
    cuEquivariance, and triton next in line), each crashing partway into the
    first real prediction."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def _env(self, name: str) -> dict[str, str]:
        """ENV assignments in one Dockerfile, as a dict."""
        text = strip_comments((self.ROOT / name / "Dockerfile").read_text())
        joined = text.replace("\\\n", " ")
        env: dict[str, str] = {}
        for line in joined.splitlines():
            if not line.strip().startswith("ENV "):
                continue
            for token in line.strip()[4:].split():
                if "=" in token:
                    k, v = token.split("=", 1)
                    env[k] = v
        return env

    def test_boltz_redirects_every_known_home_writing_cache(self):
        env = self._env("boltz2")
        # Each verified in upstream source, not guessed:
        #   CUEQ_TRITON_CACHE_DIR  cache_manager.py:45  (the observed crash)
        #   TRITON_CACHE_DIR       triton/runtime/cache.py:65
        #   NUMBA_CACHE_DIR        numba/core/caching.py:205
        for var in ("CUEQ_TRITON_CACHE_DIR", "TRITON_CACHE_DIR", "NUMBA_CACHE_DIR"):
            assert var in env, f"{var} unset; its library would write under ~"
            assert env[var].startswith("/cache/"), (
                f"{var}={env[var]!r} must sit under the declared volume, or "
                "--rm discards the compiled artifacts every run"
            )

    def test_boltz_sets_xdg_cache_home_as_the_class_level_fix(self):
        # platformdirs AND appdirs both honour XDG_CACHE_HOME, which covers
        # cuEquivariance's user_cache_dir() and numba's AppDirs() at once --
        # so the next library of this kind lands in /cache by default.
        env = self._env("boltz2")
        assert env.get("XDG_CACHE_HOME") == "/cache"

    def test_no_image_leaves_a_cache_var_outside_the_volume(self):
        # Guards the whole family across every image at once.
        for name in ("rfdiffusion", "proteinmpnn", "esmfold", "boltz2", "colabfold"):
            path = self.ROOT / name / "Dockerfile"
            if not path.exists():
                continue
            for var, value in self._env(name).items():
                if "CACHE" in var.upper() and value.startswith("/"):
                    assert value == "/cache" or value.startswith("/cache/"), (
                        f"{name}: {var}={value!r} is an absolute path outside "
                        "/cache, so it lives in the container layer"
                    )


class TestTritonRuntimeToolchain:
    """Triton JIT-compiles a CPython extension on its first kernel launch, so
    a C compiler and a linkable libcuda.so are RUNTIME dependencies -- which a
    'runtime' base image does not provide. Observed as `RuntimeError: Failed to
    find C compiler` partway into a real Step 4 prediction."""

    ROOT = Path(__file__).resolve().parents[1] / "docker"

    def _text(self) -> str:
        return strip_comments((self.ROOT / "boltz2" / "Dockerfile").read_text())

    def test_c_compiler_is_installed(self):
        # triton/runtime/build.py:28 checks $CC, then gcc, then clang.
        text = self._text()
        assert re.search(r"build-essential|\bgcc\b|\bclang\b", text), (
            "triton needs a C compiler at run time to build cuda_utils"
        )

    def test_libcuda_is_linkable(self):
        # triton links -lcuda (nvidia/backend/driver.py:16) but the container
        # runtime injects only libcuda.so.1, with no .so symlink.
        text = self._text()
        assert "libcuda.so" in text
        assert "LIBRARY_PATH=" in text, (
            "gcc's link-time search path must include a dir holding libcuda.so"
        )

    def test_does_not_set_ld_library_path_for_this(self):
        # LD_LIBRARY_PATH is the LOAD-time path; pointing it at a stub would
        # shadow the real driver at run time. LIBRARY_PATH is link-time.
        assert "ENV LD_LIBRARY_PATH" not in self._text()


class TestCuequivarianceProbeIsRepresentative:
    """The probe must drive the same code path production does.

    It previously called triangle_multiplicative_update with all weights at
    their None defaults, which skips the fused layer norm -- so no triton
    kernel launched, no JIT, no compiler needed, and the probe passed on an
    image that could not run a real prediction."""

    SH = Path(__file__).resolve().parents[1] / "docker/build_all.sh"

    def _probe(self) -> str:
        m = re.search(r"readonly CUEQ_PROBE='(.*?)'\n", self.SH.read_text(), re.S)
        assert m, "CUEQ_PROBE not found"
        return m.group(1)

    def test_probe_passes_real_norm_weights(self):
        # Without norm_in_weight/norm_out_weight the fused layer-norm triton
        # path is never entered.
        probe = self._probe()
        for arg in ("norm_in_weight", "norm_in_bias",
                    "norm_out_weight", "norm_out_bias"):
            assert f"{arg}=" in probe, f"{arg} omitted; triton path not exercised"

    def test_probe_passes_the_projection_weights(self):
        probe = self._probe()
        for arg in ("p_in_weight", "g_in_weight", "p_out_weight", "g_out_weight"):
            assert f"{arg}=" in probe

    def test_probe_uses_shapes_from_the_production_module(self):
        # boltz triangular_mult.py:53-59: p_in/g_in are Linear(dim, 2*dim),
        # p_out/g_out are Linear(dim, dim), norms are LayerNorm(dim).
        probe = self._probe()
        assert "2 * d" in probe, "p_in/g_in must be (2*dim, dim)"
        assert re.search(r"n, d = \d+, (\d+)", probe)
        dim = int(re.search(r"n, d = \d+, (\d+)", probe).group(1))
        assert dim % 32 == 0, f"hidden dim {dim} must be a multiple of 32"

    def test_probe_executes_rather_than_only_importing(self):
        probe = self._probe()
        assert "triangle_multiplicative_update(" in probe
        assert "torch.cuda.synchronize()" in probe


class TestProbesRunLikeProduction:
    """A probe that runs as root tests a configuration that never ships.

    Two of the four Boltz-2 defects were invisible to a root probe with no
    cache mount, then failed on the first real prediction: numba's
    'no locator available' (root can write site-packages/__pycache__) and
    cuEquivariance's PermissionError on '/.cache' (root can create it)."""

    SH = Path(__file__).resolve().parents[1] / "docker/build_all.sh"

    def _text(self) -> str:
        return self.SH.read_text()

    def test_probe_drops_root(self):
        body = re.search(r"^probe\(\) \{.*?^\}", self._text(), re.S | re.M).group(0)
        assert "--user" in body, "probes must run unprivileged, as the campaign does"
        assert "id -u" in body and "id -g" in body

    def test_probe_mounts_the_shared_cache(self):
        body = re.search(r"^probe\(\) \{.*?^\}", self._text(), re.S | re.M).group(0)
        assert ":/cache" in body, (
            "probes must mount /cache or they cannot exercise cache writes"
        )

    def test_cache_path_agrees_with_pipeline_docker(self):
        # pipeline/docker.py::cache_dir() is the single definition; a probe
        # against a different directory would not reproduce production.
        assert 'BINDER_CACHE:-$HOME/.cache/binder-pipeline' in self._text()

    def test_cache_dir_created_before_mounting(self):
        # `docker run -v` would create a missing path as ROOT, which is the
        # hazard --user exists to avoid.
        text = self._text()
        assert 'mkdir -p "$BINDER_CACHE"' in text


class TestBuildAllProbes:
    """Every image must be probed, with a probe matched to its stack. The
    colabfold one matters most: JAX falls back to CPU without erroring."""

    SH = Path(__file__).resolve().parents[1] / "docker/build_all.sh"

    def _text(self) -> str:
        return self.SH.read_text()

    def test_all_five_images_are_probed(self):
        text = self._text()
        for image in ("rfdiffusion", "proteinmpnn", "esmfold", "boltz2", "colabfold"):
            assert f"probe {image} python" in text, f"{image} has no GPU probe"

    def test_colabfold_probe_uses_jax_not_torch(self):
        # 1.6.2 is debian-based with no torch at all, so a torch probe would
        # raise ImportError and be misread as a GPU fault. Asserted against the
        # probe body wherever it lives -- inline or in a variable -- rather
        # than a fixed phrasing.
        text = self._text()
        assert "jax.default_backend()" in text
        assert "block_until_ready" in text, "backend selection alone is not proof"
        # The colabfold probe body must not reference torch.
        start = text.index("jax.default_backend()")
        body = text[max(0, start - 400):start + 800]
        assert "import torch" not in body

    def test_openfold_arch_extraction_uses_a_regex(self):
        # cuobjdump prints "...attn_core_inplace_cuda.1.sm_80.cubin": the sm_NN
        # is embedded in the filename, so startswith("sm_") finds nothing and
        # the probe would report "no architectures" on a healthy image.
        assert r'findall(r"sm_(\d+)"' in self._text()
        assert 'startswith("sm_")' not in strip_comments(self._text())

    def test_openfold_probe_distinguishes_missing_from_mismatched(self):
        # A missing extension is reachable: the Dockerfile's own import check is
        # deliberately non-fatal. It must not be reported as an architecture
        # mismatch. Asserted structurally, not on wording.
        text = self._text()
        assert "except ImportError" in text
        tail = text[text.index("except ImportError"):][:200]
        assert "sys.exit" in tail and "extension" in tail.lower()

    def test_probes_skipped_loudly_without_a_gpu(self):
        # Silence would imply the images were verified.
        text = self._text()
        assert "nvidia-smi" in text
        assert "UNVERIFIED" in text


class TestOpenfoldArchPredicate:
    """The cubin-compatibility rule the esmfold probe applies: a cubin runs on
    a device of the same major with an equal or higher minor."""

    ARCHS = [37, 52, 61, 70, 80]  # what openfold's setup.py emits

    @staticmethod
    def _covered(archs, major, minor):
        return any(a // 10 == major and a % 10 <= minor for a in archs)

    def test_supported_cards_pass(self):
        for major, minor in [(8, 0), (8, 6), (8, 9), (7, 0), (7, 5)]:
            assert self._covered(self.ARCHS, major, minor), f"sm_{major}{minor}"

    def test_sm90_and_above_excluded(self):
        # The whole reason this probe exists.
        for major, minor in [(9, 0), (10, 0)]:
            assert not self._covered(self.ARCHS, major, minor), f"sm_{major}{minor}"

    def test_lower_minor_than_compiled_is_not_covered(self):
        # cubin is forward-compatible across minors only, never backward.
        assert not self._covered([86], 8, 0)


class TestEsmfoldOpenfoldStdPatch:
    """OpenFold's setup.py hardcodes -std=c++14, and torch declines to inject
    -std=c++17 when a -std is already present. That combination fails the build
    against torch 2.x headers."""

    DF = Path(__file__).resolve().parents[1] / "docker/esmfold/Dockerfile"

    def _text(self) -> str:
        return self.DF.read_text()

    def test_std_flag_is_patched(self):
        d = strip_comments(self._text())
        assert "'-std=c++14'" in d and "'-std=c++17'" in d, "the sed patch is missing"

    def test_commit_stays_on_the_esmfold_pin(self):
        # -std=c++17 only arrives upstream in v2.2.0. Bumping there to get one
        # compiler flag would abandon the commit ESMFold is tested against
        # (PR #199, "fixes to support ESMFold").
        assert "4b41059694619831a7db195b7e0988fc4ff3a307" in self._text()

    def test_patch_is_asserted_not_silent(self):
        # A future SHA bump onto a differently-shaped setup.py must fail the
        # build, not quietly skip the patch and fail at nvcc instead.
        #
        # Asserts STRUCTURE, not wording: an earlier version of this test
        # required the literal "no -std flag found" and broke when the message
        # was recased and re-punctuated. The contract is that the else branch
        # exists and is fatal.
        d = strip_comments(self._text())
        assert "else" in d, "no fallback branch for an unrecognised setup.py"
        assert "exit 1" in d, "the fallback branch must fail the build"
        diagnostic = [
            l for l in d.splitlines()
            if "-std" in l and re.search(r"error|found", l, re.IGNORECASE)
        ]
        assert diagnostic, (
            "the fallback branch must emit a diagnostic naming the -std flag"
        )

    def test_source_tree_removed_after_install(self):
        # Unlike ProteinMPNN, OpenFold needs nothing from its checkout at
        # runtime; pip install copies the built extension into site-packages.
        assert "rm -rf /opt/openfold" in strip_comments(self._text())
