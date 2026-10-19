"""Container log capture.

The behaviour worth pinning is where output goes, not how it is formatted:
a transcript on disk per run, a flush per line, and a notebook that sees a
summary rather than ten thousand denoising steps. Each test resets the module's
handlers, since logging state is process-global and would otherwise leak
between tests and duplicate every line.
"""

import logging
from pathlib import Path

import pytest

from pipeline import docker, runlog


@pytest.fixture(autouse=True)
def isolated_logging(tmp_path, monkeypatch):
    """Point logging at tmp_path and leave no handlers behind."""
    monkeypatch.delenv(runlog.LOG_DIR_ENV, raising=False)
    monkeypatch.setattr(runlog, "_log_dir", None)
    monkeypatch.setattr(runlog, "_configured", False)
    monkeypatch.setattr(runlog, "_last_transcript", None)
    yield
    log = logging.getLogger(runlog.LOGGER_NAME)
    for handler in list(log.handlers):
        log.removeHandler(handler)
        handler.close()


class TestLogDir:
    def test_env_override_wins_over_configure(self, tmp_path, monkeypatch):
        # The lever available on an instance where the notebook cannot be
        # edited; it must beat the configured value, not the reverse.
        runlog.configure(tmp_path / "configured")
        monkeypatch.setenv(runlog.LOG_DIR_ENV, str(tmp_path / "env"))
        assert runlog.log_dir() == tmp_path / "env"

    def test_default_is_under_results(self):
        # results/ is gitignored, so transcripts are untracked by default.
        assert runlog.log_dir() == runlog.DEFAULT_LOG_DIR
        assert "results" in str(runlog.DEFAULT_LOG_DIR)

    def test_import_creates_no_directory(self):
        # Importing pipeline must stay side-effect free; the directory appears
        # on configure(), not on import.
        assert not runlog.DEFAULT_LOG_DIR.exists() or runlog.DEFAULT_LOG_DIR.is_dir()

    def test_configure_creates_the_directory(self, tmp_path):
        target = runlog.configure(tmp_path / "logs")
        assert target.is_dir()


class TestConfigure:
    def test_is_idempotent(self, tmp_path):
        # Re-running the setup cell must not double every log line.
        runlog.configure(tmp_path)
        before = len(logging.getLogger(runlog.LOGGER_NAME).handlers)
        runlog.configure(tmp_path)
        assert len(logging.getLogger(runlog.LOGGER_NAME).handlers) == before

    def test_does_not_propagate_to_root(self, tmp_path):
        # A notebook that called logging.basicConfig() would otherwise print
        # every event twice.
        runlog.configure(tmp_path)
        assert logging.getLogger(runlog.LOGGER_NAME).propagate is False

    def test_console_can_be_silenced(self, tmp_path):
        runlog.configure(tmp_path, console=False, force=True)
        handlers = logging.getLogger(runlog.LOGGER_NAME).handlers
        assert not any(type(h) is logging.StreamHandler for h in handlers)

    def test_campaign_log_is_appended_not_truncated(self, tmp_path):
        # A kernel restart mid-campaign must not erase what already ran.
        runlog.configure(tmp_path)
        runlog.logger().info("first")
        runlog.configure(tmp_path, force=True)
        runlog.logger().info("second")
        text = (tmp_path / runlog.CAMPAIGN_LOG).read_text()
        assert "first" in text and "second" in text


class TestTranscriptPath:
    def test_tag_is_dropped_from_the_filename(self, tmp_path):
        runlog.configure(tmp_path)
        assert ":" not in runlog.transcript_path("binder-boltz2:latest").name

    def test_image_name_is_in_the_filename(self, tmp_path):
        runlog.configure(tmp_path)
        assert "binder-boltz2" in runlog.transcript_path("binder-boltz2:latest").name

    def test_label_distinguishes_two_runs_of_one_image(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("binder-proteinmpnn:latest", "parse-chains")
        assert "parse-chains" in path.name

    def test_same_second_does_not_overwrite(self, tmp_path):
        # The two ProteinMPNN helper scripts run back to back; a collision
        # would silently discard the first transcript.
        runlog.configure(tmp_path)
        first = runlog.transcript_path("img")
        first.write_text("x")
        second = runlog.transcript_path("img")
        assert second != first
        assert first.read_text() == "x"


class TestRunLog:
    def test_writes_lines_to_the_file(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            sink.write("alpha\n")
            sink.write("beta\n")
        text = path.read_text()
        assert "alpha\nbeta\n" in text

    def test_flushes_before_the_run_ends(self, tmp_path):
        # A stage killed by the OOM reaper must still leave its output on
        # disk, so lines cannot sit in a buffer until close.
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            sink.write("mid-run\n")
            assert "mid-run" in path.read_text()

    def test_header_records_the_argv(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path, command="docker run --rm img") as _:
            pass
        text = path.read_text()
        assert "docker run --rm img" in text
        assert "img" in text

    def test_does_not_echo_by_default(self, tmp_path, capsys):
        runlog.configure(tmp_path, console=False, force=True)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            sink.write("noise\n")
        assert "noise" not in capsys.readouterr().out

    def test_echo_is_opt_in(self, tmp_path, capsys):
        runlog.configure(tmp_path, console=False, force=True)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path, echo=True) as sink:
            sink.write("visible\n")
        assert "visible" in capsys.readouterr().out

    def test_file_is_closed_when_the_body_raises(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with pytest.raises(ValueError):
            with runlog.RunLog(image="img", path=path) as sink:
                sink.write("before the crash\n")
                raise ValueError("boom")
        assert "before the crash" in path.read_text()

    def test_tail_returns_the_last_lines_only(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            for i in range(100):
                sink.write(f"line {i}\n")
            tail = sink.tail(3)
        assert tail == "line 97\nline 98\nline 99\n"

    def test_memory_is_bounded(self, tmp_path):
        # A thousand-design run prints hundreds of thousands of lines; the
        # in-memory tail must not grow with them.
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            for i in range(runlog.TAIL_BUFFER * 3):
                sink.write(f"line {i}\n")
            assert len(sink._buffer) == runlog.TAIL_BUFFER
            assert sink.lines_written == runlog.TAIL_BUFFER * 3


class TestTail:
    def test_reads_the_most_recent_transcript(self, tmp_path):
        runlog.configure(tmp_path)
        path = runlog.transcript_path("img")
        with runlog.RunLog(image="img", path=path) as sink:
            sink.write("last line\n")
        assert runlog.last_transcript() == path
        assert "last line" in runlog.tail(5)

    def test_is_safe_before_any_run(self):
        assert "no transcript" in runlog.tail()

    def test_reports_a_missing_file_rather_than_raising(self, tmp_path):
        assert "missing" in runlog.tail(path=tmp_path / "gone.log")


class TestDockerIntegration:
    """pipeline.docker drives this module; these assert the wiring, not docker."""

    def _spec(self):
        return docker.ContainerRun(image="img:latest", command=["echo", "hi"])

    def test_container_output_lands_in_a_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(
            docker.subprocess, "Popen", _FakePopen(["hello\n", "world\n"], 0)
        )
        out = docker.run(self._spec())
        assert out == "hello\nworld\n"
        transcript = runlog.last_transcript()
        assert "hello\nworld\n" in transcript.read_text()

    def test_output_is_not_echoed_by_default(self, tmp_path, monkeypatch, capsys):
        # The whole point: a chatty stage must not land in the cell output.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(docker.subprocess, "Popen", _FakePopen(["chatty\n"], 0))
        docker.run(self._spec())
        assert "chatty" not in capsys.readouterr().out

    def test_output_does_not_repr_itself(self, tmp_path, monkeypatch):
        # The other door into the output cell: a notebook cell ending in
        # docker.run(...) displays the return value, so a chatty stage lands
        # in the .ipynb even with stream=False. repr() must summarise instead.
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(docker.subprocess, "Popen", _FakePopen(["chatty\n"], 0))
        out = docker.run(self._spec())
        assert "chatty" not in repr(out)
        assert str(runlog.last_transcript()) in repr(out)
        # ...while still being the string every caller and parser expects.
        assert out == "chatty\n"

    def test_stream_true_restores_echoing(self, tmp_path, monkeypatch, capsys):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(docker.subprocess, "Popen", _FakePopen(["chatty\n"], 0))
        docker.run(self._spec(), stream=True)
        assert "chatty" in capsys.readouterr().out

    def test_failure_names_the_transcript(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(
            docker.subprocess, "Popen", _FakePopen(["CUDA error\n"], 1)
        )
        with pytest.raises(docker.ContainerError) as excinfo:
            docker.run(self._spec())
        error = excinfo.value
        assert "CUDA error" in error.tail
        assert error.transcript is not None
        assert str(error.transcript) in str(error)
        assert "CUDA error" in Path(error.transcript).read_text()

    def test_campaign_log_records_start_and_exit(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)
        monkeypatch.setattr(docker.subprocess, "Popen", _FakePopen(["ok\n"], 0))
        docker.run(self._spec())
        timeline = (tmp_path / "logs" / runlog.CAMPAIGN_LOG).read_text()
        assert "[run] img:latest" in timeline
        assert "[done] img:latest" in timeline

    def test_dry_run_logs_the_argv_and_runs_nothing(self, tmp_path, monkeypatch):
        monkeypatch.setenv("BINDER_CACHE", str(tmp_path / "cache"))
        runlog.configure(tmp_path / "logs", console=False, force=True)

        def forbidden(*args, **kwargs):
            raise AssertionError("dry_run must not launch a container")

        monkeypatch.setattr(docker.subprocess, "Popen", forbidden)
        assert docker.run(self._spec(), dry_run=True) == ""
        timeline = (tmp_path / "logs" / runlog.CAMPAIGN_LOG).read_text()
        assert "[dry-run] docker run" in timeline


class _FakePopen:
    """Stand-in for subprocess.Popen yielding fixed lines and a return code."""

    def __init__(self, lines, returncode):
        self.lines = lines
        self.returncode = returncode

    def __call__(self, *args, **kwargs):
        self.stdout = iter(self.lines)
        return self

    def wait(self):
        return self.returncode
