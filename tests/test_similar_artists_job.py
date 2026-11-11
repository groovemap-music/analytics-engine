"""Tests for the standalone, deadline-bounded similar-artist run (gm-analytics-engine-dhit).

Every vector and id here is synthetic. The deadline is driven by a fake clock that advances a
fixed amount per finished row block, so each window's stopping point is deterministic.
"""

from __future__ import annotations

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

from insights import similar_artists as sa
from insights import similar_artists_job as job
from insights.embedding_pipeline import DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR
from tests.test_similar_artists import _expected, _FakeConnection, _FakeCursor, _FakePool, _FakeRegistry, _quiet_guard, _stored_rows, _vectors


if TYPE_CHECKING:
    import argparse


BLOCK_ROWS = 64
K = 8
LINEAGE = ("dump-11", date(2026, 11, 1))


class _BlockClock:
    """A monotonic clock that only moves when a block finishes: `MemoryGuard` reads peak RSS
    once before the compute and once after every row block, so using `block` as the guard's
    ``peak_rss`` makes every block take ``block_s`` seconds."""

    def __init__(self, block_s: float = 10.0) -> None:
        self.now = 1000.0
        self.block_s = block_s

    def __call__(self) -> float:
        return self.now

    def block(self) -> int:
        self.now += self.block_s
        return 1


# ── Compute level: the spool itself ──────────────────────────────────────────────────────────


def _spool_bytes(directory: Path) -> tuple[bytes, bytes]:
    return (directory / "positions.i32").read_bytes(), (directory / "scores.f32").read_bytes()


class TestDeadlineBoundedSpool:
    def test_several_windows_produce_a_byte_identical_spool(self, tmp_path: Path) -> None:
        vectors = _vectors(450)  # 8 row blocks of 64, the last one ragged
        reference = sa.compute_to_spool(vectors, tmp_path / "one-go", model_version="m1", k=K, block_rows=BLOCK_ROWS, threads=2, guard=_quiet_guard())

        clock = _BlockClock()
        directory = tmp_path / "windows"
        pauses: list[sa.SpoolProgress] = []
        for _window in range(20):
            try:
                spool = sa.compute_to_spool(
                    vectors,
                    directory,
                    model_version="m1",
                    k=K,
                    block_rows=BLOCK_ROWS,
                    threads=2,
                    checkpoint_every_s=10**9,  # only the deadline checkpoints
                    guard=_quiet_guard(peak_rss=clock.block),
                    clock=clock,
                    deadline=clock.now + 25,
                )
                break
            except sa.PausedAtDeadline as paused:
                pauses.append(paused.progress)
                # A pause is a block boundary with a valid checkpoint, and no half-written file.
                with np.load(directory / "checkpoint.npz") as saved:
                    assert int(saved["next_block"]) == paused.progress.blocks_done
                assert json.loads((directory / "meta.json").read_text())["complete"] is False
                assert sorted(p.name for p in directory.iterdir()) == ["checkpoint.npz", "meta.json", "positions.i32", "scores.f32"]
                # Every block before the boundary is already final in the spool.
                done_rows = paused.progress.blocks_done * BLOCK_ROWS
                np.testing.assert_array_equal(sa.Spool(directory, 450, K).read(0, done_rows)[0], reference.read(0, done_rows)[0])
        else:
            pytest.fail("never finished")

        assert len(pauses) >= 3
        assert [p.blocks_done for p in pauses] == sorted({p.blocks_done for p in pauses})
        assert all(p.blocks_total == 8 for p in pauses)
        assert all(p.eta_s is not None and p.eta_s > 0 for p in pauses)
        assert _spool_bytes(spool.directory) == _spool_bytes(reference.directory)
        assert not spool.checkpoint_path.exists()

    def test_a_deadline_already_past_computes_nothing(self, tmp_path: Path) -> None:
        calls: list[int] = []

        def peak() -> int:
            calls.append(1)
            return 1

        with pytest.raises(sa.PausedAtDeadline) as paused:
            sa.compute_to_spool(
                _vectors(200),
                tmp_path,
                model_version="m1",
                k=5,
                block_rows=BLOCK_ROWS,
                threads=1,
                guard=_quiet_guard(peak_rss=peak),
                clock=lambda: 50.0,
                deadline=50.0,
            )

        assert paused.value.progress == sa.SpoolProgress(0, 4, None)
        assert calls == []
        assert not (tmp_path / "checkpoint.npz").exists()

    def test_the_last_block_completes_even_past_the_deadline(self, tmp_path: Path) -> None:
        clock = _BlockClock()
        vectors = _vectors(128)  # two blocks; the deadline passes during the second
        spool = sa.compute_to_spool(
            vectors,
            tmp_path,
            model_version="m1",
            k=5,
            block_rows=BLOCK_ROWS,
            threads=1,
            guard=_quiet_guard(peak_rss=clock.block),
            clock=clock,
            deadline=clock.now + 25,
        )
        np.testing.assert_array_equal(spool.read(0, 128)[0], _expected(vectors, 5)[0])

    def test_a_slice_reports_progress_too(self, tmp_path: Path) -> None:
        clock = _BlockClock()
        with pytest.raises(sa.PausedAtDeadline) as paused:
            sa.compute_to_spool(
                _vectors(300),
                tmp_path,
                model_version="m1",
                k=5,
                block_rows=BLOCK_ROWS,
                threads=1,
                guard=_quiet_guard(peak_rss=clock.block),
                clock=clock,
                deadline=clock.now + 15,
                query_rows=192,
            )
        # One of three equal-cost slice blocks done, in one block's time: two left.
        assert paused.value.progress == sa.SpoolProgress(1, 3, 20.0)

    def test_atomic_writes_leave_no_temporary_file(self, tmp_path: Path) -> None:
        target = tmp_path / "meta.json"
        sa._atomic_write(target, lambda p: p.write_text("{}"))
        assert [p.name for p in tmp_path.iterdir()] == ["meta.json"]


# ── Job level: target, publish once, status ──────────────────────────────────────────────────


class _Db(_FakeConnection):
    """A fake catalog database: one or more embedding versions, a set of published releases."""

    def __init__(self, vectors: np.ndarray, *, versions: tuple[str, ...] = ("m2",), published: tuple[str, ...] = (), lineages: int = 1) -> None:
        super().__init__(fetchone_by_sql={sa._COUNT_EMBEDDINGS_SQL: (len(vectors),)}, streamed_rows=_stored_rows(vectors))
        self.versions = list(versions)
        self.published = list(published)
        self.lineages = lineages

    def cursor(self, name: str | None = None) -> _DbCursor:
        return _DbCursor(self, name)


class _DbCursor(_FakeCursor):
    conn: _Db

    async def execute(self, sql: str, params: Any = None, *, binary: bool | None = None) -> None:
        await super().execute(sql, params, binary=binary)
        self._params = params

    async def fetchall(self) -> list[tuple[Any, ...]]:
        sql = self._last_sql
        if sql == job._EMBEDDING_VERSIONS_SQL:
            return [(v,) for v in self.conn.versions]
        if sql == job._LATCH_SQL:
            return []  # no loader latch open: the load check passes
        if sql == job._WRITES_IN_FLIGHT_SQL:
            return [(0,)]  # no embeddings write in flight
        if sql == job._PIPELINE_LOCK_HELD_SQL:
            return [(0,)]  # the embeddings pipeline does not hold its run lock
        if sql in (job._PUBLISHED_VERSIONS_SQL, sa._RELEASES_BY_RECENCY_SQL):
            return [(v,) for v in reversed(self.conn.published)]
        if sql == job._EMBEDDING_LINEAGE_SQL:
            return [(f"{LINEAGE[0]}-{i}" if i else LINEAGE[0], LINEAGE[1]) for i in range(self.conn.lineages)]
        if sql == job._RELEASE_LINEAGE_SQL:
            return [("dump-old", date(2026, 10, 1))]
        return []

    async def fetchone(self) -> tuple[Any, ...] | None:
        if self._last_sql == sa._RELEASE_EXISTS_SQL:
            return (1,) if self._params[0] in self.conn.published else None
        if self._last_sql == sa._CURRENT_RELEASE_SQL:
            return (self.conn.published[-1],) if self.conn.published else None
        return await super().fetchone()


class _PublishingRegistry(_FakeRegistry):
    def __init__(self, db: _Db) -> None:
        super().__init__()
        self.db = db

    async def publish(self, cursor: Any, model_version: str, **kwargs: Any) -> int:
        self.db.published.append(model_version)
        return await super().publish(cursor, model_version, **kwargs)

    @property
    def publishes(self) -> int:
        return sum(call[0] == "publish" for call in self.calls)


async def _run(
    db: _Db, root: Path, registry: _FakeRegistry, clock: _BlockClock | None = None, *, window_s: float | None = None, model_version: str | None = None
) -> job.JobStatus:
    deadline = None if clock is None or window_s is None else clock.now + window_s
    return await job.run_job(
        _FakePool(db),
        model_version=model_version,
        spool_root=root,
        deadline=deadline,
        clock=clock or _BlockClock(),
        registry=registry,
        threads=2,
        guard=_quiet_guard(peak_rss=clock.block) if clock else _quiet_guard(),
        checkpoint_every_s=10**9,
        k=K,
        block_rows=BLOCK_ROWS,
    )


class TestRunJob:
    @pytest.mark.asyncio
    async def test_windowed_runs_publish_once_the_lists_of_one_uninterrupted_run(self, tmp_path: Path) -> None:
        vectors = _vectors(450, dim=8)
        one_go_db = _Db(vectors)
        one_go = await _run(one_go_db, tmp_path / "one-go", _PublishingRegistry(one_go_db))
        assert (one_go.status, one_go.exit_code, one_go.published) == ("done", job.EXIT_DONE, True)

        db = _Db(vectors)
        registry = _PublishingRegistry(db)
        clock = _BlockClock()
        root = tmp_path / "nightly"
        statuses: list[job.JobStatus] = []
        while not statuses or statuses[-1].status == "paused":
            statuses.append(await _run(db, root, registry, clock, window_s=25))
            assert len(statuses) < 20

        *paused, done = statuses
        assert len(paused) >= 3
        assert {(s.status, s.exit_code, s.published, s.rows_written) for s in paused} == {("paused", job.EXIT_PAUSED, False, 0)}
        assert [s.blocks_done for s in paused] == sorted({s.blocks_done for s in paused if s.blocks_done is not None})
        assert (done.status, done.exit_code, done.published, done.blocks_done, done.blocks_total, done.model_version) == ("done", 0, True, 8, 8, "m2")
        # Byte-identical lists (scores are float32 values, so equal floats are equal bits) and one publish.
        assert db.copied == one_go_db.copied
        assert len(db.copied) == 450
        assert registry.publishes == 1
        assert registry.calls[0] == ("create", "m2", {"source_dump_id": LINEAGE[0], "source_dump_date": LINEAGE[1], "k": K})
        # The spool is gone after publishing; only the lock file stays in the root.
        assert [p.name for p in root.iterdir()] == [".lock"]

        # Re-running after publish is a no-op that reports done, by name or by default.
        again = await _run(db, root, registry, clock, window_s=25, model_version="m2")
        assert (again.status, again.exit_code, again.already_published, again.published) == ("done", 0, True, False)
        nothing = await _run(db, root, registry, clock, window_s=25)
        assert (nothing.status, nothing.exit_code, nothing.model_version) == ("done", 0, None)
        assert registry.publishes == 1

    @pytest.mark.asyncio
    async def test_a_compute_finished_past_the_deadline_publishes_next_window(self, tmp_path: Path) -> None:
        db = _Db(_vectors(450, dim=8))
        registry = _PublishingRegistry(db)
        clock = _BlockClock()
        # 1 guard check before + 8 blocks = 90 s; the deadline passes during the last block.
        first = await _run(db, tmp_path, registry, clock, window_s=85)
        assert (first.status, first.blocks_done, first.blocks_total, first.eta_s, first.rows_written) == ("paused", 8, 8, 0.0, 0)
        assert registry.calls == []

        second = await _run(db, tmp_path, registry, clock, window_s=25)
        assert (second.status, second.published, second.rows_written) == ("done", True, 450)
        assert registry.publishes == 1

    @pytest.mark.asyncio
    async def test_a_published_release_with_a_leftover_spool_cleans_it(self, tmp_path: Path) -> None:
        db = _Db(_vectors(100, dim=8), published=("m2",))
        leftover = sa.spool_directory(tmp_path, "m2")
        leftover.mkdir(parents=True)
        (leftover / "meta.json").write_text("{}")

        status = await _run(db, tmp_path, _PublishingRegistry(db), model_version="m2")

        assert (status.status, status.already_published) == ("done", True)
        assert not leftover.exists()

    @pytest.mark.asyncio
    async def test_a_held_lock_fails_the_run(self, tmp_path: Path) -> None:
        db = _Db(_vectors(100, dim=8))
        with job.spool_lock(tmp_path), pytest.raises(job.SpoolBusyError):
            await _run(db, tmp_path, _PublishingRegistry(db))


class TestResolveTarget:
    @pytest.mark.asyncio
    async def test_defaults_to_the_one_unpublished_version(self) -> None:
        db = _Db(_vectors(10), versions=("m1", "m2"), published=("m1",))
        assert await job.resolve_target(db, None) == job.Target("m2", *LINEAGE)

    @pytest.mark.asyncio
    async def test_several_unpublished_versions_need_a_name(self) -> None:
        db = _Db(_vectors(10), versions=("m1", "m2"))
        with pytest.raises(job.UsageError, match="pass --model-version"):
            await job.resolve_target(db, None)
        assert await job.resolve_target(db, "m1") == job.Target("m1", *LINEAGE)

    @pytest.mark.asyncio
    async def test_an_unknown_version_is_refused(self) -> None:
        with pytest.raises(job.UsageError, match="not in artist_embeddings"):
            await job.resolve_target(_Db(_vectors(10)), "m9")

    @pytest.mark.asyncio
    async def test_a_published_version_without_embeddings_takes_the_release_lineage(self) -> None:
        db = _Db(_vectors(10), versions=("m2",), published=("m1",))
        assert await job.resolve_target(db, "m1") == job.Target("m1", "dump-old", date(2026, 10, 1))

    @pytest.mark.asyncio
    async def test_mixed_lineage_is_an_error(self) -> None:
        with pytest.raises(ValueError, match="2 dump lineages"):
            await job.resolve_target(_Db(_vectors(10), lineages=2), "m2")


# ── Arguments ────────────────────────────────────────────────────────────────────────────────


class TestSpoolRoot:
    def test_requires_an_explicit_directory(self) -> None:
        with pytest.raises(job.UsageError, match="--spool-dir"):
            job.resolve_spool_root(None, None)
        with pytest.raises(job.UsageError, match="--spool-dir"):
            job.resolve_spool_root(None, "")

    @pytest.mark.parametrize("path", [DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR, Path("/tmp"), Path("/var/tmp/spool"), Path("/dev/shm/x")])  # noqa: S108
    def test_refuses_non_persistent_directories(self, path: Path) -> None:
        with pytest.raises(job.UsageError, match="does not persist"):
            job.resolve_spool_root(path, None)

    def test_refuses_a_relative_path(self) -> None:
        with pytest.raises(job.UsageError, match="absolute"):
            job.resolve_spool_root(Path("spool"), None)

    def test_refuses_through_a_symlink(self, tmp_path: Path) -> None:
        volatile = tmp_path / "volatile"
        volatile.mkdir()
        (tmp_path / "link").symlink_to(volatile)
        with pytest.raises(job.UsageError, match="does not persist"):
            job.resolve_spool_root(tmp_path / "link" / "spool", None, non_persistent_roots=(volatile,))

    def test_accepts_a_persistent_directory_from_the_argument_or_env(self, tmp_path: Path) -> None:
        assert job.resolve_spool_root(tmp_path / "a", str(tmp_path / "b"), non_persistent_roots=()) == tmp_path / "a"
        assert job.resolve_spool_root(None, str(tmp_path / "b"), non_persistent_roots=()) == tmp_path / "b"


class TestDeadline:
    NOW = datetime(2026, 11, 11, 22, 0, tzinfo=UTC)

    def test_unbounded_without_either(self) -> None:
        assert job.resolve_deadline(None, None, wall_now=self.NOW, mono_now=5.0) is None

    def test_the_earlier_bound_wins(self) -> None:
        deadline = self.NOW + timedelta(hours=6)
        assert job.resolve_deadline(deadline, None, wall_now=self.NOW, mono_now=5.0) == 5.0 + 6 * 3600
        assert job.resolve_deadline(deadline, 3600, wall_now=self.NOW, mono_now=5.0) == 5.0 + 3600
        assert job.resolve_deadline(deadline, 9 * 3600, wall_now=self.NOW, mono_now=5.0) == 5.0 + 6 * 3600

    def test_a_deadline_in_another_offset_is_absolute(self) -> None:
        deadline = datetime.fromisoformat("2026-11-12T00:00:00+02:00")  # 22:00Z, i.e. now
        assert job.resolve_deadline(deadline, None, wall_now=self.NOW, mono_now=5.0) == 5.0

    def test_refuses_a_naive_deadline_or_a_non_positive_runtime(self) -> None:
        with pytest.raises(job.UsageError, match="UTC offset"):
            job.resolve_deadline(datetime(2026, 11, 12, 6), None, wall_now=self.NOW, mono_now=0.0)
        with pytest.raises(job.UsageError, match="positive"):
            job.resolve_deadline(None, 0, wall_now=self.NOW, mono_now=0.0)


# ── main: exit codes and the status line ─────────────────────────────────────────────────────


def _main(argv: list[str], capsys: pytest.CaptureFixture[str], run: Any = None) -> tuple[int, dict[str, Any]]:
    with pytest.raises(SystemExit) as exited:
        job.main(argv, run=run)
    last = capsys.readouterr().out.strip().splitlines()[-1]
    return int(exited.value.code or 0), json.loads(last)


@pytest.fixture(autouse=True)
def _no_logging_setup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(job, "setup_logging", lambda *_a, **_k: None)
    monkeypatch.delenv(job.SPOOL_DIR_ENV, raising=False)


class TestMain:
    def test_paused_exits_75_and_writes_the_status_file(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(job, "NON_PERSISTENT_ROOTS", ())  # pytest's tmp_path is under /tmp on Linux
        seen: dict[str, Any] = {}

        async def run(args: argparse.Namespace, spool_root: Path, deadline: float | None) -> job.JobStatus:
            seen.update(spool_root=spool_root, deadline=deadline, threads=args.threads)
            return job.JobStatus(status="paused", exit_code=job.EXIT_PAUSED, model_version="m2", blocks_done=3, blocks_total=8, eta_s=120.0)

        status_file = tmp_path / "status.json"
        code, line = _main(
            ["--spool-dir", str(tmp_path / "spool"), "--max-runtime", "60", "--threads", "4", "--status-file", str(status_file)], capsys, run
        )

        assert code == 75
        assert line["kind"] == "similar_artists_status"
        assert (line["status"], line["exit_code"], line["blocks_done"], line["blocks_total"], line["eta_s"]) == ("paused", 75, 3, 8, 120.0)
        assert json.loads(status_file.read_text()) == line
        assert seen["spool_root"] == tmp_path / "spool"
        assert seen["deadline"] is not None
        assert seen["threads"] == 4

    def test_done_exits_0(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv(job.SPOOL_DIR_ENV, str(tmp_path))
        monkeypatch.setattr(job, "NON_PERSISTENT_ROOTS", ())  # pytest's tmp_path is under /tmp on Linux

        async def run(_args: argparse.Namespace, spool_root: Path, deadline: float | None) -> job.JobStatus:
            assert deadline is None
            return job.JobStatus(status="done", exit_code=job.EXIT_DONE, model_version="m2", published=True)

        code, line = _main(["--model-version", "m2"], capsys, run)
        assert (code, line["status"], line["published"]) == (0, "done", True)

    def test_the_tmp_default_is_refused_with_exit_2(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, line = _main(["--spool-dir", str(DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR)], capsys)
        assert (code, line["status"]) == (2, "failed")
        assert "does not persist" in line["error"]

    def test_bad_arguments_still_end_in_a_status_line(self, capsys: pytest.CaptureFixture[str]) -> None:
        code, line = _main(["--deadline", "tomorrow"], capsys)
        assert (code, line["status"], line["exit_code"]) == (2, "failed", 2)

    def test_a_failed_run_exits_1(self, tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(job, "NON_PERSISTENT_ROOTS", ())  # pytest's tmp_path is under /tmp on Linux

        async def run(*_args: Any) -> job.JobStatus:
            raise sa.MemoryBudgetExceededError("block 3/8: over budget")

        code, line = _main(["--spool-dir", str(tmp_path)], capsys, run)
        assert (code, line["status"], line["exit_code"]) == (1, "failed", 1)
        assert line["error"] == "MemoryBudgetExceededError: block 3/8: over budget"
