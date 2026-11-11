"""Tests for the load check that makes the nightly similar-artist run skip while a catalog load or
the embeddings pipeline is running (gm-analytics-engine-d6vm).

Synthetic rows only. The fake database answers the latch and write-lock queries with the rows a
test sets, and reuses the job-test fakes for everything else.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any

import pytest

from insights import similar_artists_job as job
from tests.test_similar_artists import _FakePool, _vectors
from tests.test_similar_artists_job import _BlockClock, _Db, _DbCursor, _main, _PublishingRegistry, _run


if TYPE_CHECKING:
    import argparse
    from pathlib import Path

    from _pytest.capture import CaptureFixture


NOW = datetime(2026, 11, 12, 1, 0, tzinfo=UTC)


def _latch(loader: str, version: str, *, age_h: float, refreshed: bool, signals: tuple[str, ...] = ()) -> job.LatchRow:
    created = NOW - timedelta(hours=age_h)
    return job.LatchRow(loader, version, signals, created, NOW - timedelta(hours=1) if refreshed else None)


# ── The decision, as a pure function ─────────────────────────────────────────────────────────


class TestJudgeLoads:
    def test_an_open_latch_inside_the_bound_is_an_extraction_in_progress(self) -> None:
        check = job.judge_loads([_latch("discogs", "20261101", age_h=2, refreshed=False, signals=("artist", "label"))], 0, now=NOW)

        assert check.skip is True
        assert check.reason is not None
        assert "discogs extraction 20261101 latch open since" in check.reason
        assert "signals: artist, label" in check.reason
        assert "extraction in progress" in check.reason

    def test_a_completed_latch_does_not_block(self) -> None:
        check = job.judge_loads(
            [_latch("discogs", "20261101", age_h=3, refreshed=True, signals=("artist", "label", "master", "release"))], 0, now=NOW
        )

        assert (check.skip, check.reason) == (False, None)

    def test_an_open_latch_past_the_bound_is_reported_and_does_not_block(self) -> None:
        check = job.judge_loads([_latch("musicbrainz", "20261001", age_h=49, refreshed=False)], 0, now=NOW)

        assert check.skip is False
        assert check.reason is not None
        assert check.reason.startswith("musicbrainz extraction 20261001 latch open since")
        assert "signals: none recorded" in check.reason
        assert "open 49 h, past the 48 h bound: ignored" in check.reason

    def test_an_uncommitted_embeddings_write_blocks_whatever_the_latches_say(self) -> None:
        check = job.judge_loads([_latch("discogs", "20261101", age_h=60, refreshed=True)], 1, now=NOW)

        assert check.skip is True
        assert check.reason == "1 session(s) have uncommitted writes to public.artist_embeddings: the embeddings pipeline is loading"

    def test_the_pipeline_lock_held_means_skip_even_with_nothing_else_running(self) -> None:
        check = job.judge_loads([], 0, now=NOW, pipeline_lock_held=True)

        assert check.skip is True
        assert check.reason == "the embeddings pipeline holds its run lock: an embeddings run is in progress"

    def test_an_unheld_pipeline_lock_is_not_a_signal(self) -> None:
        assert job.judge_loads([], 0, now=NOW, pipeline_lock_held=False) == job.LoadCheck(skip=False, reason=None)

    def test_the_lock_key_is_split_the_way_pg_locks_reports_it(self) -> None:
        key = job._PIPELINE_LOCK_KEY
        high, low = key >> 32, key & 0xFFFFFFFF

        assert (high << 32) + low == key
        assert 0 < high < 2**31 and 0 <= low < 2**32  # positive, so classid and objid compare as plain ints

    def test_a_normal_start_has_no_reason(self) -> None:
        check = job.judge_loads([], 0, now=NOW)

        assert (check.skip, check.reason) == (False, None)

    def test_the_bound_is_inclusive_so_a_latch_exactly_48_hours_old_still_blocks(self) -> None:
        check = job.judge_loads([_latch("discogs", "20261101", age_h=48, refreshed=False)], 0, now=NOW)

        assert check.skip is True


# ── The read: the two queries, judged ────────────────────────────────────────────────────────


class _LoadCursor(_DbCursor):
    conn: _LoadDb

    async def fetchall(self) -> list[tuple[Any, ...]]:
        if self._last_sql == job._LATCH_SQL:
            return list(self.conn.latches)
        if self._last_sql == job._WRITES_IN_FLIGHT_SQL:
            return [(self.conn.writes,)]
        if self._last_sql == job._PIPELINE_LOCK_HELD_SQL:
            return [(self.conn.lock_holders,)]
        return await super().fetchall()


class _LoadDb(_Db):
    def __init__(self, vectors: Any, *, latches: tuple[tuple[Any, ...], ...] = (), writes: int = 0, lock_holders: int = 0, **kwargs: Any) -> None:
        super().__init__(vectors, **kwargs)
        self.latches = list(latches)
        self.writes = writes
        self.lock_holders = lock_holders

    def cursor(self, name: str | None = None) -> _LoadCursor:
        return _LoadCursor(self, name)


class TestReadLoadCheck:
    @pytest.mark.asyncio
    async def test_the_latch_rows_and_write_count_are_read_and_judged(self) -> None:
        db = _LoadDb(_vectors(10, dim=8), latches=(("discogs", "20261101", ["artist"], NOW - timedelta(hours=1), None),), writes=0)

        check = await job.read_load_check(db, NOW)

        assert check.skip is True
        assert check.reason is not None
        assert "discogs extraction 20261101" in check.reason

    @pytest.mark.asyncio
    async def test_a_held_pipeline_lock_is_read_and_skips(self) -> None:
        db = _LoadDb(_vectors(10, dim=8), lock_holders=1)

        check = await job.read_load_check(db, NOW)

        assert check.skip is True
        assert "run lock" in (check.reason or "")
        lock_calls = [params for sql, params in db.log if sql == job._PIPELINE_LOCK_HELD_SQL]
        assert lock_calls == [((job._PIPELINE_LOCK_KEY >> 32), (job._PIPELINE_LOCK_KEY & 0xFFFFFFFF))]

    @pytest.mark.asyncio
    async def test_a_null_signals_array_reads_as_none_recorded(self) -> None:
        db = _LoadDb(_vectors(10, dim=8), latches=(("musicbrainz", "20261101", None, NOW - timedelta(hours=1), None),))

        check = await job.read_load_check(db, NOW)

        assert "signals: none recorded" in (check.reason or "")


# ── Through run_job: nothing is computed or published on a skip ──────────────────────────────


class TestRunJobLoadCheck:
    @pytest.mark.asyncio
    async def test_an_extraction_in_progress_skips_with_status_skipped_and_exit_0(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), latches=(("discogs", "20261101", ["artist"], NOW - timedelta(hours=2), None),))
        registry = _PublishingRegistry(db)

        status = await _run(db, tmp_path, registry)

        assert (status.status, status.exit_code, status.decision, status.published) == ("skipped", 0, "skip", False)
        assert status.model_version == "m2"
        assert status.reason is not None
        assert "discogs extraction 20261101" in status.reason
        assert registry.calls == []
        assert [p.name for p in tmp_path.iterdir()] == [".lock"]

    @pytest.mark.asyncio
    async def test_an_embeddings_write_in_flight_skips(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), writes=1)
        registry = _PublishingRegistry(db)

        status = await _run(db, tmp_path, registry)

        assert (status.status, status.exit_code, status.decision) == ("skipped", 0, "skip")
        assert "uncommitted writes to public.artist_embeddings" in (status.reason or "")
        assert registry.calls == []
        assert not (tmp_path / "m2").exists()

    @pytest.mark.asyncio
    async def test_a_held_pipeline_lock_skips_the_run_without_computing(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), lock_holders=1)
        registry = _PublishingRegistry(db)

        status = await _run(db, tmp_path, registry)

        assert (status.status, status.exit_code, status.decision, status.published) == ("skipped", 0, "skip", False)
        assert "run lock" in (status.reason or "")
        assert registry.calls == []
        assert [p.name for p in tmp_path.iterdir()] == [".lock"]

    @pytest.mark.asyncio
    async def test_a_stale_latch_proceeds_and_says_so_in_the_reason(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), latches=(("musicbrainz", "20261001", None, NOW - timedelta(hours=400), None),))
        registry = _PublishingRegistry(db)

        status = await _run(db, tmp_path, registry, _BlockClock())

        assert (status.status, status.exit_code, status.decision, status.published) == ("done", 0, "run", True)
        assert "past the 48 h bound: ignored" in (status.reason or "")
        assert registry.publishes == 1

    @pytest.mark.asyncio
    async def test_a_completed_latch_and_no_writes_run_as_before(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), latches=(("discogs", "20261101", ["artist", "label"], NOW - timedelta(hours=3), NOW),))
        registry = _PublishingRegistry(db)

        status = await _run(db, tmp_path, registry, _BlockClock())

        assert (status.status, status.exit_code, status.decision, status.reason, status.published) == ("done", 0, "run", None, True)
        assert registry.publishes == 1

    @pytest.mark.asyncio
    async def test_check_only_reports_a_skip_without_computing(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8), latches=(("discogs", "20261101", ["artist"], NOW - timedelta(hours=2), None),))
        registry = _PublishingRegistry(db)

        status = await job.run_job(_FakePool(db), model_version=None, spool_root=tmp_path, registry=registry, check_only=True, wall_now=NOW)

        assert (status.status, status.exit_code, status.decision) == ("checked", 0, "skip")
        assert status.reason is not None
        assert registry.calls == []
        assert [p.name for p in tmp_path.iterdir()] == [".lock"]

    @pytest.mark.asyncio
    async def test_check_only_reports_a_run_when_nothing_is_loading(self, tmp_path: Path) -> None:
        db = _LoadDb(_vectors(100, dim=8))
        registry = _PublishingRegistry(db)

        status = await job.run_job(_FakePool(db), model_version=None, spool_root=tmp_path, registry=registry, check_only=True, wall_now=NOW)

        assert (status.status, status.exit_code, status.decision, status.reason) == ("checked", 0, "run", None)
        assert registry.calls == []


# ── CLI surface ──────────────────────────────────────────────────────────────────────────────


class TestCli:
    def test_the_check_only_flag_is_parsed(self) -> None:
        assert job.parse_args(["--spool-dir", "/srv/x", "--skip-check-only"]).skip_check_only is True
        assert job.parse_args(["--spool-dir", "/srv/x"]).skip_check_only is False

    def test_a_skipped_status_exits_0_and_names_the_reason(
        self, tmp_path: Path, capsys: CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(job, "NON_PERSISTENT_ROOTS", ())  # pytest's tmp_path is under /tmp on Linux

        async def run(_args: argparse.Namespace, _root: Path, _deadline: float | None) -> job.JobStatus:
            return job.JobStatus(status="skipped", exit_code=job.EXIT_SKIPPED, decision="skip", reason="discogs extraction 20261101 latch open")

        code, line = _main(["--spool-dir", str(tmp_path)], capsys, run)

        assert (code, line["status"], line["exit_code"], line["decision"]) == (0, "skipped", 0, "skip")
        assert line["reason"] == "discogs extraction 20261101 latch open"
