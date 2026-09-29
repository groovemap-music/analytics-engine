"""gm-analytics-engine-8ts, 2026-09-29: unit coverage for wait_for_memory's footprint-aware
guard, specifically the paged-out-while-waiting failure this bead's own re-embedding run hit.

Synthetic data only -- no Discogs/MusicBrainz-derived data is read or committed here.
September's real run measured its own LIVE (psutil) footprint collapse from 6.22 GB to
0.02 GB across five consecutive 60s polls while idling in `wait_for_memory`'s loop (nothing
was touching its allocations between polls, so macOS paged/compressed them out), pushing
`required_free` from 13.78 GB toward 19.98 GB -- a bound this host structurally cannot ever
satisfy while Colima holds 16 GiB of it. `process_footprint_bytes` now prefers
`resource.getrusage`'s peak RSS (monotonic, never decreasing for the life of the process)
over that live reading; these tests cover the preference order and that the guard converges
under a footprint reading that stays put instead of collapsing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from scripts import embeddings_from_dump as efd


if TYPE_CHECKING:
    import pytest


class TestProcessFootprintBytes:
    def test_prefers_the_monotonic_getrusage_peak_over_a_live_reading(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(efd, "peak_rss_bytes", lambda: 12_010_000_000)

        def _psutil_process_should_not_be_constructed() -> None:
            raise AssertionError("psutil.Process() was constructed even though resource.getrusage succeeded")

        import psutil

        monkeypatch.setattr(psutil, "Process", _psutil_process_should_not_be_constructed)

        assert efd.process_footprint_bytes() == 12_010_000_000

    def test_falls_back_to_psutil_only_if_getrusage_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _boom() -> int:
            raise OSError("no getrusage here")

        monkeypatch.setattr(efd, "peak_rss_bytes", _boom)

        class _FakeMemInfo:
            rss = 4_000_000_000

        class _FakeProcess:
            def memory_info(self) -> _FakeMemInfo:
                return _FakeMemInfo()

        import psutil

        monkeypatch.setattr(psutil, "Process", _FakeProcess)

        assert efd.process_footprint_bytes() == 4_000_000_000

    def test_peak_rss_does_not_shrink_after_memory_is_freed(self) -> None:
        # The real-world failure mode in one line: a LIVE reading (psutil RSS) can drop after
        # memory is freed, paged out, or compressed; resource.getrusage's ru_maxrss cannot --
        # it is a monotonic high-water mark for the life of the process. Allocate, measure,
        # free, measure again: the second reading must never be smaller than the first.
        before = efd.peak_rss_bytes()
        block = bytearray(64 * 1024 * 1024)  # 64 MiB -- enough to move ru_maxrss, not enough to be slow.
        block[0] = 1  # touch it so the OS actually commits the pages, not just reserves them.
        after_alloc = efd.peak_rss_bytes()
        del block
        after_free = efd.peak_rss_bytes()

        assert after_alloc >= before
        assert after_free >= after_alloc  # never shrinks, even though the allocation was freed.


class TestWaitForMemoryConvergence:
    def test_converges_immediately_with_a_stable_footprint_and_the_corrected_expected_peak(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # This bead's own re-embedding run measured a real peak of 12.01 GB for edges-v3
        # (7.34 GB post-parse, 12.01 GB post-build+fastrp) -- the corrected --expected-peak-gb,
        # replacing the stale 18 GB carried over from an older edges-v2-era measurement. A
        # STABLE footprint reading (the fix: no collapse while idling) at that same peak,
        # against the ~7.6 GB free+inactive the real host had, converges without a single wait.
        monkeypatch.setattr(efd, "process_footprint_bytes", lambda: int(12.01e9))
        monkeypatch.setattr(efd, "free_plus_inactive_bytes", lambda: int(7.61e9))
        sleeps: list[float] = []
        monkeypatch.setattr(efd.time, "sleep", lambda s: sleeps.append(s))

        efd.wait_for_memory(int(12.5e9), margin_bytes=int(2e9), poll_interval_s=60.0)

        assert sleeps == []  # required_free = max(12.5 - 12.01 + 2, 0) = 2.49 GB < 7.61 GB available.

    def test_waits_then_converges_once_available_memory_rises_with_a_stable_footprint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # The footprint reading (now peak-based) stays put across every poll -- it must NOT
        # need to grow for the loop to converge; only the host's free+inactive needs to rise,
        # exactly the legitimate "genuinely wait for memory to free up" case this guard is
        # for. Three polls: not enough, not enough, then enough.
        monkeypatch.setattr(efd, "process_footprint_bytes", lambda: int(10e9))
        available_gb_sequence = iter([1.0, 1.5, 3.0])
        monkeypatch.setattr(efd, "free_plus_inactive_bytes", lambda: int(next(available_gb_sequence) * 1e9))
        sleeps: list[float] = []
        monkeypatch.setattr(efd.time, "sleep", lambda s: sleeps.append(s))

        # required_free = max(10 - 10 + 2, 0) = 2 GB -- satisfied once available reaches 3 GB.
        efd.wait_for_memory(int(10e9), margin_bytes=int(2e9), poll_interval_s=60.0)

        assert sleeps == [60.0, 60.0]  # waited for the first two (insufficient) polls, not the third.

    def test_expected_peak_bytes_zero_or_less_skips_the_wait_entirely(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _should_not_be_called() -> int:
            raise AssertionError("process_footprint_bytes should not be read when the guard is disabled")

        monkeypatch.setattr(efd, "process_footprint_bytes", _should_not_be_called)
        efd.wait_for_memory(0, margin_bytes=int(2e9))  # 0 = disabled, the CLI default.
