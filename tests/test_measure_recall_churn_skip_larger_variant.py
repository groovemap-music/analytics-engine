"""gm-analytics-engine-i37: `--skip-larger-variant` unit coverage.

Synthetic data only -- no Discogs/MusicBrainz-derived data is read or committed here.
`scripts/measure_recall_churn.py`'s real flow needs a live Postgres+pgvector container and
real embedding vectors, neither of which belong in this suite, so these tests exercise the
skip/skip-not control flow around the larger-index (m=32) variant directly: `AsyncPostgreSQLPool`,
`_apply_schema`, `measure_month`, `measure_index_variant`, and `_drop_index` are all replaced
with fakes, and the "months" are a handful of random unit vectors.
"""

from __future__ import annotations

import argparse
import json
from typing import TYPE_CHECKING, Any

import numpy as np
import pytest


if TYPE_CHECKING:
    from pathlib import Path

from scripts import measure_recall_churn as mrc


def _synthetic_month(*, dump_id: str, dump_date: str, n: int, seed: int) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    vectors = rng.normal(size=(n, 8)).astype(np.float32)
    return {
        "artist_ids": [f"a{i}" for i in range(n)],
        "vectors": vectors,
        "degrees": None,
        "method_version": "test-method-version",
        "model_version": f"test-model-version:{dump_id}",
        "dump_id": dump_id,
        "dump_date": dump_date,
    }


class _FakeConnCtx:
    def __init__(self, conn: Any) -> None:
        self._conn = conn

    async def __aenter__(self) -> Any:
        return self._conn

    async def __aexit__(self, *exc: object) -> bool:
        return False


class _FakePool:
    """Stands in for `common.AsyncPostgreSQLPool` -- no real connection is ever made."""

    def __init__(self, **_kwargs: object) -> None:
        self._conn = object()

    async def initialize(self) -> None:
        pass

    async def close(self) -> None:
        pass

    def connection(self) -> _FakeConnCtx:
        return _FakeConnCtx(self._conn)


async def _fake_apply_schema(_conn: Any) -> None:
    pass


async def _fake_measure_month(_conn: Any, month: dict[str, Any], *, label: str, **_kwargs: object) -> dict[str, Any]:
    return {
        "write": {"rows_written": len(month["artist_ids"])},
        "exact_elapsed_s": 0.0,
        "index": {"index_name": f"fake_idx_{label}"},
        "production_ef_search": 64,
        "churn_ef_search_used": 64,
    }


@pytest.fixture(autouse=True)
def _patch_infra(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mrc, "AsyncPostgreSQLPool", _FakePool)
    monkeypatch.setattr(mrc, "_apply_schema", _fake_apply_schema)
    monkeypatch.setattr(mrc, "measure_month", _fake_measure_month)


def _trimmed_args(tmp_path: Path, *, skip_larger_variant: bool) -> argparse.Namespace:
    return argparse.Namespace(
        host="127.0.0.1",
        port=5432,
        database="groovemap",
        username="groovemap",
        password="x",  # noqa: S106 -- not a real credential, never reaches a real connection (see _FakePool)
        out=tmp_path / "result.json",
        sept_maintenance_work_mem="2GB",
        skip_larger_variant=skip_larger_variant,
    )


def _months(n: int = 15) -> tuple[dict[str, Any], dict[str, Any], list[str], list[str], list[int]]:
    aug = _synthetic_month(dump_id="aug-dump", dump_date="2026-08-01", n=n, seed=1)
    sept = _synthetic_month(dump_id="sept-dump", dump_date="2026-09-01", n=n, seed=2)
    common_ids = sorted(set(aug["artist_ids"]) & set(sept["artist_ids"]))
    churn_sample_ids = common_ids
    sept_query_ids = sept["artist_ids"][:3]
    sept_query_positions = [0, 1, 2]
    return aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions


@pytest.mark.asyncio
async def test_skip_larger_variant_never_calls_the_build_and_records_the_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"measure_index_variant": 0, "drop_index": 0}

    async def _unexpected_measure_index_variant(*_args: object, **_kwargs: object) -> dict[str, Any]:
        calls["measure_index_variant"] += 1
        raise AssertionError("measure_index_variant must not be called when --skip-larger-variant is set")

    async def _unexpected_drop_index(*_args: object, **_kwargs: object) -> None:
        calls["drop_index"] += 1
        raise AssertionError("_drop_index must not be called when --skip-larger-variant is set")

    monkeypatch.setattr(mrc, "measure_index_variant", _unexpected_measure_index_variant)
    monkeypatch.setattr(mrc, "_drop_index", _unexpected_drop_index)

    aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions = _months()
    args = _trimmed_args(tmp_path, skip_larger_variant=True)

    result = await mrc._main_async_trimmed(args, aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions)

    assert calls == {"measure_index_variant": 0, "drop_index": 0}
    assert result["partial"] is False
    assert result["september"]["larger_index_variant"] == {"skipped": True, "reason": mrc.LARGER_VARIANT_SKIP_REASON}

    # The partial checkpoint (standard-variant recall + exact churn) must already be on disk,
    # written BEFORE the (here, skipped) larger-variant step -- a cancel during that step must
    # never lose these numbers. It must not itself carry a larger_index_variant key yet.
    on_disk = json.loads(args.out.read_text())
    assert on_disk["partial"] is True
    assert on_disk["churn_exact_cosine"]["n"] == len(churn_sample_ids)
    assert "larger_index_variant" not in on_disk["september"]


@pytest.mark.asyncio
async def test_without_the_flag_the_larger_variant_is_still_attempted(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"measure_index_variant": 0, "drop_index": 0}

    async def _fake_drop_index(_conn: Any, _model_version: str) -> None:
        calls["drop_index"] += 1

    async def _fake_measure_index_variant(_conn: Any, _month: dict[str, Any], *, label: str, **_kwargs: object) -> dict[str, Any]:
        calls["measure_index_variant"] += 1
        assert label == "larger"
        return {"m": mrc.HNSW_LARGER_M, "ef_construction": mrc.HNSW_LARGER_EF_CONSTRUCTION}

    monkeypatch.setattr(mrc, "measure_index_variant", _fake_measure_index_variant)
    monkeypatch.setattr(mrc, "_drop_index", _fake_drop_index)

    aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions = _months()
    args = _trimmed_args(tmp_path, skip_larger_variant=False)

    result = await mrc._main_async_trimmed(args, aug, sept, common_ids, churn_sample_ids, sept_query_ids, sept_query_positions)

    assert calls == {"measure_index_variant": 1, "drop_index": 1}
    assert result["september"]["larger_index_variant"] == {"m": mrc.HNSW_LARGER_M, "ef_construction": mrc.HNSW_LARGER_EF_CONSTRUCTION}


def _main_argv(tmp_path: Path, *extra: str) -> list[str]:
    return [
        "measure_recall_churn.py",
        str(tmp_path / "aug.npz"),
        str(tmp_path / "sept.npz"),
        "--host",
        "127.0.0.1",
        "--port",
        "5432",
        "--database",
        "groovemap",
        "--username",
        "groovemap",
        "--password",
        "x",
        "--out",
        str(tmp_path / "out.json"),
        *extra,
    ]


def test_cli_flag_is_parsed_and_reaches_main_async(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_main_async(args: argparse.Namespace) -> dict[str, Any]:
        captured["skip_larger_variant"] = args.skip_larger_variant
        return {"ok": True}

    monkeypatch.setattr(mrc, "main_async", _fake_main_async)
    monkeypatch.setattr("sys.argv", _main_argv(tmp_path, "--skip-larger-variant"))

    mrc.main()

    assert captured["skip_larger_variant"] is True
    assert json.loads((tmp_path / "out.json").read_text()) == {"ok": True}


def test_cli_flag_defaults_to_false(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, Any] = {}

    async def _fake_main_async(args: argparse.Namespace) -> dict[str, Any]:
        captured["skip_larger_variant"] = args.skip_larger_variant
        return {"ok": True}

    monkeypatch.setattr(mrc, "main_async", _fake_main_async)
    monkeypatch.setattr("sys.argv", _main_argv(tmp_path))

    mrc.main()

    assert captured["skip_larger_variant"] is False
