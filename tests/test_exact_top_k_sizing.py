"""Synthetic snapshot-reader checks for the bounded local sizing harness."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


def _reader():
    path = Path(__file__).parents[1] / "scripts/measure-exact-top-k-sizing.py"
    spec = importlib.util.spec_from_file_location("exact_sizing", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.read_slice


def test_reads_only_requested_vector_prefix_without_unpickling_artist_ids(tmp_path: Path) -> None:
    vectors = np.random.default_rng(7).normal(size=(20, 128)).astype(np.float16)
    snapshot = tmp_path / "synthetic.npz"
    # Object ids would fail with allow_pickle=False if the reader tried to load them.
    np.savez(
        snapshot,
        vectors=vectors,
        artist_ids=np.array([object()] * 20, dtype=object),
        weights=[0, 1, 1, 1, 1],
        self_weight=0.05,
        model_version="fastrp-v2:dim=128:synthetic",
    )
    read, catalog_rows = _reader()(snapshot, 3)
    assert catalog_rows == 20
    assert read.dtype == np.float16
    np.testing.assert_array_equal(read, vectors[:3])


@pytest.mark.parametrize(
    "weights,self_weight,dtype", [([1, 1, 1, 1, 1], 0.05, np.float16), ([0, 1, 1, 1, 1], 0.0, np.float16), ([0, 1, 1, 1, 1], 0.05, np.float32)]
)
def test_rejects_incompatible_configuration_or_precision(tmp_path: Path, weights, self_weight, dtype) -> None:
    snapshot = tmp_path / "synthetic.npz"
    np.savez(snapshot, vectors=np.ones((10, 128), dtype=dtype), weights=weights, self_weight=self_weight, model_version="fastrp-v2:dim=128:synthetic")
    with pytest.raises(ValueError):
        _reader()(snapshot, 3)
