"""Bounded, local-only sizing pass before any full-catalog exact top-K job.

Reports aggregates only. A snapshot reads only the requested prefix of vectors.npy,
never artist ids or the whole archive; temporary spools are removed on exit.
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
from numpy.lib import format

from insights.similar_artists import MemoryGuard, compute_to_spool, estimate_peak_bytes, peak_rss_bytes


def read_slice(path: Path, rows: int) -> tuple[np.ndarray, int]:
    # Read only scalar/config metadata, never the object artist-id array.
    with np.load(path, allow_pickle=False) as metadata:
        if not np.array_equal(metadata["weights"], [0, 1, 1, 1, 1]) or abs(float(metadata["self_weight"]) - 0.05) > 1e-12:
            raise ValueError("snapshot is not the production w0=0, self=0.05 configuration")
        model = str(metadata["model_version"])
        if not model.startswith("fastrp-v2:dim=128:"):
            raise ValueError("snapshot is not the production v2/128-dimensional algorithm")
    with zipfile.ZipFile(path) as archive, archive.open("vectors.npy") as stream:
        version = format.read_magic(stream)
        if version not in {(1, 0), (2, 0)}:
            raise ValueError(f"unsupported NPY header: {version}")
        reader = format.read_array_header_1_0 if version == (1, 0) else format.read_array_header_2_0
        shape, fortran, dtype = reader(stream)
        if len(shape) != 2 or shape[1] != 128 or shape[0] < rows or fortran or dtype != np.float16:
            raise ValueError("snapshot must contain C-order 128-dimensional float16 vectors")
        expected = rows * shape[1] * dtype.itemsize
        raw = stream.read(expected)
        if len(raw) != expected:
            raise ValueError("truncated vector slice")
        return np.frombuffer(raw, dtype=dtype).reshape(rows, shape[1]), shape[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path)
    parser.add_argument("--rows", type=int, default=500_000)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--block-rows", type=int, default=4096)
    parser.add_argument("--scratch-root", type=Path, default=Path(tempfile.gettempdir()))
    args = parser.parse_args()
    if not 500_000 <= args.rows <= 1_000_000:
        parser.error("--rows must be between 500000 and 1000000")
    # Spool plus old and replacement checkpoint; check before even reading vectors.
    required_disk = args.rows * (50 * 8 * 3 + 8 * 2) + 256 * 1024**2
    free_disk = shutil.disk_usage(args.scratch_root).free
    if free_disk < required_disk:
        raise RuntimeError(f"sizing needs {required_disk} free bytes; available {free_disk}")
    if args.snapshot:
        vectors, catalog_rows = read_slice(args.snapshot, args.rows)
    else:
        catalog_rows = 9_366_416
        vectors = np.empty((args.rows, 128), dtype=np.float16)
        rng = np.random.default_rng(20260924)
        for start in range(0, args.rows, 65536):
            stop = min(start + 65536, args.rows)
            vectors[start:stop] = rng.standard_normal((stop - start, 128), dtype=np.float32)
    guard = MemoryGuard()
    guard.check("before sizing")
    with tempfile.TemporaryDirectory(prefix="gm-exact-sizing-", dir=args.scratch_root) as scratch:
        started = time.monotonic()
        compute_to_spool(vectors, Path(scratch), model_version="local-sizing", k=50, block_rows=args.block_rows, threads=args.threads, guard=guard)
        wall = time.monotonic() - started
        estimate = estimate_peak_bytes(catalog_rows, 128, threads=args.threads, block_rows=args.block_rows)
        pipeline_estimate = estimate + catalog_rows * 64  # conservative held artist-id list allowance
        # Both startup selection and CPU contention remain in this conservative n² scaling.
        full_seconds = wall * (catalog_rows / args.rows) ** 2
        print(
            json.dumps(
                {
                    "source_kind": "real_snapshot_slice" if args.snapshot else "synthetic",
                    "rows": args.rows,
                    "catalog_rows": catalog_rows,
                    "dimensions": 128,
                    "k": 50,
                    "block_rows": args.block_rows,
                    "threads": args.threads,
                    "wall_seconds": wall,
                    "peak_rss_bytes": peak_rss_bytes(),
                    "estimated_full_kernel_peak_bytes": estimate,
                    "estimated_full_pipeline_peak_bytes": pipeline_estimate,
                    "quadratic_full_seconds": full_seconds,
                    "within_six_hours": full_seconds <= 6 * 3600,
                    "within_memory_budget": pipeline_estimate <= guard.budget_bytes,
                    "free_disk_before_bytes": free_disk,
                }
            )
        )


if __name__ == "__main__":
    main()
