"""FastRP (Chen et al., 2019) with the configuration ADR 0013 adopted.

The embedding of node ``v`` is ``sum_k w_k * normalize((P^(k+1) R)[v])`` over the
iteration weights ``w``, where ``P = D^-1 A`` is the random-walk transition matrix,
``R`` the very sparse projection, and ``normalize`` scales a row to unit length.
That is the spike's ``embed.fastrp`` (design ``docs/spikes/gm-design-chw.2/``) with
two changes, neither of which changes the method:

- ``R`` is ``HashedProjection``: each node's row is a function of its stable key and
  the pinned seed, not a draw from a global random stream.
- The computation is column-blocked. The spike held three ``n x 128`` ``float32``
  matrices, about 50 GB for the 32.8M-node catalog. Here the embedding columns are
  processed ``block_columns`` at a time. Columns propagate independently; only the
  per-power row norms couple them, so a first pass accumulates each power's squared
  row norms across all blocks and a second pass recomputes each block and adds its
  normalized contribution. With a single block the two passes fuse into one.

Determinism. Every step is a fixed sequence of row-local floating-point operations:
SciPy's CSR-times-dense kernel sums each output row over that row's neighbours in
column-index order, which ``AdjacencyBuilder`` sorts into node-key order, and the
squared norms are accumulated column by column in ``float64`` in a fixed order. So:

- the same graph gives byte-identical output on every run, whatever order its edges
  arrived in and whatever ``block_columns`` is;
- a node with no added or removed edge touching any vertex within ``len(weights) - 1``
  hops of it gets a byte-identical vector, even when vertices elsewhere are added or
  removed and every node position shifts.
"""

from __future__ import annotations

import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, cast

import numpy as np
import scipy.sparse as sp

from insights.embeddings.projection import HashedProjection


if TYPE_CHECKING:
    from concurrent.futures import Executor

    from numpy.typing import ArrayLike, DTypeLike, NDArray

    from insights.embeddings.graph import Adjacency
    from insights.embeddings.projection import Projection


# Bump when a code change alters any output bit for the same graph and config; it
# is part of model_version, so old and new vectors never share a key.
FASTRP_ALGORITHM_VERSION: Final = 1
# The pinned projection seed (the date ADR 0013's data-access amendment landed).
DEFAULT_SEED: Final = 20260924
# Four columns per block keeps the full-catalog run near 10 GB; see docs/embeddings.md.
DEFAULT_BLOCK_COLUMNS: Final = 4
# Rows per threaded task.
_TASK_ROWS: Final = 1 << 18


@dataclass(frozen=True)
class FastRPConfig:
    """Method parameters. The defaults are the configuration ADR 0013 adopted."""

    dim: int = 128
    weights: tuple[float, ...] = field(default=(0.0, 1.0, 1.0, 1.0, 1.0))
    beta: float = 0.0
    seed: int = DEFAULT_SEED

    def __post_init__(self) -> None:
        if self.dim < 1:
            raise ValueError("dim must be positive")
        if not any(self.weights):
            raise ValueError("at least one iteration weight must be non-zero")

    @property
    def model_version(self) -> str:
        """The ``artist_embeddings.model_version`` naming method, parameters, and seed rule."""
        weights = ",".join(f"{w:g}" for w in self.weights)
        return (
            f"fastrp-v{FASTRP_ALGORITHM_VERSION}:dim={self.dim}:weights={weights}:beta={self.beta:g}"
            f":proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed={self.seed}"
        )


def fastrp(
    adjacency: Adjacency,
    config: FastRPConfig | None = None,
    *,
    rows: ArrayLike | None = None,
    block_columns: int = DEFAULT_BLOCK_COLUMNS,
    projection: Projection | None = None,
    out_dtype: DTypeLike = np.float32,
    threads: int = 1,
) -> NDArray[np.floating]:
    """Return the FastRP embeddings of ``rows`` (node positions; all nodes if omitted).

    ``out_dtype`` sets only the storage of the returned array; accumulation is always
    ``float32``, so ``float16`` output equals ``float32`` output rounded once, which is
    what storing into a ``halfvec`` column does anyway. ``projection`` replaces the
    hashed projection, for parity tests against a given ``R``. ``threads`` splits each
    propagation step and projection block across a thread pool (SciPy and NumPy release
    the GIL); the work is row- and column-local, so the output does not depend on it.
    """
    config = config or FastRPConfig()
    if block_columns < 1 or threads < 1:
        raise ValueError("block_columns and threads must be positive")
    if threads == 1:
        return _fastrp(adjacency, config, rows, block_columns, projection, out_dtype, None, 1)
    with ThreadPoolExecutor(max_workers=threads, thread_name_prefix="fastrp") as executor:
        return _fastrp(adjacency, config, rows, block_columns, projection, out_dtype, executor, threads)


def _fastrp(
    adjacency: Adjacency,
    config: FastRPConfig,
    rows: ArrayLike | None,
    block_columns: int,
    projection: Projection | None,
    out_dtype: DTypeLike,
    executor: Executor | None,
    threads: int,
) -> NDArray[np.floating]:
    n = adjacency.transition.shape[0]
    propagate = _Propagator(adjacency.transition, executor, threads)
    if projection is None:
        projection = HashedProjection(adjacency.nodes.keys, config.seed, degree=adjacency.degree, beta=config.beta)
    project = _Projector(projection, n, executor)
    selected = None if rows is None else np.asarray(rows, dtype=np.int64)
    out = np.zeros((n if selected is None else selected.size, config.dim), dtype=out_dtype)
    # Trailing zero weights contribute nothing, so their powers are never computed.
    powers = max(k for k, w in enumerate(config.weights) if w) + 1
    blocks = [(start, min(start + block_columns, config.dim)) for start in range(0, config.dim, block_columns)]

    # Per-power scale w_k / |row| of the returned rows only; other rows' norms are
    # needed solely to be summed, never kept.
    scales: dict[int, NDArray[np.float32]] = {}
    if len(blocks) > 1:
        sums = {k: np.zeros(n, dtype=np.float64) for k, w in enumerate(config.weights) if w}
        for start, stop in blocks:
            current = project(start, stop)
            for k in range(powers):
                current = propagate(current)
                if k in sums:
                    _accumulate_squares(current, sums[k])
            del current
        for k in list(sums):
            scales[k] = _scale(config.weights[k], sums.pop(k), selected)

    for start, stop in blocks:
        current = project(start, stop)
        block = np.zeros((out.shape[0], stop - start), dtype=np.float32)
        picked = None if selected is None else np.empty_like(block)
        for k in range(powers):
            current = propagate(current)
            weight = config.weights[k]
            if not weight:
                continue
            if len(blocks) == 1:
                total = np.zeros(n, dtype=np.float64)
                _accumulate_squares(current, total)
                scales[k] = _scale(weight, total, selected)
                del total
            if picked is None or selected is None:
                block += scales[k][:, None] * current
            else:
                np.take(current, selected, axis=0, out=picked)
                picked *= scales[k][:, None]
                block += picked
        out[:, start:stop] = block
    return out


class _Propagator:
    """``transition @ block``, optionally split into row slices of about equal nnz."""

    def __init__(self, transition: sp.csr_array, executor: Executor | None, threads: int) -> None:
        self._transition = transition
        self._executor = executor
        self._slices: list[tuple[int, int, sp.csr_array]] = []
        if executor is None:
            return
        n = transition.shape[0]
        # Small slices keep each task's temporary result small, so memory freed by
        # worker threads does not pile up in per-thread allocator arenas.
        parts = max(4 * threads, -(-n // _TASK_ROWS))
        targets = np.linspace(0, transition.nnz, parts + 1)[1:-1]
        bounds = [0, *np.unique(np.searchsorted(transition.indptr, targets)).tolist(), n]
        for lo_row, hi_row in itertools.pairwise(bounds):
            if lo_row >= hi_row:
                continue
            lo, hi = int(transition.indptr[lo_row]), int(transition.indptr[hi_row])
            # Assigned rather than passed to the constructor: its prune step copies any
            # view much smaller than its base, which would duplicate the whole matrix.
            part = sp.csr_array((hi_row - lo_row, n), dtype=transition.dtype)
            part.indptr = transition.indptr[lo_row : hi_row + 1] - lo
            part.indices = transition.indices[lo:hi]
            part.data = transition.data[lo:hi]
            self._slices.append((lo_row, hi_row, part))

    def __call__(self, block: NDArray[np.float32]) -> NDArray[np.float32]:
        if self._executor is None:
            return cast("NDArray[np.float32]", self._transition @ block)
        out = np.empty_like(block)

        def run(part: tuple[int, int, sp.csr_array]) -> None:
            lo_row, hi_row, matrix = part
            out[lo_row:hi_row] = matrix @ block

        for _ in self._executor.map(run, self._slices):
            pass
        return out


class _Projector:
    """Projection columns ``[start, stop)``, one row slice per task when threaded."""

    def __init__(self, projection: Projection, n: int, executor: Executor | None) -> None:
        self._projection = projection
        self._n = n
        self._executor = executor

    def __call__(self, start: int, stop: int) -> NDArray[np.float32]:
        if self._executor is None:
            return self._projection.columns(start, stop)
        block = np.empty((self._n, stop - start), dtype=np.float32)

        def run(lo: int) -> None:
            rows = slice(lo, min(lo + _TASK_ROWS, self._n))
            block[rows] = self._projection.columns(start, stop, rows)

        for _ in self._executor.map(run, range(0, self._n, _TASK_ROWS)):
            pass
        return block


def _accumulate_squares(block: NDArray[np.float32], total: NDArray[np.float64]) -> None:
    # Column by column, so the summation order is the same for every block size.
    for j in range(block.shape[1]):
        column = block[:, j].astype(np.float64)
        total += column * column


def _scale(weight: float, squared_norms: NDArray[np.float64], selected: NDArray[np.int64] | None) -> NDArray[np.float32]:
    norms = np.sqrt(squared_norms if selected is None else squared_norms[selected])
    norms[norms == 0] = 1.0
    return (weight / norms).astype(np.float32)


def estimate_peak_bytes(
    n_nodes: int,
    undirected_edges: int,
    n_rows: int,
    config: FastRPConfig | None = None,
    *,
    block_columns: int = DEFAULT_BLOCK_COLUMNS,
    out_itemsize: int = 4,
) -> dict[str, int]:
    """Estimate peak array memory of building the adjacency and running ``fastrp``.

    Counts the NumPy/SciPy arrays alive at each phase's high-water mark. The measured
    process footprint runs above it by the interpreter and allocator overhead: see
    ``docs/embeddings.md`` for the scaling test. Returns the build and compute phase
    peaks and their max, in bytes.
    """
    config = config or FastRPConfig()
    entries = 2 * undirected_edges
    index = 4 if entries <= np.iinfo(np.int32).max else 8
    keys = 8 * n_nodes
    csr = entries * (index + 4) + (n_nodes + 1) * index
    # Buffered int32 edge pairs, CSR indices and indptr, and the per-node counts and
    # cursor, all live while the counting sort scatters.
    build = 8 * undirected_edges + entries * index + (n_nodes + 1) * index + 16 * n_nodes + keys
    width = min(block_columns, config.dim)
    active = sum(1 for w in config.weights if w)
    resident = csr + keys + 8 * n_nodes + n_rows * config.dim * out_itemsize  # + degree, output
    blocks = 2 * 4 * n_nodes * width  # a block and its product with P
    # Pass 1 (two-pass only): float64 squared norms per active power, plus the float64
    # column and its square while accumulating.
    first = 8 * active * n_nodes + 16 * n_nodes if block_columns < config.dim else 0
    # Pass 2: the returned rows' scales, the block accumulator, and the gathered rows;
    # single-pass also holds one power's float64 squared norms and temporaries.
    second = 4 * active * n_rows + 2 * 4 * n_rows * width + (0 if first else 24 * n_nodes)
    compute = resident + blocks + max(first, second)
    return {"build": build, "compute": compute, "peak": max(build, compute)}
