"""Stable node identity and the row-normalized adjacency FastRP propagates over.

A catalog vertex is the pair ``(kind, key)`` that ``graph.vertex_degree`` keys on:
a one-character kind (``a`` artist, ``r`` release, ``l`` label, ``m`` master,
``g`` genre, ``s`` style) and the provider key read as text. ``node_key`` hashes
that pair to an unsigned 64-bit integer, and every other piece of the pipeline
works on those integers.

Node positions are the rank of each key in ascending order. That ordering is what
makes unchanged graph regions reproduce bit for bit when the rest of the graph
changes: adding or removing vertices shifts positions, but never reorders the
vertices that remain, so each row's neighbours stay in the same order and every
floating-point sum over them runs in the same order.

``AdjacencyBuilder`` accepts edges in blocks, the way a database cursor returns
them, and buffers only compact ``int32`` position pairs. ``build`` then assembles
the CSR matrix with a counting sort, so no ``int64`` coordinate arrays of the full
edge count are ever materialized.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np
import scipy.sparse as sp


if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy.typing import ArrayLike, NDArray


KEY_DTYPE: Final = np.uint64
POSITION_DTYPE: Final = np.int32
_MAX_NODES: Final = int(np.iinfo(np.int32).max)
# Rows per slice when writing the per-entry 1/degree values, bounding the
# temporary that np.repeat allocates.
_ROW_SLICE: Final = 1 << 20


def node_key(kind: str, key: str) -> int:
    """Return the stable unsigned 64-bit identity of vertex ``(kind, key)``.

    It is the first eight bytes, little-endian, of BLAKE2b over the UTF-8 bytes of
    ``kind``, a unit separator, and ``key``. It depends on nothing but the pair, so
    the same vertex gets the same identity in every monthly dump.
    """
    digest = hashlib.blake2b(f"{kind}\x1f{key}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def node_keys(vertices: Iterable[tuple[str, str]]) -> NDArray[np.uint64]:
    """Return ``node_key`` for every ``(kind, key)`` pair, as a ``uint64`` array."""
    return np.fromiter((node_key(kind, key) for kind, key in vertices), dtype=KEY_DTYPE)


class NodeIndex:
    """The graph's vertex set, positioned by ascending node key."""

    def __init__(self, keys: ArrayLike) -> None:
        ordered = np.sort(np.asarray(keys, dtype=KEY_DTYPE).ravel())
        if ordered.size > _MAX_NODES:
            raise ValueError(f"{ordered.size} nodes exceed the int32 position range")
        if ordered.size > 1 and bool(np.any(ordered[1:] == ordered[:-1])):
            duplicates = np.unique(ordered[1:][ordered[1:] == ordered[:-1]])
            raise ValueError(f"{duplicates.size} duplicate node keys, e.g. {int(duplicates[0]):#018x}")
        ordered.flags.writeable = False
        self.keys: NDArray[np.uint64] = ordered

    def __len__(self) -> int:
        return int(self.keys.size)

    def positions(self, keys: ArrayLike) -> NDArray[np.int32]:
        """Return the position of every key; raise ``KeyError`` if one is not a node."""
        wanted = np.asarray(keys, dtype=KEY_DTYPE)
        found = np.searchsorted(self.keys, wanted)
        clipped = np.minimum(found, max(len(self) - 1, 0))
        missing = (found >= len(self)) | (self.keys[clipped] != wanted) if len(self) else np.ones(wanted.shape, dtype=bool)
        if bool(np.any(missing)):
            raise KeyError(f"{int(np.count_nonzero(missing))} keys are not nodes, e.g. {int(wanted[missing].flat[0]):#018x}")
        return found.astype(POSITION_DTYPE)


@dataclass(frozen=True)
class Adjacency:
    """The random-walk transition matrix ``D^-1 A`` of an undirected, unweighted graph.

    ``transition`` is CSR with sorted column indices and ``float32`` entries
    ``1 / degree(row)``. ``degree`` counts distinct neighbours; parallel edges
    collapse to one and self-loops are dropped.
    """

    nodes: NodeIndex
    transition: sp.csr_array
    degree: NDArray[np.int64]

    @property
    def undirected_edges(self) -> int:
        return int(self.transition.nnz // 2)


class AdjacencyBuilder:
    """Accumulate undirected edges in blocks, then build an ``Adjacency`` once."""

    def __init__(self, nodes: NodeIndex) -> None:
        self._nodes = nodes
        self._blocks: list[tuple[NDArray[np.int32], NDArray[np.int32]]] = []
        self._built = False

    def add_edges(self, source_keys: ArrayLike, target_keys: ArrayLike) -> None:
        """Add one block of edges given as node-key pairs."""
        self.add_edge_positions(self._nodes.positions(source_keys), self._nodes.positions(target_keys))

    def add_edge_positions(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add one block of edges given as node-position pairs. Direction is ignored."""
        if self._built:
            raise RuntimeError("the adjacency has already been built")
        src = np.asarray(sources)
        dst = np.asarray(targets)
        if src.ndim != 1 or src.shape != dst.shape:
            raise ValueError("sources and targets must be one-dimensional and the same length")
        if src.size and (min(int(src.min()), int(dst.min())) < 0 or max(int(src.max()), int(dst.max())) >= len(self._nodes)):
            raise ValueError("edge endpoint outside the node range")
        keep = src != dst
        self._blocks.append((src[keep].astype(POSITION_DTYPE), dst[keep].astype(POSITION_DTYPE)))

    def build(self) -> Adjacency:
        """Assemble the transition matrix and release the buffered edge blocks."""
        if self._built:
            raise RuntimeError("the adjacency has already been built")
        self._built = True
        n = len(self._nodes)
        counts = np.zeros(n, dtype=np.int64)
        for src, dst in self._blocks:
            counts += np.bincount(src, minlength=n)
            counts += np.bincount(dst, minlength=n)
        nnz = int(counts.sum())
        index_dtype = np.int32 if nnz <= _MAX_NODES else np.int64
        indptr: NDArray[np.integer] = np.zeros(n + 1, dtype=index_dtype)
        np.cumsum(counts, out=indptr[1:])
        del counts
        cursor = indptr[:-1].astype(np.int64)
        indices: NDArray[np.integer] = np.empty(nnz, dtype=index_dtype)
        while self._blocks:
            src, dst = self._blocks.pop()
            _scatter(src, dst, cursor, indices)
            _scatter(dst, src, cursor, indices)
        del cursor
        transition = sp.csr_array((np.ones(nnz, dtype=np.float32), indices, indptr), shape=(n, n), copy=False)
        # Sorts each row's column indices (so sums run in node-key order whatever
        # order the edges arrived in) and collapses parallel edges.
        transition.sum_duplicates()
        degree = np.diff(transition.indptr).astype(np.int64)
        inverse = (1.0 / np.maximum(degree, 1).astype(np.float64)).astype(np.float32)
        for start in range(0, n, _ROW_SLICE):
            stop = min(start + _ROW_SLICE, n)
            lo, hi = int(transition.indptr[start]), int(transition.indptr[stop])
            transition.data[lo:hi] = np.repeat(inverse[start:stop], degree[start:stop])
        return Adjacency(nodes=self._nodes, transition=transition, degree=degree)


def _scatter(rows: NDArray[np.int32], cols: NDArray[np.int32], cursor: NDArray[np.int64], indices: NDArray[np.integer]) -> None:
    """Counting-sort one block of directed entries into their CSR row slots."""
    if not rows.size:
        return
    order = np.argsort(rows, kind="stable")
    sorted_rows = rows[order]
    starts = np.flatnonzero(np.diff(sorted_rows, prepend=-1))
    sizes = np.diff(starts, append=sorted_rows.size)
    rank = np.arange(sorted_rows.size, dtype=np.int64) - np.repeat(starts, sizes)
    indices[cursor[sorted_rows] + rank] = cols[order]
    cursor[sorted_rows[starts]] += sizes
