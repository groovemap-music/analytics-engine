"""Exact all-pairs top-K cosine neighbours over a whole embedding catalog.

Every artist's ``k`` most cosine-similar other artists (self excluded), computed by
brute force -- no index, no approximation -- for the precomputed similar-artist lists
the D serving-mode verdict chose (``docs/embedding_tie_break.md``). The method and its
measured sizing are in ``docs/similar_artists.md``; the short version:

- Vectors stay in their half-precision storage (``n x d`` ``float16``, 2.4 GB for the
  9.37M-artist catalog). Each row block is upcast to ``float32`` and scaled by its
  row's reciprocal norm just before it is multiplied, so every score is a ``float32``
  dot product of ``float32``-normalized vectors: the same number an independent
  ``normalize(v.astype(float32))`` brute force computes.
- The score matrix is symmetric, so only the upper block triangle ``(i, j), j >= i``
  is multiplied. Each ``B x B`` block of scores updates block ``i``'s running lists
  from its rows and block ``j``'s from its columns, halving the multiply work.
- Row blocks are finalized in order. Once every task ``(i', i)`` with ``i' <= i`` has
  run, block ``i`` has seen every column and its lists are final, so they are handed
  to ``on_block`` and can be streamed out while later blocks are still running. That
  is also the checkpoint boundary: ``state`` plus the next block index resumes a run.
- The per-task tasks for one ``i`` run on a thread pool. NumPy releases the GIL in
  BLAS and in the elementwise passes, so the pool overlaps the multiply with
  candidate selection. The running lists are merged under a per-block lock; scores
  and candidate masks are computed outside it, against a possibly stale threshold,
  which is safe because thresholds only ever rise (a stale one admits a superset).

Candidate selection keeps the per-score work to one comparison against each row's
current ``k``-th best score. Only the few scores that clear it are gathered and merged
into that row's list; rows seeing a block for the first time, where nearly every score
clears ``-inf``, are first cut to their block-local top ``k`` with ``np.partition``.

Tie order is deterministic and independent of scheduling: lists are ordered by score
descending, then by catalog position ascending, and the ``>=`` threshold test admits
exact ties so the lower position always wins a tie at the ``k``-th place, whichever
block reached the row first.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Callable

    from numpy.typing import NDArray


DEFAULT_K: Final = 50
DEFAULT_BLOCK_ROWS: Final = 4096
EMPTY_POSITION: Final = -1
SEGMENT: Final = 64
_PAD_POSITION: Final = np.iinfo(np.int32).max
# The threshold floor: below every real cosine score but above the ``-inf`` that marks
# self and unfilled slots, so ``scores >= threshold`` never admits either.
_FLOOR: Final = np.float32(np.finfo(np.float32).min)


@dataclass
class TopKState:
    """The running per-row lists: ``scores[r]`` / ``positions[r]`` hold row ``r``'s best
    ``k`` so far, unsorted, with ``-inf`` / ``EMPTY_POSITION`` in unfilled slots, and
    ``thresholds[r]`` its current ``k``-th best score (a finite floor until ``k`` are
    known)."""

    scores: NDArray[np.float32]
    positions: NDArray[np.int32]
    thresholds: NDArray[np.float32]

    @classmethod
    def empty(cls, n_rows: int, k: int) -> TopKState:
        return cls(
            scores=np.full((n_rows, k), -np.inf, dtype=np.float32),
            positions=np.full((n_rows, k), EMPTY_POSITION, dtype=np.int32),
            thresholds=np.full(n_rows, _FLOOR, dtype=np.float32),
        )


def reciprocal_norms(vectors: NDArray[np.floating], *, block_rows: int = 1 << 16) -> NDArray[np.float32]:
    """Each row's ``1 / ||v||`` in ``float32`` (``0`` for a zero row, so it scores ``0``
    against everything rather than ``nan``), computed ``block_rows`` at a time so a
    ``float16`` catalog is never upcast whole."""
    out = np.empty(vectors.shape[0], dtype=np.float32)
    for start in range(0, vectors.shape[0], block_rows):
        block = vectors[start : start + block_rows].astype(np.float32)
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))
        with np.errstate(divide="ignore"):
            out[start : start + block_rows] = np.where(norms > 0, 1.0 / norms, 0.0)
    return out


def _rank_keys(scores: NDArray[np.float32], positions: NDArray[np.int32]) -> NDArray[np.int64]:
    """One ``int64`` per entry ordering exactly as (score desc, position asc): the high
    word is ``-score``'s bits mapped to an order-preserving signed integer, the low word
    the position. A single-key partition or sort is several times faster than
    ``np.lexsort`` on two keys, and this is on the per-task merge path."""
    negated = np.negative(scores) + np.float32(0.0)  # + 0.0 folds -0.0 into 0.0: they tie.
    bits = negated.view(np.int32)
    high = bits ^ ((bits >> 31) & np.int32(0x7FFFFFFF))
    return (high.astype(np.int64) << 32) | (positions.astype(np.int64) & 0xFFFFFFFF)


def sorted_lists(scores: NDArray[np.float32], positions: NDArray[np.int32]) -> tuple[NDArray[np.float32], NDArray[np.int32]]:
    """Order each row by score descending, then position ascending -- the published
    rank order. Unfilled slots (``-inf``, ``EMPTY_POSITION``) sort last."""
    order = np.argsort(_rank_keys(scores, positions), axis=-1)
    return np.take_along_axis(scores, order, axis=-1), np.take_along_axis(positions, order, axis=-1)


def _merge(
    state: TopKState, rows: NDArray[np.intp], cand_rows: NDArray[np.intp], cand_pos: NDArray[np.int32], cand_scores: NDArray[np.float32]
) -> None:
    """Merge candidates into ``state`` for global ``rows``. ``cand_rows`` indexes into
    ``rows`` and is sorted ascending; each row keeps its best ``k`` of old + new by
    (score desc, position asc)."""
    k = state.scores.shape[1]
    counts = np.bincount(cand_rows, minlength=len(rows))
    width = int(counts.max())
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    slot = np.arange(len(cand_rows)) - starts[cand_rows]
    pad_scores = np.full((len(rows), k + width), -np.inf, dtype=np.float32)
    pad_pos = np.full((len(rows), k + width), _PAD_POSITION, dtype=np.int32)
    pad_scores[:, :k] = state.scores[rows]
    pad_pos[:, :k] = state.positions[rows]
    pad_scores[cand_rows, k + slot] = cand_scores
    pad_pos[cand_rows, k + slot] = cand_pos
    order = np.argpartition(_rank_keys(pad_scores, pad_pos), k - 1, axis=-1)[:, :k]
    best_scores = np.take_along_axis(pad_scores, order, axis=-1)
    best_pos = np.take_along_axis(pad_pos, order, axis=-1)
    best_pos[best_pos == _PAD_POSITION] = EMPTY_POSITION
    state.scores[rows] = best_scores
    state.positions[rows] = best_pos
    state.thresholds[rows] = np.maximum(best_scores.min(axis=-1), _FLOOR)


def _update(
    state: TopKState,
    lock: threading.Lock,
    row_start: int,
    col_start: int,
    scores: NDArray[np.float32],
    valid_rows: int,
    segments: NDArray[np.float32],
) -> None:
    """Fold a padded block of scores into the lists of rows ``row_start + r`` for
    ``r < valid_rows`` (candidates are columns ``col_start + c``).

    ``scores`` may be a transposed view. ``segments`` is the same block viewed as
    ``(row, segment, SEGMENT)`` -- possibly with the last two axes swapped in memory --
    so that ``segments[r, s]`` is row ``r``'s ``s``-th run of SEGMENT scores.
    """
    n_rows, n_cols = scores.shape
    k = state.scores.shape[1]
    thresholds = np.full(n_rows, np.inf, dtype=np.float32)
    thresholds[:valid_rows] = state.thresholds[row_start : row_start + valid_rows]
    parts: list[tuple[NDArray[np.intp], NDArray[np.intp], NDArray[np.float32]]] = []

    unfilled = np.flatnonzero(thresholds == _FLOOR)
    if unfilled.size:
        # First sight of these rows: nearly every score clears the floor, so cut straight
        # to the block-local k-th best, keeping ties at the cut for position to decide.
        # They are then kept out of the segment pass below.
        full = scores[unfilled]
        cut = min(k, n_cols)
        kth = np.maximum(np.partition(full, n_cols - cut, axis=1)[:, n_cols - cut], _FLOOR)
        rows, cols = np.nonzero(full >= kth[:, None])
        parts.append((unfilled[rows], cols, full[rows, cols]))
        thresholds[unfilled] = np.inf

    # Everything else: one max per SEGMENT run, and only the runs whose max clears the
    # row's threshold are looked at score by score.
    hit_rows, hit_segs = np.nonzero(segments.max(axis=2) >= thresholds[:, None])
    if hit_rows.size:
        vals = segments[hit_rows, hit_segs]
        sel_r, sel_c = np.nonzero(vals >= thresholds[hit_rows, None])
        parts.append((hit_rows[sel_r], hit_segs[sel_r] * SEGMENT + sel_c, vals[sel_r, sel_c]))

    rows = np.concatenate([p[0] for p in parts]) if parts else np.empty(0, dtype=np.intp)
    if not rows.size:
        return
    order = np.argsort(rows, kind="stable")
    touched, local_rows = np.unique(rows[order], return_inverse=True)
    cols = np.concatenate([p[1] for p in parts])[order]
    cand_scores = np.concatenate([p[2] for p in parts])[order]
    with lock:
        _merge(state, row_start + touched, local_rows, (col_start + cols).astype(np.int32), cand_scores)


def exact_top_k(
    vectors: NDArray[np.floating],
    k: int = DEFAULT_K,
    *,
    block_rows: int = DEFAULT_BLOCK_ROWS,
    threads: int = 8,
    state: TopKState | None = None,
    start_block: int = 0,
    on_block: Callable[[int, NDArray[np.float32], NDArray[np.int32]], None] | None = None,
    after_block: Callable[[int, TopKState], None] | None = None,
    inv_norms: NDArray[np.float32] | None = None,
) -> TopKState:
    """Exact top-``k`` cosine neighbours of every row of ``vectors`` among all other rows.

    ``on_block(row_start, scores, positions)`` receives each finalized row block's
    lists, sorted (see ``sorted_lists``); ``after_block(next_block, state)`` runs after
    it, for checkpointing. Pass a checkpointed ``state`` with ``start_block`` to resume.
    Returns the final state (unsorted lists).
    """
    if block_rows % SEGMENT:
        raise ValueError(f"block_rows must be a multiple of {SEGMENT}, got {block_rows}")
    n_rows = vectors.shape[0]
    if inv_norms is None:
        inv_norms = reciprocal_norms(vectors)
    if state is None:
        state = TopKState.empty(n_rows, k)
    n_blocks = -(-n_rows // block_rows)
    locks = [threading.Lock() for _ in range(n_blocks)]

    def upcast(block: int) -> NDArray[np.float32]:
        # Zero-padded to a whole number of segments; padding scores are set to -inf.
        lo, hi = block * block_rows, min(n_rows, (block + 1) * block_rows)
        out = np.zeros((-(-(hi - lo) // SEGMENT) * SEGMENT, vectors.shape[1]), dtype=np.float32)
        np.multiply(vectors[lo:hi], inv_norms[lo:hi, None], out=out[: hi - lo])
        return out

    def task(i: int, left: NDArray[np.float32], j: int) -> None:
        right = left if j == i else upcast(j)
        scores = left @ right.T
        valid_i = min(n_rows - i * block_rows, block_rows)
        valid_j = min(n_rows - j * block_rows, block_rows)
        scores[valid_i:] = -np.inf
        scores[:, valid_j:] = -np.inf
        if j == i:
            np.fill_diagonal(scores, -np.inf)
        # Each segment is one list wide -- a run of one row, or of one column -- so it is
        # tested against that list's own threshold. (Taller tiles are cheaper to reduce but
        # test against the lowest threshold among their lists, and thresholds vary too much
        # between artists for that to prune: 94% of 8 x 64 tiles still hit on real data.)
        _update(state, locks[i], i * block_rows, j * block_rows, scores, valid_i, scores.reshape(scores.shape[0], -1, SEGMENT))
        if j != i:
            # (column, row group, row in group): reducing over the last axis walks
            # contiguous rows, the cheap direction.
            by_column = scores.reshape(-1, SEGMENT, scores.shape[1]).transpose(2, 0, 1)
            _update(state, locks[j], j * block_rows, i * block_rows, scores.T, valid_j, by_column)

    with ThreadPoolExecutor(max_workers=threads) as pool:
        for i in range(start_block, n_blocks):
            left = upcast(i)
            for future in [pool.submit(task, i, left, j) for j in range(i, n_blocks)]:
                future.result()
            lo, hi = i * block_rows, min(n_rows, (i + 1) * block_rows)
            if on_block is not None:
                on_block(lo, *sorted_lists(state.scores[lo:hi], state.positions[lo:hi]))
            if after_block is not None:
                after_block(i + 1, state)
    return state
