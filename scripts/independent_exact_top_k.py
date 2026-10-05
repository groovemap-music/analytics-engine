"""Bounded independent acceptance comparator; does not import the production kernel.

Input vectors must be ordered by artist_id so position breaks exact score ties.
The full candidate catalog is streamed once per bounded query batch.
"""

from __future__ import annotations

import numpy as np


def _select(scores, positions, k):
    count = min(k, len(scores))
    boundary = np.partition(scores, len(scores) - count)[len(scores) - count]
    above = np.flatnonzero(scores > boundary)
    tied = np.flatnonzero(scores == boundary)
    tied = tied[np.argsort(positions[tied], kind="stable")[: count - len(above)]]
    selected = np.concatenate((above, tied))
    order = np.lexsort((positions[selected], -scores[selected]))
    return scores[selected[order]], positions[selected[order]]


def independent_top_k(vectors, queries, *, k=10, query_rows=32, candidate_rows=65536):
    """Return exact cosine rankings using bounded float32 dot products and merges.

    At defaults the score matrix is at most 32*65536 float32 (~8MiB), not
    2000*9.37M. Zero vectors have zero cosine, and every query excludes itself.
    """
    if k < 1 or k >= len(vectors) or query_rows < 1 or candidate_rows < 1:
        raise ValueError("invalid catalog, k, or chunk sizes")
    queries = np.asarray(queries, dtype=np.int64)
    if np.any(queries < 0) or np.any(queries >= len(vectors)):
        raise ValueError("query outside catalog")
    result_positions = np.empty((len(queries), k), dtype=np.int64)
    result_scores = np.empty((len(queries), k), dtype=np.float32)
    for lo in range(0, len(queries), query_rows):
        current = queries[lo : lo + query_rows]
        query_vectors = np.array(vectors[current], dtype=np.float32)
        query_norms = np.linalg.norm(query_vectors, axis=1, keepdims=True)
        np.divide(query_vectors, query_norms, out=query_vectors, where=query_norms != 0)
        scores = [np.empty(0, dtype=np.float32) for _ in current]
        positions = [np.empty(0, dtype=np.int64) for _ in current]
        for start in range(0, len(vectors), candidate_rows):
            stop = min(len(vectors), start + candidate_rows)
            candidates = np.array(vectors[start:stop], dtype=np.float32)
            norms = np.linalg.norm(candidates, axis=1, keepdims=True)
            np.divide(candidates, norms, out=candidates, where=norms != 0)
            similarities = query_vectors @ candidates.T
            candidate_positions = np.arange(start, stop, dtype=np.int64)
            for row, own in enumerate(current):
                keep = candidate_positions != own
                if not np.any(keep):
                    continue
                local_scores, local_positions = _select(similarities[row, keep], candidate_positions[keep], k)
                scores[row], positions[row] = _select(
                    np.concatenate((scores[row], local_scores)), np.concatenate((positions[row], local_positions)), k
                )
        result_positions[lo : lo + len(current)] = np.stack(positions)
        result_scores[lo : lo + len(current)] = np.stack(scores)
    return result_positions, result_scores


def compare_stored_top10(vectors, queries, stored_positions, stored_scores, exact_positions, exact_scores, *, tolerance=1e-4):
    """Rescore each DB neighbour independently; DB scores never establish recall.

    All catalog positions refer to the same lexically ordered snapshot. Structural
    corruption raises; inaccurate persisted scores produce explicit failing metrics.
    """
    queries = np.asarray(queries, dtype=np.int64)
    shape = (len(queries), 10)
    if not len(queries) or any(array.shape != shape for array in (stored_positions, stored_scores, exact_positions, exact_scores)):
        raise ValueError("inconsistent sample dimensions")
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError("invalid tolerance")
    if np.any(queries < 0) or np.any(queries >= len(vectors)):
        raise ValueError("query outside catalog")
    for positions in (stored_positions, exact_positions):
        if not np.issubdtype(positions.dtype, np.integer) or np.any(positions < 0) or np.any(positions >= len(vectors)):
            raise ValueError("neighbour outside catalog")
    if not np.isfinite(stored_scores).all() or not np.isfinite(exact_scores).all():
        raise ValueError("nonfinite score")
    exact_hits = tolerant_hits = 0
    violations = order_violations = 0
    maximum_error = 0.0
    total = stored_positions.size
    per_query_recall = []
    for own, stored, persisted, expected, expected_scores in zip(
        queries, stored_positions, stored_scores, exact_positions, exact_scores, strict=True
    ):
        if len(set(stored)) != 10 or len(set(expected)) != 10:
            raise ValueError("duplicate neighbours")
        if own in stored or own in expected:
            raise ValueError("self neighbour")
        query = np.array(vectors[own], dtype=np.float32)
        candidates = np.array(vectors[stored], dtype=np.float32)
        if not np.isfinite(query).all() or not np.isfinite(candidates).all():
            raise ValueError("nonfinite vector")
        norm = np.linalg.norm(query)
        if norm:
            query /= norm
        norms = np.linalg.norm(candidates, axis=1, keepdims=True)
        np.divide(candidates, norms, out=candidates, where=norms != 0)
        independently_scored = candidates @ query
        errors = np.abs(independently_scored - persisted)
        violations += int(np.count_nonzero(errors > tolerance))
        maximum_error = max(maximum_error, float(errors.max()))
        if not np.array_equal(np.lexsort((stored, -persisted)), np.arange(10)):
            order_violations += 1
        expected_set = set(expected)
        exact_hits += sum(position in expected_set for position in stored)
        boundary = expected_scores[-1]
        hits = int(np.count_nonzero(independently_scored >= boundary - tolerance))
        tolerant_hits += hits
        per_query_recall.append(hits / 10)
    return {
        "queries": len(stored_positions),
        "exact_set_recall": exact_hits / total,
        "tie_tolerant_recall": tolerant_hits / total,
        "minimum_query_tie_tolerant_recall": min(per_query_recall),
        "score_max_abs_error": maximum_error,
        "score_violation_count": violations,
        "canonical_order_violations": order_violations,
        "passed": tolerant_hits == total and violations == 0 and order_violations == 0,
        "tolerance": tolerance,
    }
