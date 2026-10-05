"""Synthetic independent-comparator validation; no real catalog payloads."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest


spec = importlib.util.spec_from_file_location("independent", Path(__file__).parents[1] / "scripts/independent_exact_top_k.py")
assert spec and spec.loader
independent = importlib.util.module_from_spec(spec)
spec.loader.exec_module(independent)


def test_every_candidate_chunk_matches_brute_force_with_self_and_ties():
    rng = np.random.default_rng(123)
    vectors = rng.normal(size=(73, 8)).astype(np.float16)
    vectors[0] = 0
    vectors[-1] = vectors[1]
    queries = [0, 1, 19, 72]
    positions, scores = independent.independent_top_k(vectors, queries, k=10, query_rows=2, candidate_rows=11)
    normalized = vectors.astype(np.float32)
    norms = np.linalg.norm(normalized, axis=1, keepdims=True)
    np.divide(normalized, norms, out=normalized, where=norms != 0)
    for row, query in enumerate(queries):
        brute = normalized @ normalized[query]
        brute[query] = -np.inf
        order = np.lexsort((np.arange(len(vectors)), -brute))[:10]
        np.testing.assert_array_equal(positions[row], order)
        np.testing.assert_allclose(scores[row], brute[order], atol=1e-6)
        assert query not in positions[row]
    assert positions[1, 0] == 72  # Final candidate chunk cannot be omitted.
    np.testing.assert_array_equal(positions[0], np.arange(1, 11))


def test_real_rescoring_rejects_forged_db_scores_even_for_exact_members():
    vectors = np.eye(14, dtype=np.float16)
    positions, scores = independent.independent_top_k(vectors, [13], k=10, candidate_rows=3)
    result = independent.compare_stored_top10(vectors, [13], positions, scores, positions, scores)
    assert result["passed"]
    forged = scores.copy()
    forged[0, 0] = 0.75
    result = independent.compare_stored_top10(vectors, [13], positions, forged, positions, scores)
    assert result["exact_set_recall"] == 1.0
    assert result["score_violation_count"] == 1
    assert result["score_max_abs_error"] == 0.75
    assert not result["passed"]


@pytest.mark.parametrize(
    "corruption,error",
    [("duplicate", "duplicate"), ("self", "self"), ("outofrange", "outside"), ("nonfinite", "nonfinite"), ("cardinality", "dimensions")],
)
def test_structural_db_corruption_cannot_be_accepted(corruption, error):
    vectors = np.eye(14, dtype=np.float16)
    expected, expected_scores = independent.independent_top_k(vectors, [13], k=10)
    positions, scores = expected.copy(), expected_scores.copy()
    if corruption == "duplicate":
        positions[0, 1] = positions[0, 0]
    elif corruption == "self":
        positions[0, 1] = 13
    elif corruption == "outofrange":
        positions[0, 1] = 14
    elif corruption == "nonfinite":
        scores[0, 1] = np.nan
    else:
        positions = positions[:, :9]
    with pytest.raises(ValueError, match=error):
        independent.compare_stored_top10(vectors, [13], positions, scores, expected, expected_scores)


def test_equal_scores_require_lexical_order_and_boundary_hits_use_real_vectors():
    vectors = np.eye(14, dtype=np.float16)
    expected, scores = independent.independent_top_k(vectors, [13], k=10)
    reordered = expected[:, ::-1]
    result = independent.compare_stored_top10(vectors, [13], reordered, scores, expected, scores)
    assert result["canonical_order_violations"] == 1
    assert not result["passed"]
    # A non-member with genuinely equal score is an allowed boundary tie.
    tied = expected.copy()
    tied[0, -1] = 11
    result = independent.compare_stored_top10(vectors, [13], tied, scores, expected, scores)
    assert result["exact_set_recall"] == 0.9
    assert result["tie_tolerant_recall"] == 1.0
    assert result["passed"]
    # Giving that neighbour an adverse true cosine cannot be rescued by a forged DB score.
    vectors[11] = -vectors[13]
    result = independent.compare_stored_top10(vectors, [13], tied, scores, expected, scores)
    assert result["tie_tolerant_recall"] == 0.9
    assert result["score_violation_count"] == 1
    assert not result["passed"]
