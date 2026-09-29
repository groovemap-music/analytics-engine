"""Tests for the exact all-pairs top-K kernel.

Every vector here is synthetic. No provider data, and nothing derived from it, may
appear in this repository (ADR 0013, data rights).
"""

from __future__ import annotations

import numpy as np
import pytest

from insights.embeddings.exact_top_k import EMPTY_POSITION, SEGMENT, TopKState, exact_top_k, reciprocal_norms, sorted_lists


def _brute_force(vectors: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """The whole score matrix, sorted by (score desc, position asc), self excluded."""
    normalized = vectors.astype(np.float32) * reciprocal_norms(vectors)[:, None]
    scores = normalized @ normalized.T
    np.fill_diagonal(scores, -np.inf)
    positions = np.broadcast_to(np.arange(len(vectors), dtype=np.int32), scores.shape)
    order = np.lexsort((positions, -scores), axis=-1)[:, :k]
    return np.take_along_axis(scores, order, axis=-1), order.astype(np.int32)


def _collect(vectors: np.ndarray, k: int, **kwargs: object) -> tuple[np.ndarray, np.ndarray]:
    blocks: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exact_top_k(vectors, k, on_block=lambda lo, s, p: blocks.__setitem__(lo, (s, p)), **kwargs)  # type: ignore[arg-type]
    starts = sorted(blocks)
    return np.concatenate([blocks[s][0] for s in starts]), np.concatenate([blocks[s][1] for s in starts])


def _integer_vectors(n: int, dim: int, seed: int) -> np.ndarray:
    """Vectors of four 0.5 entries: unit norm exactly, and every dot product a multiple
    of 0.25, so each score is exact in float32 whatever the summation order -- ties are
    real ties, not accidents of rounding."""
    rng = np.random.default_rng(seed)
    vectors = np.zeros((n, dim), dtype=np.float16)
    for row in vectors:
        row[rng.choice(dim, size=4, replace=False)] = 0.5
    return vectors


@pytest.mark.parametrize(
    ("n", "dim", "block_rows", "k", "threads"),
    [
        (1500, 16, 256, 10, 4),  # several full blocks and a ragged last one
        (700, 8, 128, 50, 3),
        (64, 4, 64, 50, 1),  # one block, fewer than k other rows
        (2000, 32, 1024, 50, 2),
        (901, 16, 192, 7, 4),
    ],
)
def test_matches_brute_force(n: int, dim: int, block_rows: int, k: int, threads: int) -> None:
    vectors = np.random.default_rng(n).standard_normal((n, dim)).astype(np.float16)
    scores, positions = _collect(vectors, k, block_rows=block_rows, threads=threads)
    want_k = min(k, n - 1)
    expected_scores, expected_positions = _brute_force(vectors, want_k)

    np.testing.assert_allclose(scores[:, :want_k], expected_scores, atol=1e-6)
    # Continuous random vectors: positions agree except where two scores differ only in
    # the last ulp (BLAS block shapes sum in different orders), and then the set agrees.
    for row in range(n):
        assert set(positions[row, :want_k]) == set(expected_positions[row])
    assert (positions[:, want_k:] == EMPTY_POSITION).all()
    assert np.isneginf(scores[:, want_k:]).all()


def test_excludes_self_even_when_a_duplicate_exists() -> None:
    vectors = np.random.default_rng(0).standard_normal((300, 8)).astype(np.float16)
    vectors[150] = vectors[10]  # an exact duplicate must still appear, at score 1
    _scores, positions = _collect(vectors, 20, block_rows=64, threads=4)

    assert not (positions == np.arange(300)[:, None]).any()
    assert positions[10, 0] == 150
    assert positions[150, 0] == 10


def test_ties_order_by_position_and_do_not_depend_on_blocking() -> None:
    vectors = _integer_vectors(900, 12, seed=3)
    expected_scores, expected_positions = _brute_force(vectors, 25)

    for block_rows, threads in [(64, 1), (128, 4), (320, 8), (960, 3)]:
        scores, positions = _collect(vectors, 25, block_rows=block_rows, threads=threads)
        np.testing.assert_array_equal(scores, expected_scores)
        np.testing.assert_array_equal(positions, expected_positions)


def test_zero_vector_scores_zero_against_everything() -> None:
    vectors = np.random.default_rng(1).standard_normal((100, 8)).astype(np.float16)
    vectors[5] = 0
    scores, positions = _collect(vectors, 10, block_rows=64, threads=2)

    np.testing.assert_array_equal(scores[5], np.zeros(10, dtype=np.float32))
    np.testing.assert_array_equal(positions[5], [0, 1, 2, 3, 4, 6, 7, 8, 9, 10])  # all tied at 0: lowest positions win


def test_resumes_from_a_checkpoint_to_the_same_answer() -> None:
    vectors = np.random.default_rng(2).standard_normal((700, 16)).astype(np.float16)
    expected = _collect(vectors, 10, block_rows=128, threads=2)

    saved: dict[int, TopKState] = {}

    def checkpoint(next_block: int, state: TopKState) -> None:
        if next_block == 3:
            saved[next_block] = TopKState(state.scores.copy(), state.positions.copy(), state.thresholds.copy())

    first: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exact_top_k(vectors, 10, block_rows=128, threads=2, on_block=lambda lo, s, p: first.__setitem__(lo, (s, p)), after_block=checkpoint)
    resumed: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    exact_top_k(vectors, 10, block_rows=128, threads=2, state=saved[3], start_block=3, on_block=lambda lo, s, p: resumed.__setitem__(lo, (s, p)))

    assert sorted(resumed) == [3 * 128, 4 * 128, 5 * 128]
    combined = {**{lo: first[lo] for lo in first if lo < 3 * 128}, **resumed}
    np.testing.assert_array_equal(np.concatenate([combined[lo][1] for lo in sorted(combined)]), expected[1])


def test_block_rows_must_be_a_whole_number_of_segments() -> None:
    with pytest.raises(ValueError, match="multiple of"):
        exact_top_k(np.ones((10, 4), dtype=np.float16), 3, block_rows=SEGMENT + 1)


def test_sorted_lists_puts_unfilled_slots_last() -> None:
    scores = np.array([[-np.inf, 0.5, 0.5, 0.9]], dtype=np.float32)
    positions = np.array([[EMPTY_POSITION, 7, 3, 9]], dtype=np.int32)
    sorted_scores, sorted_positions = sorted_lists(scores, positions)

    np.testing.assert_array_equal(sorted_positions, [[9, 3, 7, EMPTY_POSITION]])
    np.testing.assert_array_equal(sorted_scores, np.array([[0.9, 0.5, 0.5, -np.inf]], dtype=np.float32))
