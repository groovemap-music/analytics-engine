"""Tests for deterministic FastRP embeddings.

Every graph here is synthetic. No provider data, and nothing derived from it, may
appear in this repository (ADR 0013, data rights).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import scipy.sparse as sp

from insights.embeddings import (
    AdjacencyBuilder,
    FastRPConfig,
    HashedProjection,
    NodeIndex,
    estimate_peak_bytes,
    fastrp,
    node_key,
    node_keys,
)
from insights.embeddings.fastrp import DEFAULT_SEED, _Propagator


GiB = 1 << 30


def synthetic_catalog(seed: int, releases: int = 3000, prefix: str = "") -> list[tuple[tuple[str, str], tuple[str, str]]]:
    """Release-centred bipartite edges shaped like the catalog graph: artists with a
    skewed popularity, labels, genres, styles, and masters."""
    rng = np.random.default_rng(seed)
    artists = max(releases // 3, 2)
    popularity = 1.0 / np.arange(1, artists + 1) ** 0.8
    popularity /= popularity.sum()
    edges = []
    for r in range(releases):
        release = ("r", f"{prefix}{r}")
        for a in rng.choice(artists, size=int(rng.integers(1, 4)), p=popularity):
            edges.append((release, ("a", f"{prefix}{a}")))
        edges.append((release, ("l", f"{prefix}{rng.integers(releases // 12 + 1)}")))
        edges.append((release, ("g", f"{prefix}{rng.integers(15)}")))
        edges.extend((release, ("s", f"{prefix}{s}")) for s in rng.integers(0, 200, size=int(rng.integers(1, 3))))
        if rng.random() < 0.5:
            edges.append((release, ("m", f"{prefix}{r // 2}")))
    return edges


def build(
    edges: list[tuple[tuple[str, str], tuple[str, str]]], *, block: int = 1000, extra_vertices: tuple[tuple[str, str], ...] = ()
) -> AdjacencyBuilder:
    vertices = sorted({v for edge in edges for v in edge} | set(extra_vertices))
    builder = AdjacencyBuilder(NodeIndex(node_keys(vertices)))
    sources = node_keys(s for s, _ in edges)
    targets = node_keys(t for _, t in edges)
    for start in range(0, len(edges), block):
        builder.add_edges(sources[start : start + block], targets[start : start + block])
    return builder


def rows_by_vertex(edges: list[tuple[tuple[str, str], tuple[str, str]]], embedding: np.ndarray, nodes: NodeIndex) -> dict[tuple[str, str], bytes]:
    vertices = sorted({v for edge in edges for v in edge})
    positions = nodes.positions(node_keys(vertices))
    return {v: embedding[p].tobytes() for v, p in zip(vertices, positions, strict=True)}


def spike_fastrp(adjacency: sp.csr_array, projection: np.ndarray, weights: tuple[float, ...]) -> np.ndarray:
    """The spike's embed.fastrp (design docs/spikes/gm-design-chw.2/embed.py, MIT), with
    its global random draw replaced by a given R and beta fixed at 0."""
    deg = np.asarray(adjacency.sum(axis=1)).ravel().astype(np.float64)
    deg[deg == 0] = 1.0
    p = (sp.diags((1.0 / deg).astype(np.float32)) @ adjacency).tocsr()
    out = np.zeros(projection.shape, dtype=np.float32)
    current = projection
    for w in weights:
        current = p @ current
        if w:
            norms = np.linalg.norm(current, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out += (w / norms) * current
    return out


class FixedProjection:
    def __init__(self, matrix: np.ndarray) -> None:
        self.matrix = matrix

    def columns(self, start: int, stop: int, rows: slice = slice(None)) -> np.ndarray:
        return np.ascontiguousarray(self.matrix[rows, start:stop])


@pytest.fixture(scope="module")
def catalog_edges() -> list[tuple[tuple[str, str], tuple[str, str]]]:
    return synthetic_catalog(seed=11)


class TestNodeIdentity:
    def test_node_key_is_pinned(self) -> None:
        # Changing this value re-keys every projection row; it must bump FASTRP_ALGORITHM_VERSION.
        assert node_key("a", "1") == 0xC13CD84A6BC18181
        assert node_key("r", "25802761") == 0x389AF9597E6BF572

    def test_kind_is_part_of_identity(self) -> None:
        assert node_key("a", "1") != node_key("r", "1")
        assert node_keys([("a", "1"), ("r", "1")]).dtype == np.uint64

    def test_positions_follow_key_order(self) -> None:
        nodes = NodeIndex(np.array([30, 10, 20], dtype=np.uint64))
        assert nodes.positions([10, 20, 30]).tolist() == [0, 1, 2]
        assert len(nodes) == 3

    def test_duplicate_keys_are_rejected(self) -> None:
        with pytest.raises(ValueError, match="duplicate node keys"):
            NodeIndex(np.array([1, 2, 2], dtype=np.uint64))

    def test_unknown_key_is_rejected(self) -> None:
        nodes = NodeIndex(np.array([10, 20], dtype=np.uint64))
        for missing in ([15], [5], [25]):
            with pytest.raises(KeyError, match="not nodes"):
                nodes.positions(missing)
        with pytest.raises(KeyError):
            NodeIndex(np.array([], dtype=np.uint64)).positions([1])


class TestAdjacency:
    def test_parallel_edges_collapse_and_self_loops_drop(self) -> None:
        builder = AdjacencyBuilder(NodeIndex(np.arange(4, dtype=np.uint64)))
        builder.add_edge_positions([0, 1, 0, 2], [1, 0, 1, 2])
        builder.add_edge_positions([1], [2])
        adjacency = builder.build()
        assert adjacency.degree.tolist() == [1, 2, 1, 0]
        assert adjacency.undirected_edges == 2
        assert adjacency.transition.toarray().tolist() == [[0, 1, 0, 0], [0.5, 0, 0.5, 0], [0, 1, 0, 0], [0, 0, 0, 0]]
        assert adjacency.transition.has_sorted_indices

    def test_edge_order_and_blocking_do_not_matter(self, catalog_edges: list) -> None:
        shuffled = [catalog_edges[i] for i in np.random.default_rng(3).permutation(len(catalog_edges))]
        reversed_edges = [(t, s) for s, t in shuffled] + shuffled[:500]
        first = build(catalog_edges).build().transition
        second = build(reversed_edges, block=77).build().transition
        for attribute in ("indptr", "indices", "data"):
            assert np.array_equal(getattr(first, attribute), getattr(second, attribute))

    def test_builder_rejects_bad_input_and_reuse(self) -> None:
        builder = AdjacencyBuilder(NodeIndex(np.arange(3, dtype=np.uint64)))
        with pytest.raises(ValueError, match="same length"):
            builder.add_edge_positions([0, 1], [1])
        with pytest.raises(ValueError, match="outside the node range"):
            builder.add_edge_positions([0], [3])
        builder.add_edge_positions([], [])
        builder.build()
        with pytest.raises(RuntimeError):
            builder.add_edge_positions([0], [1])
        with pytest.raises(RuntimeError):
            builder.build()


class TestProjection:
    def test_projection_is_pinned(self) -> None:
        # Signs of R for four artists and eight columns under the pinned seed. A change
        # here changes every embedding and must bump FASTRP_ALGORITHM_VERSION.
        keys = node_keys(("a", str(i)) for i in range(4))
        signs = np.sign(HashedProjection(keys, DEFAULT_SEED).columns(0, 8)).astype(int).tolist()
        assert signs == [
            [0, 0, 0, 0, 0, 0, 1, 1],
            [1, 1, 0, 1, 0, 0, 0, -1],
            [-1, 0, 0, -1, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0],
        ]

    def test_distribution_is_very_sparse_s3(self) -> None:
        keys = np.random.default_rng(0).integers(0, 2**63, size=50_000, dtype=np.uint64)
        values = HashedProjection(keys, DEFAULT_SEED).columns(0, 8)
        assert set(np.unique(values).tolist()) == {float(np.float32(-np.sqrt(3))), 0.0, float(np.float32(np.sqrt(3)))}
        assert abs(np.mean(values < 0) - 1 / 6) < 0.005
        assert abs(np.mean(values > 0) - 1 / 6) < 0.005
        assert abs(np.corrcoef(values[:, 0], values[:, 1])[0, 1]) < 0.02

    def test_rows_depend_only_on_key_and_seed(self) -> None:
        keys = np.arange(100, 110, dtype=np.uint64)
        full = HashedProjection(keys, 5).columns(0, 16)
        assert np.array_equal(HashedProjection(keys[::-1], 5).columns(0, 16), full[::-1])
        assert np.array_equal(HashedProjection(keys, 5).columns(4, 9), full[:, 4:9])
        assert np.array_equal(HashedProjection(keys, 5).columns(4, 9, slice(3, 7)), full[3:7, 4:9])
        assert not np.array_equal(HashedProjection(keys, 6).columns(0, 16), full)

    def test_beta_scales_rows_by_degree(self) -> None:
        keys = np.arange(1, 400, dtype=np.uint64)
        degree = np.arange(399, dtype=np.int64)
        plain = HashedProjection(keys, 1).columns(0, 4)
        scaled = HashedProjection(keys, 1, degree=degree, beta=-0.5).columns(0, 4)
        expected = plain * (np.maximum(degree, 1).astype(np.float64) ** -0.5).astype(np.float32)[:, None]
        assert np.array_equal(scaled, expected)
        with pytest.raises(ValueError, match="degrees"):
            HashedProjection(keys, 1, beta=-0.5)
        with pytest.raises(ValueError, match="seed"):
            HashedProjection(keys, -1)


class TestFastRP:
    def test_config_defaults_are_the_adopted_configuration(self) -> None:
        config = FastRPConfig()
        assert (config.dim, config.weights, config.beta) == (128, (0.0, 1.0, 1.0, 1.0, 1.0), 0.0)
        assert config.model_version == (
            "fastrp-v1:dim=128:weights=0,1,1,1,1:beta=0:proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed=20260924"
        )
        with pytest.raises(ValueError, match="dim"):
            FastRPConfig(dim=0)
        with pytest.raises(ValueError, match="weight"):
            FastRPConfig(weights=(0.0, 0.0))

    def test_two_runs_are_byte_identical(self, catalog_edges: list) -> None:
        first = fastrp(build(catalog_edges).build())
        second = fastrp(build(catalog_edges, block=313).build())
        assert first.shape == (len({v for e in catalog_edges for v in e}), 128)
        assert first.dtype == np.float32
        assert first.tobytes() == second.tobytes()

    @pytest.mark.parametrize("block_columns", [1, 3, 16, 128, 500])
    def test_block_size_does_not_change_a_bit(self, catalog_edges: list, block_columns: int) -> None:
        adjacency = build(catalog_edges).build()
        reference = fastrp(adjacency, block_columns=32)
        assert fastrp(adjacency, block_columns=block_columns).tobytes() == reference.tobytes()

    @pytest.mark.parametrize(("threads", "block_columns"), [(2, 128), (3, 16), (8, 5)])
    def test_thread_count_does_not_change_a_bit(self, catalog_edges: list, threads: int, block_columns: int) -> None:
        adjacency = build(catalog_edges).build()
        reference = fastrp(adjacency)
        rows = np.arange(0, len(adjacency.nodes), 7)
        assert fastrp(adjacency, threads=threads, block_columns=block_columns).tobytes() == reference.tobytes()
        assert fastrp(adjacency, rows=rows, threads=threads).tobytes() == reference[rows].tobytes()

    def test_threaded_row_slices_share_the_matrix(self, catalog_edges: list) -> None:
        transition = build(catalog_edges).build().transition
        with ThreadPoolExecutor(max_workers=3) as executor:
            slices = _Propagator(transition, executor, 3)._slices
        assert len(slices) > 1
        assert sum(part.nnz for _, _, part in slices) == transition.nnz
        for _, _, part in slices:
            assert np.shares_memory(part.data, transition.data)
            assert np.shares_memory(part.indices, transition.indices)

    def test_unrelated_component_leaves_vectors_unchanged(self, catalog_edges: list) -> None:
        # The added component's keys interleave the original ones, so almost every
        # original node's position shifts; its vector must not.
        added = synthetic_catalog(seed=99, releases=800, prefix="new-")
        base = build(catalog_edges).build()
        grown = build(catalog_edges + added).build()
        assert len(grown.nodes) > len(base.nodes)
        before = rows_by_vertex(catalog_edges, fastrp(base), base.nodes)
        after = rows_by_vertex(catalog_edges, fastrp(grown), grown.nodes)
        assert before == after

    def test_local_edit_changes_only_its_neighbourhood(self) -> None:
        # A path 0 - 1 - ... - 59 with a pendant vertex, plus one new edge at the far end.
        path = [(("a", str(i)), ("a", str(i + 1))) for i in range(59)]
        edited = [*path, (("a", "59"), ("a", "pendant"))]
        base = build(path, extra_vertices=(("a", "pendant"),)).build()
        after = build(edited).build()
        old = rows_by_vertex(path, fastrp(base), base.nodes)
        new = rows_by_vertex(path, fastrp(after), after.nodes)
        # The edit changes row 59 of P; P^5 R at v reads rows of P within 4 hops of v.
        # Nodes 55..59 are within 4 hops of 59, and nothing further out may move.
        changed = sorted(int(v[1]) for v in old if old[v] != new[v])
        assert changed == list(range(55, 60))

    def test_parity_with_the_spike(self, catalog_edges: list) -> None:
        adjacency = build(catalog_edges).build()
        binary = adjacency.transition.copy()
        binary.data[:] = 1.0
        projection = HashedProjection(adjacency.nodes.keys, DEFAULT_SEED).columns(0, 128)
        expected = spike_fastrp(sp.csr_matrix(binary), projection, FastRPConfig().weights)
        actual = fastrp(adjacency, projection=FixedProjection(projection), block_columns=16)
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-6)
        cosine = (actual * expected).sum(axis=1) / (np.linalg.norm(actual, axis=1) * np.linalg.norm(expected, axis=1))
        assert cosine.min() > 0.99999

    def test_row_selection_and_half_precision(self, catalog_edges: list) -> None:
        adjacency = build(catalog_edges).build()
        full = fastrp(adjacency)
        rows = np.array([5, 0, 17, 17, len(adjacency.nodes) - 1])
        assert fastrp(adjacency, rows=rows, block_columns=40).tobytes() == full[rows].tobytes()
        half = fastrp(adjacency, rows=rows, out_dtype=np.float16)
        assert half.dtype == np.float16
        assert half.tobytes() == full[rows].astype(np.float16).tobytes()

    def test_isolated_node_and_trailing_zero_weights(self) -> None:
        builder = AdjacencyBuilder(NodeIndex(np.arange(4, dtype=np.uint64)))
        builder.add_edge_positions([0, 1], [1, 2])
        adjacency = builder.build()
        embedding = fastrp(adjacency, FastRPConfig(dim=8, weights=(1.0, 0.0, 0.0)))
        assert not embedding[3].any()
        assert np.allclose(np.linalg.norm(embedding[:3], axis=1), 1.0)
        with pytest.raises(ValueError, match="block_columns"):
            fastrp(adjacency, block_columns=0)
        with pytest.raises(ValueError, match="threads"):
            fastrp(adjacency, threads=0)


class TestMemoryEstimate:
    def test_full_catalog_blocking_strategy_fits_12_gib(self) -> None:
        # 32.8M nodes and at most 222M undirected edges (spike full-catalog extrapolation),
        # 10.2M artist rows returned as float16. The scaling test in docs/embeddings.md
        # measured about 1 GB of process overhead above this array estimate.
        nodes, edges, artists = 32_800_000, 222_000_000, 10_200_000
        recommended = estimate_peak_bytes(nodes, edges, artists, out_itemsize=2)
        assert recommended["peak"] < 10 * GiB
        assert recommended["build"] < recommended["compute"]
        assert estimate_peak_bytes(nodes, edges, artists, block_columns=128)["peak"] > 40 * GiB
        assert estimate_peak_bytes(10, 3_000_000_000, 1)["build"] > 3_000_000_000 * 16
