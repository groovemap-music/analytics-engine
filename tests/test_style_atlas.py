"""The geography layout, its pinning, and the hull and movement measures, on invented inputs."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from insights.style_atlas import (
    FRAME_TOLERANCE,
    GENRE_NODE_RADIUS,
    RING_EXACT_MAX_GENRES,
    TAU,
    Node,
    Sector,
    build_layout,
    genre_affinity,
    hull_check,
    movement,
    pinned_ring,
    ring_order,
    sector_frames,
    style_positions,
)


def _toy() -> tuple[dict[str, dict[str, float]], list[tuple[str, str, float, bool]], dict[str, int | None]]:
    """Four genres; three pure styles each, one style split across adjacent genres, one across opposite ones."""
    genres = ["A", "B", "C", "D"]
    shares: dict[str, dict[str, float]] = {}
    parents: list[tuple[str, str, float, bool]] = []
    for g, genre in enumerate(genres):
        for k in range(3):
            style = f"{genre}{k}"
            shares[style] = {genre: 1.0, genres[(g + 1) % 4]: 0.02 * k}
            parents.append((style, genre, 1.0, True))
    shares["AB"] = {"A": 0.6, "B": 0.4}
    parents += [("AB", "A", 0.6, True), ("AB", "B", 0.4, False)]
    shares["AC"] = {"A": 0.55, "C": 0.45}
    parents += [("AC", "A", 0.55, True), ("AC", "C", 0.45, False)]
    emergence: dict[str, int | None] = {style: 1960 + i for i, style in enumerate(sorted(shares))}
    return shares, parents, emergence


def _angle_in(theta: float, start: float, end: float) -> bool:
    return (theta - start) % TAU <= (end - start)


class TestTheRing:
    def test_ring_order_matches_brute_force(self) -> None:
        rng = np.random.default_rng(3)
        w = rng.random((8, 8))
        w = w + w.T
        np.fill_diagonal(w, 0)

        def score(order: list[int]) -> float:
            return float(sum(w[order[i], order[(i + 1) % len(order)]] for i in range(len(order))))

        best = max(score([0, *p]) for p in itertools.permutations(range(1, 8)))
        assert score(ring_order(w)) == pytest.approx(best, abs=1e-9)

    def test_ring_order_is_canonical_in_start_and_direction(self) -> None:
        w = np.zeros((5, 5))
        for a, b in [(0, 3), (3, 1), (1, 4), (4, 2), (2, 0)]:
            w[a, b] = w[b, a] = 10
        w[0, 1] = w[1, 0] = 1
        assert ring_order(w) == [0, 2, 4, 1, 3]

    def test_ring_order_refuses_more_genres_than_the_exact_bound(self) -> None:
        with pytest.raises(ValueError, match="exact ring"):
            ring_order(np.zeros((RING_EXACT_MAX_GENRES + 1, RING_EXACT_MAX_GENRES + 1)))

    def test_affinity_is_the_soft_overlap_of_shares(self) -> None:
        w = genre_affinity(["A", "B"], {"s": {"A": 0.6, "B": 0.4}, "t": {"A": 1.0, "B": 0.1}})
        assert np.allclose(w, [[0.0, 0.5], [0.5, 0.0]])

    def test_a_pinned_ring_keeps_its_order_even_when_the_overlap_now_prefers_another(self) -> None:
        w = np.zeros((4, 4))
        w[0, 2] = w[2, 0] = 100  # the new data wants A next to C
        assert pinned_ring(["A", "B", "C", "D"], ["A", "B", "C", "D"], w) == ["A", "B", "C", "D"]

    def test_a_new_genre_goes_where_it_adds_the_most_overlap_and_a_vanished_one_is_dropped(self) -> None:
        genres = ["A", "B", "C", "E"]
        w = np.zeros((4, 4))
        w[1, 3] = w[3, 1] = 5  # E overlaps B
        w[2, 3] = w[3, 2] = 4  # and C
        assert pinned_ring(["A", "B", "D", "C"], genres, w) == ["A", "B", "E", "C"]

    def test_a_pinned_ring_of_one_genre_grows_by_appending(self) -> None:
        assert pinned_ring(["A"], ["A", "B"], np.zeros((2, 2))) == ["A", "B"]


class TestSectorFrames:
    def test_widths_follow_the_square_root_of_membership(self) -> None:
        frames, pinned = sector_frames(["A", "B"], {"A": 15, "B": 3})

        assert not pinned
        assert frames["A"] == (0.0, pytest.approx(TAU * 4 / 6))
        assert frames["B"][1] == pytest.approx(TAU)

    def test_pinned_frames_are_kept_while_widths_stay_within_tolerance(self) -> None:
        previous = [Sector("A", 0, 1.0, 1.0 + TAU / 2), Sector("B", 1, 1.0 + TAU / 2, 1.0 + TAU)]

        frames, pinned = sector_frames(["A", "B"], {"A": 9, "B": 8}, previous)

        assert pinned
        assert frames == {"A": (1.0, 1.0 + TAU / 2), "B": (1.0 + TAU / 2, 1.0 + TAU)}

    def test_frames_resize_from_the_pinned_start_when_a_width_drifts_past_tolerance(self) -> None:
        previous = [Sector("A", 0, 1.0, 1.0 + TAU / 2), Sector("B", 1, 1.0 + TAU / 2, 1.0 + TAU)]

        frames, pinned = sector_frames(["A", "B"], {"A": 24, "B": 3}, previous)

        assert not pinned
        assert frames["A"] == (1.0, pytest.approx(1.0 + TAU * 5 / 7))
        assert frames["B"][1] == pytest.approx(1.0 + TAU)  # past 2 pi, as the schema allows

    def test_tolerance_is_ten_percent(self) -> None:
        assert FRAME_TOLERANCE == 0.10

    @pytest.mark.parametrize(
        "previous",
        [
            [Sector("A", 0, 0.0, TAU / 2), Sector("C", 1, TAU / 2, TAU)],
            [Sector("A", 0, 0.0, TAU / 2), Sector("B", 1, TAU / 2 + 0.1, TAU)],
        ],
        ids=["genre-set-changed", "gap-between-sectors"],
    )
    def test_frames_that_do_not_tile_this_ring_are_not_kept(self, previous: list[Sector]) -> None:
        _frames, pinned = sector_frames(["A", "B"], {"A": 1, "B": 1}, previous)
        assert not pinned


class TestBuildLayout:
    def test_every_secondary_parent_gets_a_ghost_and_hulls_are_clean(self) -> None:
        shares, parents, emergence = _toy()

        layout = build_layout(shares, parents, emergence)

        ghosts = {(node.style, node.genre) for node in layout.nodes if node.kind == "ghost"}
        assert ghosts == {("AB", "B"), ("AC", "C")}
        assert {node.style for node in layout.nodes if node.kind == "style"} == set(shares)
        check = hull_check(layout.nodes, parents)
        assert (check.intruding_nodes, check.omitted_pairs) == (0, 0)

    def test_nodes_sit_inside_their_host_sector_in_the_unit_disc(self) -> None:
        shares, parents, emergence = _toy()

        layout = build_layout(shares, parents, emergence)

        frames = {sector.genre: sector for sector in layout.sectors}
        for node in layout.nodes:
            rho = math.hypot(node.x, node.y)
            assert 0.35 <= rho <= 0.95
            assert _angle_in(math.atan2(node.y, node.x), frames[node.genre].start, frames[node.genre].end)
        for sector in layout.sectors:
            assert math.hypot(sector.x, sector.y) == pytest.approx(GENRE_NODE_RADIUS)

    def test_sectors_tile_the_ring(self) -> None:
        shares, parents, emergence = _toy()

        layout = build_layout(shares, parents, emergence)

        assert [sector.ring_order for sector in layout.sectors] == [0, 1, 2, 3]
        for a, b in itertools.pairwise(layout.sectors):
            assert a.end == pytest.approx(b.start)
        assert layout.sectors[-1].end - layout.sectors[0].start == pytest.approx(TAU)

    def test_radius_follows_emergence_rank_and_an_undated_style_sits_outermost(self) -> None:
        shares, parents, emergence = _toy()
        emergence["A0"] = None

        layout = build_layout(shares, parents, emergence)

        in_a = sorted((math.hypot(n.x, n.y), n.style) for n in layout.nodes if n.genre == "A")
        assert in_a[-1][1] == "A0"
        assert [style for _rho, style in in_a[:-1]] == sorted((s for s in ("A1", "A2", "AB", "AC")), key=lambda s: emergence[s] or 0)

    def test_angle_follows_lean_toward_the_following_neighbour(self) -> None:
        shares, parents, emergence = _toy()

        layout = build_layout(shares, parents, emergence)

        ring = [sector.genre for sector in layout.sectors]
        sector = layout.sectors[ring.index("B")]
        following = ring[(ring.index("B") + 1) % 4]
        hosted = [n for n in layout.nodes if n.genre == "B" and n.style.startswith("B")]
        by_angle = [n.style for n in sorted(hosted, key=lambda n: (math.atan2(n.y, n.x) - sector.start) % TAU)]
        assert by_angle == sorted(by_angle, key=lambda s: shares[s].get(following, 0.0))

    def test_is_deterministic_and_independent_of_input_order(self) -> None:
        shares, parents, emergence = _toy()

        first = build_layout(shares, parents, emergence)
        again = build_layout(dict(reversed(shares.items())), list(reversed(parents)), emergence)

        assert first == again

    def test_a_genre_with_no_parent_members_still_gets_a_sector(self) -> None:
        shares, parents, emergence = _toy()
        shares["A0"]["E"] = 0.05

        layout = build_layout(shares, parents, emergence)

        assert "E" in {sector.genre for sector in layout.sectors}
        assert "E" not in {node.genre for node in layout.nodes}

    def test_a_style_without_a_parent_is_not_drawn(self) -> None:
        shares, parents, emergence = _toy()
        shares["Orphan"] = {"A": 0.1}

        layout = build_layout(shares, parents, emergence)

        assert "Orphan" not in {node.style for node in layout.nodes}

    def test_pinning_keeps_ring_and_frames_from_the_previous_month(self) -> None:
        shares, parents, emergence = _toy()
        first = build_layout(shares, parents, emergence)
        rotated = tuple(Sector(s.genre, s.ring_order, s.start + 2.0, s.end + 2.0, s.x, s.y) for s in first.sectors)

        second = build_layout(shares, parents, emergence, rotated)

        assert second.ring_pinned and second.frames_pinned
        assert [(s.genre, s.start, s.end) for s in second.sectors] == [(s.genre, s.start, s.end) for s in rotated]


class TestMeasures:
    def test_an_intrusion_and_an_omission_are_counted(self) -> None:
        nodes = [Node("style", f"a{i}", "A", x, y) for i, (x, y) in enumerate([(0, 0), (2, 0), (0, 2), (2, 2)])]
        nodes.append(Node("style", "b0", "B", 1, 1))  # inside A's square, not an A style
        parents = [(f"a{i}", "A", 1.0, True) for i in range(4)] + [("b0", "B", 1.0, True), ("b0", "C", 0.3, False)]

        check = hull_check(nodes, parents)

        assert check.intruding_nodes == 1
        assert check.omitted_pairs == 1  # b0 has no copy under C
        assert check.parent_pairs == 6

    def test_movement_is_in_units_of_the_median_nearest_neighbour_distance(self) -> None:
        before = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (3.0, 0.0)}
        after = {"a": (0.0, 0.0), "b": (1.0, 0.0), "c": (3.0, 2.0), "d": (9.0, 9.0)}

        moved = movement(before, after)

        assert moved is not None
        assert moved.styles == 3
        assert moved.max_nn == pytest.approx(2.0)
        assert moved.share_over_1nn == pytest.approx(1 / 3)

    def test_movement_needs_two_shared_styles(self) -> None:
        assert movement({"a": (0.0, 0.0)}, {"a": (1.0, 1.0)}) is None

    def test_style_positions_leave_ghosts_out(self) -> None:
        nodes = [Node("style", "s", "A", 0.1, 0.2), Node("ghost", "s", "B", 0.3, 0.4)]
        assert style_positions(nodes) == {"s": (0.1, 0.2)}
