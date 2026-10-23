"""The epic's measured bars, as regression tests over two synthetic catalogue months.

gm-analytics-engine-1i8 keeps the bars gm-design-4804 measured on two consecutive Discogs
dumps: lifecycle flips <= 2% between consecutive runs, <= 10% under any single +-10% threshold
change, zero hull intrusions and zero omissions, and p95 style movement under one
nearest-neighbour distance per month (with at most 5% of styles moving more than that).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from insights.computations import _style_parent_rows
from insights.style_atlas import build_layout, hull_check, movement, style_positions
from insights.style_lifecycle import LIFECYCLE_RULE, StyleLifecycles, flip_share, stages, style_lifecycles
from tests.style_atlas_fixtures import LAST_YEAR, SyntheticMonth, next_month, synthetic_month


if TYPE_CHECKING:
    from insights.style_atlas import AtlasLayout


SEEDS = (4804, 1, 2, 3)
FLIP_BAR = 0.02
PERTURBATION_BAR = 0.10
MOVEMENT_BAR_NN = 1.0
MOVING_SHARE_BAR = 0.05


def _months(seed: int) -> tuple[SyntheticMonth, SyntheticMonth]:
    early = synthetic_month(seed)
    return early, next_month(early, seed + 1000)


def _lifecycles(month: SyntheticMonth) -> StyleLifecycles:
    return style_lifecycles(((row["style"], row["year"], row["release_count"]) for row in month.style_years()), LAST_YEAR)


def _layouts(seed: int) -> tuple[AtlasLayout, AtlasLayout, list[tuple[str, str, float, bool]], list[tuple[str, str, float, bool]]]:
    early, late = _months(seed)
    early_parents = _style_parent_rows(early.style_parents())
    late_parents = _style_parent_rows(late.style_parents())
    first = build_layout(early.shares, early_parents, _lifecycles(early).emergence_years())
    second = build_layout(late.shares, late_parents, _lifecycles(late).emergence_years(), first.sectors)
    return first, second, early_parents, late_parents


@pytest.mark.parametrize("seed", SEEDS)
class TestLifecycleBars:
    def test_at_most_two_percent_of_styles_change_stage_between_months(self, seed: int) -> None:
        early, late = (_lifecycles(month) for month in _months(seed))
        assert early.styles == late.styles

        latest = flip_share(early.stages[:, [-1]], late.stages[:, [-1]])
        timeline = flip_share(early.stages, late.stages)

        assert latest <= FLIP_BAR
        assert timeline <= FLIP_BAR

    def test_at_most_ten_percent_change_under_any_single_ten_percent_threshold_change(self, seed: int) -> None:
        lifecycles = _lifecycles(_months(seed)[1])

        for scale in (0.9, 1.1):
            for name, rule in LIFECYCLE_RULE.perturbed(scale):
                perturbed = stages(lifecycles.counts, rule)
                assert flip_share(lifecycles.stages[:, [-1]], perturbed[:, [-1]]) <= PERTURBATION_BAR, f"{name} x{scale}"
                assert flip_share(lifecycles.stages, perturbed) <= PERTURBATION_BAR, f"{name} x{scale}"


@pytest.mark.parametrize("seed", SEEDS)
class TestLayoutBars:
    def test_zero_hull_intrusions_and_zero_omissions_in_both_months(self, seed: int) -> None:
        first, second, early_parents, late_parents = _layouts(seed)

        for layout, parents in ((first, early_parents), (second, late_parents)):
            check = hull_check(layout.nodes, parents)
            assert check.intruding_nodes == 0
            assert check.omitted_pairs == 0
            assert check.parent_pairs == len(parents)

    def test_p95_movement_is_under_one_nearest_neighbour_distance(self, seed: int) -> None:
        first, second, _early, _late = _layouts(seed)

        moved = movement(style_positions(first.nodes), style_positions(second.nodes))

        assert moved is not None
        assert moved.p95_nn < MOVEMENT_BAR_NN
        assert moved.share_over_1nn <= MOVING_SHARE_BAR

    def test_the_second_month_is_pinned_to_the_first(self, seed: int) -> None:
        first, second, _early, _late = _layouts(seed)

        assert second.ring_pinned and second.frames_pinned
        assert [(s.genre, s.ring_order, s.start, s.end) for s in second.sectors] == [(s.genre, s.ring_order, s.start, s.end) for s in first.sectors]

    def test_the_fixture_exercises_ghosts_and_a_real_change(self, seed: int) -> None:
        early, late = _months(seed)
        _first, second, early_parents, late_parents = _layouts(seed)

        assert sum(node.kind == "ghost" for node in second.nodes) > 100
        dominant = {style: genre for style, genre, _share, is_dominant in early_parents if is_dominant}
        assert any(dominant[style] != genre for style, genre, _share, is_dominant in late_parents if is_dominant)
        assert set(early_parents) != set(late_parents)
        assert early.shares != late.shares
