"""The lifecycle rule on invented yearly series (ported from the gm-design-4804 harness)."""

from __future__ import annotations

from datetime import date

import numpy as np

from insights.style_lifecycle import (
    ABSENT,
    DECLINING,
    EMERGING,
    ESTABLISHED,
    LIFECYCLE_RULE,
    REVIVING,
    STAGES,
    LifecycleRule,
    flip_share,
    last_complete_year,
    stages,
    style_lifecycles,
)


def series(*values: float) -> np.ndarray:
    return np.array([values], dtype=float)


class TestTheRule:
    def test_rise_plateau_decline_revival(self) -> None:
        counts = series(0, 0, 10, 20, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 10, 5, 5, 5, 5, 20, 30, 30)
        s = stages(counts, LifecycleRule(window=1))[0]
        assert s[0] == ABSENT and s[1] == ABSENT
        assert s[3] == EMERGING  # cumulative 30 >= 25 first at index 3
        assert s[13] == ESTABLISHED  # 10 years after emergence
        assert s[15] == DECLINING  # 10 < 0.5 * 40
        assert s[19] == DECLINING
        assert s[20] == ESTABLISHED  # 20 >= 0.5 * 40 again
        s2 = stages(series(0, 0, 10, 20, 30, 40, 40, 40, 40, 40, 40, 40, 40, 40, 40, 4, 4, 4, 12, 4), LifecycleRule(window=1))[0]
        assert s2[17] == DECLINING and s2[18] == REVIVING  # 12 >= 1.5 * 4 and >= 0.25 * 40

    def test_is_causal(self) -> None:
        counts = np.array([[0, 5, 30, 30, 30, 0, 0, 0]], dtype=float)
        full = stages(counts)
        for y in range(1, counts.shape[1]):
            assert (stages(counts[:, : y + 1])[0] == full[0, : y + 1]).all()

    def test_the_defaults_are_the_spike_thresholds(self) -> None:
        expected = LifecycleRule(
            emerge_total=25.0, active_share=0.5, establish_years=10.0, revive_ratio=1.5, revive_floor=0.25, window=3, provisional_years=2
        )
        assert expected == LIFECYCLE_RULE

    def test_perturbed_rules_scale_one_threshold_each(self) -> None:
        names = [name for name, _ in LIFECYCLE_RULE.perturbed(1.1)]
        assert names == ["emerge_total", "active_share", "establish_years", "revive_ratio", "revive_floor"]
        eleven = dict(LIFECYCLE_RULE.perturbed(1.1))["establish_years"]
        nine = dict(LIFECYCLE_RULE.perturbed(0.9))["establish_years"]
        assert (eleven.establish_years, nine.establish_years) == (11.0, 9.0)
        assert dict(LIFECYCLE_RULE.perturbed(0.9))["active_share"].active_share == 0.45

    def test_flip_share_ignores_cells_absent_on_both_sides(self) -> None:
        a = np.array([[ABSENT, EMERGING, ESTABLISHED]], dtype=np.int8)
        b = np.array([[ABSENT, EMERGING, DECLINING]], dtype=np.int8)
        assert flip_share(a, b) == 0.5
        assert flip_share(a, a) == 0.0
        assert flip_share(a[:, :1], b[:, :1]) == 0.0

    def test_describe_carries_the_basis_stages_and_every_threshold(self) -> None:
        described = LIFECYCLE_RULE.describe()
        assert described["basis"] == "release_year"
        assert described["stages"] == list(STAGES)
        for name in ("emerge_total", "active_share", "establish_years", "revive_ratio", "revive_floor", "window", "provisional_years"):
            assert described[name] == getattr(LIFECYCLE_RULE, name)


class TestStyleLifecycles:
    def test_emergence_is_the_25_release_year_not_the_first_sighting(self) -> None:
        # One mis-dated release in 1960 sets a first_year; emergence waits for 25 releases.
        rows = [("Dub", 1960, 1), ("Dub", 1975, 10), ("Dub", 1976, 20), ("Dub", 1977, 30)]
        lifecycles = style_lifecycles(rows, last_year=1980)

        assert lifecycles.emergence_years() == {"Dub": 1976}
        assert lifecycles.stage("Dub", 1970) == "absent"
        assert lifecycles.stage("Dub", 1976) == "emerging"

    def test_a_style_short_of_25_releases_has_not_emerged(self) -> None:
        lifecycles = style_lifecycles([("Rare", 2000, 3), ("Common", 2000, 30)], last_year=2005)

        assert lifecycles.emergence_years() == {"Common": 2000, "Rare": None}
        assert all(lifecycles.stage("Rare", year) == "absent" for year in lifecycles.years)

    def test_years_after_the_last_complete_year_are_left_out(self) -> None:
        lifecycles = style_lifecycles([("House", 1990, 40), ("House", 2026, 500)], last_year=2025)

        assert lifecycles.years == range(1990, 2026)
        assert lifecycles.counts.sum() == 40

    def test_counts_land_in_a_dense_name_ordered_matrix(self) -> None:
        lifecycles = style_lifecycles([("Techno", 1992, 5), ("House", 1990, 7), ("House", 1990, 1)], last_year=1993)

        assert lifecycles.styles == ("House", "Techno")
        assert lifecycles.first_year == 1990
        assert lifecycles.counts.tolist() == [[8, 0, 0, 0], [0, 0, 5, 0]]

    def test_the_latest_two_years_are_provisional(self) -> None:
        lifecycles = style_lifecycles([("House", 2000, 40)], last_year=2025)

        assert [year for year in lifecycles.years if lifecycles.is_provisional(year)] == [2024, 2025]

    def test_the_timeline_ends_the_year_before_the_month_it_is_computed_in(self) -> None:
        assert last_complete_year(date(2026, 10, 1)) == 2025
        assert last_complete_year(date(2027, 1, 1)) == 2026
