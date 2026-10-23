"""Style lifecycle stages over release-year counts (design spike gm-design-4804, section B4).

The rule is causal: the stage of a style in year ``y`` reads only counts from years up to
``y``, so the Genre Atlas can scrub valid time without knowing the future. Ported from the
spike harness's ``scripts/lifecycle.py``, where it flipped 0.54% of styles between two
consecutive Discogs dumps (bar 2%) and at most 4.69% under any single +-10% threshold change
(bar 10%).

    c(y)   trailing mean of the yearly release count over ``window`` years ending at y
    cum(y) releases up to and including y
    y0     the emergence year: the first year with cum >= emerge_total
    P(y)   max of c over years <= y (the peak so far)
    T(y)   min of c since the style was last active (the trough)

    absent       y < y0
    emerging     c >= active_share * P and y - y0 <  establish_years
    established  c >= active_share * P and y - y0 >= establish_years
    reviving     c <  active_share * P, c >= revive_ratio * T, and c >= revive_floor * P
    declining    otherwise

The emergence year is ``y0``, never graphinator's ``first_year``: one mis-dated release sets
``first_year``, which sits more than 10 years before ``y0`` for 46.6% of styles. The same
``y0`` is the date for the "emerges after its parent genre" rule.

Release year is the basis, not master year (0.54% against 1.69% flips). The timeline ends at
the last complete calendar year, and its latest ``provisional_years`` are marked provisional:
catalogue entry lags release, so their counts still grow 0.8-1.4% a month.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, fields, replace
from typing import TYPE_CHECKING, Any, Final

import numpy as np


if TYPE_CHECKING:
    from collections.abc import Iterable
    from datetime import date

    from numpy.typing import NDArray


ABSENT, EMERGING, ESTABLISHED, DECLINING, REVIVING = range(5)
STAGES: Final = ("absent", "emerging", "established", "declining", "reviving")

# Fixed parameters and the version label, not stage thresholds, so a +-10% sensitivity run leaves them alone.
_NON_THRESHOLDS: Final = frozenset({"window", "provisional_years", "rule_version"})


@dataclass(frozen=True)
class LifecycleRule:
    """The stage rule and its thresholds, kept together so the UI can show what it reads."""

    emerge_total: float = 25.0
    active_share: float = 0.5
    establish_years: float = 10.0
    revive_ratio: float = 1.5
    revive_floor: float = 0.25
    window: int = 3
    provisional_years: int = 2
    # Names the rule and its thresholds, so a later change is visible to clients and caches.
    # Bump it whenever a default above changes.
    rule_version: str = "gm-design-4804/1"

    def perturbed(self, scale: float) -> list[tuple[str, LifecycleRule]]:
        """One rule per threshold, with that threshold scaled by ``scale``.

        ``establish_years`` is a whole number of years, so it rounds away from the base
        (10 x 0.9 = 9, 10 x 1.1 = 11).
        """
        out: list[tuple[str, LifecycleRule]] = []
        for field in fields(self):
            if field.name in _NON_THRESHOLDS:
                continue
            value = getattr(self, field.name) * scale
            if field.name == "establish_years":
                value = float(math.floor(value) if scale < 1 else math.ceil(value))
            out.append((field.name, replace(self, **{field.name: value})))
        return out

    def describe(self) -> dict[str, Any]:
        """The rule as data: the basis, the stage names, the version, and every threshold."""
        return {"basis": "release_year", "stages": list(STAGES), **asdict(self)}


LIFECYCLE_RULE: Final = LifecycleRule()


def last_complete_year(month: date) -> int:
    """The last calendar year the timeline covers when computed in ``month``.

    The current year is partial by construction, and its trailing mean would read as a
    decline, so the timeline stops at the year before.
    """
    return month.year - 1


def trailing_mean(counts: NDArray[np.float64], window: int) -> NDArray[np.float64]:
    """Mean over the ``window`` years ending at each year (fewer at the start)."""
    csum = np.cumsum(counts, axis=-1, dtype=np.float64)
    out = csum.copy()
    out[..., window:] = csum[..., window:] - csum[..., :-window]
    n = np.minimum(np.arange(1, counts.shape[-1] + 1), window)
    result: NDArray[np.float64] = out / n
    return result


def emergence_index(counts: NDArray[np.float64], rule: LifecycleRule = LIFECYCLE_RULE) -> NDArray[np.int64]:
    """Per row, the index of the first year with ``emerge_total`` cumulative releases.

    A row that never gets there has the number of years, one past the last index.
    """
    emerged = np.cumsum(counts, axis=1) >= rule.emerge_total
    index: NDArray[np.int64] = np.where(emerged.any(axis=1), emerged.argmax(axis=1), counts.shape[1])
    return index


def stages(counts: NDArray[np.float64], rule: LifecycleRule = LIFECYCLE_RULE) -> NDArray[np.int8]:
    """Stage per (row, year) for an R x Y matrix of yearly release counts."""
    counts = np.asarray(counts, dtype=np.float64)
    rows, years = counts.shape
    c = trailing_mean(counts, rule.window)
    y0 = emergence_index(counts, rule)
    peak = np.maximum.accumulate(c, axis=1)
    out = np.full((rows, years), ABSENT, dtype=np.int8)
    trough = np.full(rows, np.inf)
    for y in range(years):
        live = y >= y0
        active = live & (c[:, y] >= rule.active_share * peak[:, y])
        young = (y - y0) < rule.establish_years
        out[active & young, y] = EMERGING
        out[active & ~young, y] = ESTABLISHED
        trough = np.where(active | ~live, np.inf, np.minimum(trough, c[:, y]))
        low = live & ~active
        reviving = low & (c[:, y] >= rule.revive_ratio * trough) & (c[:, y] >= rule.revive_floor * peak[:, y])
        out[low & ~reviving, y] = DECLINING
        out[reviving, y] = REVIVING
    return out


def flip_share(a: NDArray[np.int8], b: NDArray[np.int8], mask: NDArray[np.bool_] | None = None) -> float:
    """Share of cells whose stage differs, among cells where either side is not absent."""
    considered = (a != ABSENT) | (b != ABSENT)
    if mask is not None:
        considered &= mask
    n = int(considered.sum())
    return float(((a != b) & considered).sum() / n) if n else 0.0


@dataclass(frozen=True)
class StyleLifecycles:
    """Every style's stage per year, over one dense year range."""

    styles: tuple[str, ...]
    first_year: int
    last_year: int
    counts: NDArray[np.float64]
    stages: NDArray[np.int8]
    rule: LifecycleRule

    @property
    def years(self) -> range:
        return range(self.first_year, self.last_year + 1)

    def is_provisional(self, year: int) -> bool:
        """Whether ``year`` is among the latest ``provisional_years`` of the timeline."""
        return year > self.last_year - self.rule.provisional_years

    def emergence_years(self) -> dict[str, int | None]:
        """Each style's emergence year, or ``None`` if it has not reached ``emerge_total``."""
        index = emergence_index(self.counts, self.rule)
        n_years = self.counts.shape[1]
        return {style: (self.first_year + int(i) if i < n_years else None) for style, i in zip(self.styles, index, strict=True)}

    def stage(self, style: str, year: int) -> str:
        """The named stage of one style in one year of the timeline."""
        return STAGES[int(self.stages[self.styles.index(style), year - self.first_year])]


def style_lifecycles(rows: Iterable[tuple[str, int, int]], last_year: int, rule: LifecycleRule = LIFECYCLE_RULE) -> StyleLifecycles:
    """Stages for every style from ``(style, year, release_count)`` rows.

    Years after ``last_year`` are left out, so a partial current year never reads as a
    decline. Styles are in name order, and the year range starts at the earliest year any
    style has a release.
    """
    kept = [(style, int(year), int(count)) for style, year, count in rows if int(year) <= last_year]
    styles = tuple(sorted({style for style, _year, _count in kept}))
    first_year = min((year for _style, year, _count in kept), default=last_year)
    counts = np.zeros((len(styles), last_year - first_year + 1), dtype=np.float64)
    index = {style: i for i, style in enumerate(styles)}
    for style, year, count in kept:
        counts[index[style], year - first_year] += count
    return StyleLifecycles(styles, first_year, last_year, counts, stages(counts, rule), rule)
