"""Synthetic catalogue months for the Genre Atlas regression bars.

No real catalogue data: every style, count, and share is drawn from a seeded generator shaped
after what the spike gm-design-4804 measured on two Discogs dumps.

- 16 genres with the Discogs names, on a hidden circle, so neighbouring genres share styles.
- Each style has a dominant genre at share 0.85-1.0. About 45% have a secondary parent at
  0.20-0.60, mostly a hidden neighbour, and some a third. Every style also carries small
  sub-threshold shares, which is what the lean reads.
- Release-year counts follow a rise, plateau, decay, and sometimes revival curve with Poisson
  noise, and catalogue lag thins the latest years.
- The next month grows every year's count the way one month of catalogue entry did: about
  1.4% for the latest year, 0.8% for the one before, and 0.2% for older years. Shares wobble a
  little, a few secondary parents cross the 0.20 threshold, and one style changes its
  dominant genre.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


GENRES = (
    "Blues",
    "Brass & Military",
    "Children's",
    "Classical",
    "Electronic",
    "Folk, World, & Country",
    "Funk / Soul",
    "Hip Hop",
    "Jazz",
    "Latin",
    "Non-Music",
    "Pop",
    "Reggae",
    "Rock",
    "Stage & Screen",
    "Spoken Word",
)
FIRST_YEAR = 1950
LAST_YEAR = 2025
# The month the synthetic catalogue is laid out in: its last complete year is LAST_YEAR.
COMPUTED_IN = 2026
# Monthly count growth by how far a year is from the latest: catalogue entry lags release.
_GROWTH = {0: 0.014, 1: 0.008}
_OLDER_GROWTH = 0.002
_LAG = {0: 0.30, 1: 0.15, 2: 0.08, 3: 0.04}


@dataclass
class SyntheticMonth:
    """One producer response's worth of data: counts per (style, year) and raw genre shares."""

    counts: dict[str, np.ndarray]
    shares: dict[str, dict[str, float]]
    years: range = field(default_factory=lambda: range(FIRST_YEAR, LAST_YEAR + 1))

    def style_years(self) -> list[dict[str, Any]]:
        return [
            {"style": style, "year": year, "release_count": int(n), "master_count": int(n) // 2}
            for style, row in sorted(self.counts.items())
            for year, n in zip(self.years, row, strict=True)
            if n > 0
        ]

    def style_parents(self) -> list[dict[str, Any]]:
        return [
            {"style": style, "genre": genre, "release_count": 0, "style_release_count": 0, "share": share}
            for style, genres in sorted(self.shares.items())
            for genre, share in sorted(genres.items())
        ]

    def genre_years(self) -> list[dict[str, Any]]:
        """Each genre's yearly releases: its styles' counts weighted by their share of the genre."""
        totals: dict[tuple[str, int], float] = {}
        for style, row in self.counts.items():
            for genre, share in self.shares.get(style, {}).items():
                for year, n in zip(self.years, row, strict=True):
                    totals[genre, year] = totals.get((genre, year), 0.0) + float(n) * share
        return [{"genre": genre, "year": year, "release_count": int(n)} for (genre, year), n in sorted(totals.items()) if int(n) > 0]

    def producer_body(self) -> dict[str, Any]:
        return {"style_years": self.style_years(), "style_parents": self.style_parents(), "genre_years": self.genre_years()}


def _curve(rng: np.random.Generator, n_years: int) -> np.ndarray:
    start = int(rng.integers(0, n_years - 8))
    peak = float(np.exp(rng.normal(3.5, 1.3)))
    rise = int(rng.integers(2, 9))
    plateau = int(rng.integers(3, 25))
    decay = float(rng.uniform(0.0, 0.2))
    curve = np.zeros(n_years)
    for k in range(start, n_years):
        t = k - start
        if t < rise:
            curve[k] = peak * (t + 1) / rise
        elif t < rise + plateau:
            curve[k] = peak
        else:
            curve[k] = peak * np.exp(-decay * (t - rise - plateau))
    if rng.random() < 0.25:
        revive_at = int(rng.integers(start + rise, n_years))
        curve[revive_at:] *= rng.uniform(1.5, 3.0)
    for back, lag in _LAG.items():
        curve[n_years - 1 - back] *= 1 - lag
    return curve


def synthetic_month(seed: int = 4804, styles: int = 400) -> SyntheticMonth:
    """A catalogue of ``styles`` styles over 16 genres, deterministic for a seed."""
    rng = np.random.default_rng(seed)
    n_genres = len(GENRES)
    hidden = rng.permutation(n_genres)
    neighbours = {int(hidden[k]): (int(hidden[(k - 1) % n_genres]), int(hidden[(k + 1) % n_genres])) for k in range(n_genres)}
    years = range(FIRST_YEAR, LAST_YEAR + 1)
    counts: dict[str, np.ndarray] = {}
    shares: dict[str, dict[str, float]] = {}
    for i in range(styles):
        style = f"Style {i:04d}"
        home = int(rng.integers(n_genres))
        genre_shares = {home: float(rng.uniform(0.85, 1.0))}
        near = neighbours[home]
        if rng.random() < 0.45:
            second = int(rng.choice(near)) if rng.random() < 0.7 else int(rng.integers(n_genres))
            if second != home:
                genre_shares[second] = float(rng.uniform(0.2, 0.6))
        if rng.random() < 0.12:
            third = int(rng.integers(n_genres))
            genre_shares.setdefault(third, float(rng.uniform(0.2, 0.35)))
        for other in {*near, int(rng.integers(n_genres)), int(rng.integers(n_genres))}:
            genre_shares.setdefault(other, float(rng.uniform(0.005, 0.15)))
        shares[style] = {GENRES[g]: round(s, 4) for g, s in genre_shares.items()}
        counts[style] = rng.poisson(_curve(rng, len(years))).astype(np.int64)
    return SyntheticMonth(counts, shares, years)


def next_month(month: SyntheticMonth, seed: int = 4805, threshold_crossings: int = 6) -> SyntheticMonth:
    """The same catalogue one month later: counts grow with catalogue lag, shares drift."""
    rng = np.random.default_rng(seed)
    last = len(month.years) - 1
    growth = np.array([_GROWTH.get(last - k, _OLDER_GROWTH) for k in range(len(month.years))])
    counts = {style: row + rng.poisson(row * growth) for style, row in month.counts.items()}
    shares = {
        style: {genre: float(np.clip(round(s + rng.normal(0, 0.003), 4), 0.0001, 1.0)) for genre, s in genres.items()}
        for style, genres in month.shares.items()
    }
    near_threshold = sorted(
        ((style, genre) for style, genres in shares.items() for genre, s in genres.items() if 0.12 < s < 0.3),
        key=lambda pair: (abs(shares[pair[0]][pair[1]] - 0.2), pair),
    )
    for style, genre in near_threshold[:threshold_crossings]:
        shares[style][genre] = 0.25 if shares[style][genre] < 0.2 else 0.15
    # One style's data really changes: its strongest secondary genre overtakes the home genre.
    switch = next(style for style, genres in sorted(shares.items()) if sorted(genres.values())[-2] >= 0.2)
    ranked = sorted(shares[switch], key=lambda genre: -shares[switch][genre])
    shares[switch][ranked[0]], shares[switch][ranked[1]] = shares[switch][ranked[1]], shares[switch][ranked[0]]
    return SyntheticMonth(counts, shares, month.years)
