"""The Genre Atlas geography layout (design spike gm-design-4804, sections B2 and B3).

Ported from the spike harness's ``scripts/layout.py``, with the placement rule and the layout
source the spike recommends:

- **Genre ring.** Genres sit on a ring in the exact maximum-weight Hamiltonian cycle over
  their soft overlap, ``sum over styles of min(share_g, share_h)``, so genres that share
  styles are neighbours. The order starts at the first genre by name and runs toward its
  smaller-named neighbour, so equal inputs give one order.
- **Sectors.** Each genre owns one contiguous sector. Its width is proportional to the square
  root of its member count plus one, so an empty genre still has a sector.
- **Members and ghosts.** A style's parents are the genres at share >= 0.20, as
  ``insights.style_parents`` stores them. The style is drawn in its dominant parent's sector,
  and a ghost copy is drawn in every secondary parent's sector. A ghost counts as a member of
  its host's hull. This is the only rule the spike measured with zero intrusions and zero
  omissions; drawing the dominant parent alone omitted 470 of 1,227 parent pairs.
- **In-sector placement: lean x emergence rank.** A node's angle inside its sector is the rank
  of its lean, its share of the following ring neighbour minus its share of the preceding one.
  Its radius is the rank of its emergence year. Ranks, not raw values, so a month of edits moves
  a node only when it swaps rank with another. The spike measured p95 0.69 nearest-neighbour
  distances of movement between consecutive dumps. Co-tag MDS inside each sector is the
  measured alternative: it roughly halves edge crossings (11,976 down to 5,722) but moves p95
  1.13 nearest-neighbour distances, over the 1.0 bar.
- **Pinning.** The previous month's ``insights.genre_atlas_sectors`` rows pin the ring order
  and every sector's frame. A genre new to the catalogue is inserted where it adds the most
  overlap, and a genre that disappears is dropped. The frames are reused unchanged while
  every sector's square-root width stays within ``FRAME_TOLERANCE`` of its pinned width and
  the genre set is unchanged. Otherwise every sector is resized from the current membership,
  starting at the first ring genre's pinned start angle, so the ring does not rotate. Sector
  bounds are radians, and a bound may pass 2 pi.

``graph.part_of`` is never an input. One single-genre document adds a part_of parent, so 445
of 757 styles have five or more. Hulls built on part_of are meaningless.

Every coordinate is in the unit disc. Style and ghost nodes lie at radius 0.35 to 0.95, and
inside the middle 80% of their sector's angle, so a sector's nodes stay inside its wedge.
A genre node sits at its sector's middle angle, at radius 0.65.
"""

from __future__ import annotations

import contextlib
import itertools
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Final, Literal

import numpy as np
from scipy.spatial import Delaunay, KDTree, QhullError


if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from numpy.typing import NDArray


TAU: Final = 2 * math.pi
# The exact ring is a Held-Karp search over 2^n subsets. Discogs has 16 genres, and pinning
# means it runs only on the first month or when a genre appears with no pinned ring.
RING_EXACT_MAX_GENRES: Final = 20
# How far a pinned sector's width may drift from its square-root width before the ring resizes.
FRAME_TOLERANCE: Final = 0.10
STYLE_RADIUS_MIN: Final = 0.35
STYLE_RADIUS_SPAN: Final = 0.60
SECTOR_ANGLE_MARGIN: Final = 0.10
GENRE_NODE_RADIUS: Final = 0.65
_CONTIGUITY_EPSILON: Final = 1e-9

NodeKind = Literal["style", "ghost"]


@dataclass(frozen=True)
class Sector:
    """One genre's place on the ring: an ``insights.genre_atlas_sectors`` row."""

    genre: str
    ring_order: int
    start: float
    end: float
    x: float = 0.0
    y: float = 0.0

    @property
    def width(self) -> float:
        return self.end - self.start


@dataclass(frozen=True)
class Node:
    """One drawn copy of a style: an ``insights.style_atlas_layout`` row.

    ``genre`` is the host sector: the dominant parent for a ``style`` node and the secondary
    parent for a ``ghost``.
    """

    kind: NodeKind
    style: str
    genre: str
    x: float
    y: float


@dataclass(frozen=True)
class AtlasLayout:
    """A month's layout: sectors in ring order and nodes in key order."""

    sectors: tuple[Sector, ...]
    nodes: tuple[Node, ...]
    ring_pinned: bool
    frames_pinned: bool


# ── Genre ring ──────────────────────────────────────────────────────────────────────────────


def genre_affinity(genres: Sequence[str], shares: Mapping[str, Mapping[str, float]]) -> NDArray[np.float64]:
    """G x G soft overlap: the sum over styles of ``min(share_g, share_h)``."""
    column = {genre: j for j, genre in enumerate(genres)}
    matrix = np.zeros((len(shares), len(genres)))
    for i, style in enumerate(sorted(shares)):
        for genre, share in shares[style].items():
            matrix[i, column[genre]] = share
    weight = np.zeros((len(genres), len(genres)))
    for j in range(len(genres)):
        weight[j] = np.minimum(matrix[:, [j]], matrix).sum(axis=0)
    np.fill_diagonal(weight, 0.0)
    return weight


def ring_order(weight: NDArray[np.float64]) -> list[int]:
    """Exact maximum-weight Hamiltonian cycle (Held-Karp), in canonical start and direction.

    The cycle starts at index 0 and runs toward its smaller-index neighbour, so a ring read
    from any start or in either direction comes back as one order. Ties between equal-weight
    cycles go to the smallest predecessor index at each step.

    Raises:
        ValueError: With more than ``RING_EXACT_MAX_GENRES`` genres.
    """
    n = len(weight)
    if n > RING_EXACT_MAX_GENRES:
        raise ValueError(f"an exact ring over {n} genres exceeds the {RING_EXACT_MAX_GENRES}-genre bound")
    if n <= 3:
        return list(range(n))
    full = 1 << n
    best = np.full((full, n), -np.inf)
    previous = np.full((full, n), -1, dtype=np.int64)
    best[1, 0] = 0.0
    masks = np.arange(full)
    popcount = np.array([bin(mask).count("1") for mask in range(full)])
    with_start = masks[(masks & 1) == 1]
    for size in range(2, n + 1):
        layer = with_start[popcount[with_start] == size]
        for last in range(1, n):
            bit = 1 << last
            targets = layer[(layer & bit) != 0]
            if not len(targets):
                continue
            candidates = best[targets ^ bit] + weight[:, last][None, :]
            chosen = candidates.argmax(axis=1)
            best[targets, last] = candidates[np.arange(len(targets)), chosen]
            previous[targets, last] = chosen
    closing = best[full - 1] + weight[:, 0]
    closing[0] = -np.inf
    last = int(np.argmax(closing))
    order: list[int] = []
    mask = full - 1
    while last != -1:
        order.append(last)
        prior = int(previous[mask, last])
        mask ^= 1 << last
        last = prior
    order.reverse()
    if order[1] > order[-1]:
        order = [order[0], *order[1:][::-1]]
    return order


def pinned_ring(pinned: Sequence[str], genres: Sequence[str], weight: NDArray[np.float64]) -> list[str]:
    """The pinned ring, with vanished genres dropped and new genres inserted.

    Each new genre, in name order, goes between the adjacent pair where it adds the most
    overlap, ``w(a, new) + w(new, b) - w(a, b)``. Ties go to the earliest pair.
    """
    index = {genre: j for j, genre in enumerate(genres)}
    ring = [genre for genre in pinned if genre in index]
    for genre in sorted(set(genres) - set(ring)):
        if len(ring) < 2:
            ring.append(genre)
            continue
        g = index[genre]
        gains = [weight[index[a], g] + weight[g, index[b]] - weight[index[a], index[b]] for a, b in zip(ring, [*ring[1:], ring[0]], strict=True)]
        ring.insert(int(np.argmax(gains)) + 1, genre)
    return ring


# ── Sector frames ───────────────────────────────────────────────────────────────────────────


def _square_root_widths(ring: Sequence[str], members: Mapping[str, int]) -> dict[str, float]:
    raw = {genre: math.sqrt(members.get(genre, 0) + 1) for genre in ring}
    total = sum(raw.values())
    return {genre: TAU * value / total for genre, value in raw.items()}


def _is_contiguous_ring(ring: Sequence[str], frames: Mapping[str, Sector]) -> bool:
    for a, b in itertools.pairwise(ring):
        if abs(frames[a].end - frames[b].start) > _CONTIGUITY_EPSILON:
            return False
    return abs(frames[ring[-1]].end - frames[ring[0]].start - TAU) <= _CONTIGUITY_EPSILON


def sector_frames(ring: Sequence[str], members: Mapping[str, int], pinned: Sequence[Sector] = ()) -> tuple[dict[str, tuple[float, float]], bool]:
    """Every genre's ``(start, end)``, and whether the pinned frames were kept.

    Pinned frames are kept when they cover exactly this ring, in this order, with no gap, and
    every sector's square-root width is within ``FRAME_TOLERANCE`` of its pinned width.
    Otherwise every sector is resized, starting from the first ring genre's pinned start.
    """
    widths = _square_root_widths(ring, members)
    by_genre = {sector.genre: sector for sector in pinned}
    pinned_ring_order = [sector.genre for sector in sorted(pinned, key=lambda sector: sector.ring_order)]
    if (
        ring
        and pinned_ring_order == list(ring)
        and _is_contiguous_ring(ring, by_genre)
        and all(abs(by_genre[genre].width - widths[genre]) <= FRAME_TOLERANCE * widths[genre] for genre in ring)
    ):
        return {genre: (by_genre[genre].start, by_genre[genre].end) for genre in ring}, True
    start = by_genre[ring[0]].start if ring and ring[0] in by_genre else 0.0
    frames: dict[str, tuple[float, float]] = {}
    for genre in ring:
        frames[genre] = (start, start + widths[genre])
        start += widths[genre]
    return frames, False


# ── Placement ───────────────────────────────────────────────────────────────────────────────


def _entries(parents: Iterable[tuple[str, str, float, bool]]) -> list[tuple[NodeKind, str, str]]:
    """``(kind, style, host)`` per drawn copy: the dominant parent's node and every ghost."""
    return [("style" if is_dominant else "ghost", style, genre) for style, genre, _share, is_dominant in parents]


def _emergence_key(year: int | None) -> float:
    return math.inf if year is None else float(year)


def _ranks(keys: Sequence[tuple[float, str]]) -> list[float]:
    """Each key's rank, as the middle of its slot in [0, 1). The style name breaks ties."""
    order = sorted(range(len(keys)), key=lambda i: keys[i])
    out = [0.0] * len(keys)
    for rank, i in enumerate(order):
        out[i] = (rank + 0.5) / len(keys)
    return out


def build_layout(
    shares: Mapping[str, Mapping[str, float]],
    parents: Iterable[tuple[str, str, float, bool]],
    emergence: Mapping[str, int | None],
    pinned: Sequence[Sector] = (),
) -> AtlasLayout:
    """Lay out one month's atlas.

    Args:
        shares: Each style's raw genre shares, with no floor applied. They feed the ring's
            overlap and each node's lean, and every genre in them gets a sector.
        parents: ``(style, genre, share, is_dominant)`` rows at the parent threshold, exactly
            one dominant per style, as ``insights.style_parents`` holds them. A style with no
            row is not drawn.
        emergence: Each style's 25-release emergence year. A style that has not emerged, or
            is missing, ranks after every dated style, at the outer edge.
        pinned: The previous month's sectors. Empty on the first month.
    """
    entries = _entries(parents)
    genres = sorted({genre for style_shares in shares.values() for genre in style_shares} | {host for _kind, _style, host in entries})
    weight = genre_affinity(genres, shares)
    if pinned:
        ring = pinned_ring([sector.genre for sector in sorted(pinned, key=lambda sector: sector.ring_order)], genres, weight)
    else:
        ring = [genres[j] for j in ring_order(weight)]

    members: dict[str, int] = {}
    for _kind, _style, host in entries:
        members[host] = members.get(host, 0) + 1
    frames, frames_pinned = sector_frames(ring, members, pinned)

    position = {genre: k for k, genre in enumerate(ring)}
    nodes: list[Node] = []
    for genre in ring:
        hosted = [(kind, style) for kind, style, host in entries if host == genre]
        if not hosted:
            continue
        following = ring[(position[genre] + 1) % len(ring)]
        preceding = ring[(position[genre] - 1) % len(ring)]
        start, end = frames[genre]
        lean = _ranks([(shares.get(style, {}).get(following, 0.0) - shares.get(style, {}).get(preceding, 0.0), style) for _kind, style in hosted])
        age = _ranks([(_emergence_key(emergence.get(style)), style) for _kind, style in hosted])
        for (kind, style), u, v in zip(hosted, lean, age, strict=True):
            theta = start + (end - start) * (SECTOR_ANGLE_MARGIN + (1 - 2 * SECTOR_ANGLE_MARGIN) * u)
            rho = STYLE_RADIUS_MIN + STYLE_RADIUS_SPAN * v
            nodes.append(Node(kind, style, genre, rho * math.cos(theta), rho * math.sin(theta)))

    sectors = tuple(
        Sector(
            genre,
            k,
            frames[genre][0],
            frames[genre][1],
            GENRE_NODE_RADIUS * math.cos((frames[genre][0] + frames[genre][1]) / 2),
            GENRE_NODE_RADIUS * math.sin((frames[genre][0] + frames[genre][1]) / 2),
        )
        for k, genre in enumerate(ring)
    )
    nodes.sort(key=lambda node: (node.kind, node.style, node.genre))
    return AtlasLayout(sectors, tuple(nodes), ring_pinned=bool(pinned), frames_pinned=frames_pinned)


# ── Measures ────────────────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class HullCheck:
    """Intrusions: nodes inside a hull of a genre that is not their style's parent.

    Omissions: (style, parent) pairs with no drawn copy inside that parent's hull. Either one
    breaks the atlas's promise that a hull holds exactly a genre's styles.
    """

    intruding_nodes: int
    omitted_pairs: int
    parent_pairs: int


def hull_check(nodes: Sequence[Node], parents: Iterable[tuple[str, str, float, bool]]) -> HullCheck:
    """Measure intrusions and omissions against each genre's convex hull of hosted nodes."""
    parent_set: dict[str, set[str]] = {}
    for style, genre, _share, _dominant in parents:
        parent_set.setdefault(style, set()).add(genre)
    genres = sorted({node.genre for node in nodes})
    xy = np.array([(node.x, node.y) for node in nodes], dtype=np.float64).reshape(-1, 2)
    inside = np.zeros((len(nodes), len(genres)), dtype=bool)
    for j, genre in enumerate(genres):
        hosted = np.array([node.genre == genre for node in nodes])
        if hosted.sum() >= 3:
            # Collinear points have no area, so nothing is inside their hull.
            with contextlib.suppress(QhullError):
                inside[:, j] = Delaunay(xy[hosted]).find_simplex(xy) >= 0
        inside[hosted, j] = True
    intruding = sum(
        1 for i, node in enumerate(nodes) if any(inside[i, j] and genre not in parent_set.get(node.style, set()) for j, genre in enumerate(genres))
    )
    covered: dict[str, set[str]] = {}
    for i, node in enumerate(nodes):
        covered.setdefault(node.style, set()).update(genre for j, genre in enumerate(genres) if inside[i, j])
    drawn = {node.style for node in nodes}
    pairs = [(style, genre) for style in sorted(parent_set) for genre in sorted(parent_set[style])]
    omitted = sum(1 for style, genre in pairs if style not in drawn or genre not in covered.get(style, set()))
    return HullCheck(intruding_nodes=intruding, omitted_pairs=omitted, parent_pairs=len(pairs))


@dataclass(frozen=True)
class Movement:
    """How far styles moved between two months, in nearest-neighbour distances (NN).

    NN is the earlier layout's median nearest-neighbour distance over the styles both months
    draw. The spike's bar: p95 at most 1 NN, and at most 5% of styles moving more than 1 NN.
    """

    styles: int
    median_nn: float
    p95_nn: float
    max_nn: float
    share_over_1nn: float


def style_positions(nodes: Iterable[Node]) -> dict[str, tuple[float, float]]:
    """Each style's primary node position. Ghosts are copies, so they are left out."""
    return {node.style: (node.x, node.y) for node in nodes if node.kind == "style"}


def movement(before: Mapping[str, tuple[float, float]], after: Mapping[str, tuple[float, float]]) -> Movement | None:
    """Displacement of every style both months draw, or ``None`` with fewer than two."""
    common = sorted(set(before) & set(after))
    if len(common) < 2:
        return None
    a = np.array([before[style] for style in common], dtype=np.float64)
    b = np.array([after[style] for style in common], dtype=np.float64)
    distances, _ = KDTree(a).query(a, k=2)
    scale = float(np.median(distances[:, 1]))
    if scale <= 0:
        return None
    moved = np.linalg.norm(a - b, axis=1) / scale
    return Movement(
        styles=len(common),
        median_nn=float(np.median(moved)),
        p95_nn=float(np.percentile(moved, 95)),
        max_nn=float(moved.max()),
        share_over_1nn=float((moved > 1).mean()),
    )
