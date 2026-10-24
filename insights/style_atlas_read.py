"""The Genre Atlas read path: the latest month's stored layout, with lifecycles derived on read.

``insights.genre_atlas_sectors`` and ``insights.style_atlas_layout`` hold the month's geography,
and ``insights.style_parents`` the parents. Lifecycle stages are not stored (see
:mod:`insights.style_lifecycle`), so they are computed here from ``insights.style_year_counts``
with the same module and rule that dated each style's emergence when the layout was built.
Each genre's emergence year is dated from ``insights.genre_year_counts`` with that same rule, and
a style's emergence edge to a parent is drawn when the style emerged in a later year.
``insights.style_landmark_masters`` supplies each placed style's landmark masters for the master
plane, unpositioned.

The read is bounded by the catalogue's size, not by the request: one month of sectors and
nodes, one row per (style, year) and per (genre, year) up to the timeline's last year, one row per
parent, and at most 25 landmark masters per style.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

from insights.models import AtlasGenre, AtlasGhost, AtlasMaster, AtlasParent, AtlasStyle, AtlasTimeline, StyleAtlasResponse
from insights.style_lifecycle import ABSENT, LIFECYCLE_RULE, STAGES, last_complete_year, style_lifecycles


if TYPE_CHECKING:
    from datetime import date


_LATEST_MONTH_SQL: Final = "SELECT max(month) FROM insights.genre_atlas_sectors"
_SECTORS_SQL: Final = """
    SELECT genre, ring_order, sector_start, sector_end, x, y, computed_at
    FROM insights.genre_atlas_sectors
    WHERE month = %s
    ORDER BY ring_order
"""
_NODES_SQL: Final = """
    SELECT node_kind, style, genre, x, y
    FROM insights.style_atlas_layout
    WHERE month = %s
    ORDER BY node_kind, style, genre
"""
_YEAR_COUNTS_SQL: Final = "SELECT style, year, release_count FROM insights.style_year_counts WHERE year <= %s"
_GENRE_YEAR_COUNTS_SQL: Final = "SELECT genre, year, release_count FROM insights.genre_year_counts WHERE year <= %s"
_MASTERS_SQL: Final = """
    SELECT master_id, title, year, style, release_count, rank
    FROM insights.style_landmark_masters
    WHERE style IN (SELECT style FROM insights.style_atlas_layout WHERE month = %s AND node_kind = 'style')
    ORDER BY style, rank
"""
_PARENTS_SQL: Final = """
    SELECT style, genre, share, is_dominant
    FROM insights.style_parents
    WHERE style IN (SELECT style FROM insights.style_atlas_layout WHERE month = %s AND node_kind = 'style')
    ORDER BY style, share DESC, genre
"""


def _emerges_after(style_year: int | None, genre_year: int | None) -> bool:
    """Whether a style that emerged in ``style_year`` emerged after a genre that emerged in ``genre_year``.

    A year of ``None`` is an emergence that has not happened, so there is nothing to order and no edge.
    """
    return style_year is not None and genre_year is not None and style_year > genre_year


def empty_style_atlas() -> StyleAtlasResponse:
    """The response before the first monthly run: no month, no geography, and the rule still shown."""
    return StyleAtlasResponse(
        month=None, computed_at=None, rule=LIFECYCLE_RULE.describe(), timeline=None, genres=[], styles=[], ghosts=[], masters=[]
    )


async def read_style_atlas(cursor: Any) -> StyleAtlasResponse:
    """Assemble the latest month's atlas from an open cursor."""
    cursor = cast("Any", cursor)
    await cursor.execute(_LATEST_MONTH_SQL)
    row = await cursor.fetchone()
    month: date | None = row[0] if row else None
    if month is None:
        return empty_style_atlas()

    await cursor.execute(_SECTORS_SQL, (month,))
    sectors = await cursor.fetchall()
    await cursor.execute(_NODES_SQL, (month,))
    nodes = await cursor.fetchall()
    last_year = last_complete_year(month)
    await cursor.execute(_YEAR_COUNTS_SQL, (last_year,))
    lifecycles = style_lifecycles(((style, year, count) for style, year, count in await cursor.fetchall()), last_year)
    emergence = lifecycles.emergence_years()
    await cursor.execute(_GENRE_YEAR_COUNTS_SQL, (last_year,))
    genre_emergence = style_lifecycles(
        ((genre, year, count) for genre, year, count in await cursor.fetchall()), last_year, lifecycles.rule
    ).emergence_years()
    await cursor.execute(_PARENTS_SQL, (month,))
    parents: dict[str, list[AtlasParent]] = {}
    for style, genre, share, dominant in await cursor.fetchall():
        parents.setdefault(style, []).append(
            AtlasParent(genre=genre, share=share, dominant=dominant, emergence_edge=_emerges_after(emergence.get(style), genre_emergence.get(genre)))
        )

    await cursor.execute(_MASTERS_SQL, (month,))
    masters = [
        AtlasMaster(master_id=master_id, title=title, year=year, style=style, release_count=release_count, rank=rank)
        for master_id, title, year, style, release_count, rank in await cursor.fetchall()
    ]

    row_of = {style: i for i, style in enumerate(lifecycles.styles)}
    n_years = len(lifecycles.years)
    provisional_from = last_year - lifecycles.rule.provisional_years + 1
    settled_index = provisional_from - 1 - lifecycles.first_year
    styles: list[AtlasStyle] = []
    ghosts: list[AtlasGhost] = []
    for kind, style, genre, x, y in nodes:
        if kind == "ghost":
            ghosts.append(AtlasGhost(style=style, genre=genre, x=x, y=y))
            continue
        # A style the year counts have no row for has no releases to date: absent throughout.
        timeline = lifecycles.stages[row_of[style]].tolist() if style in row_of else [ABSENT] * n_years
        styles.append(
            AtlasStyle(
                style=style,
                genre=genre,
                x=x,
                y=y,
                stage=STAGES[timeline[-1]] if timeline else STAGES[ABSENT],
                settled_stage=STAGES[timeline[settled_index]] if 0 <= settled_index < len(timeline) else None,
                emergence_year=emergence.get(style),
                stages=timeline,
                parents=parents.get(style, []),
            )
        )

    return StyleAtlasResponse(
        month=month,
        computed_at=max((sector[6] for sector in sectors), default=None),
        rule=lifecycles.rule.describe(),
        timeline=AtlasTimeline(
            first_year=lifecycles.first_year,
            last_year=last_year,
            provisional_from=provisional_from,
        ),
        genres=[
            AtlasGenre(genre=g, ring_order=order, sector_start=start, sector_end=end, x=x, y=y, emergence_year=genre_emergence.get(g))
            for g, order, start, end, x, y, _at in sectors
        ],
        styles=styles,
        ghosts=ghosts,
        masters=masters,
    )
