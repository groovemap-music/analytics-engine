"""Pydantic response models for the GrooveMap analytics engine."""

from datetime import date, datetime  # noqa: TC003  # Pydantic resolves these annotations at runtime.
from typing import Any

from pydantic import BaseModel, Field


class ArtistCentralityItem(BaseModel):
    """A single artist's centrality ranking."""

    rank: int
    artist_id: str
    artist_name: str
    edge_count: int


class GenreTrendItem(BaseModel):
    """Release count for a genre in a specific decade."""

    decade: int
    release_count: int


class GenreTrendsResponse(BaseModel):
    """Genre trend data across decades."""

    genre: str
    trends: list[GenreTrendItem]
    peak_decade: int | None = None


class LabelLongevityItem(BaseModel):
    """A label's longevity ranking."""

    rank: int
    label_id: str
    label_name: str
    first_year: int
    last_year: int | None
    years_active: int
    total_releases: int
    peak_decade: int | None = None
    still_active: bool = False


class AnniversaryItem(BaseModel):
    """A release with a notable anniversary this month."""

    master_id: str
    title: str
    artist_name: str | None = None
    release_year: int
    anniversary: int


class DataCompletenessItem(BaseModel):
    """Data completeness metrics for an entity type."""

    entity_type: str
    total_count: int
    with_image: int = 0
    with_year: int = 0
    with_country: int = 0
    with_genre: int = 0
    completeness_pct: float = 0.0


class ReleaseRarity(BaseModel):
    """A single release's precomputed rarity score (ADR 0007 media-neutral core plus family extensions)."""

    release_id: int
    title: str
    artist_name: str
    year: int | None
    rarity_score: float
    tier: str
    hidden_gem_score: float
    pressing_scarcity: float | None = Field(
        default=None,
        description="Grooved-only signal (vinyl, shellac, grooved_other); null when no family extension claims the release.",
    )
    label_catalog: float
    format_rarity: float = Field(
        default=0.0,
        deprecated=True,
        description="Deprecated for one minor version: raw Discogs format name signal, superseded by medium_rarity.",
    )
    temporal_scarcity: float
    graph_isolation: float
    collection_prevalence: float
    medium_rarity: float | None = Field(
        default=None,
        description="Canonical-medium rarity score (ADR 0007); replaces format_rarity.",
    )
    media_families: list[str] = Field(
        default_factory=list,
        description="Canonical ADR 0007 media family ids the release covers.",
    )
    family_signals: dict[str, dict[str, float]] = Field(
        default_factory=dict,
        description="Per-family-extension signal scores, keyed by module id (e.g. grooved) to a signal-name-to-score mapping.",
    )


class ComputationStatus(BaseModel):
    """Status of a specific insight computation."""

    insight_type: str
    last_computed: datetime | None = None
    status: str
    duration_ms: int | None = None


class ActivitySummaryItem(BaseModel):
    """One day's consented first-party activity for one dimension key.

    Two row shapes share the table, told apart by ``dimension``. ``event_type`` rows count
    events of that type; ``policy_id`` rows count impressions served by that ranking policy.
    ``candidate_set_count`` is therefore populated only on the impression rows: a candidate
    set describes a ranking decision, and an event has none to count.

    Every count here was produced by the consent-aware read path, so a subject who has
    revoked consent contributes to no row computed after the revocation.
    """

    summary_date: date
    dimension: str = Field(description="Which grouping the row belongs to: event_type or policy_id.")
    dimension_key: str = Field(description="The event type, or the ranking policy id.")
    record_count: int = Field(description="Events of this type, or impressions served by this policy, on this day.")
    subject_count: int = Field(description="Distinct consenting subjects behind those records.")
    candidate_set_count: int | None = Field(
        default=None,
        description="Distinct candidate sets, on policy_id rows only; null on event_type rows.",
    )


class AtlasGenre(BaseModel):
    """One genre's sector on the Genre Atlas ring. Angles are radians; ``sector_end`` may pass 2 pi."""

    genre: str
    ring_order: int
    sector_start: float
    sector_end: float
    x: float
    y: float
    emergence_year: int | None = Field(description="The year the genre reached 25 cumulative releases, or null if it has not.")


class AtlasParent(BaseModel):
    """One parent genre of a style, at a share of at least the parent threshold.

    ``emergence_edge`` is whether the atlas draws the style's emergence edge to this genre: the
    style emerged in a later year than the genre did. It is false for a style or genre that has
    not reached 25 releases, because there is no emergence year to order.
    """

    genre: str
    share: float
    dominant: bool
    emergence_edge: bool


class AtlasStyle(BaseModel):
    """A style's primary node, with its lifecycle derived on read.

    ``stages`` holds one index per year of the response's timeline, into ``rule.stages``.
    ``stage`` is the stage in the timeline's last year, which is provisional. ``settled_stage`` is
    the stage in the last year before ``timeline.provisional_from``, or null if that year is before
    the timeline starts, so the UI can show it next to the provisional latest stage.
    """

    style: str
    genre: str = Field(description="The host sector: the style's dominant parent genre.")
    x: float
    y: float
    stage: str
    settled_stage: str | None
    emergence_year: int | None = Field(description="The first year with 25 cumulative releases, null if the style has not got there.")
    stages: list[int]
    parents: list[AtlasParent]


class AtlasGhost(BaseModel):
    """A ghost copy of a style, drawn in a secondary parent's sector."""

    style: str
    genre: str = Field(description="The host sector: the secondary parent genre.")
    x: float
    y: float


class AtlasMaster(BaseModel):
    """A landmark master of a style: one of its top 25 by release count, ranked 1 to 25.

    The atlas places no master; the client positions them. ``title`` is empty when the master has
    none, and ``year`` is null when it has no year.
    """

    master_id: str
    title: str
    year: int | None
    style: str
    release_count: int = Field(description="Distinct releases tagged with the style that derive from the master.")
    rank: int = Field(description="1 to 25 within the style, by release count descending, then master id.")


class StyleCotag(BaseModel):
    """A style that shares releases with the requested one, ranked 1 to 25 by shared releases."""

    other: str
    release_count: int = Field(description="Distinct releases tagged with both styles.")
    rank: int = Field(description="1 to 25 within the style, by release count descending, then name.")


class AtlasTimeline(BaseModel):
    """The years each style's ``stages`` list covers."""

    first_year: int
    last_year: int
    provisional_from: int = Field(description="The first provisional year; every year from here to last_year is provisional.")


class StyleAtlasResponse(BaseModel):
    """The latest month's Genre Atlas: geography, nodes, and lifecycles. Empty before the first monthly run."""

    month: date | None
    computed_at: datetime | None
    rule: dict[str, Any]
    timeline: AtlasTimeline | None
    genres: list[AtlasGenre]
    styles: list[AtlasStyle]
    ghosts: list[AtlasGhost]
    masters: list[AtlasMaster] = Field(description="Landmark masters of the styles in the layout, by style then rank. No positions.")
