"""Pydantic response models for the GrooveMap analytics engine."""

from datetime import datetime  # noqa: TC003  # Pydantic resolves this annotation at runtime.

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
