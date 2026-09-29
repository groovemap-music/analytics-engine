"""The monthly FastRP embedding load, under the `embedding_pipeline` role (ADR 0013).

`insights/embeddings/` holds the algorithm and its interfaces; this module is "the pipeline
that calls it" docs/embeddings.md defers to — reading the graph from PostgreSQL, running
`fastrp`, and writing `public.artist_embeddings`. It is a separate entry point
(`analytics-engine-embeddings`), not part of the always-on FastAPI service in
`insights.insights`, because it connects under a different, deliberately narrower role.

ADR 0013's 2026-09-24 amendment grants `embedding_pipeline` `SELECT` on every relation in the
`graph` schema and `SELECT, INSERT, UPDATE, DELETE` on `public.artist_embeddings` alone — no
DDL, no ownership, nothing on any other schema. Three consequences shape this module:

- **No index DDL, ever.** Building or rebuilding the ANN index over a `model_version` needs
  table ownership this role does not have and never will; `_log_operator_step` logs the
  statement an operator with a different, more privileged credential runs after this job's
  transaction commits (`build_artist_embeddings_index`-equivalent — see
  database-schema's "Building the artist HNSW index"). Per-`model_version` partial indexes
  (`gm-database-schema-19g5`) are not landed as of this writing, so the statement logged is the
  forward-looking, `model_version`-filtered shape rather than something database-schema
  documents today. Retiring a superseded `model_version`'s rows and index is that follow-on's
  business, never this job's.
- **No writes outside its own tables.** Every read below is a `SELECT` against `graph`; the
  load's one write statement targets `public.artist_embeddings`, and the similar-artist stage
  that follows it (`insights.similar_artists`, gm-analytics-engine-d4d) writes only
  `public.artist_similar_artists` and `public.artist_embedding_releases`, which
  gm-database-schema-2xe0 grants this role. A connection authenticated as this role gets a
  permission error on anything else — see `tests/integration/test_embedding_pipeline_integration.py`.
- **No log table.** Other scheduled computations in `insights/computations.py` write their
  outcome to `insights.computation_log`; this role holds nothing on the `insights` schema, so
  this job cannot do that. `public.artist_embeddings` itself — `source_dump_id`,
  `source_dump_date`, `computed_at` — is the only durable lineage record it can leave, and
  doubles as the idempotency check below.

## The stored `model_version` is per dump, not per algorithm

`FastRPConfig.model_version` (`insights/embeddings/fastrp.py`) names only the method, its
parameters, and the projection seed rule — the same string every month an operator does not
change the algorithm. Writing that string directly into `artist_embeddings.model_version`
would mean a second month's load lands on the *same* primary key, `(artist_id, model_version)`,
as the first: it would upsert the first month's rows in place under a version `catalog-api` may
still be serving, drive row-by-row HNSW maintenance on that live index instead of a clean
`CREATE INDEX CONCURRENTLY`, discard the first dump's rows before a purge-by-dump could ever
reach them, and leave ieu.3's month-over-month churn measurement with only one month on disk to
compare.

`stored_model_version(config, dump_id)` — `f"{config.model_version}:{_EDGE_SET_VERSION}@{dump_id}"`
— is what this module actually reads and writes as `model_version`. Composing the dump id in is
what lets two months coexist: each dump gets its own primary-key value, so a second month's load
is a brand-new set of rows, never a write to the first month's. `_EDGE_SET_VERSION` is the same
idea applied to the *graph* rather than the dump: it names which relations `_EDGE_RELATIONS`
reads, bumped whenever one is added, removed, or refiltered (ieu.6 added the release-level
credited-artist relation and bumped it to `"edges-v2"`; this bead, x3d, adds the track-credited-
artist and track-performer relations and bumps it to `"edges-v3"`), so a dump reprocessed under
a changed edge set also lands on its own rows rather than upserting or being skipped as the old
edge set's.
`config.model_version` (the pure method string) is recorded separately, in every log line here,
as `method_version` — see "Bit-identity and lineage" below.

## Idempotency

The job is idempotent per stored `model_version` (which already encodes `(dump_id,
method_version)`): re-running it for a dump already recorded under this algorithm is a no-op.
The read-only idempotency check (`_already_loaded`) and the write (`_write_embeddings`) are not
one transaction — the read runs before the (multi-minute, CPU-bound) graph read and `fastrp`
compute, and only the write itself is wrapped in a transaction. That is deliberate rather than a
race: if the write transaction never commits (a crash, a killed process), no row carries the new
stored `model_version`, so a retry's idempotency check correctly reports "not loaded" and redoes
the full load; holding one long-lived transaction across the whole compute would only add lock
and connection-lifetime risk for no additional safety.

`_write_embeddings`' `ON CONFLICT (artist_id, model_version) DO UPDATE` is retry safety for
exactly that crash case — a partial previous attempt's rows under *this same* stored
`model_version` — not a cross-dump upsert: since the stored value already differs per dump, the
conflict target can never match a row from a different month. Every query and write in this
module is scoped by the stored `model_version` throughout, so a version `catalog-api` is
currently serving is untouched by a load of a new one.

## Bit-identity and lineage

docs/embeddings.md's "Determinism" section notes that bit-identity holds for one build of NumPy
and SciPy, so cross-month comparisons need to know which build produced which vectors. Every
run logs `numpy_version`/`scipy_version` alongside `method_version`/`model_version`/`dump_id`
at the start of `load_embeddings`, since `public.artist_embeddings` itself has no column for
them — the pipeline role's lineage columns (`source_dump_id`, `source_dump_date`, `computed_at`)
are fixed by database-schema's DDL, so logs are the durable record for this.

## Scheduling

There is no in-process scheduler loop here, unlike `insights.insights`'s `_scheduler_loop`.
This is a one-shot script (`analytics-engine-embeddings`, `main()` below), meant to be invoked
by the deployment layer once a month, after that month's dump has loaded and `SOURCE_DUMP_ID`/
`SOURCE_DUMP_DATE` are known to the invoker. Wiring the exact monthly trigger (cron, a
`CronJob`, an operator running it by hand) is a deployment-repo concern outside this bead.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, date, datetime
from os import getenv
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np
import scipy
import structlog
from common import (
    AsyncPostgreSQLPool,
    describe_exception,
    parse_postgres_host_port,
    setup_logging,
    setup_telemetry,
    shutdown_telemetry,
)
from common.config import _build_postgres_connstr, get_secret

from insights.embeddings import AdjacencyBuilder, FastRPConfig, NodeIndex, fastrp, node_keys
from insights.similar_artists import run_similar_artists
from insights.telemetry import computation_span, record_computation, record_embedding_pipeline_failure, record_embedding_rows_written


if TYPE_CHECKING:
    from collections.abc import Sequence

    from numpy.typing import NDArray


logger = structlog.get_logger(__name__)

SERVICE_NAME: Final = "analytics-engine-embeddings"
COMPUTATION_NAME: Final = "embedding_pipeline"

ARTIST_EMBEDDINGS_TABLE: Final = "public.artist_embeddings"

# The six vertex kinds insights/embeddings/graph.py's node identity covers (its module
# docstring: "a one-character kind (a artist, r release, l label, m master, g genre, s
# style)"). `graph.vertex_degree` carries one row per vertex of the *whole* property-graph
# path-traversal surface (person, company, medium, ...); filtering to these six is what makes
# this the FastRP subgraph rather than that wider graph.
_VERTEX_KINDS: Final = ("a", "r", "l", "m", "g", "s")

# Kept `common.credit_roles` categories for the release-level credited-artist edge below,
# exactly the chw.2 spike's `KEPT_CREDIT_CATEGORIES` (design
# docs/spikes/gm-design-chw.2/parse_dump.py): a producer, engineer, session musician, or
# otherwise-uncategorized credit says something about musical similarity. Mastering, design,
# and management credits are dropped -- a cutting engineer or sleeve photographer links
# releases by vendor, not by sound. `common.credit_roles.ALL_CATEGORIES` (`groovemap-runtime`,
# already a dependency here) is the taxonomy these two tuples partition;
# `tests/test_embedding_pipeline.py` guards that they still cover it completely and disjointly,
# so a category added upstream fails a test here instead of silently landing in neither list.
_KEPT_CREDIT_CATEGORIES: Final[tuple[str, ...]] = ("production", "engineering", "session", "other")
_DROPPED_CREDIT_CATEGORIES: Final[tuple[str, ...]] = ("mastering", "design", "management")

# The release-level credited-artist edge (ieu.6). `graph.credited_on` is `(person_name,
# release_id, role)` with a GENERATED `role_category` column over the same taxonomy
# `common.credit_roles.categorize_role` implements (`graph.credit_role_category`,
# database-schema); it carries no artist id of its own. `graph.same_as`, `(person_name,
# artist_id)`, is the separate, additive table that resolves a credited name to zero, one, or
# more catalog artists. An INNER JOIN on `person_name` gives exactly the resolution rule this
# pipeline uses, with no extra Python-side logic: a name `same_as` never resolved (no id was
# ever recorded for it) joins to nothing and the credit is silently dropped -- there is no
# artist to point an edge at; a name `same_as` resolved to more than one artist id (seen across
# different releases, since `same_as` has no release column to disambiguate by) joins to every
# one of them, fanning the credit out to all of them rather than guessing which is "the" match.
# Both are accepted, documented behaviour, not an error: dropping an unresolvable credit loses
# no real artist, and fanning out an ambiguous one at worst adds a handful of noisy edges from
# genuine name collisions, which FastRP's propagation is already robust to by construction (one
# release among thousands touching a hub node changes it negligibly). The chw.2 spike itself
# never faced this: its harness read Discogs artist ids straight out of the dump's
# `extraartists/artist/id` element (`parse_dump.py`'s `_ids`), which production's
# `graph.credited_on`/`graph.same_as` split does not carry forward, so this rule has no spike
# precedent to match -- it is this bead's own design decision. `person_name` keeps Discogs' own
# `(2)`/`(3)` disambiguation suffix (discogs-sql-loader's `_credits` reads it verbatim), so an
# exact-string collision is the narrower case of an un-merged duplicate profile or a data-entry
# error, not the common "two same-named musicians" case Discogs' own numbering already
# separates -- see docs/embeddings.md's "Release-level credited-artist edges" for the full
# argument and the real-catalog measurement this fan-out-vs-drop choice is pending.
_CREDITED_ARTIST_EDGE_SQL: Final = """
SELECT DISTINCT credited_on.release_id AS release_id, same_as.artist_id AS artist_id
FROM graph.credited_on AS credited_on
JOIN graph.same_as AS same_as ON same_as.person_name = credited_on.person_name
WHERE credited_on.role_category = ANY(%s)
"""

# The two vertex-discovery queries `_read_vertices` runs beyond `graph.vertex_degree` -- see
# that function's docstring for why a credited-only artist or release needs one at all.
_CREDITED_ARTIST_IDS_SQL: Final = """
SELECT DISTINCT same_as.artist_id AS artist_id
FROM graph.credited_on AS credited_on
JOIN graph.same_as AS same_as ON same_as.person_name = credited_on.person_name
WHERE credited_on.role_category = ANY(%s)
"""
_CREDITED_RELEASE_IDS_SQL: Final = """
SELECT DISTINCT release_id
FROM graph.credited_on
WHERE role_category = ANY(%s)
"""

# Track and sub-track credits (gm-analytics-engine-x3d, follow-up to ieu.6). `graph.track_credited_on`
# is `(person_name, release_id, track_ordinal, sub_track_ordinal, track_position, role, role_category)`
# -- database-schema's `gm-database-schema-ug3v` -- one row per `tracklist[].extraartists` or
# `tracklist[].sub_tracks[].extraartists` credit, the same free-text-name/role shape
# `graph.credited_on` carries at release level, and it resolves through the identical
# `graph.same_as` join: `person_name` is the key, never `artist_id`, since `same_as` has no
# notion of "found at track level" versus "found at release level" to key on. Same kept/dropped
# category split, same fan-out-on-ambiguity and drop-on-unresolved rules as
# `_CREDITED_ARTIST_EDGE_SQL` above -- see that constant's comment for the full reasoning, which
# applies unchanged here.
#
# **This is deliberately the *same* release<->artist edge as `_CREDITED_ARTIST_EDGE_SQL`, not a
# second one.** A track-level credit and a release-level credit for the same (release, artist)
# pair assert the same fact -- "this artist is credited on this release" -- discovered one
# nesting level apart; `Adjacency`'s `build()` already collapses parallel edges between the same
# two positions regardless of which `_EDGE_RELATIONS` entry contributed them (`insights/embeddings/
# graph.py`: "parallel edges collapse to one"), so registering this as its own `_EDGE_RELATIONS`
# entry, resolving through `same_as` exactly like the release-level query, reproduces the chw.2
# spike's own treatment (`design/docs/spikes/gm-design-chw.2/parse_dump.py`: release, track, and
# sub-track extraartists are unioned into one `credits` set per release *before* it ever becomes
# an edge) without needing a hand-written `UNION` in SQL to get there -- the CSR build is where
# the dedup already has to happen for every other pair of relations that can name the same edge
# twice, and this pair is no exception. A track credit that never matches an existing
# release-level one still contributes a genuinely new edge; one that does match contributes
# nothing beyond what the release-level relation already said.
_TRACK_CREDITED_ARTIST_EDGE_SQL: Final = """
SELECT DISTINCT track_credited_on.release_id AS release_id, same_as.artist_id AS artist_id
FROM graph.track_credited_on AS track_credited_on
JOIN graph.same_as AS same_as ON same_as.person_name = track_credited_on.person_name
WHERE track_credited_on.role_category = ANY(%s)
"""
_TRACK_CREDITED_ARTIST_IDS_SQL: Final = """
SELECT DISTINCT same_as.artist_id AS artist_id
FROM graph.track_credited_on AS track_credited_on
JOIN graph.same_as AS same_as ON same_as.person_name = track_credited_on.person_name
WHERE track_credited_on.role_category = ANY(%s)
"""
_TRACK_CREDITED_RELEASE_IDS_SQL: Final = """
SELECT DISTINCT release_id
FROM graph.track_credited_on
WHERE role_category = ANY(%s)
"""

# Track performers (gm-analytics-engine-x3d). `graph.track_by_artist` is `(release_id,
# track_ordinal, sub_track_ordinal, track_position, artist_id)` -- the same formal, id-bearing
# `<artists>` shape `graph.by_artist` reads at release level, read instead from
# `tracklist[].artists`/`tracklist[].sub_tracks[].artists`, so it carries `artist_id` directly
# with no name-based `same_as` resolution step, exactly as `graph.by_artist` does.
#
# **This is a genuinely different signal from the credited-artist edge above, so it gets its
# own edge rather than merging with it.** A various-artists compilation's track performer is
# very often a different artist than whichever name the release itself is credited to --
# that is the whole reason `by_artist` and `credited_on` are already two separate relations at
# release level -- and the chw.2 spike keeps the same split: `trackartists` is its own CSR list,
# disjoint from `credits` (`parse_dump.py`'s `LISTS` and the `_parse_chunk` row tuple never
# combine them). One row per distinct (release, artist) mirrors the release-level `by_artist`
# scan, which also does not deduplicate a repeated main-artist row itself -- the `DISTINCT` here
# only collapses the same artist performing on more than one track of the same release, not
# anything role- or track-identity-related the way the credited-artist queries collapse role
# fan-out.
_TRACK_PERFORMER_EDGE_SQL: Final = "SELECT DISTINCT release_id, artist_id FROM graph.track_by_artist"
_TRACK_PERFORMER_ARTIST_IDS_SQL: Final = "SELECT DISTINCT artist_id FROM graph.track_by_artist"
_TRACK_PERFORMER_RELEASE_IDS_SQL: Final = "SELECT DISTINCT release_id FROM graph.track_by_artist"

# (name, query, params, source kind, target kind) for every edge relation that connects two of
# the six kinds above -- release<->artist/label/master/genre/style and master<->artist/genre/
# style, plus the release<->artist credited-artist, track-credited-artist, and track-performer
# edges ieu.6 and this bead added. The first eight are plain table scans, declared in
# database-schema's `graph.catalog` property graph (docs/architecture.md, "Property graph");
# the last three are filtered joins or DISTINCT scans over base tables that are not (yet) a
# `graph.catalog` edge label, so each carries its own query and bind params rather than being
# built from a bare table/column pair like the first eight.
_EDGE_RELATIONS: Final[tuple[tuple[str, str, tuple[Any, ...], str, str], ...]] = (
    ("graph.by_artist", "SELECT release_id, artist_id FROM graph.by_artist", (), "r", "a"),
    ("graph.on_label", "SELECT release_id, label_id FROM graph.on_label", (), "r", "l"),
    ("graph.derived_from", "SELECT release_id, master_id FROM graph.derived_from", (), "r", "m"),
    ("graph.in_genre", "SELECT release_id, genre_name FROM graph.in_genre", (), "r", "g"),
    ("graph.in_style", "SELECT release_id, style_name FROM graph.in_style", (), "r", "s"),
    ("graph.master_by_artist", "SELECT master_id, artist_id FROM graph.master_by_artist", (), "m", "a"),
    ("graph.master_in_genre", "SELECT master_id, genre_name FROM graph.master_in_genre", (), "m", "g"),
    ("graph.master_in_style", "SELECT master_id, style_name FROM graph.master_in_style", (), "m", "s"),
    ("graph.credited_on", _CREDITED_ARTIST_EDGE_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "r", "a"),
    ("graph.track_credited_on", _TRACK_CREDITED_ARTIST_EDGE_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "r", "a"),
    ("graph.track_by_artist", _TRACK_PERFORMER_EDGE_SQL, (), "r", "a"),
)

# Every vertex-discovery query `_read_vertices` runs beyond `graph.vertex_degree` -- see that
# function's docstring for why a credited-only or track-only artist or release needs one at
# all. `kind` says which vertex kind the query's single returned column names ("a" artist, "r"
# release); `_read_vertices` folds every "a" query's results into the same de-duplicated
# `artist_ids` list it returns, in the order each id is first seen across all of them.
_DISCOVERY_QUERIES: Final[tuple[tuple[str, str, tuple[Any, ...], str], ...]] = (
    ("credited_artist_ids", _CREDITED_ARTIST_IDS_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "a"),
    ("credited_release_ids", _CREDITED_RELEASE_IDS_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "r"),
    ("track_credited_artist_ids", _TRACK_CREDITED_ARTIST_IDS_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "a"),
    ("track_credited_release_ids", _TRACK_CREDITED_RELEASE_IDS_SQL, (list(_KEPT_CREDIT_CATEGORIES),), "r"),
    ("track_performer_artist_ids", _TRACK_PERFORMER_ARTIST_IDS_SQL, (), "a"),
    ("track_performer_release_ids", _TRACK_PERFORMER_RELEASE_IDS_SQL, (), "r"),
)

# Rows fetched per round trip from a server-side (named) cursor. Bounds how much of one block
# a Python list holds at a time; the full result set — up to 32.8M vertices, 222M edges at
# catalog scale (docs/embeddings.md) — is never materialized in one piece.
_CURSOR_FETCH_SIZE: Final = 50_000

# Rows per multi-row upsert statement.
_UPSERT_BATCH_SIZE: Final = 5_000

_UPSERT_SQL: Final = f"""
    INSERT INTO {ARTIST_EMBEDDINGS_TABLE}
        (artist_id, model_version, embedding, source_dump_id, source_dump_date, computed_at)
    VALUES (%s, %s, %s::halfvec, %s, %s, %s)
    ON CONFLICT (artist_id, model_version) DO UPDATE SET
        embedding = EXCLUDED.embedding,
        source_dump_id = EXCLUDED.source_dump_id,
        source_dump_date = EXCLUDED.source_dump_date,
        computed_at = EXCLUDED.computed_at
"""  # noqa: S608 -- ARTIST_EMBEDDINGS_TABLE is a module constant, not caller input.

_ALREADY_LOADED_SQL: Final = f"SELECT 1 FROM {ARTIST_EMBEDDINGS_TABLE} WHERE model_version = %s LIMIT 1"  # noqa: S608

_NAME_SLUG_PATTERN: Final = re.compile(r"[^a-z0-9]+")

# The separator composing the stored `model_version` from the algorithm's own version and the
# dump id — see "The stored model_version is per dump, not per algorithm" above. Rejected
# inside a dump id so the composed string is always unambiguous to a human reading it back.
_STORED_VERSION_SEPARATOR: Final = "@"

# The edge relations `_stream_edge_blocks` reads, as a short version tag bumped whenever a
# relation is added, removed, or its filter changes. Composed into the stored `model_version`
# below, between `config.model_version` and the dump id, so a rerun of a dump already loaded
# under a different edge set can never land on, get skipped as, or silently overwrite that
# earlier set's rows -- the two edge sets read a structurally different graph for the same
# `FastRPConfig` and dump, and must never share a primary-key value. ieu.2 shipped the eight
# relations in `_EDGE_RELATIONS` before this as (implicitly) "edges-v1"; ieu.6 added the
# release-level credited-artist relation as "edges-v2"; this bead (x3d) adds the track-credited-
# artist and track-performer relations as "edges-v3".
_EDGE_SET_VERSION: Final = "edges-v3"

# The monthly job's FastRP configuration: ADR 0013's defaults (w0 = 0) plus the self term at
# 0.05 (gm-analytics-engine-8ts, docs/embedding_tie_break.md), which took edges-v3's
# byte-duplicate vectors from 42.22% to 0.0% and exact top-10 churn from 0.8886 to 0.9519.
# `FastRPConfig()`'s own default stays 0.0 so the pre-8ts sum remains reproducible bit for bit;
# this is the one place production opts in. Its `model_version` names `self=0.05` and
# `fastrp-v2`, so the stored `model_version` changes with it and a dump already loaded under
# the old configuration is recomputed rather than skipped.
PRODUCTION_FASTRP_CONFIG: Final = FastRPConfig(self_weight=0.05)

DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR: Final = Path("/tmp/analytics-engine-similar-artists")  # noqa: S108 -- local scratch, see EmbeddingPipelineConfig.


@dataclass(frozen=True)
class EmbeddingPipelineConfig:
    """Configuration for one monthly embedding load, under the `embedding_pipeline` role.

    Deliberately separate from `insights.config.InsightsConfig`: this job authenticates as a
    different, narrower-privileged role than the always-on service, so it reads its own
    username/password secrets rather than the service's `POSTGRES_USERNAME`/`POSTGRES_PASSWORD`.
    It shares the same host, port, and database, because `graph` and `public.artist_embeddings`
    live in the one catalog database every service already points at.
    """

    postgres_host: str
    postgres_username: str
    postgres_password: str
    postgres_database: str
    source_dump_id: str
    source_dump_date: date
    # Local scratch for the exact top-K spool (about 3.75 GB for the full catalog at K=50,
    # plus a same-sized checkpoint); see docs/similar_artists.md.
    similar_artists_spool_dir: Path = DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR

    @classmethod
    def from_env(cls) -> EmbeddingPipelineConfig:
        """Create configuration from environment variables.

        Raises:
            ValueError: If a required variable is missing, or `SOURCE_DUMP_DATE` is not an
                ISO date.
        """
        postgres_username = get_secret("EMBEDDING_PIPELINE_POSTGRES_USERNAME")
        postgres_password = get_secret("EMBEDDING_PIPELINE_POSTGRES_PASSWORD")
        postgres_database = getenv("POSTGRES_DATABASE")
        source_dump_id = getenv("SOURCE_DUMP_ID")
        source_dump_date_raw = getenv("SOURCE_DUMP_DATE")
        missing_vars = [
            name
            for name, value in (
                ("EMBEDDING_PIPELINE_POSTGRES_USERNAME", postgres_username),
                ("EMBEDDING_PIPELINE_POSTGRES_PASSWORD", postgres_password),
                ("POSTGRES_DATABASE", postgres_database),
                ("SOURCE_DUMP_ID", source_dump_id),
                ("SOURCE_DUMP_DATE", source_dump_date_raw),
            )
            if not value
        ]
        if missing_vars:
            raise ValueError(f"Missing required environment variables: {', '.join(missing_vars)}")
        try:
            source_dump_date = date.fromisoformat(cast("str", source_dump_date_raw))
        except ValueError as exc:
            raise ValueError(f"SOURCE_DUMP_DATE must be an ISO date (YYYY-MM-DD), got {source_dump_date_raw!r}") from exc
        return cls(
            postgres_host=_build_postgres_connstr(),
            postgres_username=cast("str", postgres_username),
            postgres_password=cast("str", postgres_password),
            postgres_database=cast("str", postgres_database),
            source_dump_id=cast("str", source_dump_id),
            source_dump_date=source_dump_date,
            similar_artists_spool_dir=Path(getenv("SIMILAR_ARTISTS_SPOOL_DIR") or DEFAULT_SIMILAR_ARTISTS_SPOOL_DIR),
        )


@dataclass(frozen=True)
class LoadResult:
    """The outcome of one embedding load attempt.

    `method_version` is the pure `FastRPConfig.model_version` (algorithm, parameters, seed);
    `model_version` is `stored_model_version(config, dump_id)`, the value actually written to
    and read from `artist_embeddings.model_version`. See the module docstring.
    """

    method_version: str
    model_version: str
    rows_written: int
    skipped: bool


def stored_model_version(config: FastRPConfig, dump_id: str) -> str:
    """The `artist_embeddings.model_version` value one dump's load reads and writes.

    Composes the algorithm's own version, the edge-set version (`_EDGE_SET_VERSION`), and the
    dump id, so two dumps under an unchanged algorithm land on different primary-key values
    instead of one upserting the other's rows in place, and so do two edge sets under an
    unchanged algorithm and dump — see the module docstring. `model_version` is `TEXT`, so
    there is no length bound to enforce here beyond what a reasonable `dump_id` already is.

    Args:
        config: The FastRP method configuration.
        dump_id: The current dump's identifier.

    Raises:
        ValueError: If `dump_id` contains the separator (`@`), which would make the composed
            string ambiguous to read back.
    """
    if _STORED_VERSION_SEPARATOR in dump_id:
        raise ValueError(f"dump_id must not contain {_STORED_VERSION_SEPARATOR!r}, got {dump_id!r}")
    return f"{config.model_version}:{_EDGE_SET_VERSION}{_STORED_VERSION_SEPARATOR}{dump_id}"


def _sql_string_literal(value: str) -> str:
    """A single-quoted SQL string literal, escaped for the operator step's logged statement.

    Logged only, never executed by this job — but an operator may copy it verbatim, and a
    `dump_id` is free text that could otherwise carry a quote that breaks the pasted statement.
    """
    return "'" + value.replace("'", "''") + "'"


def _halfvec_literal(vector: NDArray[np.floating]) -> str:
    """Render one embedding row as pgvector's `halfvec` text input format.

    No `pgvector` Python dependency is added for this: the text format
    (``"[v1,v2,...]"``, cast with ``::halfvec`` in `_UPSERT_SQL`) is part of pgvector's SQL
    input syntax and needs no client-side adapter.
    """
    return "[" + ",".join(f"{value:g}" for value in vector.tolist()) + "]"


def _index_name_slug(value: str, *, max_length: int = 48) -> str:
    """Turn a string into a safe, lowercase SQL-identifier fragment of at most `max_length`.

    Logged only — this job never executes a statement built from it.
    """
    return _NAME_SLUG_PATTERN.sub("_", value.lower()).strip("_")[:max_length]


# PostgreSQL identifiers are silently truncated at NAMEDATALEN - 1 bytes, never rejected — two
# different names that agree up to this length become the same object. 63 is NAMEDATALEN - 1
# on every supported build.
_INDEX_NAME_MAX_LENGTH: Final = 63
_INDEX_NAME_PREFIX: Final = "idx_artist_embeddings_"
_INDEX_NAME_SUFFIX: Final = "_hnsw"
# Hex characters of a BLAKE2b digest of the *full* stored model_version, giving the name
# per-stored-version uniqueness independent of how much of it is human-legible.
_INDEX_NAME_DIGEST_LENGTH: Final = 12


def _index_name(model_version: str) -> str:
    """A deterministic HNSW index name for one stored `model_version`, always <= 63 bytes.

    `model_version` (`FastRPConfig.model_version`) alone routinely exceeds the 63-byte budget
    on its own — `docs/embeddings.md`'s example is over 100 characters — so a plain truncated
    slug of the *stored* value (`method_version@dump_id`) never reaches the `@dump_id` suffix
    that makes two months distinct: every dump would get the identical, silently-truncated
    name, and `CREATE INDEX CONCURRENTLY IF NOT EXISTS` would then skip every month after the
    first (gm-analytics-engine-ieu.2 review round 2).

    The name is composed from a short, human-legible fragment of the dump id (for readability
    in `psql \\di` output) and a fixed-width hash of the *entire* stored value (for
    uniqueness): two different stored versions can never collide on the same name, whatever
    the dump id looks like, and the same stored version always names the same index.
    """
    digest = hashlib.blake2b(model_version.encode(), digest_size=_INDEX_NAME_DIGEST_LENGTH // 2).hexdigest()
    dump_id_part = model_version.rsplit(_STORED_VERSION_SEPARATOR, 1)[-1]
    budget = _INDEX_NAME_MAX_LENGTH - len(_INDEX_NAME_PREFIX) - len(_INDEX_NAME_SUFFIX) - len(digest) - 1
    dump_slug = _index_name_slug(dump_id_part, max_length=max(budget, 0))
    middle = f"{dump_slug}_{digest}" if dump_slug else digest
    return f"{_INDEX_NAME_PREFIX}{middle}{_INDEX_NAME_SUFFIX}"


def _log_operator_step(model_version: str) -> None:
    """Log the ANN-index build this job never runs itself. See the module docstring.

    `model_version` here is the *stored* value (`stored_model_version(...)`, including the
    dump id) — the WHERE clause must match what is actually in the table. The index name comes
    from `_index_name`; the WHERE clause's value from `_sql_string_literal`. `m = 16` and
    `ef_construction = 64` are ADR 0013's fixed HNSW parameters (docs/architecture.md,
    "Building the artist HNSW index" in database-schema) — stated explicitly rather than left
    to pgvector's own defaults, so the logged statement matches what an upgrade of pgvector's
    defaults would not silently change underneath it.
    """
    index_name = _index_name(model_version)
    statement = (
        f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} "
        f"ON {ARTIST_EMBEDDINGS_TABLE} USING hnsw (embedding halfvec_cosine_ops) WITH (m = 16, ef_construction = 64) "
        f"WHERE model_version = {_sql_string_literal(model_version)}"
    )
    logger.info(
        "🛠️ Operator step required — build the ANN index for this model_version",
        model_version=model_version,
        statement=statement,
    )


async def _already_loaded(conn: Any, model_version: str) -> bool:
    """Return whether this stored `model_version` already has rows in `artist_embeddings`."""
    async with conn.cursor() as cursor:
        await cursor.execute(_ALREADY_LOADED_SQL, (model_version,))
        return await cursor.fetchone() is not None


async def _discover_ids(conn: Any, cursor_name: str, query: str, params: tuple[Any, ...]) -> list[str]:
    """Fetch every id a `_DISCOVERY_QUERIES` entry's single-column query returns, in blocks.

    A thin wrapper so `_read_vertices` runs each discovery query the same way it runs every
    other named-cursor scan in this module, without repeating the fetch loop once per query.
    """
    ids: list[str] = []
    async with conn.cursor(name=f"embedding_pipeline_{cursor_name}") as cursor:
        await cursor.execute(query, params)
        while True:
            batch = await cursor.fetchmany(_CURSOR_FETCH_SIZE)
            if not batch:
                break
            ids.extend(row[0] for row in batch)
    return ids


async def _read_vertices(conn: Any) -> tuple[NodeIndex, list[str]]:
    """Stream every `(kind, key)` vertex of the six FastRP kinds; return it and the artist ids.

    Named (server-side) cursors, so no full result set is ever materialized as a Python list
    in one piece. PostgreSQL only allows `DECLARE CURSOR` inside a transaction block, so this
    opens one read-only transaction for the duration of the scan — `AsyncPostgreSQLPool`
    connections default to autocommit, unlike a plain `psycopg.AsyncConnection`.

    `graph.vertex_degree` sums only the ten path-traversal relations database-schema declares
    for it (by_artist, master_by_artist, on_label, in_genre/in_style, master_in_genre/in_style,
    derived_from, alias_of, artist_member_of) — `credited_on`, `same_as`, `track_credited_on`,
    and `track_by_artist` are deliberately absent from that list. An artist credited only via
    `extraartists` (release- or track-level) — never a main artist, an alias, or a group member
    — therefore has no `graph.vertex_degree` row at all, and the credited-artist edges
    (`_CREDITED_ARTIST_EDGE_SQL`, `_TRACK_CREDITED_ARTIST_EDGE_SQL`) or the track-performer edge
    (`_TRACK_PERFORMER_EDGE_SQL`) would then name a node this pipeline had never seen; the same
    is true, in principle, of a release asserting only credits or track data and none of the
    other eight relations. `_DISCOVERY_QUERIES` runs one extra query per artist- or
    release-shaped gap each of the three added relations can open, over the same kept-category
    filter the credit queries already use, so every endpoint `_stream_edge_blocks` will later
    ask `NodeIndex.positions` for already has a node. Every discovery query is an unordered
    `SELECT DISTINCT`; the final `np.unique` sorts and dedupes the combined key array, so the
    resulting node set (and therefore the resulting embeddings) does not depend on the order
    PostgreSQL happens to return any of them in.
    """
    key_chunks: list[NDArray[np.uint64]] = []
    artist_ids: list[str] = []
    seen_artist_ids: set[str] = set()
    async with conn.transaction():
        async with conn.cursor(name="embedding_pipeline_vertices") as cursor:
            await cursor.execute("SELECT kind, key FROM graph.vertex_degree WHERE kind = ANY(%s)", (list(_VERTEX_KINDS),))
            while True:
                batch = await cursor.fetchmany(_CURSOR_FETCH_SIZE)
                if not batch:
                    break
                key_chunks.append(node_keys(batch))
                for kind, key in batch:
                    if kind == "a" and key not in seen_artist_ids:
                        seen_artist_ids.add(key)
                        artist_ids.append(key)
        for cursor_name, query, params, kind in _DISCOVERY_QUERIES:
            ids = await _discover_ids(conn, cursor_name, query, params)
            if kind == "a":
                new_ids = [artist_id for artist_id in ids if artist_id not in seen_artist_ids]
                seen_artist_ids.update(new_ids)
                if new_ids:
                    key_chunks.append(node_keys((kind, artist_id) for artist_id in new_ids))
                    artist_ids.extend(new_ids)
            elif ids:
                key_chunks.append(node_keys((kind, key) for key in ids))
    keys = np.unique(np.concatenate(key_chunks)) if key_chunks else np.zeros(0, dtype=np.uint64)
    return NodeIndex(keys), artist_ids


async def _stream_edge_blocks(conn: Any, builder: AdjacencyBuilder) -> None:
    """Add every edge of the eleven relations that connect the six FastRP kinds, in blocks.

    One read-only transaction for the whole scan — see `_read_vertices` on why a named
    cursor needs one.
    """
    async with conn.transaction():
        for name, query, params, source_kind, target_kind in _EDGE_RELATIONS:
            cursor_name = f"embedding_pipeline_{name.replace('.', '_')}"
            async with conn.cursor(name=cursor_name) as cursor:
                await cursor.execute(query, params)
                while True:
                    batch = await cursor.fetchmany(_CURSOR_FETCH_SIZE)
                    if not batch:
                        break
                    sources = node_keys((source_kind, str(source_key)) for source_key, _target_key in batch)
                    targets = node_keys((target_kind, str(target_key)) for _source_key, target_key in batch)
                    builder.add_edges(sources, targets)


async def _write_embeddings(
    conn: Any,
    *,
    model_version: str,
    dump_id: str,
    dump_date: date,
    artist_ids: Sequence[str],
    vectors: NDArray[np.floating],
) -> int:
    """Upsert every artist's embedding row for the stored `model_version`, in one transaction.

    `model_version` here is the *stored* value (`stored_model_version(...)`), already unique
    per dump — so `ON CONFLICT (artist_id, model_version) DO UPDATE` can only ever match a row
    this same dump's own, possibly-partial, previous attempt wrote, never another dump's. It
    exists purely for that crash-retry case: `COPY` has no `ON CONFLICT` clause, and a plain
    re-`INSERT` would fail outright on a retry that reaches those previously-committed rows
    (which cannot happen after this function returns, by the idempotency check above, but a
    retry that resumes mid-run before that check would still see them).
    """
    computed_at = datetime.now(UTC)
    rows_written = 0
    async with conn.transaction(), conn.cursor() as cursor:
        for start in range(0, len(artist_ids), _UPSERT_BATCH_SIZE):
            stop = min(start + _UPSERT_BATCH_SIZE, len(artist_ids))
            batch = [
                (
                    artist_ids[index],
                    model_version,
                    _halfvec_literal(vectors[index]),
                    dump_id,
                    dump_date,
                    computed_at,
                )
                for index in range(start, stop)
            ]
            await cursor.executemany(_UPSERT_SQL, batch)
            rows_written += len(batch)
    return rows_written


async def load_embeddings(pool: AsyncPostgreSQLPool, config: FastRPConfig, dump_id: str, dump_date: date) -> LoadResult:
    """Load one month's FastRP embeddings for `dump_id`, or no-op if already loaded.

    Args:
        pool: A pool connected as the `embedding_pipeline` role (or a role holding it).
        config: The FastRP method configuration; `config.model_version` is recorded as
            `method_version` in every log line here.
        dump_id: The current dump's identifier, recorded as `source_dump_id` lineage and
            composed into the stored `model_version` (see the module docstring).
        dump_date: The current dump's date, recorded as `source_dump_date` lineage.

    Returns:
        The load outcome — `skipped=True` when this stored `model_version` was already loaded.

    Raises:
        ValueError: If `dump_id` contains `stored_model_version`'s separator (`@`).
    """
    version = stored_model_version(config, dump_id)
    logger.info(
        "🔢 Embedding pipeline run starting",
        method_version=config.model_version,
        model_version=version,
        dump_id=dump_id,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )
    async with pool.connection() as conn:
        if await _already_loaded(conn, version):
            logger.info("⏭️ Embedding load skipped — already loaded", model_version=version, dump_id=dump_id)
            return LoadResult(method_version=config.model_version, model_version=version, rows_written=0, skipped=True)

        nodes, artist_ids = await _read_vertices(conn)
        if not artist_ids:
            logger.warning("⚠️ No artist vertices found in the graph — nothing to embed", model_version=version, dump_id=dump_id)
            return LoadResult(method_version=config.model_version, model_version=version, rows_written=0, skipped=False)

        builder = AdjacencyBuilder(nodes)
        await _stream_edge_blocks(conn, builder)
        adjacency = builder.build()

        artist_positions = nodes.positions(node_keys(("a", artist_id) for artist_id in artist_ids))
        # float16 output: the stored column is `halfvec`, so a wider dtype buys nothing and
        # doubles the resident array (docs/embeddings.md, "Memory and time at catalog scale").
        # block_columns defaults to 4 and threads=6 for the same reason: the pair the 12 GB
        # full-catalog budget was measured against.
        vectors = fastrp(adjacency, config, rows=artist_positions, out_dtype=np.float16, threads=6)

        rows_written = await _write_embeddings(
            conn,
            model_version=version,
            dump_id=dump_id,
            dump_date=dump_date,
            artist_ids=artist_ids,
            vectors=vectors,
        )

    logger.info("💾 Embedding load complete", model_version=version, dump_id=dump_id, rows_written=rows_written)
    _log_operator_step(version)
    return LoadResult(method_version=config.model_version, model_version=version, rows_written=rows_written, skipped=False)


async def run_embedding_pipeline(
    pool: AsyncPostgreSQLPool,
    dump_id: str,
    dump_date: date,
    config: FastRPConfig | None = None,
    similar_artists_spool_dir: Path | None = None,
) -> LoadResult:
    """Run one embedding load, recording duration, rows written, and failure metrics.

    Mirrors `insights.computations.run_all_computations`'s span-and-metric shape, without the
    `insights.computation_log` write that function's `_record_lifecycle` also does — this role
    cannot make it (see the module docstring).

    With `similar_artists_spool_dir`, then computes and publishes that `model_version`'s exact
    similar-artist lists (`insights.similar_artists.run_similar_artists`, docs/similar_artists.md).
    That stage reads the vectors back from `artist_embeddings` rather than reusing the in-memory
    array, so FastRP's peak and the top-K peak never overlap, and it runs whether this load wrote
    the vectors or found them already loaded, since it has its own idempotency check.
    """
    config = config or PRODUCTION_FASTRP_CONFIG
    started = time.perf_counter()
    try:
        with computation_span(COMPUTATION_NAME):
            result = await load_embeddings(pool, config, dump_id, dump_date)
            if similar_artists_spool_dir is not None:
                await run_similar_artists(
                    pool,
                    model_version=result.model_version,
                    source_dump_id=dump_id,
                    source_dump_date=dump_date,
                    spool_root=similar_artists_spool_dir,
                )
    except Exception as error:
        record_computation(COMPUTATION_NAME, time.perf_counter() - started, success=False)
        record_embedding_pipeline_failure()
        logger.error(
            "❌ Embedding pipeline failed",
            error=describe_exception(error),
            method_version=config.model_version,
            dump_id=dump_id,
        )
        raise
    record_computation(COMPUTATION_NAME, time.perf_counter() - started, success=True)
    record_embedding_rows_written(result.rows_written)
    return result


async def _initialize_pool(config: EmbeddingPipelineConfig) -> AsyncPostgreSQLPool:
    """Connect the one-shot pool, as the `embedding_pipeline`-scoped login."""
    host, port = parse_postgres_host_port(config.postgres_host)
    pool = AsyncPostgreSQLPool(
        connection_params={
            "host": host,
            "port": port,
            "dbname": config.postgres_database,
            "user": config.postgres_username,
            "password": config.postgres_password,
        },
        min_connections=1,
        max_connections=1,
    )
    await pool.initialize()
    return pool


async def _run(config: EmbeddingPipelineConfig) -> LoadResult:
    pool = await _initialize_pool(config)
    try:
        return await run_embedding_pipeline(
            pool, config.source_dump_id, config.source_dump_date, similar_artists_spool_dir=config.similar_artists_spool_dir
        )
    finally:
        await pool.close()


def main() -> None:  # pragma: no cover -- exercised through _run/run_embedding_pipeline in tests.
    """Run one monthly embedding load; exit non-zero on failure.

    Invoked as `analytics-engine-embeddings` (see `pyproject.toml`), by the deployment layer's
    own monthly schedule, after `SOURCE_DUMP_ID`/`SOURCE_DUMP_DATE` are known — see the module
    docstring's "Scheduling" section.
    """
    setup_logging(SERVICE_NAME, log_file=Path(f"/logs/{SERVICE_NAME}.log"))
    setup_telemetry(SERVICE_NAME)
    try:
        config = EmbeddingPipelineConfig.from_env()
        result = asyncio.run(_run(config))
    except Exception:
        logger.exception("❌ Embedding pipeline run failed")
        shutdown_telemetry()
        sys.exit(1)
    logger.info("✅ Embedding pipeline run complete", model_version=result.model_version, rows_written=result.rows_written, skipped=result.skipped)
    shutdown_telemetry()


if __name__ == "__main__":  # pragma: no cover
    main()
