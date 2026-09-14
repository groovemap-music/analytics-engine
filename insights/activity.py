"""The consent-aware read path over ``activity.events`` and ``activity.impressions``.

ADR 0010 enforces consent twice, and the two checks are not redundant. A writer snapshots
the purposes active at the moment of the write onto the row, which is what makes an old row
interpretable later without reconstructing the grant history. A reader filters against
``activity.consent_grants`` again, so a revocation is honoured going forward for data that
was lawfully collected before it. The snapshot records what was true; the re-check enforces
what is true now. Neither replaces the other, so every read here applies both.

The whole point of this module is that there is one place to apply that filter and no way
past it. :func:`read_events` and :func:`read_impressions` compute the eligible subject set
themselves rather than accepting one from the caller — the ``subjects`` argument can only
*narrow* the result, never widen it — so a training job that reaches for activity rows
cannot skip the re-check by calling the reader differently.

The module is read-only by construction: it issues no ``INSERT``, ``UPDATE``, or ``DELETE``
(``tests/test_activity.py`` asserts that against the source), and it never selects
``users.id``. The join to ``activity.user_subjects`` is what resolves a grant to a
pseudonymous subject, and the subject is the only identity that leaves this module.

Both reads are bounded by ``occurred_at`` because both tables are ``PARTITION BY RANGE
(occurred_at)``: the bound is what lets the planner prune to the months actually asked for
instead of scanning every partition. Within that window the reads are keyset-paged on the
tables' own primary keys — ``(occurred_at, event_id)`` and ``(occurred_at, impression_id)``
— so a long scan holds no server-side cursor and a page costs the same at the end of the
window as at the start.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, cast

import structlog
from common.events import Event, Impression, consent_purposes


if TYPE_CHECKING:
    from collections.abc import AsyncIterator, Iterable, Sequence
    from datetime import datetime
    from uuid import UUID


logger = structlog.get_logger(__name__)

# The purpose a training reader filters on, and the default for every read here.
MODEL_TRAINING: Final = "model_training"

# The purpose an analytics reader filters on. The activity-summary computation uses this
# one: counting what happened is product analytics, not model training, and a subject who
# consented to one has not thereby consented to the other.
PRODUCT_ANALYTICS: Final = "product_analytics"

# Rows per keyset page. Large enough that a full month is not thousands of round trips,
# small enough that one page is a bounded amount of memory in the reader.
DEFAULT_BATCH_SIZE: Final = 1_000

# Columns of `activity.events`, in `Event` field order.
_EVENT_COLUMNS: Final = (
    "event_id, event_type, schema_version, subject_id, session_id, occurred_at, "
    "recorded_at, producer, consent_purposes, model_version, feature_version, "
    "idempotency_key, payload"
)

# Columns of `activity.impressions`, in `Impression` field order.
_IMPRESSION_COLUMNS: Final = (
    "impression_id, subject_id, surface, policy_id, candidate_set_id, position, "
    "item_id, score, propensity, request_id, occurred_at, recorded_at, consent_purposes"
)

# The subject ids whose user currently holds an unrevoked grant for one purpose.
#
# `revoked_at IS NULL` is the revocation check; `granted_at <= NOW()` excludes a grant
# that has not taken effect yet. Both only narrow the set, which is the direction a
# consent filter should err in. Only `subject_id` is selected: the user id is what the
# join is on, never what the read returns.
_ELIGIBLE_SUBJECTS_SQL: Final = """
    SELECT subjects.subject_id
    FROM activity.user_subjects AS subjects
    JOIN activity.consent_grants AS grants ON grants.user_id = subjects.user_id
    WHERE grants.purpose = %s
      AND grants.revoked_at IS NULL
      AND grants.granted_at <= NOW()
"""


class ConsentPurposeError(ValueError):
    """A read asked for a purpose the published vocabulary does not carry.

    Raised rather than filtered silently. An unknown purpose matches no snapshot and no
    grant, so a read that accepted one would return zero rows and look like an absence of
    consented data rather than the typo it is.
    """


def _validated_purposes(purposes: Sequence[str]) -> tuple[str, ...]:
    """Return ``purposes`` as a tuple, rejecting an empty or unpublished one.

    Args:
        purposes: The consent purposes a read is filtered on.

    Returns:
        The purposes, de-duplicated and in the order given.

    Raises:
        ConsentPurposeError: If no purpose was given, or one is not in the vocabulary
            :func:`common.events.consent_purposes` publishes.
    """
    ordered = tuple(dict.fromkeys(purposes))
    if not ordered:
        raise ConsentPurposeError("at least one consent purpose is required; an unfiltered activity read is not available")
    published = consent_purposes()
    unknown = [purpose for purpose in ordered if purpose not in published]
    if unknown:
        raise ConsentPurposeError(
            f"unknown consent {'purposes' if len(unknown) > 1 else 'purpose'} {unknown}; published purposes are {list(published)}"
        )
    return ordered


def _validated_window(since: datetime, until: datetime) -> None:
    """Check the ``occurred_at`` bound both reads are pruned by.

    Args:
        since: Inclusive lower bound.
        until: Exclusive upper bound.

    Raises:
        ValueError: If either bound is naive, or the window is empty.
    """
    if since.tzinfo is None or until.tzinfo is None:
        raise ValueError("since and until must be timezone-aware; activity.occurred_at is TIMESTAMPTZ")
    if since >= until:
        raise ValueError(f"since must be before until, got since={since.isoformat()} until={until.isoformat()}")


async def training_eligible_subjects(conn: Any, purpose: str = MODEL_TRAINING) -> set[UUID]:
    """Return the subjects whose user currently consents to ``purpose``.

    This is *the* consent filter. Every purpose-bound read in this module goes through it,
    and it is deliberately recomputed per read rather than cached: a revocation that lands
    between two reads has to take effect on the second one.

    Args:
        conn: An open connection to the database holding the ``activity`` schema.
        purpose: A purpose from :func:`common.events.consent_purposes`.

    Returns:
        The eligible ``subject_id`` values. Empty when nobody consents, which callers treat
        as "read nothing" rather than as "read everything".

    Raises:
        ConsentPurposeError: If ``purpose`` is not in the published vocabulary.
    """
    _validated_purposes((purpose,))
    async with conn.cursor() as cursor:
        cursor = cast("Any", cursor)
        await cursor.execute(_ELIGIBLE_SUBJECTS_SQL, (purpose,))
        rows = await cursor.fetchall()
    return {row[0] for row in rows}


async def _eligible_for_all(conn: Any, purposes: Sequence[str], subjects: Iterable[UUID] | None) -> list[UUID]:
    """Return the subjects eligible for *every* purpose, narrowed by ``subjects``.

    Asking for two purposes at once means asking for rows both are permitted for, so the
    per-purpose sets intersect rather than union. ``subjects`` intersects too: a caller can
    restrict a read to subjects it already cares about, and cannot use the argument to
    reach a subject the grant table does not allow.
    """
    eligible: set[UUID] | None = None
    for purpose in purposes:
        granted = await training_eligible_subjects(conn, purpose)
        eligible = granted if eligible is None else eligible & granted
        if not eligible:
            return []
    if subjects is not None:
        eligible = (eligible or set()) & set(subjects)
    return sorted(eligible or set())


# The write-time snapshot check, as one conjunct. A read asking for several purposes repeats
# it, because a row must carry every purpose asked for, not merely one of them.
_SNAPSHOT_PREDICATE: Final = "\n              AND %s = ANY(consent_purposes)"

# The read-time re-check, as a membership test against the set the grant table just yielded.
_GRANT_PREDICATE: Final = "\n              AND subject_id = ANY(%s)"

# The `occurred_at` bound. Half-open so a window boundary belongs to exactly one read, and
# first in the predicate order so the planner prunes partitions before evaluating anything
# else.
_WINDOW_PREDICATE: Final = "occurred_at >= %s\n              AND occurred_at < %s"


def _page_query(*, columns: str, table: str, key_column: str, purposes: Sequence[str], extra_clause: str, resuming: bool) -> str:
    """Assemble one keyset page statement.

    Every fragment is a module constant or a placeholder. The only things that vary are the
    table and its key column, both chosen here rather than by a caller, and the number of
    repetitions of the constant snapshot predicate. No caller value reaches the statement
    text: values travel as bound parameters, which is why the S608 heuristic is suppressed
    rather than satisfied.
    """
    where = _WINDOW_PREDICATE + _SNAPSHOT_PREDICATE * len(purposes) + _GRANT_PREDICATE + extra_clause
    if resuming:
        where += f"\n              AND (occurred_at, {key_column}) > (%s, %s)"
    return (
        f"SELECT {columns}\n            FROM {table}\n            WHERE {where}\n            ORDER BY occurred_at, {key_column}\n            LIMIT %s"  # noqa: S608
    )


async def _keyset_pages(
    conn: Any,
    *,
    columns: str,
    table: str,
    key_column: str,
    purposes: Sequence[str],
    eligible: Sequence[UUID],
    since: datetime,
    until: datetime,
    extra_clause: str,
    extra_params: Sequence[Any],
    batch_size: int,
) -> AsyncIterator[Any]:
    """Yield rows of one partitioned activity table, a keyset page at a time.

    The predicate order is the shared contract of both reads: the ``occurred_at`` window
    first so the planner prunes partitions, then the write-time snapshot check, then the
    read-time grant re-check as a subject-set membership, then whatever the caller narrowed
    by, then the keyset cursor.
    """

    def page(*, resuming: bool) -> str:
        return _page_query(columns=columns, table=table, key_column=key_column, purposes=purposes, extra_clause=extra_clause, resuming=resuming)

    first_page = page(resuming=False)
    next_page = page(resuming=True)

    shared = [since, until, *purposes, list(eligible), *extra_params]
    cursor_key: tuple[datetime, UUID] | None = None
    async with conn.cursor() as cursor:
        cursor = cast("Any", cursor)
        while True:
            if cursor_key is None:
                await cursor.execute(first_page, (*shared, batch_size))
            else:
                await cursor.execute(next_page, (*shared, cursor_key[0], cursor_key[1], batch_size))
            rows = await cursor.fetchall()
            if not rows:
                return
            for row in rows:
                yield row
            if len(rows) < batch_size:
                return
            cursor_key = _cursor_key_of(rows[-1], key_column)


def _cursor_key_of(row: Any, key_column: str) -> tuple[datetime, UUID]:
    """Return the ``(occurred_at, key)`` pair the next page resumes after."""
    # `activity.events` selects the key first and `occurred_at` sixth; `activity.impressions`
    # selects the key first and `occurred_at` eleventh. Both orders are fixed by the column
    # constants above, which are in the dataclass field order the row builders read.
    occurred_at_index = 5 if key_column == "event_id" else 10
    return (row[occurred_at_index], row[0])


def _event_of(row: Any) -> Event:
    """Build an :class:`~common.events.Event` from one ``activity.events`` row."""
    return Event(
        event_id=row[0],
        event_type=row[1],
        schema_version=row[2],
        subject_id=row[3],
        session_id=row[4],
        occurred_at=row[5],
        recorded_at=row[6],
        producer=row[7],
        consent_purposes=tuple(row[8] or ()),
        model_version=row[9],
        feature_version=row[10],
        idempotency_key=row[11],
        payload=row[12] or {},
    )


def _impression_of(row: Any) -> Impression:
    """Build an :class:`~common.events.Impression` from one ``activity.impressions`` row.

    ``score``, ``propensity``, and ``request_id`` are nullable columns that the published
    envelope declares required, so a row written before those fields were populated reads
    back with ``None`` in them. The stored value is surfaced as it is rather than defaulted
    to a number nobody recorded: an offline evaluation that needs a propensity has to see
    that it is missing.
    """
    return Impression(
        impression_id=row[0],
        subject_id=row[1],
        surface=row[2],
        policy_id=row[3],
        candidate_set_id=row[4],
        position=row[5],
        item_id=row[6],
        score=row[7],
        propensity=row[8],
        request_id=row[9],
        occurred_at=row[10],
        recorded_at=row[11],
        consent_purposes=tuple(row[12] or ()),
    )


async def read_events(
    conn: Any,
    *,
    since: datetime,
    until: datetime,
    purposes: Sequence[str] = (MODEL_TRAINING,),
    event_types: Sequence[str] | None = None,
    subjects: Iterable[UUID] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> AsyncIterator[Event]:
    """Iterate ``activity.events`` rows both consent checks admit.

    A row is yielded only when its write-time ``consent_purposes`` snapshot contains every
    requested purpose *and* its subject is currently granted every requested purpose. When
    nobody is eligible the read issues no query at all and yields nothing.

    Args:
        conn: An open connection to the database holding the ``activity`` schema.
        since: Inclusive lower bound on ``occurred_at``.
        until: Exclusive upper bound on ``occurred_at``.
        purposes: The consent purposes to filter on. Defaults to model training.
        event_types: Optional restriction to these types.
        subjects: Optional restriction to these subjects. Narrows the eligible set, never
            widens it.
        batch_size: Rows per keyset page.

    Yields:
        One :class:`~common.events.Event` per admitted row, ordered by
        ``(occurred_at, event_id)``.

    Raises:
        ConsentPurposeError: If a purpose is not in the published vocabulary.
        ValueError: If the ``occurred_at`` window is naive or empty.
    """
    checked = _validated_purposes(purposes)
    _validated_window(since, until)
    eligible = await _eligible_for_all(conn, checked, subjects)
    if not eligible:
        logger.info("🔒 No subjects consent to this activity read", purposes=list(checked), table="activity.events")
        return

    extra_clause, extra_params = ("\n              AND event_type = ANY(%s)", [list(event_types)]) if event_types else ("", [])
    pages = _keyset_pages(
        conn,
        columns=_EVENT_COLUMNS,
        table="activity.events",
        key_column="event_id",
        purposes=checked,
        eligible=eligible,
        since=since,
        until=until,
        extra_clause=extra_clause,
        extra_params=extra_params,
        batch_size=batch_size,
    )
    async for row in pages:
        yield _event_of(row)


async def read_impressions(
    conn: Any,
    *,
    since: datetime,
    until: datetime,
    purposes: Sequence[str] = (MODEL_TRAINING,),
    policy_ids: Sequence[str] | None = None,
    subjects: Iterable[UUID] | None = None,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> AsyncIterator[Impression]:
    """Iterate ``activity.impressions`` rows both consent checks admit.

    Identical consent semantics to :func:`read_events`, over the impression table and its
    ``(occurred_at, impression_id)`` key.

    Args:
        conn: An open connection to the database holding the ``activity`` schema.
        since: Inclusive lower bound on ``occurred_at``.
        until: Exclusive upper bound on ``occurred_at``.
        purposes: The consent purposes to filter on. Defaults to model training.
        policy_ids: Optional restriction to these ranking policies.
        subjects: Optional restriction to these subjects. Narrows the eligible set, never
            widens it.
        batch_size: Rows per keyset page.

    Yields:
        One :class:`~common.events.Impression` per admitted row, ordered by
        ``(occurred_at, impression_id)``.

    Raises:
        ConsentPurposeError: If a purpose is not in the published vocabulary.
        ValueError: If the ``occurred_at`` window is naive or empty.
    """
    checked = _validated_purposes(purposes)
    _validated_window(since, until)
    eligible = await _eligible_for_all(conn, checked, subjects)
    if not eligible:
        logger.info("🔒 No subjects consent to this activity read", purposes=list(checked), table="activity.impressions")
        return

    extra_clause, extra_params = ("\n              AND policy_id = ANY(%s)", [list(policy_ids)]) if policy_ids else ("", [])
    pages = _keyset_pages(
        conn,
        columns=_IMPRESSION_COLUMNS,
        table="activity.impressions",
        key_column="impression_id",
        purposes=checked,
        eligible=eligible,
        since=since,
        until=until,
        extra_clause=extra_clause,
        extra_params=extra_params,
        batch_size=batch_size,
    )
    async for row in pages:
        yield _impression_of(row)
