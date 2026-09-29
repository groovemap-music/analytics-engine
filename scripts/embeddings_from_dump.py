"""Build the real FastRP input graph directly from two Discogs monthly dumps and run
the real ``insights.embeddings.fastrp`` over it, for gm-analytics-engine-ieu.3.

Reads ``<dump>_releases.xml.gz`` and ``<dump>_masters.xml.gz`` and derives the edge
relations and vertex kinds the embedding pipeline reads from ``graph.*`` in PostgreSQL
-- copied one-to-one from discogs-sql-loader's ``tableinator/graph_derivation.py`` (the
producer of those tables), not from the design spike's own subset builder, which differs
(it also keeps per-track credits/performers and filters placeholder artists; production
does neither):

- ``by_artist``: a release's own ``artists`` list only (the main "BY" artists), never
  ``extraartists``.
- ``on_label``: a release's ``labels`` list, every entry.
- ``derived_from``: a release's own ``master_id`` field.
- ``in_genre`` / ``in_style``: a release's own ``genres`` / ``styles`` lists, verbatim.
- ``master_by_artist`` / ``master_in_genre`` / ``master_in_style``: the **master's own**
  document fields -- not aggregated from its releases. This is why the masters dump is
  needed at all.
- ``credited_by_artist`` (ieu.6, landed 2026-09-26, merge commit 4c0d4de): a release's
  own **release-level** ``extraartists`` only -- never per-track or sub-track -- with a
  kept role category (production, engineering, session, `common.credit_roles`' catch-all
  "other"; mastering/design/management dropped), resolved to an artist id through the
  SAME name-join `graph.credited_on` INNER JOIN `graph.same_as` ON `person_name` performs
  in `insights/embedding_pipeline.py`'s `_CREDITED_ARTIST_EDGE_SQL`: an unresolvable name
  (never paired with any id, anywhere in the catalog) drops the credit; a name resolved to
  more than one id fans out to all of them. This needs the GLOBAL same_as map -- built
  from every release-level `extraartists` entry with a resolvable id, **regardless of
  role** (same_as has no category filter) -- built once over the whole releases dump
  before any `credited_by_artist` edge can be resolved. See `_build_same_as_map` below.
  `categorize_role` does **not** split a compound "A, B" role on comma; it substring-
  matches the whole lowered string, longest fragment first, exactly like the
  `graph.credit_role_category` SQL it mirrors -- confirmed by reading both directly, and
  the one place this script's first draft (pre-ieu.6) had it wrong.
- An id is dropped only when blank or the Discogs "no entity" sentinel ``"0"``
  (``graph_derivation._entity_id``); no placeholder-artist filtering.

Deliberate simplification (approved for this measurement, see the bead comments): the
vertex set for every kind is derived from these relations' own endpoints, not from a
separate full read of ``artists.xml``/``labels.xml``. Production embeds every artist
document, including the ones with no graph edge at all, which get an all-zero FastRP
vector regardless of source -- degenerate for both recall and churn.

gm-analytics-engine-i37 closes the gap the paragraph above used to describe: two more
relations, mirroring x3d's landed `insights/embedding_pipeline.py` (`_EDGE_SET_VERSION =
"edges-v3"`) and, one level further down, discogs-sql-loader's
`tableinator/graph_derivation.py` (`_track_credits`/`_track_performers`, the reference
implementation for these two relations -- there is no enricher function to mirror, see
that module's docstring) and database-schema's `_TRACK_CREDIT_SOURCE`/
`_TRACK_PERFORMER_SOURCE`:

- ``track_credited_by_artist`` (``graph.track_credited_on`` JOIN ``graph.same_as``): every
  usable ``extraartists`` credit on a track or one of its sub-tracks, same kept-category
  filter and same-name resolution rule as the release-level credit above. Landed as its
  own `_EDGE_RELATIONS` entry in `embedding_pipeline.py`, not merged into the release-level
  one, but see "Adjacency collapses them anyway" below for why this script also keeps them
  as two separate relation names without that mattering to the output graph.
- ``track_by_artist`` (``graph.track_by_artist``): every usable formal ``<artists>``
  performer named on a track or one of its sub-tracks -- the same id-bearing shape
  ``by_artist`` reads at release level, most often naming a different artist than the
  release's own credit on a various-artists compilation. No name resolution: the id is
  right there on the element, exactly like ``by_artist``.

Both are release-scoped in the loader's `tracklist`/`sub_tracks` structure but resolve or
land on the SAME node kinds `by_artist`/`credited_by_artist` do (release <-> artist), so no
new vertex kind or `node_key` prefix is needed.

**xmltodict-wrapper unwrapping is a non-issue here.** `graph_derivation._xmltodict_array`
exists because the loader's `data` argument is JSON already round-tripped through
xmltodict, which collapses a single-child XML list into a bare dict instead of a
one-element list -- `tracklist.track` is a dict when there's exactly one track, a list
otherwise. This script parses the raw XML with `ElementTree` directly, where
`element.findall(tag)` always returns a list (0, 1, or many), so the single/many
distinction `_xmltodict_array` exists to paper over never arises here. `_track_position`
mirrors `track.get("position")` as `track.findtext("position")` for the same reason: no
JSON dict to `.get()` from, just an XML child to look up.

**The same_as map must include track-level credits too, not just release-level ones.**
`graph_derivation.derive_release`'s own `same_as` PUT statement unions
`[(name, artist_id) for name, role, artist_id in credits if artist_id is not None]` (release-
level) with the same list comprehension over `track_credits` -- `graph.same_as` has no
release or track-nesting column to key by, so a name paired with an id *anywhere* in the
catalog, at either level, resolves everywhere that name is credited, exactly the way ieu.6's
own comment on `_CREDITED_ARTIST_EDGE_SQL` already describes for the release-level case. Pass
1 (`_build_same_as_map`) now folds in every track/sub-track `extraartists` entry with a
resolvable id alongside the release-level ones it already read, before pass 2 resolves either
`credited_by_artist` or `track_credited_by_artist`.

**Adjacency collapses them anyway.** `insights/embedding_pipeline.py`'s own comment on
`_TRACK_CREDITED_ARTIST_EDGE_SQL` notes a track-level and release-level credit for the same
(release, artist) pair assert the same fact, and `Adjacency.build()`'s `sum_duplicates()`
already collapses parallel edges between the same two positions regardless of which relation
contributed them. Keeping `track_credited_by_artist`/`track_by_artist` as their own
`ALL_RELATIONS` entries (rather than folding their pairs into `credited_by_artist`/
`by_artist` in Python) costs nothing and keeps this script's per-relation edge counts
legible against `embedding_pipeline.py`'s own relation names in the parity report.

Sized on the 2026-08 dump (script: scripts/count_credit_slices.py, not committed -- see the
bead comments): the spike's full credit scope (release+track+subtrack) reaches 6,917,277
distinct artists (+5,593,003 beyond main); track performers alone add a further
2,633,830 distinct artists (+1,271,244 beyond main, partially overlapping the above).

No provider-derived data is written by this script except to the local scratch ``--out``
file (a numpy ``.npz`` of artist id strings and their float16 vectors), which is never
committed -- see ADR 0013's data-rights section and this repo's docs/embeddings.md.

    uv run python scripts/embeddings_from_dump.py \\
        ~/.cache/groovemap-spikes/dumps/discogs_20260801_releases.xml.gz \\
        ~/.cache/groovemap-spikes/dumps/discogs_20260801_masters.xml.gz \\
        --dump-id discogs_20260801 --dump-date 2026-08-01 \\
        --out /path/to/scratch/aug.npz
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import multiprocessing as mp
import os
import re
import resource
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING, Final
from xml.etree import ElementTree as ET


if TYPE_CHECKING:
    from collections.abc import Callable, Iterable


# Resolved once to a full path, matching this repo's own `DOCKER = shutil.which("docker")`
# convention (scripts/measure_recall_churn.py) -- S607 wants a full executable path, not a
# bare name resolved via $PATH at call time.
CURL: Final = shutil.which("curl") or "curl"
VM_STAT: Final = shutil.which("vm_stat") or "vm_stat"
VMMAP: Final = shutil.which("vmmap") or "vmmap"

THREADS = os.environ.get("FASTRP_THREADS", "6")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, THREADS)

import numpy as np  # noqa: E402
import scipy  # noqa: E402
import scipy.sparse as sp  # noqa: E402
from common.credit_roles import categorize_role  # noqa: E402

from insights.embeddings import Adjacency, AdjacencyBuilder, FastRPConfig, NodeIndex, fastrp, node_key  # noqa: E402


CHUNK_BYTES = 16 << 20

# The eight relations `insights/embedding_pipeline.py`'s `_EDGE_RELATIONS` reads at
# "edges-v1", plus `credited_by_artist` (ieu.6, "edges-v2"), plus the two this bead (x3d's
# follow-on measurement) adds for "edges-v3": `track_credited_by_artist` (this script's name
# for `embedding_pipeline.py`'s `graph.track_credited_on` entry) and `track_by_artist`
# (`graph.track_by_artist`, same name on both sides -- it is a plain table scan there too).
RELEASE_RELATIONS: tuple[str, ...] = ("by_artist", "on_label", "derived_from", "in_genre", "in_style")
MASTER_RELATIONS: tuple[str, ...] = ("master_by_artist", "master_in_genre", "master_in_style")
CREDIT_RELATION: str = "credited_by_artist"
TRACK_CREDIT_RELATION: str = "track_credited_by_artist"
TRACK_PERFORMER_RELATION: str = "track_by_artist"
ALL_RELATIONS: tuple[str, ...] = (*RELEASE_RELATIONS, *MASTER_RELATIONS, CREDIT_RELATION, TRACK_CREDIT_RELATION, TRACK_PERFORMER_RELATION)
# Release-scoped relations whose SOURCE endpoint is a release id -- used only to widen the
# parity report's distinct-release count so a release credited/performed on only at track
# level (no release-level relation of its own) still counts as a release the graph touched.
_RELEASE_SOURCED_RELATIONS: tuple[str, ...] = (*RELEASE_RELATIONS, CREDIT_RELATION, TRACK_CREDIT_RELATION, TRACK_PERFORMER_RELATION)

# The chw.2 spike's kept role categories (production, engineering, session, and
# `common.credit_roles`' catch-all "other"), matching ieu.6's stated scope exactly:
# mastering/design/management credits are dropped -- a cutting engineer or sleeve
# photographer links releases by vendor, not by sound (ADR 0013).
KEPT_CREDIT_CATEGORIES: frozenset[str] = frozenset({"production", "engineering", "session", "other"})


def peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes, Linux in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


_VM_STAT_PAGE_SIZE_RE: Final = re.compile(rb"page size of (\d+) bytes")
_VM_STAT_FIELD_RE: Final = re.compile(rb"^(Pages [a-z ]+):\s+(\d+)\.$", re.MULTILINE)


def free_plus_inactive_bytes() -> int:
    """`vm_stat`'s free+inactive pages, in bytes -- macOS-only (this pipeline only runs here).

    Neither category alone is the right memory-pressure signal on this host: "free" alone
    undercounts because macOS keeps recently-used pages "inactive" rather than evicting them
    immediately (they're reclaimed on demand, same as free pages, just not yet); "free +
    inactive" is the same sum Activity Monitor's own "Memory Used" gauge is the complement of,
    and what the bead's dispatch message keys its 18 GiB threshold on -- see `wait_for_memory`.
    Colima's VM (used by other work on this host) can inflate to hold up to 16 GB on its own,
    so this checks the CURRENT number at call time, never a cached one.
    """
    output = subprocess.run([VM_STAT], capture_output=True, check=True).stdout  # noqa: S603 -- VM_STAT is a resolved full path (S607), no arguments.
    page_size_match = _VM_STAT_PAGE_SIZE_RE.search(output)
    if page_size_match is None:
        raise RuntimeError(f"could not parse vm_stat's page size from: {output!r}")
    page_size = int(page_size_match.group(1))
    fields = {name.decode(): int(value) for name, value in _VM_STAT_FIELD_RE.findall(output)}
    missing = {"Pages free", "Pages inactive"} - fields.keys()
    if missing:
        raise RuntimeError(f"vm_stat output is missing {sorted(missing)}: {output!r}")
    return (fields["Pages free"] + fields["Pages inactive"]) * page_size


_VMMAP_FOOTPRINT_RE: Final = re.compile(rb"Physical footprint:\s*([\d.]+)([KMGT])")
_VMMAP_UNIT_SCALE: Final = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}


def _vmmap_physical_footprint_bytes() -> int:
    """This process's "Physical footprint" per `vmmap <pid>` -- the last-resort fallback
    `process_footprint_bytes` uses if neither `psutil` nor `resource.getrusage` works. The
    same figure a human running `vmmap` by hand on this host would read off directly."""
    output = subprocess.run([VMMAP, str(os.getpid())], capture_output=True, check=True).stdout  # noqa: S603 -- VMMAP is a resolved full path (S607); the only argument is this process's own pid.
    match = _VMMAP_FOOTPRINT_RE.search(output)
    if match is None:
        raise RuntimeError(f"could not parse vmmap's Physical footprint line from output of length {len(output)}")
    return int(float(match.group(1)) * _VMMAP_UNIT_SCALE[match.group(2).decode()])


def process_footprint_bytes() -> int:
    """This process's physical memory footprint, in bytes -- a monotonic high-water mark,
    never a live/current reading.

    `wait_for_memory`'s footprint-aware threshold needs this: whatever THIS process already
    holds is memory it won't need to additionally acquire from the host's free pool to reach
    a given peak, so subtracting it out of the flat requirement (rather than requiring it
    system-wide on top of what's already ours) is what makes the threshold accurate instead
    of double-counting.

    Prefers `resource.getrusage`'s peak RSS (`peak_rss_bytes`, `ru_maxrss` -- monotonic,
    never decreasing for the life of this process) over a LIVE reading such as psutil's
    `Process().memory_info().rss`. gm-analytics-engine-8ts, 2026-09-29: `wait_for_memory`'s
    own poll loop is exactly where this process is idling (nothing touching its allocations
    between polls), which is exactly when macOS is most likely to page out or compress its
    inactive resident pages -- the underlying allocations (the parsed same_as map, the graph
    being built) are still live and get paged straight back in the moment the next phase
    touches them, so a live RSS reading collapsing toward 0 during the wait is not "no
    longer needed", it's a false signal. Because `required_free = expected_peak_bytes -
    footprint + margin`, a footprint that shrinks while waiting makes `required_free` RISE
    each poll instead of converging -- caught live: September's run measured
    `this_process_footprint` fall from 6.22 GB to 0.02 GB across five consecutive 60s polls,
    pushing `required_free` from 13.78 GB to 19.98 GB, a bound this host cannot ever satisfy
    while Colima holds 16 GiB of it -- an unconditional deadlock, not a slow wait.
    `peak_rss_bytes()` doesn't have this failure mode: it only ever grows, so a value read
    before any paging/compression happened stays valid afterward. `psutil` is tried only as
    a fallback should `resource.getrusage` itself fail (not observed on macOS, where it's a
    lightweight syscall wrapper); `vmmap` (the dispatcher's own manual measurement tool for
    this host) is the last resort.
    """
    try:
        return peak_rss_bytes()
    except Exception as error:
        print(f"⚠️  resource.getrusage footprint read failed ({error!r}), falling back to psutil", file=sys.stderr)
    try:
        import psutil  # noqa: PLC0415 -- optional fallback import, see docstring.

        return int(psutil.Process().memory_info().rss)
    except Exception as error:
        print(f"⚠️  psutil footprint read failed ({error!r}), falling back to vmmap", file=sys.stderr)
    return _vmmap_physical_footprint_bytes()


def wait_for_memory(expected_peak_bytes: int, *, margin_bytes: int = 2_000_000_000, poll_interval_s: float = 60.0) -> None:
    """Block until this process's OWN footprint plus whatever else is free+inactive covers
    `expected_peak_bytes + margin_bytes`, logging while waiting -- footprint-AWARE, not a flat
    system-wide free-memory floor.

    A flat threshold (this pipeline's original shape) double-counts: by the time this runs,
    this process has already parsed the whole graph into memory, and that memory is itself
    PART OF the eventual build/FastRP peak, not separate from it. Requiring the full expected
    peak to ALSO be free elsewhere demands more total host memory than the peak actually
    needs -- on a host where the live (flat 18 GiB) guard could never be satisfied at all
    (~11.5 GB host free, Colima's VM holding 16 GiB of its own), that difference is the whole
    ballgame. `required_free = expected_peak_bytes - process_footprint_bytes() + margin_bytes`
    is what this process still needs to ACQUIRE, not what the peak totals.

    Called right before the graph-build/FastRP phase (`build_graph`'s `AdjacencyBuilder.build()`
    step) -- the point this pipeline's own memory checkpoints show host memory actually spikes
    (docs/recall_and_churn.md's "Memory at catalog scale"), not before parsing itself, which
    this run's own instrumentation shows costs far less. `expected_peak_bytes <= 0` skips the
    wait entirely (a smoke test on a tiny slice never needs a multi-GB peak).
    """
    if expected_peak_bytes <= 0:
        return
    while True:
        footprint = process_footprint_bytes()
        available = free_plus_inactive_bytes()
        required_free = max(expected_peak_bytes - footprint + margin_bytes, 0)
        detail = (
            f"expected_peak={expected_peak_bytes / 1e9:.2f}GB, this_process_footprint={footprint / 1e9:.2f}GB, "
            f"margin={margin_bytes / 1e9:.2f}GB -> required_free={required_free / 1e9:.2f}GB, free+inactive={available / 1e9:.2f}GB"
        )
        if available >= required_free:
            print(f"✅ {detail} -- proceeding", file=sys.stderr, flush=True)
            return
        print(f"⏳ {detail} -- waiting {poll_interval_s:.0f}s before the graph-build/FastRP phase", file=sys.stderr, flush=True)
        time.sleep(poll_interval_s)


def free_disk_bytes(path: Path) -> int:
    """Free bytes on the filesystem holding `path` -- `path` itself if it exists, else the
    nearest existing ancestor directory (the usual case: checking room for a file that is
    about to be created)."""
    existing = path if path.exists() else next((parent for parent in path.parents if parent.exists()), Path(path.anchor or "/"))
    return shutil.disk_usage(existing).free


def wait_for_disk(path: Path, expected_bytes: int, *, margin_bytes: int, poll_interval_s: float = 60.0) -> None:
    """Block until `path`'s filesystem has `expected_bytes + margin_bytes` free, logging while
    waiting -- called right before writing a graph checkpoint or a `--w0` output npz. Host
    disk here has repeatedly run critically low from *other*, unrelated work sharing the same
    volume (gm-analytics-engine-i37's dispatch, 2026-09-27: free space dropped from ~15 GB to
    ~4.5 GB over the course of one run), so a write that would leave less than `margin_bytes`
    afterward waits instead of racing a `No space left on device` mid-write -- np.savez_
    compressed/scipy.sparse.save_npz have no atomic-rename-on-success behavior of their own,
    so a write that runs out of room partway leaves a truncated, corrupt file behind rather
    than simply failing cleanly. `margin_bytes <= 0` (with `expected_bytes` also <= 0) skips
    the wait entirely, for the same smoke-test reason `wait_for_memory`'s `min_bytes <= 0` does.
    """
    required = expected_bytes + margin_bytes
    if required <= 0:
        return
    while True:
        available = free_disk_bytes(path)
        if available >= required:
            print(
                f"✅ disk free {available / 1e9:.2f} GB >= {required / 1e9:.2f} GB required "
                f"({expected_bytes / 1e9:.2f} GB expected write + {margin_bytes / 1e9:.2f} GB margin) -- proceeding to write {path}",
                file=sys.stderr,
                flush=True,
            )
            return
        print(
            f"⏳ disk free {available / 1e9:.2f} GB < {required / 1e9:.2f} GB required for {path} -- waiting {poll_interval_s:.0f}s",
            file=sys.stderr,
            flush=True,
        )
        time.sleep(poll_interval_s)


def _artist_ids_nbytes(artist_ids: Iterable[str]) -> int:
    """A conservative estimate of `artist_ids`' size once written: each id's UTF-8 byte
    length plus a flat per-string overhead (CPython's own small-string object overhead, ~50
    bytes, rounded up) -- not exact (numpy's `dtype=object`/pickle framing adds a bit more,
    and `savez_compressed` shrinks it again), but `wait_for_disk`'s job is a safety margin,
    not a byte-perfect prediction, and this errs generous (an overestimate waits a little
    longer, never writes with too little room)."""
    return sum(len(aid.encode()) + 56 for aid in artist_ids)


_ENTITY_MARKERS: tuple[bytes, ...] = (b"<!DOCTYPE", b"<!ENTITY")


def _reject_entity_declarations(chunk: bytes) -> None:
    """Raise if `chunk` contains a DOCTYPE or ENTITY declaration.

    Both classic XXE (external entity fetches) and billion-laughs (internal entity
    expansion) need one of these declared in the fed bytes before they can do anything,
    so refusing them outright -- the same mitigation `defusedxml` applies, done here at
    the byte level since this Python's `ElementTree.XMLParser` no longer exposes the
    underlying expat parser to install a handler on directly -- closes both without
    adding that dependency. Belt-and-suspenders: `_chunks` already only ever hands this
    a `<record ...>...</record>` fragment cut well past the dump's own top-level
    preamble, so a real Discogs dump's DOCTYPE (if any) never reaches here regardless.
    """
    for marker in _ENTITY_MARKERS:
        if marker in chunk:
            raise ValueError(f"refusing a chunk containing {marker!r}")


def _usable_id(value: str | None) -> str | None:
    """Drop a blank id or the Discogs "no entity" sentinel "0" -- `graph_derivation._entity_id`."""
    if value is None:
        return None
    text = value.strip()
    return None if not text or text == "0" else text


def _curl_stream(url: str) -> subprocess.Popen:
    """Start `curl` streaming URL's raw (still-gzipped) bytes to its own stdout pipe -- never
    to disk. HTTP/1.1 per the bead brief (HTTP/2 resets have hit this hive's streams before);
    curl's own `--retry` covers a pre-connect hiccup, `--speed-limit`/`--speed-time` gives up
    on a stalled-but-still-open connection rather than hanging forever, and the CALLER
    (`_chunks`'s `finally` below) is what detects and reports a hiccup that happens mid-
    transfer, after curl has already connected."""
    return subprocess.Popen(  # noqa: S603 -- CURL is a resolved full path (S607) and url is this script's own `releases` CLI argument, not attacker input.
        [
            CURL,
            "--http1.1",
            "--fail",
            "--location",
            "--silent",
            "--show-error",
            "--retry",
            "5",
            "--retry-all-errors",
            "--retry-delay",
            "5",
            "--connect-timeout",
            "30",
            "--speed-limit",
            "1024",
            "--speed-time",
            "60",
            url,
        ],
        stdout=subprocess.PIPE,
    )


def _chunks(source: Path | str, record_tag: bytes):
    """Yield whole ``<record_tag ...>...</record_tag>`` chunks, decompressing as a stream.

    Mirrors the design spike's `parse_dump.py._chunks`: the gzip is never decompressed to
    disk, and the container is cut only at a closing tag boundary, so every yielded chunk
    is well-formed XML on its own once wrapped in a throwaway root element.

    `source` is either a local path (`gzip.open` re-opens a `Path` fresh on every call, which
    matters because this function runs twice over the releases dump -- see `_build_same_as_map`'s
    docstring) or an `http(s)://` URL, streamed via a fresh `curl` subprocess every call instead
    -- never written to disk (the bead brief: the releases dump is ~11 GB and disk here is
    tight; masters/artists/labels are small enough to cache locally and always take the `Path`
    branch). A stream that breaks mid-transfer (curl exiting non-zero, or a `gzip.BadGzipFile`/
    `EOFError` from a truncated read) raises out of this generator uncaught -- restarting a
    gzip member from the middle isn't practical, so the caller's job (`build_graph`'s
    `_retry_whole_pass`) is to redo the WHOLE pass with a brand new stream, not to resume this
    one.
    """
    open_marker = b"<" + record_tag + b" "
    close_marker = b"</" + record_tag + b">"
    tail = b""
    proc = _curl_stream(source) if isinstance(source, str) else None
    proc_stdout = proc.stdout if proc is not None else None
    if proc is not None:
        assert proc_stdout is not None  # always true: `_curl_stream` always passes stdout=PIPE.
    reached_eof = False
    try:
        with gzip.open(proc_stdout if proc_stdout is not None else source, "rb") as stream:
            first = True
            while True:
                block = stream.read(CHUNK_BYTES)
                if not block:
                    break
                data = tail + block
                if first:
                    index = data.find(open_marker)
                    if index >= 0:
                        data = data[index:]
                    first = False
                cut = data.rfind(close_marker)
                if cut < 0:
                    tail = data
                    continue
                cut += len(close_marker)
                yield data[:cut]
                tail = data[cut:]
        reached_eof = True
        rest = tail.strip()
        if rest:
            yield rest
    finally:
        if proc is not None and proc_stdout is not None:
            proc_stdout.close()
            if reached_eof:
                # The gzip stream ended cleanly; curl should have exited 0 by now (or will
                # within moments) -- a non-zero exit here means curl itself reported a
                # transfer error (e.g. a truncated Content-Length) that the gzip decoder
                # didn't happen to notice, which is exactly the "looked fine, wasn't" case
                # this bead's dispatch calls out. Surface it so the whole pass gets retried.
                return_code = proc.wait()
                if return_code != 0:
                    raise RuntimeError(f"curl exited {return_code} after an apparently-complete stream of {source}")
            else:
                # Either an exception is already propagating (a real stream failure -- let it
                # through unmasked) or the caller stopped early on purpose (--limit-chunks): in
                # both cases the remaining transfer is simply unwanted, not an error. SIGTERM,
                # not SIGKILL, so curl gets to unwind (close its own socket) instead of leaving
                # a half-torn-down connection; reap it either way so it never becomes a zombie.
                proc.terminate()
                proc.wait()


def _artist_ids(container: ET.Element | None) -> list[str]:
    if container is None:
        return []
    out = []
    for artist in container.findall("artist"):
        id_el = artist.find("id")
        aid = _usable_id(id_el.text if id_el is not None else None)
        if aid is not None:
            out.append(aid)
    return out


def _role_kept(role: str) -> bool:
    """Whether ROLE's single resolved category is one of the kept ones.

    `categorize_role` does NOT split a compound "A, B" role by comma -- it substring-
    matches the whole lowered/stripped string, longest fragment first, exactly like the
    `graph.credit_role_category` SQL rendering `credited_on.role_category` is a GENERATED
    column over. A per-comma-segment check (what this script's pre-ieu.6 draft, and the
    chw.2 spike's own harness, both did) is not what production's stored `role_category`
    actually is -- it is one category for the whole string.
    """
    return categorize_role(role) in KEPT_CREDIT_CATEGORIES


def _release_extraartists(container: ET.Element | None) -> list[tuple[str, str, str | None]]:
    """Return ``(name, role, xml_id_or_None)`` for every release-level ``extraartists``
    entry with both a name and a role -- exactly what `graph_derivation._credits()` keeps,
    with or without a resolvable id (an id-less entry still asserts `graph.credited_on`,
    just not `graph.same_as`)."""
    if container is None:
        return []
    out: list[tuple[str, str, str | None]] = []
    for artist in container.findall("artist"):
        name = artist.findtext("name")
        role = artist.findtext("role")
        if not name or not role:
            continue
        id_el = artist.find("id")
        out.append((name, role, _usable_id(id_el.text if id_el is not None else None)))
    return out


def _track_position(track: ET.Element) -> str | None:
    """A track or sub-track's raw ``position`` label, or ``None`` -- `graph_derivation.
    _track_position`'s ``track.get("position")`` re-read as an XML child lookup: an absent
    or empty element and a missing key read the same, `findtext`'s own `None` default."""
    return track.findtext("position") or None


def _iter_track_credits(tracklist: ET.Element | None) -> list[tuple[int, int, str, str, str | None]]:
    """Return ``(track_ordinal, sub_track_ordinal, name, role, xml_id)`` for every usable
    ``extraartists`` credit on a track or one of its sub-tracks -- `graph_derivation.
    _track_credits`, minus the `track_position` field this script's callers don't need.
    ``track_ordinal``/``sub_track_ordinal`` are 1-based positions over `findall` order (the
    thing `WITH ORDINALITY` counts in the schema's SQL body), not the dump's own `position`
    string, which can be empty or repeated; ``sub_track_ordinal`` is ``0`` for a credit on
    the track itself."""
    if tracklist is None:
        return []
    rows: list[tuple[int, int, str, str, str | None]] = []
    for track_ordinal, track in enumerate(tracklist.findall("track"), start=1):
        rows.extend((track_ordinal, 0, name, role, xml_id) for name, role, xml_id in _release_extraartists(track.find("extraartists")))
        sub_tracks = track.find("sub_tracks")
        if sub_tracks is None:
            continue
        for sub_track_ordinal, sub_track in enumerate(sub_tracks.findall("track"), start=1):
            rows.extend(
                (track_ordinal, sub_track_ordinal, name, role, xml_id) for name, role, xml_id in _release_extraartists(sub_track.find("extraartists"))
            )
    return rows


def _iter_track_performers(tracklist: ET.Element | None) -> list[str]:
    """Return every usable formal ``<artists>`` performer id named on a track or one of its
    sub-tracks -- `graph_derivation._track_performers`, minus the ordinal/position fields
    this script's caller only needs deduplicated per release (it mirrors
    `_TRACK_PERFORMER_EDGE_SQL`'s ``SELECT DISTINCT release_id, artist_id``, which already
    discards which track(s) contributed a given pair)."""
    if tracklist is None:
        return []
    ids: list[str] = []
    for track in tracklist.findall("track"):
        ids.extend(_artist_ids(track.find("artists")))
        sub_tracks = track.find("sub_tracks")
        if sub_tracks is None:
            continue
        for sub_track in sub_tracks.findall("track"):
            ids.extend(_artist_ids(sub_track.find("artists")))
    return ids


# Set once per worker process by `_pass2_worker_init` (a `Pool(initializer=...)`), not
# passed as part of every chunk's task args: pickling and re-sending a several-million-
# entry map on every one of ~1,200 chunk tasks (instead of once per worker at pool
# startup) turned this into a multi-hour run the first time this was tried -- see the
# bead comments' same_as-comparison investigation.
_SAME_AS: dict[str, frozenset[str]] = {}


def _pass2_worker_init(same_as: dict[str, frozenset[str]]) -> None:
    global _SAME_AS
    _SAME_AS = same_as


def _pass1_same_as_chunk(chunk: bytes) -> dict[str, set[str]]:
    """This chunk's contribution to the global `same_as` map: every release-level AND
    track/sub-track-level `extraartists` entry with a resolvable id, of ANY role --
    `graph.same_as` carries no category filter (only `graph.credited_on`'s/`graph.
    track_credited_on`'s queries do) and no release-or-track-nesting column either.
    `graph_derivation.derive_release`'s own `same_as` PUT statement unions the release-level
    list with the identical list comprehension over `_track_credits`, so this pass must too
    -- a name paired with an id only at track level, never at release level, still has to
    resolve when it turns up again as a release-level or another track's credit."""
    partial: dict[str, set[str]] = {}
    if b"<!DOCTYPE" in chunk or b"<!ENTITY" in chunk:
        return partial
    try:
        root = ET.fromstring(b"<r>" + chunk + b"</r>")  # noqa: S314 -- see _reject_entity_declarations above this file's DOCTYPE/ENTITY guard.
    except ET.ParseError:
        return partial
    for rel in root.findall("release"):
        if rel.get("id") is None:
            continue
        for name, _role, xml_id in _release_extraartists(rel.find("extraartists")):
            if xml_id is not None:
                partial.setdefault(name, set()).add(xml_id)
        for _track_ordinal, _sub_track_ordinal, name, _role, xml_id in _iter_track_credits(rel.find("tracklist")):
            if xml_id is not None:
                partial.setdefault(name, set()).add(xml_id)
    return partial


def _build_same_as_map(releases_source: Path | str, workers: int, limit_chunks: int) -> dict[str, frozenset[str]]:
    """Pass 1: the global `person_name -> {artist_id, ...}` map `credited_by_artist`
    resolves through, built once over the whole releases dump before pass 2 can compute
    any credited edge -- `graph.same_as` is additive and catalog-wide, not per-release.

    This is pass 1 of TWO full reads of `releases_source` (`_run_pool`'s release call, right
    after this one in `build_graph`, is pass 2) -- `_SAME_AS` has to be complete before pass 2
    can resolve a single credit, so the two cannot be fused into one read of the stream. For a
    local `Path` that costs nothing extra (`gzip.open` just reopens the file); for an
    `http(s)://` URL (`--releases-url`, no local copy of the ~11 GB releases dump) it means the
    whole dump is fetched over the network twice, once per pass -- see the bead's dispatch
    message and docs/embeddings.md for why the disk budget here forces that trade.
    """
    print("🔗 pass 1: building the global same_as map (release + track/sub-track extraartists, any role)...", file=sys.stderr)
    same_as: dict[str, set[str]] = {}
    started = time.time()
    ctx = mp.get_context("spawn")
    chunks = _chunks(releases_source, b"release")
    if limit_chunks:
        chunks = (c for _, c in zip(range(limit_chunks), chunks, strict=False))
    processed = 0
    with ctx.Pool(workers) as pool:
        for partial in pool.imap_unordered(_pass1_same_as_chunk, chunks, chunksize=1):
            for name, ids in partial.items():
                existing = same_as.get(name)
                if existing is None:
                    same_as[name] = ids
                else:
                    existing |= ids
            processed += 1
            if processed % 200 == 0:
                print(f"  pass 1: {processed} chunks, {len(same_as):,} distinct names, {time.time() - started:.0f}s", file=sys.stderr, flush=True)
    print(f"🔗 pass 1 done: {len(same_as):,} distinct names, {time.time() - started:.0f}s", file=sys.stderr)
    return {name: frozenset(ids) for name, ids in same_as.items()}


def _tag_names(container: ET.Element | None, tag: str) -> list[str]:
    if container is None:
        return []
    return [el.text for el in container.findall(tag) if el.text]


class ChunkResult:
    """One chunk's edge endpoint pairs, as ``(source_key, target_key)`` uint64 lists, plus
    the raw artist id strings it saw (for the final artist_id -> vector output)."""

    __slots__ = ("artist_ids", "count", "edges")

    def __init__(self) -> None:
        self.edges: dict[str, list[tuple[int, int]]] = {name: [] for name in ALL_RELATIONS}
        self.artist_ids: list[str] = []
        self.count = 0


def _parse_release_chunk(chunk: bytes) -> ChunkResult:
    result = ChunkResult()
    try:
        _reject_entity_declarations(chunk)
        root = ET.fromstring(b"<r>" + chunk + b"</r>")  # noqa: S314 -- see _reject_entity_declarations above this file's DOCTYPE/ENTITY guard.
    except (ET.ParseError, ValueError) as error:  # pragma: no cover -- defensive; well-formed chunks expected
        print(f"⚠️  skipping malformed release chunk: {error}", file=sys.stderr)
        return result
    for rel in root.findall("release"):
        rid = _usable_id(rel.get("id"))
        if rid is None:
            continue
        result.count += 1
        rkey = node_key("r", rid)
        artist_ids = _artist_ids(rel.find("artists"))
        for aid in artist_ids:
            result.edges["by_artist"].append((rkey, node_key("a", aid)))
        result.artist_ids.extend(artist_ids)
        labels = rel.find("labels")
        if labels is not None:
            for label in labels.findall("label"):
                lid = _usable_id(label.get("id"))
                if lid is not None:
                    result.edges["on_label"].append((rkey, node_key("l", lid)))
        master_el = rel.find("master_id")
        mid = _usable_id(master_el.text if master_el is not None else None)
        if mid is not None:
            result.edges["derived_from"].append((rkey, node_key("m", mid)))
        for genre in _tag_names(rel.find("genres"), "genre"):
            result.edges["in_genre"].append((rkey, node_key("g", genre)))
        for style in _tag_names(rel.find("styles"), "style"):
            result.edges["in_style"].append((rkey, node_key("s", style)))
        # credited_on JOIN same_as ON person_name (ieu.6, _CREDITED_ARTIST_EDGE_SQL): a
        # kept-category credit resolves through the GLOBAL same_as map, not its own XML
        # id -- an unresolvable name (never paired with any id anywhere) is dropped, an
        # ambiguous one (paired with more than one id somewhere) fans out to all of them.
        # SELECT DISTINCT release_id, artist_id: dedup per release, whatever the fan-out.
        credited_ids: set[str] = set()
        for name, role, _xml_id in _release_extraartists(rel.find("extraartists")):
            if not _role_kept(role):
                continue
            resolved = _SAME_AS.get(name)
            if resolved:
                credited_ids |= resolved
        for aid in credited_ids:
            result.edges[CREDIT_RELATION].append((rkey, node_key("a", aid)))
        result.artist_ids.extend(credited_ids)

        # graph.track_credited_on JOIN graph.same_as (x3d): same kept-category filter and
        # same-name resolution rule as the release-level credit above, over every track's and
        # sub-track's own extraartists. SELECT DISTINCT release_id, artist_id: dedup per
        # release across every track/sub-track that credited the same resolved artist.
        tracklist = rel.find("tracklist")
        track_credited_ids: set[str] = set()
        for _track_ordinal, _sub_track_ordinal, name, role, _xml_id in _iter_track_credits(tracklist):
            if not _role_kept(role):
                continue
            resolved = _SAME_AS.get(name)
            if resolved:
                track_credited_ids |= resolved
        for aid in track_credited_ids:
            result.edges[TRACK_CREDIT_RELATION].append((rkey, node_key("a", aid)))
        result.artist_ids.extend(track_credited_ids)

        # graph.track_by_artist (x3d): the formal, id-bearing <artists> performer named on a
        # track or sub-track -- no same_as resolution, the id is right on the element, same
        # as by_artist. SELECT DISTINCT release_id, artist_id: dedup per release the same way.
        track_performer_ids = set(_iter_track_performers(tracklist))
        for aid in track_performer_ids:
            result.edges[TRACK_PERFORMER_RELATION].append((rkey, node_key("a", aid)))
        result.artist_ids.extend(track_performer_ids)
    return result


def _parse_master_chunk(chunk: bytes) -> ChunkResult:
    result = ChunkResult()
    try:
        _reject_entity_declarations(chunk)
        root = ET.fromstring(b"<r>" + chunk + b"</r>")  # noqa: S314 -- see _reject_entity_declarations above this file's DOCTYPE/ENTITY guard.
    except (ET.ParseError, ValueError) as error:  # pragma: no cover -- defensive; well-formed chunks expected
        print(f"⚠️  skipping malformed master chunk: {error}", file=sys.stderr)
        return result
    for master in root.findall("master"):
        mid = _usable_id(master.get("id"))
        if mid is None:
            continue
        result.count += 1
        mkey = node_key("m", mid)
        artist_ids = _artist_ids(master.find("artists"))
        for aid in artist_ids:
            result.edges["master_by_artist"].append((mkey, node_key("a", aid)))
        result.artist_ids.extend(artist_ids)
        for genre in _tag_names(master.find("genres"), "genre"):
            result.edges["master_in_genre"].append((mkey, node_key("g", genre)))
        for style in _tag_names(master.find("styles"), "style"):
            result.edges["master_in_style"].append((mkey, node_key("s", style)))
    return result


def _run_pool(
    path: Path | str,
    record_tag: bytes,
    parse_chunk,
    workers: int,
    limit_chunks: int,
    *,
    initializer=None,
    initargs: tuple = (),
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], set[str], int]:
    """Stream `path` through a worker pool; return per-relation (source, target) uint64
    array lists, the DISTINCT artist id strings seen, and the number of records parsed.

    `artist_ids` is a `set`, deduplicated incrementally as each chunk's result arrives, not a
    flat list of every occurrence. The only downstream use of these ids (`build_graph`'s
    `artist_key_to_id` loop) already collapses them by `node_key` anyway -- which duplicate
    string object survives is irrelevant, since equal strings hash identically -- so holding
    every occurrence simultaneously bought nothing but memory. At catalog scale ieu.3 measured
    this flat list at ~5.35 GB for ~77M occurrences of ~9-10M distinct artists; deduplicating
    on the way in instead of at the end keeps peak memory near the distinct count instead of
    the occurrence count (gm-analytics-engine-i37, prompted by this run's tighter host memory
    budget -- see the bead's dispatch message).
    """
    sources: dict[str, list[np.ndarray]] = {name: [] for name in ALL_RELATIONS}
    targets: dict[str, list[np.ndarray]] = {name: [] for name in ALL_RELATIONS}
    artist_ids: set[str] = set()
    count = 0
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        source_chunks = _chunks(path, record_tag)
        if limit_chunks:
            source_chunks = (c for _, c in zip(range(limit_chunks), source_chunks, strict=False))
        for result in pool.imap(parse_chunk, source_chunks, chunksize=1):
            count += result.count
            artist_ids.update(result.artist_ids)
            for name, pairs in result.edges.items():
                if not pairs:
                    continue
                array = np.asarray(pairs, dtype=np.uint64)
                sources[name].append(array[:, 0])
                targets[name].append(array[:, 1])
            if count and count % 200_000 < 1000:
                rate = count / (time.time() - started)
                print(f"  {count:,} {record_tag.decode()} records, {rate:,.0f}/s", file=sys.stderr, flush=True)
    return sources, targets, artist_ids, count


# Exception types `_retry_whole_pass` refuses to retry even under its otherwise-broad catch:
# a real bug in this script's own code, which retrying five times with backoff only delays
# discovering. Everything else is assumed to be a transient stream/subprocess/OS problem
# worth retrying rather than a bug worth enumerating -- this is deliberately a DENY-list, not
# an allow-list of named "stream" exceptions. An earlier version of this function used an
# allow-list (`RuntimeError, OSError, EOFError, gzip.BadGzipFile`) and it missed exactly the
# failure it existed for: `zlib.error` (a corrupted deflate block from a stream curl gave up
# on mid-transfer, `curl: (28) Operation too slow`) killed a real run (gm-analytics-engine-i37,
# 2026-09-27, pass 1 died at chunk ~1800/~3600) because that exception type was never added to
# the list -- the exact whack-a-mole an allow-list guarantees. A deny-list only has to name the
# few things that must NEVER be retried.
_NON_RETRYABLE_PROGRAMMER_ERRORS: Final = (
    KeyError,
    TypeError,
    ValueError,
    AttributeError,
    IndexError,
    NameError,
    AssertionError,
    SyntaxError,
    ImportError,
)


def _retry_whole_pass[T](description: str, run: Callable[[], T], *, max_attempts: int = 5, initial_delay_s: float = 10.0) -> T:
    """Run `run()`, retrying it from scratch (a brand new `curl` subprocess and worker pool,
    via `_chunks`/`_build_same_as_map`/`_run_pool`) on any exception except
    `_NON_RETRYABLE_PROGRAMMER_ERRORS` -- see that constant's comment for why this is a
    deny-list, not an allow-list of named "stream" exceptions. A local-`Path` source can raise
    a retryable-looking error too (a genuinely unreadable file), in which case the retries
    just fail the same way every time and this still reports the real error after
    `max_attempts` -- it costs nothing to share the one loop rather than branching on source
    type.

    Cleanup between attempts is automatic, not something this loop does itself: `_chunks`'s
    own `finally` block always closes curl's stdout and terminates that attempt's `curl`
    subprocess (reached_eof is False on any exception path), and `_build_same_as_map`/
    `_run_pool`'s `with ctx.Pool(...) as pool:` always calls `pool.terminate()` on the way out,
    exception or not -- both run as the exception unwinds through them, before it ever reaches
    this function's `except` clause, so a retried attempt always starts from a clean subprocess
    and a clean pool.

    Exponential backoff, capped at 120s, mirrors `scripts/fetch_aug_masters.sh`'s (uncommitted
    ops script) retry shape for the same reason: an HTTP/2 reset or mid-stream corruption is
    exactly the failure this loop exists for, per the bead's dispatch message.
    """
    delay = initial_delay_s
    for attempt in range(1, max_attempts + 1):
        print(f"▶️  {description}: attempt {attempt}/{max_attempts}", file=sys.stderr, flush=True)
        try:
            return run()
        except _NON_RETRYABLE_PROGRAMMER_ERRORS:
            raise
        except Exception as error:  # deliberately broad -- see _NON_RETRYABLE_PROGRAMMER_ERRORS' comment.
            if attempt == max_attempts:
                raise
            print(
                f"⚠️  {description}: attempt {attempt}/{max_attempts} failed ({error!r}), retrying the WHOLE pass in {delay:.0f}s",
                file=sys.stderr,
                flush=True,
            )
            time.sleep(delay)
            delay = min(delay * 2, 120.0)
    raise AssertionError("unreachable")  # pragma: no cover -- the loop above always returns or raises


def build_graph(
    releases_source: Path | str,
    masters_path: Path,
    workers: int,
    limit_chunks: int = 0,
    *,
    expected_peak_gb: float = 0.0,
    memory_margin_gb: float = 2.0,
    memory_poll_interval_s: float = 60.0,
    same_as_checkpoint: Path | None = None,
    disk_margin_bytes: int = 0,
    disk_poll_interval_s: float = 60.0,
) -> dict:
    """Parse both dumps, build the Adjacency, and return everything the parity report
    and `fastrp` need. Raises nothing on a normal run; malformed chunks are skipped and
    logged, never fatal.

    `releases_source` is a local `Path` or an `http(s)://` URL (`--releases-url`); either way
    it is read TWICE (`_build_same_as_map`'s docstring), and each of those two reads is its
    own `_retry_whole_pass` -- a stream that breaks partway through only costs a redo of the
    pass it broke in, never the other one.

    `expected_peak_gb` (0 = disabled, the default) gates the graph-build step behind the
    footprint-aware `wait_for_memory` -- see that function's docstring for why here
    specifically and why footprint-aware, and the bead's dispatch message for why this host's
    tight memory needs it at all.

    `same_as_checkpoint`, if given, is loaded instead of re-running pass 1 when it already
    exists, and saved right after pass 1 computes it otherwise -- a restart after a crash
    anywhere in pass 2, the graph build, or fastrp then skips straight back to pass 2 instead
    of re-streaming the whole releases dump a third time for the same ~10 minutes of work
    (the dispatcher's ask, after the ~50 minutes an earlier in-flight restart cost here).
    """
    # Memory checkpoints below are named phase BOUNDARIES of `resource.getrusage`'s
    # monotonic, process-lifetime high-water mark, not isolated per-phase costs -- ru_maxrss
    # never decreases, even after `del` + `gc.collect()` frees real memory back to the
    # allocator. The explicit frees between phases are still worth doing: they let the
    # *next* phase's allocations reuse that freed memory instead of growing the peak
    # further, which is what makes the checkpoint *sequence* an honest (if not perfectly
    # isolated) attribution of where this script's memory actually goes -- see the "Memory
    # at catalog scale" finding in docs/recall_and_churn.md this instrumentation feeds.
    parse_started = time.perf_counter()
    if same_as_checkpoint is not None and same_as_checkpoint.exists():
        print(f"📦 loading same_as checkpoint: {same_as_checkpoint}", file=sys.stderr)
        same_as = load_same_as_checkpoint(same_as_checkpoint)
    else:
        same_as = _retry_whole_pass("pass 1 (same_as map)", lambda: _build_same_as_map(releases_source, workers, limit_chunks))
        if same_as_checkpoint is not None:
            print(f"📦 saving same_as checkpoint: {same_as_checkpoint}", file=sys.stderr)
            save_same_as_checkpoint(same_as, same_as_checkpoint, disk_margin_bytes=disk_margin_bytes, disk_poll_interval_s=disk_poll_interval_s)
    same_as_peak_rss = peak_rss_bytes()

    print(f"📖 parsing releases: {releases_source}", file=sys.stderr)
    r_sources, r_targets, r_artist_ids, release_count = _retry_whole_pass(
        "pass 2 (release edges)",
        # `same_as=same_as` default: `_retry_whole_pass` calls this synchronously (well before
        # the `del same_as` a few lines below), so a plain closure over `same_as` would run
        # correctly too -- the default is here only because ruff's F821 liveness analysis
        # mis-attributes that later `del` to this lambda's closure and flags it as undefined; a
        # bound default sidesteps that false positive without changing runtime behaviour.
        lambda same_as=same_as: _run_pool(
            releases_source, b"release", _parse_release_chunk, workers, limit_chunks, initializer=_pass2_worker_init, initargs=(same_as,)
        ),
    )
    del same_as  # only needed by the (now-finished) release-parsing workers above.

    print(f"📖 parsing masters: {masters_path}", file=sys.stderr)
    m_sources, m_targets, m_artist_ids, master_count = _run_pool(masters_path, b"master", _parse_master_chunk, workers, limit_chunks)
    parse_elapsed = time.perf_counter() - parse_started
    parse_peak_rss = peak_rss_bytes()

    # NOT a dict-spread merge: `_run_pool` initializes every one of the 8 relation keys
    # in both `r_sources`/`m_sources` regardless of which dump produced it (the release
    # pass's `master_*` entries and the master pass's release-relation entries are both
    # always-present empty lists), so `{**r_sources, **m_sources}` would let the master
    # pass's empty release-relation lists silently overwrite the release pass's real data.
    # Concatenating each relation's two (one real, one empty) contributions is correct
    # for both directions and never loses data.
    sources = {name: r_sources[name] + m_sources[name] for name in ALL_RELATIONS}
    targets = {name: r_targets[name] + m_targets[name] for name in ALL_RELATIONS}
    del r_sources, r_targets, m_sources, m_targets

    # artist_id -> node_key, first occurrence wins (all occurrences hash identically).
    artist_key_to_id: dict[int, str] = {}
    for aid in r_artist_ids:
        artist_key_to_id.setdefault(node_key("a", aid), aid)
    for aid in m_artist_ids:
        artist_key_to_id.setdefault(node_key("a", aid), aid)
    del r_artist_ids, m_artist_ids

    relation_arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    all_keys: list[np.ndarray] = []
    relation_counts: dict[str, int] = {}
    for name in ALL_RELATIONS:
        src = np.concatenate(sources[name]) if sources[name] else np.zeros(0, dtype=np.uint64)
        dst = np.concatenate(targets[name]) if targets[name] else np.zeros(0, dtype=np.uint64)
        relation_arrays[name] = (src, dst)
        relation_counts[name] = int(src.size)
        all_keys.append(src)
        all_keys.append(dst)
    del sources, targets
    gc.collect()
    pre_build_peak_rss = peak_rss_bytes()

    # Gate the graph-build/FastRP phase -- this pipeline's own instrumentation (docs/
    # recall_and_churn.md, "Memory at catalog scale") shows THIS is where host memory spikes
    # to its peak, not parsing. Waiting here, not before parsing, is deliberate: parsing is
    # comparatively cheap and can safely proceed while memory is still recovering from
    # whatever else is using it. Footprint-aware (wait_for_memory's own docstring): this
    # process's own already-parsed graph counts toward `expected_peak_gb` for free.
    wait_for_memory(int(expected_peak_gb * 1e9), margin_bytes=int(memory_margin_gb * 1e9), poll_interval_s=memory_poll_interval_s)

    build_started = time.perf_counter()
    # Edge endpoints repeat (the same artist/release/... appears in many edges), but
    # `NodeIndex` takes the distinct vertex set -- unlike `graph.vertex_degree`, which is
    # already one row per vertex, these keys come straight from edge occurrences here.
    all_keys_concatenated = np.concatenate(all_keys) if all_keys else np.zeros(0, dtype=np.uint64)
    del all_keys
    nodes = NodeIndex(np.unique(all_keys_concatenated))
    del all_keys_concatenated
    builder = AdjacencyBuilder(nodes)
    for name in ALL_RELATIONS:
        src, dst = relation_arrays[name]
        if src.size:
            builder.add_edges(src, dst)
    adjacency = builder.build()
    build_elapsed = time.perf_counter() - build_started
    build_peak_rss = peak_rss_bytes()

    def _distinct_count(*arrays: np.ndarray) -> int:
        non_empty = [array for array in arrays if array.size]
        return int(np.unique(np.concatenate(non_empty)).size) if non_empty else 0

    artist_keys = np.fromiter(artist_key_to_id.keys(), dtype=np.uint64)
    distinct_by_kind = {
        "a": int(artist_keys.size),
        "r": _distinct_count(*(relation_arrays[n][0] for n in _RELEASE_SOURCED_RELATIONS)),
        "l": _distinct_count(relation_arrays["on_label"][1]),
        "m": _distinct_count(relation_arrays["derived_from"][1], relation_arrays["master_by_artist"][0]),
        "g": _distinct_count(relation_arrays["in_genre"][1], relation_arrays["master_in_genre"][1]),
        "s": _distinct_count(relation_arrays["in_style"][1], relation_arrays["master_in_style"][1]),
    }

    return {
        "nodes": nodes,
        "adjacency": adjacency,
        "artist_key_to_id": artist_key_to_id,
        "release_count": release_count,
        "master_count": master_count,
        "relation_counts": relation_counts,
        "distinct_by_kind": distinct_by_kind,
        "parse_elapsed_s": parse_elapsed,
        "same_as_peak_rss_gb": same_as_peak_rss / 1e9,
        "parse_peak_rss_gb": parse_peak_rss / 1e9,
        "pre_build_peak_rss_gb": pre_build_peak_rss / 1e9,
        "build_elapsed_s": build_elapsed,
        "build_peak_rss_gb": build_peak_rss / 1e9,
    }


_GRAPH_CHECKPOINT_ARRAYS: Final = ("adjacency.npz", "node_keys.npy", "degree.npy", "artist_ids.npz", "meta.json")


def save_graph_checkpoint(graph: dict, path: Path, *, disk_margin_bytes: int = 0, disk_poll_interval_s: float = 60.0) -> None:
    """Serialize enough of `build_graph`'s return value to skip both parse passes and the
    `AdjacencyBuilder.build()` step on a restart or a second invocation for the same month's
    weight sweep: the adjacency CSR matrix, its node-key order, the artist id map, and the
    scalar counts/timings the parity report and summary line print. Every weight in the
    sweep (`--w0`) reads the *same* graph -- only `FastRPConfig.weights` and thus `fastrp()`
    differ between them -- so one checkpoint written after the first `--w0` invocation lets
    every later one skip straight to `fastrp()` via `--graph-checkpoint`.

    `disk_margin_bytes > 0` gates the write behind `wait_for_disk`, sized from the CSR
    matrix's own arrays (the checkpoint's dominant cost by far) plus the artist id map --
    see `wait_for_disk`'s docstring for why this matters on this host.

    Atomic: every file is written into a TEMPORARY sibling directory
    (`<path>.tmp-<pid>`), and only once all of them succeed is that whole directory renamed
    into place at `path` with a single `os.replace` (atomic on the same filesystem). A
    process killed mid-write (gm-analytics-engine-i37, 2026-09-27: the recall-phase watcher
    killed the orchestrator mid-write of a *different* npz and the resulting truncated file
    passed a plain existence check as "done") leaves only the abandoned temp directory
    behind, never a half-written `path` that `graph_checkpoint_exists` could mistake for
    complete.
    """
    adjacency: Adjacency = graph["adjacency"]
    artist_key_to_id: dict[int, str] = graph["artist_key_to_id"]
    if disk_margin_bytes > 0:
        transition = adjacency.transition
        expected = (
            transition.data.nbytes
            + transition.indices.nbytes
            + transition.indptr.nbytes
            + adjacency.nodes.keys.nbytes
            + adjacency.degree.nbytes
            + len(artist_key_to_id) * 8
            + _artist_ids_nbytes(artist_key_to_id.values())
        )
        wait_for_disk(path, expected, margin_bytes=disk_margin_bytes, poll_interval_s=disk_poll_interval_s)

    tmp_dir = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    try:
        with (tmp_dir / "adjacency.npz").open("wb") as f:
            sp.save_npz(f, adjacency.transition)
        with (tmp_dir / "node_keys.npy").open("wb") as f:
            np.save(f, adjacency.nodes.keys)
        with (tmp_dir / "degree.npy").open("wb") as f:
            np.save(f, adjacency.degree)
        with (tmp_dir / "artist_ids.npz").open("wb") as f:
            np.savez_compressed(
                f,
                keys=np.fromiter(artist_key_to_id.keys(), dtype=np.uint64),
                ids=np.asarray(list(artist_key_to_id.values()), dtype=object),
            )
        # "nodes" is dropped, not just "adjacency"/"artist_key_to_id": it is the exact same
        # NodeIndex `adjacency.nodes` already is (build_graph's own return dict aliases the
        # two), not JSON-serializable, and fully reconstructed by `load_graph_checkpoint`
        # from the `Adjacency` it loads.
        meta = {key: value for key, value in graph.items() if key not in ("adjacency", "nodes", "artist_key_to_id")}
        (tmp_dir / "meta.json").write_text(json.dumps(meta))
        path_bak = path.with_name(f"{path.name}.replaced-{os.getpid()}")
        if path.exists():
            path.replace(path_bak)  # never leaves `path` briefly absent between the two renames below.
        tmp_dir.replace(path)
        shutil.rmtree(path_bak, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise


def graph_checkpoint_exists(path: Path) -> bool:
    return path.is_dir() and all((path / name).exists() for name in _GRAPH_CHECKPOINT_ARRAYS)


def load_graph_checkpoint(path: Path) -> dict:
    """The inverse of `save_graph_checkpoint`. Re-sorting `node_keys.npy` through `NodeIndex`
    is cheap (it is already sorted) and re-validates uniqueness; nothing here re-parses or
    re-derives the graph itself."""
    transition = sp.load_npz(path / "adjacency.npz")
    nodes = NodeIndex(np.load(path / "node_keys.npy"))
    degree = np.load(path / "degree.npy")
    adjacency = Adjacency(nodes=nodes, transition=transition, degree=degree)
    # allow_pickle=True: the "ids" array holds Python str objects (dtype=object), which numpy
    # can only round-trip through its pickle protocol -- safe here because this file is a
    # scratch checkpoint this same script wrote moments earlier (never a download or a value
    # from an untrusted source), the same trust boundary the final --out .npz below already
    # relies on for its own dtype=object artist_ids array.
    artist_npz = np.load(path / "artist_ids.npz", allow_pickle=True)
    artist_key_to_id = dict(zip(artist_npz["keys"].tolist(), artist_npz["ids"].tolist(), strict=True))
    meta = json.loads((path / "meta.json").read_text())
    return {"adjacency": adjacency, "artist_key_to_id": artist_key_to_id, **meta}


# Separator joining one name's ids into a single string for the same_as checkpoint's flat
# array storage -- ASCII unit separator, never a character a Discogs artist id (always
# digits) or a same_as-key person name from `_release_extraartists` legitimately contains.
_SAME_AS_ID_SEPARATOR: Final = "\x1f"


def _atomic_savez_compressed(path: Path, **arrays: object) -> None:
    """`np.savez_compressed(path, **arrays)`, but atomic: written to a temporary sibling file
    first, then renamed into place with `os.replace` (atomic on the same filesystem) only
    once the write has fully succeeded. A process killed mid-write leaves only the abandoned
    `.tmp-<pid>` file behind, never a truncated file visible at `path` -- see
    `save_graph_checkpoint`'s docstring for the real incident this guards against.

    Writes through an open file OBJECT, not the tmp path string, deliberately: `np.savez_
    compressed` appends a `.npz` extension to a bare string/Path argument that doesn't
    already end in one, which `<path>.tmp-<pid>` never does -- passing the already-open file
    sidesteps that renaming entirely.
    """
    tmp_path = path.with_name(f"{path.name}.tmp-{os.getpid()}")
    try:
        with tmp_path.open("wb") as f:
            np.savez_compressed(f, **arrays)
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def save_same_as_checkpoint(
    same_as: dict[str, frozenset[str]], path: Path, *, disk_margin_bytes: int = 0, disk_poll_interval_s: float = 60.0
) -> None:
    """Persist pass 1's global `same_as` map (`person_name -> {artist_id, ...}`) to disk right
    after it finishes -- pass 1 streams the WHOLE releases dump once, over the network, just
    to build this (~10 minutes at real catalog scale, per the bead's dispatch), so a restart
    of this month after a crash anywhere in pass 2 or later can load it back and skip straight
    to pass 2 instead of re-streaming the entire dump a third time.

    Flattened into two parallel object arrays (`names`, each name's ids joined by
    `_SAME_AS_ID_SEPARATOR`) and `np.savez_compressed`, matching this script's other
    checkpoints' numpy-native shape rather than JSON -- millions of names as JSON text would
    be far slower to write and parse for no benefit here.
    """
    names = np.fromiter(same_as.keys(), dtype=object, count=len(same_as))
    joined_ids = np.fromiter((_SAME_AS_ID_SEPARATOR.join(ids) for ids in same_as.values()), dtype=object, count=len(same_as))
    if disk_margin_bytes > 0:
        expected = _artist_ids_nbytes(same_as.keys()) + _artist_ids_nbytes(joined_ids.tolist())
        wait_for_disk(path, expected, margin_bytes=disk_margin_bytes, poll_interval_s=disk_poll_interval_s)
    path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_savez_compressed(path, names=names, joined_ids=joined_ids)


def load_same_as_checkpoint(path: Path) -> dict[str, frozenset[str]]:
    """The inverse of `save_same_as_checkpoint`."""
    # allow_pickle=True: both arrays hold Python str objects (dtype=object) -- the same
    # trust boundary `load_graph_checkpoint`'s own `allow_pickle=True` already documents
    # (this script's own scratch checkpoint, never untrusted input).
    with np.load(path, allow_pickle=True) as data:
        names = data["names"].tolist()
        joined_ids = data["joined_ids"].tolist()
    return {name: frozenset(joined.split(_SAME_AS_ID_SEPARATOR)) for name, joined in zip(names, joined_ids, strict=True)}


def print_parity_report(graph: dict) -> None:
    print("\n=== parity report ===", file=sys.stderr)
    print(f"releases parsed:  {graph['release_count']:,}", file=sys.stderr)
    print(f"masters parsed:   {graph['master_count']:,}", file=sys.stderr)
    print(f"total vertices:   {len(graph['adjacency'].nodes):,}", file=sys.stderr)
    for kind, label in (("a", "artists"), ("r", "releases"), ("l", "labels"), ("m", "masters"), ("g", "genres"), ("s", "styles")):
        print(f"distinct {label:9s} (kind={kind}): {graph['distinct_by_kind'][kind]:,}", file=sys.stderr)
    for name in ALL_RELATIONS:
        print(f"edges {name:20s}: {graph['relation_counts'][name]:,}", file=sys.stderr)
    print(f"peak RSS after same_as map (pass 1):        {graph['same_as_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after full parse (pass 2, both dumps): {graph['parse_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after freeing parser structures, pre-build: {graph['pre_build_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after AdjacencyBuilder.build():     {graph['build_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"parse: {graph['parse_elapsed_s']:.1f}s, build: {graph['build_elapsed_s']:.1f}s", file=sys.stderr)


def _out_path_for_config(out: Path, w0: float, self_weight: float, multiple: bool) -> Path:
    """`out` unchanged when there is only one `--w0` value and `--self-weight` is 0 (backward
    compatible with every existing caller); otherwise `out` with `.w0-<value>` and/or
    `.self-<value>` inserted before the suffix, e.g. `aug.npz` -> `aug.w0-0.1.npz` or
    `aug.npz` -> `aug.self-0.05.npz`, so a sweep's outputs, or a non-default self weight,
    never collide on one filename."""
    stem = out.stem
    if multiple:
        stem = f"{stem}.w0-{w0:g}"
    if self_weight:
        stem = f"{stem}.self-{self_weight:g}"
    return out if stem == out.stem else out.with_name(f"{stem}{out.suffix}")


def _releases_source(value: str) -> Path | str:
    """`argparse` type for the `releases` positional: an `http://`/`https://` URL is kept as a
    string (streamed fresh, twice, via `curl` -- see `_chunks`/`_build_same_as_map`'s
    docstrings); anything else is a local path, exactly like `masters` always is."""
    return value if value.startswith(("http://", "https://")) else Path(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "releases",
        type=_releases_source,
        help="a local releases.xml.gz path, or an http(s):// URL to stream (never written to "
        "disk -- see the bead brief on why the ~11 GB releases dump is streamed, not cached).",
    )
    parser.add_argument("masters", type=Path)
    parser.add_argument("--dump-id", required=True)
    parser.add_argument("--dump-date", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--w0",
        type=float,
        nargs="+",
        default=[0.0],
        help="one or more step-0 FastRP weights (weights=w0,1,1,1,1); the graph is parsed and "
        "built ONCE and fastrp() re-run once per value, each as its own model_version and its "
        "own output file (see --out's per-w0 naming when more than one value is given).",
    )
    parser.add_argument(
        "--self-weight",
        type=float,
        default=0.0,
        help="FastRPConfig.self_weight (gm-analytics-engine-8ts): weight on normalize(R[v]), "
        "the node's own hashed projection row -- breaks exact ties between artists with "
        "identical graph neighbourhoods. Applied to every --w0 value in the sweep. 0.0 (the "
        "default) is bit-identical to omitting it.",
    )
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--threads", type=int, default=6)
    parser.add_argument("--block-columns", type=int, default=4)
    parser.add_argument("--limit-chunks", type=int, default=0, help="parse only the first N chunks of each dump (smoke test)")
    parser.add_argument("--parity-only", action="store_true", help="stop after the parity report, before fastrp")
    parser.add_argument(
        "--delete-dumps-after-parse",
        action="store_true",
        help="delete the releases/masters dump files immediately after a successful parse, before fastrp runs. "
        "Safe: the gzip streams are fully consumed and closed by the time build_graph() returns, nothing after "
        "this point re-reads them. Only for a dump this invocation downloaded itself, never a shared one.",
    )
    parser.add_argument(
        "--graph-checkpoint",
        type=Path,
        default=None,
        help="directory to save the parsed graph (adjacency, node keys, artist ids) to after a "
        "fresh parse, or load it from instead of re-parsing/rebuilding when it already exists -- "
        "every --w0 value reads the identical graph, so a checkpoint written by the first "
        "invocation of a sweep lets later ones (or a restart after a crash) skip straight to "
        "fastrp(). Implies --parity-only is still honored: a checkpoint is saved even when "
        "--parity-only stops before fastrp, so a later non---parity-only run can reuse it.",
    )
    parser.add_argument(
        "--expected-peak-gb",
        type=float,
        default=0.0,
        help="the estimated whole-process peak RSS (GB) this run's graph-build/FastRP phase "
        "will reach -- gates that phase behind a FOOTPRINT-AWARE wait (polling vm_stat, macOS "
        "only): waits until (this process's own current footprint + host free+inactive memory) "
        "covers --expected-peak-gb + --memory-margin-gb, not until a flat amount is free "
        "system-wide (a flat threshold double-counts memory this process already holds -- see "
        "wait_for_memory()'s docstring). 0 (the default) disables the wait entirely; skipped "
        "altogether when --graph-checkpoint is loaded from an existing checkpoint (no fresh "
        "parse/build happens on that path).",
    )
    parser.add_argument(
        "--memory-margin-gb",
        type=float,
        default=2.0,
        help="safety margin (GB) added on top of --expected-peak-gb for the footprint-aware memory wait",
    )
    parser.add_argument("--memory-poll-interval-s", type=float, default=60.0, help="seconds between --expected-peak-gb checks while waiting")
    parser.add_argument(
        "--min-free-disk-margin-gb",
        type=float,
        default=3.0,
        help="before writing a graph checkpoint or a --w0 output npz, require this many GB free "
        "BEYOND that write's own expected size (computed from the in-memory arrays being "
        "written), waiting and logging otherwise rather than risking a mid-write "
        "'No space left on device' that leaves a truncated, corrupt file behind. 0 disables "
        "the check (a smoke test on a tiny slice never needs a margin like this).",
    )
    parser.add_argument("--disk-poll-interval-s", type=float, default=60.0, help="seconds between --min-free-disk-margin-gb checks while waiting")
    parser.add_argument(
        "--same-as-checkpoint",
        type=Path,
        default=None,
        help="file to save pass 1's same_as map to right after it's computed, or load it from "
        "instead of re-streaming the releases dump for pass 1 when it already exists -- a "
        "restart after a crash in pass 2, the graph build, or fastrp skips straight back to "
        "pass 2 instead of redoing pass 1's own ~10-minute stream. Independent of "
        "--graph-checkpoint: this covers a crash BEFORE the graph checkpoint would exist.",
    )
    args = parser.parse_args()
    disk_margin_bytes = int(args.min_free_disk_margin_gb * 1e9)

    if args.graph_checkpoint is not None and graph_checkpoint_exists(args.graph_checkpoint):
        print(f"📦 loading graph checkpoint: {args.graph_checkpoint}", file=sys.stderr)
        graph = load_graph_checkpoint(args.graph_checkpoint)
    else:
        graph = build_graph(
            args.releases,
            args.masters,
            args.workers,
            args.limit_chunks,
            expected_peak_gb=args.expected_peak_gb,
            memory_margin_gb=args.memory_margin_gb,
            memory_poll_interval_s=args.memory_poll_interval_s,
            same_as_checkpoint=args.same_as_checkpoint,
            disk_margin_bytes=disk_margin_bytes,
            disk_poll_interval_s=args.disk_poll_interval_s,
        )
        if args.graph_checkpoint is not None:
            print(f"📦 saving graph checkpoint: {args.graph_checkpoint}", file=sys.stderr)
            save_graph_checkpoint(graph, args.graph_checkpoint, disk_margin_bytes=disk_margin_bytes, disk_poll_interval_s=args.disk_poll_interval_s)

    print_parity_report(graph)

    if args.delete_dumps_after_parse:
        for dump_path in (args.releases, args.masters):
            if not isinstance(dump_path, Path) or not dump_path.exists():  # a URL or a checkpoint load: nothing local to remove
                continue
            size_gb = dump_path.stat().st_size / 1e9
            dump_path.unlink()
            print(f"🗑️  deleted {dump_path} ({size_gb:.2f} GB) -- parse is done, fastrp needs no further disk read of it", file=sys.stderr)

    if args.parity_only:
        print("🛑 --parity-only: stopping before fastrp", file=sys.stderr)
        return

    artist_key_to_id = graph["artist_key_to_id"]
    artist_ids = list(artist_key_to_id.values())
    artist_keys = np.fromiter(artist_key_to_id.keys(), dtype=np.uint64)
    positions = graph["adjacency"].nodes.positions(artist_keys)
    # Each artist's undirected degree in THIS month's graph, aligned with `artist_ids` --
    # gm-analytics-engine-i37's degree-bucketed recall breakdown (scripts/measure_recall_churn.py)
    # needs it per query artist and this script is the only place that ever builds the
    # `Adjacency` those degrees come from; `measure_recall_churn.py` has no graph of its own.
    degrees = graph["adjacency"].degree[positions]

    multiple = len(args.w0) > 1
    for w0 in args.w0:
        out = _out_path_for_config(args.out, w0, args.self_weight, multiple)
        config = FastRPConfig(weights=(w0, 1.0, 1.0, 1.0, 1.0), self_weight=args.self_weight)
        print(f"\n🔢 running fastrp: {config.model_version}", file=sys.stderr)
        print(f"numpy {np.__version__}, scipy {scipy.__version__}", file=sys.stderr)
        fastrp_started = time.perf_counter()
        vectors = fastrp(graph["adjacency"], config, rows=positions, out_dtype=np.float16, block_columns=args.block_columns, threads=args.threads)
        fastrp_elapsed = time.perf_counter() - fastrp_started
        fastrp_peak_rss = peak_rss_bytes()
        print(f"fastrp: {fastrp_elapsed:.1f}s, peak RSS {fastrp_peak_rss / 1e9:.2f} GB, {vectors.shape[0]:,} vectors", file=sys.stderr)

        # `vectors.nbytes` (raw float16, before savez_compressed's own compression -- an
        # overestimate, not an underestimate, of what actually lands on disk) plus the artist
        # ids: wait_for_disk's job is a safety margin against a mid-write ENOSPC, not a
        # byte-perfect prediction. See wait_for_disk's docstring for why this check exists at
        # all on this host.
        expected_npz_bytes = vectors.nbytes + _artist_ids_nbytes(artist_ids)
        wait_for_disk(out, expected_npz_bytes, margin_bytes=disk_margin_bytes, poll_interval_s=args.disk_poll_interval_s)
        out.parent.mkdir(parents=True, exist_ok=True)
        _atomic_savez_compressed(
            out,
            artist_ids=np.asarray(artist_ids, dtype=object),
            vectors=vectors,
            model_version=config.model_version,
            # `weights` (gm-analytics-engine-i37's w0 sweep): the raw tuple `config` was built
            # from, not just its already-composed `model_version` string -- a downstream
            # consumer (scripts/measure_recall_churn.py) needs a real `FastRPConfig` to call
            # `stored_model_version` with, and re-parsing one back out of `model_version`'s
            # text is both more code and more fragile than saving the tuple that produced it.
            weights=np.asarray(config.weights, dtype=np.float64),
            # `self_weight` (gm-analytics-engine-8ts): same reasoning as `weights` above --
            # without it, `measure_recall_churn.py`'s `_load_month` would reconstruct a
            # `FastRPConfig` with `self_weight`'s default (0.0) for every file, and its own
            # self-consistency check (`method_version != config.model_version`) would reject
            # any npz saved with a non-zero self weight as internally inconsistent.
            self_weight=config.self_weight,
            # `degrees` (i37's degree-bucketed recall): each row of `vectors`/`artist_ids` is
            # one artist; `degrees[i]` is that SAME artist's undirected degree in this month's
            # graph, independent of `w0` (the graph -- and therefore every artist's degree --
            # is identical across the whole sweep; only the FastRP weights change).
            degrees=degrees,
            dump_id=args.dump_id,
            dump_date=args.dump_date,
            numpy_version=np.__version__,
            scipy_version=scipy.__version__,
        )
        summary = {
            "dump_id": args.dump_id,
            "dump_date": args.dump_date,
            "method_version": config.model_version,
            "w0": w0,
            "self_weight": args.self_weight,
            "release_count": graph["release_count"],
            "master_count": graph["master_count"],
            "distinct_by_kind": graph["distinct_by_kind"],
            "relation_counts": graph["relation_counts"],
            "parse_elapsed_s": round(graph["parse_elapsed_s"], 1),
            "same_as_peak_rss_gb": round(graph["same_as_peak_rss_gb"], 2),
            "parse_peak_rss_gb": round(graph["parse_peak_rss_gb"], 2),
            "pre_build_peak_rss_gb": round(graph["pre_build_peak_rss_gb"], 2),
            "build_elapsed_s": round(graph["build_elapsed_s"], 1),
            "build_peak_rss_gb": round(graph["build_peak_rss_gb"], 2),
            "fastrp_elapsed_s": round(fastrp_elapsed, 1),
            "fastrp_peak_rss_gb": round(fastrp_peak_rss / 1e9, 2),
            "vectors_written": int(vectors.shape[0]),
            "out": str(out),
        }
        print(json.dumps(summary))


if __name__ == "__main__":
    main()
