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

Known remaining gap vs. the chw.2 spike's own (broader) graph, out of scope for this
bead and ieu.6 alike -- tracked as follow-ups gm-analytics-engine-x3d (embedding
pipeline), gm-discogs-sql-loader-b2a (loader derivation), and gm-database-schema-ug3v
(schema/graph relations): per-track and sub-track ``extraartists`` credits, and per-track
``<artists>`` (track performers). Neither exists in the `graph` schema today. Sized on
the 2026-08 dump (script: scripts/count_credit_slices.py, not committed -- see the bead
comments): the spike's full credit scope (release+track+subtrack) reaches 6,917,277
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
import resource
import sys
import time
from pathlib import Path
from xml.etree import ElementTree as ET


THREADS = os.environ.get("FASTRP_THREADS", "6")
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, THREADS)

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from common.credit_roles import categorize_role  # noqa: E402

from insights.embeddings import AdjacencyBuilder, FastRPConfig, NodeIndex, fastrp, node_key  # noqa: E402


CHUNK_BYTES = 16 << 20

# The eight relations `insights/embedding_pipeline.py`'s `_EDGE_RELATIONS` reads today,
# plus `credited_by_artist` (placeholder name; see the module docstring) for the ninth
# relation gm-analytics-engine-ieu.6 is adding, each as (source_kind, target_kind).
RELEASE_RELATIONS: tuple[str, ...] = ("by_artist", "on_label", "derived_from", "in_genre", "in_style")
MASTER_RELATIONS: tuple[str, ...] = ("master_by_artist", "master_in_genre", "master_in_style")
CREDIT_RELATION: str = "credited_by_artist"
ALL_RELATIONS: tuple[str, ...] = (*RELEASE_RELATIONS, *MASTER_RELATIONS, CREDIT_RELATION)

# The chw.2 spike's kept role categories (production, engineering, session, and
# `common.credit_roles`' catch-all "other"), matching ieu.6's stated scope exactly:
# mastering/design/management credits are dropped -- a cutting engineer or sleeve
# photographer links releases by vendor, not by sound (ADR 0013).
KEPT_CREDIT_CATEGORIES: frozenset[str] = frozenset({"production", "engineering", "session", "other"})


def peak_rss_bytes() -> int:
    # macOS reports ru_maxrss in bytes, Linux in KiB.
    scale = 1 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * scale


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


def _chunks(path: Path, record_tag: bytes):
    """Yield whole ``<record_tag ...>...</record_tag>`` chunks, decompressing as a stream.

    Mirrors the design spike's `parse_dump.py._chunks`: the gzip is never decompressed to
    disk, and the container is cut only at a closing tag boundary, so every yielded chunk
    is well-formed XML on its own once wrapped in a throwaway root element.
    """
    open_marker = b"<" + record_tag + b" "
    close_marker = b"</" + record_tag + b">"
    tail = b""
    with gzip.open(path, "rb") as stream:
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
    rest = tail.strip()
    if rest:
        yield rest


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
    """This chunk's contribution to the global `same_as` map: every release-level
    `extraartists` entry with a resolvable id, of ANY role -- `graph.same_as` carries no
    category filter, only `graph.credited_on`'s query does."""
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
    return partial


def _build_same_as_map(releases_path: Path, workers: int, limit_chunks: int) -> dict[str, frozenset[str]]:
    """Pass 1: the global `person_name -> {artist_id, ...}` map `credited_by_artist`
    resolves through, built once over the whole releases dump before pass 2 can compute
    any credited edge -- `graph.same_as` is additive and catalog-wide, not per-release."""
    print("🔗 pass 1: building the global same_as map (release-level extraartists, any role)...", file=sys.stderr)
    same_as: dict[str, set[str]] = {}
    started = time.time()
    ctx = mp.get_context("spawn")
    chunks = _chunks(releases_path, b"release")
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
    path: Path,
    record_tag: bytes,
    parse_chunk,
    workers: int,
    limit_chunks: int,
    *,
    initializer=None,
    initargs: tuple = (),
) -> tuple[dict[str, list[np.ndarray]], dict[str, list[np.ndarray]], list[str], int]:
    """Stream `path` through a worker pool; return per-relation (source, target) uint64
    array lists, the raw artist id strings seen, and the number of records parsed."""
    sources: dict[str, list[np.ndarray]] = {name: [] for name in ALL_RELATIONS}
    targets: dict[str, list[np.ndarray]] = {name: [] for name in ALL_RELATIONS}
    artist_ids: list[str] = []
    count = 0
    started = time.time()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=initializer, initargs=initargs) as pool:
        source_chunks = _chunks(path, record_tag)
        if limit_chunks:
            source_chunks = (c for _, c in zip(range(limit_chunks), source_chunks, strict=False))
        for result in pool.imap(parse_chunk, source_chunks, chunksize=1):
            count += result.count
            artist_ids.extend(result.artist_ids)
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


def build_graph(releases_path: Path, masters_path: Path, workers: int, limit_chunks: int = 0) -> dict:
    """Parse both dumps, build the Adjacency, and return everything the parity report
    and `fastrp` need. Raises nothing on a normal run; malformed chunks are skipped and
    logged, never fatal."""
    # Memory checkpoints below are named phase BOUNDARIES of `resource.getrusage`'s
    # monotonic, process-lifetime high-water mark, not isolated per-phase costs -- ru_maxrss
    # never decreases, even after `del` + `gc.collect()` frees real memory back to the
    # allocator. The explicit frees between phases are still worth doing: they let the
    # *next* phase's allocations reuse that freed memory instead of growing the peak
    # further, which is what makes the checkpoint *sequence* an honest (if not perfectly
    # isolated) attribution of where this script's memory actually goes -- see the "Memory
    # at catalog scale" finding in docs/recall_and_churn.md this instrumentation feeds.
    parse_started = time.perf_counter()
    same_as = _build_same_as_map(releases_path, workers, limit_chunks)
    same_as_peak_rss = peak_rss_bytes()

    print(f"📖 parsing releases: {releases_path}", file=sys.stderr)
    r_sources, r_targets, r_artist_ids, release_count = _run_pool(
        releases_path, b"release", _parse_release_chunk, workers, limit_chunks, initializer=_pass2_worker_init, initargs=(same_as,)
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
        "r": _distinct_count(*(relation_arrays[n][0] for n in RELEASE_RELATIONS)),
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


def print_parity_report(graph: dict) -> None:
    print("\n=== parity report ===", file=sys.stderr)
    print(f"releases parsed:  {graph['release_count']:,}", file=sys.stderr)
    print(f"masters parsed:   {graph['master_count']:,}", file=sys.stderr)
    print(f"total vertices:   {len(graph['nodes']):,}", file=sys.stderr)
    for kind, label in (("a", "artists"), ("r", "releases"), ("l", "labels"), ("m", "masters"), ("g", "genres"), ("s", "styles")):
        print(f"distinct {label:9s} (kind={kind}): {graph['distinct_by_kind'][kind]:,}", file=sys.stderr)
    for name in ALL_RELATIONS:
        print(f"edges {name:20s}: {graph['relation_counts'][name]:,}", file=sys.stderr)
    print(f"peak RSS after same_as map (pass 1):        {graph['same_as_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after full parse (pass 2, both dumps): {graph['parse_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after freeing parser structures, pre-build: {graph['pre_build_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"peak RSS after AdjacencyBuilder.build():     {graph['build_peak_rss_gb']:.2f} GB", file=sys.stderr)
    print(f"parse: {graph['parse_elapsed_s']:.1f}s, build: {graph['build_elapsed_s']:.1f}s", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("releases", type=Path)
    parser.add_argument("masters", type=Path)
    parser.add_argument("--dump-id", required=True)
    parser.add_argument("--dump-date", required=True)
    parser.add_argument("--out", type=Path, required=True)
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
    args = parser.parse_args()

    graph = build_graph(args.releases, args.masters, args.workers, args.limit_chunks)
    print_parity_report(graph)

    if args.delete_dumps_after_parse:
        for dump_path in (args.releases, args.masters):
            size_gb = dump_path.stat().st_size / 1e9
            dump_path.unlink()
            print(f"🗑️  deleted {dump_path} ({size_gb:.2f} GB) -- parse is done, fastrp needs no further disk read of it", file=sys.stderr)

    if args.parity_only:
        print("🛑 --parity-only: stopping before fastrp", file=sys.stderr)
        return

    artist_key_to_id = graph["artist_key_to_id"]
    artist_ids = list(artist_key_to_id.values())
    artist_keys = np.fromiter(artist_key_to_id.keys(), dtype=np.uint64)
    positions = graph["nodes"].positions(artist_keys)

    config = FastRPConfig()
    print(f"\n🔢 running fastrp: {config.model_version}", file=sys.stderr)
    print(f"numpy {np.__version__}, scipy {scipy.__version__}", file=sys.stderr)
    fastrp_started = time.perf_counter()
    vectors = fastrp(graph["adjacency"], config, rows=positions, out_dtype=np.float16, block_columns=args.block_columns, threads=args.threads)
    fastrp_elapsed = time.perf_counter() - fastrp_started
    fastrp_peak_rss = peak_rss_bytes()
    print(f"fastrp: {fastrp_elapsed:.1f}s, peak RSS {fastrp_peak_rss / 1e9:.2f} GB, {vectors.shape[0]:,} vectors", file=sys.stderr)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out,
        artist_ids=np.asarray(artist_ids, dtype=object),
        vectors=vectors,
        model_version=config.model_version,
        dump_id=args.dump_id,
        dump_date=args.dump_date,
        numpy_version=np.__version__,
        scipy_version=scipy.__version__,
    )
    summary = {
        "dump_id": args.dump_id,
        "dump_date": args.dump_date,
        "method_version": config.model_version,
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
        "out": str(args.out),
    }
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
