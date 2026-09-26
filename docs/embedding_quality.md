# Shipped FastRP embedding quality on the chw.2 proxy benchmark

Re-runs the `gm-design-chw.2` design-spike proxy benchmark (design repo:
`docs/spikes/gm-design-chw.2/`) against the FastRP embeddings actually shipped by
`gm-analytics-engine-ieu.3` (`sept.npz`, `model_version =
fastrp-v1:dim=128:weights=0,1,1,1,1:beta=0:proj=achlioptas-s3:rows=splitmix64(blake2b64(kind,key)):seed=20260924`,
`dump_id = discogs_20260901`), instead of the spike's own from-scratch FastRP training run.
The question: does the FastRP-vs-heuristic similar-artist recall case behind ADR 0013
(gm-design-3eo) still hold for the embedding that is actually running in production, given
that the shipped extraction graph differs from the spike's in several ways (below)?

All numbers are recall@10 on the benchmark's held-out test split (macro-averaged over query
artists, "all" view — known + novel collaborators). No artist ids, names, vectors, or edges
are recorded anywhere in this repo; only these aggregates.

## Method

1. Streamed the same `discogs_20260901_releases.xml.gz` dump the spike used (curl into
   `parse_dump.py` via a FIFO -- never written to disk as a whole file) and rebuilt the
   spike's 10% artist-seeded subset with `build_subset.py --rate 0.10` (no `--expand`), from
   a copy of the harness run outside any repo. The rebuilt `subset.json` is byte-identical to
   the spike's committed one (same `eligible_artists`, `seed_artists`, `subset_releases`,
   etc.), confirming the subset reproduces deterministically.
2. Looked up each subset artist's Discogs id in `sept.npz` and built a local-index-aligned
   embedding matrix for the harness's `evaluate.py --emb`. Artists without a shipped vector
   get an all-zero row (contributes ~0 cosine similarity everywhere).
3. Ran `evaluate.py` unmodified except for two additions: (a) also scoring the fusion weight
   fixed at the spike's headline alpha (0.9) on test, in addition to whatever alpha the
   harness selects on dev for this embedding, and (b) dumping per-query test recall@10 (by
   local array index only) to slice by duplicate-vector-group membership afterward. The
   heuristic/production-path numbers below come from this same run -- they don't depend on
   which embedding is supplied, so they double as the reproduction check.

## Coverage

| | |
|---|---|
| Subset artists | 1,092,879 |
| ... with a shipped vector | 967,504 (**88.5%**) |
| ... without a shipped vector | 125,375 (11.5%) |
| Shipped file: byte-duplicate vectors | 38.7% (matches the known FastRP-weights-`0,1,1,1,1` artifact) |

Coverage among the benchmark's **test queries** specifically was 100% (4,721 / 4,721) --
query artists are seed artists with >= 3 pre-cut releases, i.e. active enough that they
almost always land in the shipped graph. The 11.5% gap concentrates in lower-activity
artists that this benchmark never queries (see "What this benchmark can't measure" below).

## Reproduction sanity check

The freshly-rebuilt subset reproduces the spike's baseline numbers exactly:

| | this run | spike (committed) |
|---|---|---|
| Production heuristic path | 0.91641% | 0.9164% |
| All-artist heuristic (dense, same weights) | 17.9883% | 17.99% |

## Headline results (recall@10, test split)

| Model | Recall@10 | vs. all-artist heuristic |
|---|---|---|
| Production path (candidate gen + weighted cosine) | 0.9164% | -94.91% |
| All-artist heuristic (dense weighted cosine) | 17.988% | -- |
| **Shipped FastRP alone** | **24.301%** | +35.09% [+28.84%, +41.08%] |
| Shipped FastRP fused @ alpha=0.8 (dev-selected) | 25.329% | +40.81% [+34.97%, +46.67%] |
| Shipped FastRP fused @ alpha=0.9 (spike's alpha) | 25.004% | +39.00% [+32.79%, +45.42%] |
| *Spike's own fused FastRP (frp128_w01111) @ alpha=0.9* | *31.727%* | *+76.37% [+69.37%, +83.63%]* |

Relative gains and 95% CIs are the harness's paired bootstrap vs. the all-artist heuristic
(1000 resamples), same methodology the spike used.

**Gap vs. the spike's headline number:** the shipped embedding's best fused result (25.33%,
alpha=0.8) is **6.40 points** below the spike's fused 31.73% at alpha=0.9; at the spike's own
alpha=0.9 the gap is 6.72 points. The dev-selected alpha for the shipped embedding (0.8) is
also lower than the spike's (0.9), consistent with the shipped signal being weaker relative
to the heuristic than the spike's from-scratch embedding was.

### Per-family breakdown (test split, recall@10)

| Family | Queries | Shipped FastRP | Production heuristic | All-artist heuristic |
|---|---:|---:|---:|---:|
| digital | 1,223 | 27.36% | 0.31% | 23.72% |
| optical | 1,781 | 20.39% | 1.41% | 14.20% |
| tape | 158 | 34.28% | 0.74% | 22.90% |
| vinyl | 1,545 | 25.30% | 0.86% | 17.22% |
| shellac | 9 | 34.44% | 0.00% | 34.35% |
| grooved_other | 4 | 25.00% | 0.00% | 25.00% |
| video | 1 | 33.33% | 0.00% | 0.00% |

(shellac/grooved_other/video have single-digit query counts and are noise, kept only for
completeness -- same caveat the spike itself carried.) The shipped embedding beats both
heuristics in every family with a meaningful query count, same qualitative pattern as the
spike.

## Likely causes of the ~6.4-point gap

The shipped extraction graph differs from the spike's in several known ways, any of which
could plausibly account for the gap (this evaluation cannot isolate which, since the shipped
embedding was built once, not ablated):

- **No track-level edges at all.** The spike's own ablation on this same benchmark found
  that removing credit edges alone (`frp128_w0111` -> `frp128_w0111_nocredit`, both without
  track-only variation) cost about 10 points of recall@10 (30.50% -> 20.61%). The shipped
  graph drops track-artist edges entirely (not just credits), which the spike never tested in
  isolation -- so this is plausibly the single largest contributor, but not confirmed.
- **Whole-string role filter** on production's credit extraction (~1.7% fewer credit edges
  than the spike's per-token role filter).
- **Release-level credits via a `same_as` name join** rather than the spike's direct id-based
  credit edges -- a potential source of both missed and spurious credit edges.
- **9 relation types** in the shipped graph vs. the spike's smaller, more targeted edge set.
- **11.5% of subset artists have no shipped vector at all** (zero-vector fallback), diluting
  the "all artists" comparison for the dense heuristic baseline (which does score every
  artist) relative to FastRP (which effectively can't distinguish among unvectorized
  candidates).

## Duplicate-vector-group effect

FastRP with iteration weights `0,1,1,1,1` gives zero weight to a node's own projection, so
nodes with identical post-hop-0 neighbourhoods land on bit-identical vectors -- 38.7% of the
shipped file's ~6.9M vectors sit in such a group (largest observed group: 816 members).

| | Queries | Recall@10 | 95% CI |
|---|---:|---:|---:|
| All test queries | 4,721 | 24.30% | [23.27%, 25.33%] |
| Query vector unique | 4,716 | 24.27% | [23.22%, 25.30%] |
| Query vector in a duplicate group | 5 | 51.11% | [17.78%, 86.67%] |

**This benchmark cannot measure the duplicate-vector effect.** Only 5 of 4,721 test queries
(0.1%) have a shipped vector that falls in a duplicate group, because the benchmark's queries
are seed artists with >= 3 pre-cut releases -- i.e. active enough to almost always have a
distinguishing neighbourhood. The duplicate-vector long tail is concentrated among
low-activity artists this benchmark never queries (consistent with the 11.5% coverage gap
above also concentrating there). The 51% figure for the 5 in-group queries is not
statistically meaningful (CI spans from below the unique-group rate to near-perfect) and
should not be read as "duplicates recommend better" -- it is noise from an n of 5. Whether
duplicate vectors hurt similar-artist quality for the artists that actually have them (the
long tail) is a question this proxy benchmark's active-artist query set structurally cannot
answer; a targeted benchmark would need to sample queries from within duplicate groups
directly.

## Known differences: shipped graph vs. spike's graph

Recorded here as adopted context for anyone revisiting ADR 0013 or D (gm-catalog-api-2zsq):

- No track-level edges (spike measured ~10 points of recall@10 lost from credit-edge removal
  alone on a comparable config; track removal was never isolated).
- Production's whole-string role filter vs. the spike's per-token filter (~1.7% fewer credit
  edges).
- Release-level credits resolved via a `same_as` name join rather than direct ids.
- 9 relation types in production vs. the spike's narrower edge set.
- FastRP iteration weights `0,1,1,1,1` (matches the spike's best config, `frp128_w01111`) but
  trained on the above different graph, and only once (no ablations, no seed-stability check
  on the shipped run itself).

## Reproducibility

- Design-repo dump: `discogs_20260901_releases.xml.gz` (streamed, never persisted).
- Subset: `build_subset.py --rate 0.10`, no `--expand` -- `subset.json` byte-identical to the
  spike's committed one.
- Shipped embeddings: `~/.cache/groovemap-spikes/embeddings-scratch/sept.npz` (from
  gm-analytics-engine-ieu.3), matched to the subset by exact Discogs artist id.
- Harness: a copy of `design/docs/spikes/gm-design-chw.2/*.py`, run outside any repo, with a
  minimal `--also-alpha` / `--dump-per-query` addition to `evaluate.py` for the two extra
  slices above; no change to its recall/precision/bootstrap logic.
