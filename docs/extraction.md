# Extraction provenance

This repository was extracted without modifying the source monorepo. Its `main` history contains 103 commits relevant to the analytics service before the standalone-establishment commit.

Source: `SimplicityGuy/discogsography`, bead branch `wt/bead/issue/discogsography-2kpm.19`.

The reproducible extraction used an isolated clone and `git-filter-repo`:

```bash
git clone --no-local --branch wt/bead/issue/discogsography-2kpm.19 \
  /Users/Robert/workspaces/github/SimplicityGuy/discogsography analytics-engine
cd analytics-engine
git filter-repo --force \
  --path insights/ \
  --path tests/insights/ \
  --path LICENSE \
  --path docs/database-resilience.md \
  --path docs/neo4j-indexing.md \
  --path docs/performance-guide.md \
  --path docs/postgres-pool-exhaustion-analysis.md \
  --path docs/query-performance-optimizations.md \
  --path docs/superpowers/plans/2026-03-25-release-rarity-scoring-phase1.md \
  --path docs/superpowers/specs/2026-03-25-release-rarity-scoring-phase1-design.md \
  --path docs/superpowers/plans/2026-04-11-community-enrichment-rarity.md \
  --path docs/superpowers/specs/2026-04-11-community-enrichment-rarity-design.md \
  --path docs/superpowers/plans/2026-05-21-neo4j-bolt-tls.md \
  --path docs/superpowers/specs/2026-05-21-neo4j-bolt-tls-design.md \
  --path-rename tests/insights/:tests/
```

No source tags were copied because the monorepo tags do not unambiguously version this service. API query implementation tests were assigned to `catalog-api`, while service-owned tests remain here. The source PolyForm Noncommercial 1.0.0 license was retained.
