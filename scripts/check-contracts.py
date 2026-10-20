"""Verify the promoted catalog API contract and generated binding."""

import json
from hashlib import sha256
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONTRACT_ROOT = ROOT / "contracts/catalog-api/internal-insights/v1"


def digest(path: Path) -> str:
    """Return a file's hexadecimal SHA-256 digest."""
    return sha256(path.read_bytes()).hexdigest()


source = json.loads((CONTRACT_ROOT / "source.json").read_text())
assert source["producer_repository"] == "https://github.com/groovemap-music/catalog-api"
assert len(source["producer_commit"]) == 40
assert source["version"] == "1.1.0"
assert digest(CONTRACT_ROOT / "openapi.yaml") == source["contract_sha256"]
assert digest(ROOT / source["binding"]) == source["binding_sha256"]

binding = (ROOT / source["binding"]).read_text()
assert 'CONTRACT_VERSION = "1.1.0"' in binding
assert "COMMUNITY_ENRICHMENT_MAX_PROCESSING_SECONDS = 1500" in binding
