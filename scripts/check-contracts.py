"""Verify the promoted catalog API contract and generated binding."""

import ast
import importlib
import json
import tomllib
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

# The production release adapter contains only producer helpers, never its initializer.
# Compare their ASTs to the dev-only producer package; any future semantic drift fails.


schema_contract = json.loads((ROOT / "contracts/database-schema/artist-similarity/v1/source.json").read_text())
assert schema_contract["producer_repository"] == "https://github.com/groovemap-music/database-schema"
project = tomllib.loads((ROOT / "pyproject.toml").read_text())
assert project["tool"]["uv"]["sources"]["groovemap-database-schema"]["rev"] == schema_contract["producer_commit"]
assert digest(ROOT / schema_contract["binding"]) == schema_contract["binding_sha256"]
producer_module = importlib.import_module("groovemap_schema.postgres")
producer_path = Path(producer_module.__file__)
assert digest(producer_path) == schema_contract["producer_source_sha256"]


def contract_symbols(path: Path) -> dict[str, str]:
    symbols = {}
    for node in ast.parse(path.read_text()).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            symbols[node.name] = ast.dump(node)
        elif isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
            symbols[node.targets[0].id] = ast.dump(node)
    return symbols


promoted_symbols = contract_symbols(ROOT / schema_contract["binding"])
producer_symbols = contract_symbols(producer_path)
for symbol in schema_contract["symbols"]:
    assert promoted_symbols[symbol] == producer_symbols[symbol], f"release helper drift: {symbol}"
