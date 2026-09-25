"""Deterministic FastRP graph embeddings over the catalog graph (ADR 0013)."""

from insights.embeddings.fastrp import FASTRP_ALGORITHM_VERSION, FastRPConfig, estimate_peak_bytes, fastrp
from insights.embeddings.graph import Adjacency, AdjacencyBuilder, NodeIndex, node_key, node_keys
from insights.embeddings.projection import HashedProjection, Projection


__all__ = [
    "FASTRP_ALGORITHM_VERSION",
    "Adjacency",
    "AdjacencyBuilder",
    "FastRPConfig",
    "HashedProjection",
    "NodeIndex",
    "Projection",
    "estimate_peak_bytes",
    "fastrp",
    "node_key",
    "node_keys",
]
