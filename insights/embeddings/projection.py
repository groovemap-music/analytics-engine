"""The very sparse random projection, derived per node from its stable key.

The spike drew the projection matrix ``R`` from one global random stream, so a new
seed (or a single added vertex, which shifts every later draw) replaced about 72%
of each artist's top ten. ADR 0013 requires each node's row of ``R`` to come from
a hash of its provider identity instead. ``HashedProjection`` computes entry
``R[v, j]`` as a pure function of ``(node_key(v), j, seed)``:

    u = splitmix64(node_key(v) XOR splitmix64(seed + (j + 1) * GOLDEN_GAMMA))

and maps ``u`` onto the Achlioptas distribution with sparsity ``s = 3``, as the
spike did: ``-sqrt(3)`` when ``u < 2^64 / 6``, ``+sqrt(3)`` when ``u`` is in the top
sixth, and ``0`` otherwise. The thresholds are integers, so no floating-point
rounding enters the draw. Columns are produced independently, which lets FastRP
generate exactly the column block it is working on and never hold all of ``R``.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Final, Protocol

import numpy as np


if TYPE_CHECKING:
    from numpy.typing import NDArray


_MASK64: Final = (1 << 64) - 1
GOLDEN_GAMMA: Final = 0x9E3779B97F4A7C15
_MIX1: Final = 0xBF58476D1CE4E5B9
_MIX2: Final = 0x94D049BB133111EB
# P(u < LOW) = P(u >= HIGH) = 1/6 to within 2^-64.
_LOW: Final = np.uint64((1 << 64) // 6)
_HIGH: Final = np.uint64((1 << 64) - (1 << 64) // 6)
_SQRT3: Final = np.float32(math.sqrt(3.0))
_ROW_SLICE: Final = 1 << 16
_ALL: Final = slice(None)


class Projection(Protocol):
    """Columns ``[start, stop)`` of the ``n_nodes x dim`` projection matrix, ``float32``,
    restricted to the node positions in ``rows``."""

    def columns(self, start: int, stop: int, rows: slice = ...) -> NDArray[np.float32]: ...


def splitmix64(x: int) -> int:
    """Scalar SplitMix64 finalizer (Steele, Lea & Flood 2014), for column salts."""
    z = (x + GOLDEN_GAMMA) & _MASK64
    z = ((z ^ (z >> 30)) * _MIX1) & _MASK64
    z = ((z ^ (z >> 27)) * _MIX2) & _MASK64
    return z ^ (z >> 31)


def splitmix64_array(x: NDArray[np.uint64]) -> NDArray[np.uint64]:
    """Vectorized SplitMix64; ``uint64`` array arithmetic wraps modulo 2^64."""
    z = x + np.uint64(GOLDEN_GAMMA)
    z ^= z >> np.uint64(30)
    z *= np.uint64(_MIX1)
    z ^= z >> np.uint64(27)
    z *= np.uint64(_MIX2)
    z ^= z >> np.uint64(31)
    return z


def column_salt(seed: int, column: int) -> int:
    return splitmix64((seed + (column + 1) * GOLDEN_GAMMA) & _MASK64)


class HashedProjection:
    """``R[v, j]`` from ``(node_key(v), j, seed)``, optionally scaled by ``degree ** beta``."""

    def __init__(self, keys: NDArray[np.uint64], seed: int, *, degree: NDArray[np.int64] | None = None, beta: float = 0.0) -> None:
        if not 0 <= seed <= _MASK64:
            raise ValueError("seed must fit in an unsigned 64-bit integer")
        self._keys = np.asarray(keys, dtype=np.uint64)
        self._seed = seed
        self._scale: NDArray[np.float32] | None = None
        if beta != 0.0:
            if degree is None:
                raise ValueError("beta != 0 needs node degrees")
            self._scale = (np.maximum(degree, 1).astype(np.float64) ** beta).astype(np.float32)

    def columns(self, start: int, stop: int, rows: slice = _ALL) -> NDArray[np.float32]:
        keys = self._keys[rows]
        scale = None if self._scale is None else self._scale[rows]
        block = np.empty((keys.size, stop - start), dtype=np.float32)
        for offset, column in enumerate(range(start, stop)):
            salt = np.uint64(column_salt(self._seed, column))
            # Row slices keep the uint64 temporaries small at catalog scale.
            for lo in range(0, keys.size, _ROW_SLICE):
                hi = min(lo + _ROW_SLICE, keys.size)
                u = splitmix64_array(keys[lo:hi] ^ salt)
                values = np.zeros(hi - lo, dtype=np.float32)
                values[u < _LOW] = -_SQRT3
                values[u >= _HIGH] = _SQRT3
                if scale is not None:
                    values *= scale[lo:hi]
                block[lo:hi, offset] = values
        return block
