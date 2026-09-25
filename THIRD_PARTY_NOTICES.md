# Third-party dependency notices

The GrooveMap analytics engine depends on third-party software. The upstream license terms control; this file records the distribution boundary and the compliance work that must accompany releases. `uv.lock` is the authoritative version lock, and `scripts/release-dry-run.sh` writes a complete machine-readable `THIRD_PARTY_NOTICES.json` from an isolated installation of the runtime wheel.

The analytics-engine wheel declares dependencies but does not bundle their code. The OCI image installs the locked runtime dependencies and retains the license files supplied in each package's `.dist-info` directory.

## Reciprocal-license runtime dependencies

- `certifi` 2026.7.22 is licensed under `MPL-2.0` and is included through HTTPX. Source: <https://github.com/certifi/python-certifi>.
- `orjson` 3.12.0 declares `MPL-2.0 AND (Apache-2.0 OR MIT)` and is included through `groovemap-runtime`. Source: <https://github.com/ijl/orjson>.
- `psycopg` 3.3.5 and `psycopg-binary` 3.3.5 are licensed under `LGPL-3.0-only` and are included through the PostgreSQL extra of `groovemap-runtime`. Source: <https://github.com/psycopg/psycopg>.

For MPL-covered files, preserve license and copyright notices and make the source form of any distributed modifications to those files available under MPL-2.0. For the LGPL-covered Psycopg libraries, preserve their notices and license texts, provide the applicable covered source, and do not prevent replacement or reverse engineering for debugging modifications. Reassess these obligations before modifying, statically combining, vendoring, or changing how any covered dependency is distributed.

## Numerical dependencies of the FastRP embedding pipeline

- `numpy` 2.5.3 declares `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`. Source: <https://github.com/numpy/numpy>.
- `scipy` 1.18.1 is licensed under `BSD-3-Clause`. Source: <https://github.com/scipy/scipy>.

Their binary wheels bundle compiled libraries whose terms are listed in each wheel's own license file: OpenBLAS and LAPACK under BSD terms, the GCC runtime libraries (`libgfortran`, `libgcc`) under `GPL-3.0-or-later WITH GCC-exception-3.1`, and `libquadmath` under `LGPL-2.1-or-later`. The GCC Runtime Library Exception lets programs that use those runtime libraries be distributed under their own terms; the `libquadmath` obligations match the LGPL obligations above. The analytics engine uses these libraries only through NumPy and SciPy and does not modify, vendor, or statically link them. They add no MPL-covered package: `certifi` and `orjson` above were already present, and neither NumPy nor SciPy pulls in `tqdm`.

## Build and test dependencies

The locked development environment also includes `chardet` 5.2.0 under `LGPL-2.1-or-later`, plus `fqdn` 1.5.1 and `pathspec` 1.1.1 under `MPL-2.0`. They are tools or transitive development dependencies and are not installed in the runtime image. Their distribution obligations must be reassessed if that boundary changes.

## Other licenses

The remaining locked dependencies primarily use permissive `MIT`, BSD, ISC, Python, or `Apache-2.0` terms. Preserve their copyright, attribution, license, and upstream NOTICE material as required. The release-generated JSON inventory records the exact package versions, declared licenses, upstream URLs, and available license texts for review before distribution.
