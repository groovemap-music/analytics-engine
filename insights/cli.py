"""``analytics-engine``: the image's default entrypoint, a small subcommand dispatcher.

Schedulers that can only set arguments (Cronduit ``cmd``, Kubernetes ``args``, Ofelia) cannot
replace the image ENTRYPOINT, so the jobs are reachable as subcommands of it::

    analytics-engine                                # start the insights server (unchanged)
    analytics-engine similar-artists [args...]      # same as analytics-engine-similar-artists
    analytics-engine embeddings                     # same as analytics-engine-embeddings

Heavy imports (uvicorn, numpy, the database stack) happen inside the chosen branch, so usage
and ``--help`` stay cheap and never start the server.
"""

import sys
from typing import NoReturn

USAGE = """\
usage: analytics-engine [similar-artists|embeddings] [args...]

  (no arguments)          start the insights server
  similar-artists [args]  run the exact similar-artist job (see analytics-engine-similar-artists --help)
  embeddings              run one monthly embedding load
"""

EXIT_USAGE = 2


def _usage(*, error: str | None = None) -> NoReturn:
    if error:
        sys.stderr.write(f"analytics-engine: {error}\n{USAGE}")
        sys.exit(EXIT_USAGE)
    sys.stdout.write(USAGE)
    sys.exit(0)


def main(argv: list[str] | None = None) -> None:
    """Dispatch on the first argument; with none, start the insights server."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        from insights.insights import main as serve  # noqa: PLC0415

        serve()
        return
    command, rest = args[0], args[1:]
    if command in {"-h", "--help"}:
        _usage()
    if command == "similar-artists":
        from insights.similar_artists_job import main as similar_artists  # noqa: PLC0415

        similar_artists(rest)  # NoReturn: prints the status line and exits with its code
    if command == "embeddings":
        if rest:
            _usage(error="embeddings takes no arguments (it is configured through the environment)")
        from insights.embedding_pipeline import main as embeddings  # noqa: PLC0415

        embeddings()
        return
    _usage(error=f"unknown subcommand {command!r}")
