#!/usr/bin/env python3
"""
Utility script for preparing and cleaning ACE example storage artifacts.

This script helps create (setup) or remove (teardown) the SQLite databases and
FAISS index files used by the example programs. It defaults to the artifacts
used in ``examples/simple_test.py`` and ``examples/weather_agent.py`` but can
operate on custom paths as well.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List

from ace.storage.faiss_index import FAISSVectorIndex
from ace.storage.sqlite_storage import SQLiteBulletStorage

DEFAULT_DATABASES: List[str] = [
    "examples/test_ace.db",
    "examples/weather_agent.db",
]

DEFAULT_FAISS: List[str] = [
    "examples/test_ace.faiss",
    "examples/weather_agent.faiss",
]


def _configure_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(message)s")


def _resolve_targets(
    specified: Iterable[str] | None,
    fallback: List[str],
    use_defaults: bool,
) -> List[Path]:
    if specified:
        return [Path(item) for item in specified]
    if use_defaults:
        return [Path(item) for item in fallback]
    return []


def _setup_databases(paths: Iterable[Path], overwrite: bool) -> None:
    for db_path in paths:
        if db_path.exists():
            if overwrite:
                logging.debug("Removing existing database %s", db_path)
                db_path.unlink()
            else:
                logging.info("SQLite database already exists: %s (skipping)", db_path)
                continue

        db_path.parent.mkdir(parents=True, exist_ok=True)
        storage = SQLiteBulletStorage(str(db_path))
        storage.close()
        logging.info("Initialized SQLite database: %s", db_path)


def _setup_faiss_indices(paths: Iterable[Path], dimension: int, overwrite: bool) -> None:
    for index_path in paths:
        if index_path.exists():
            if overwrite:
                logging.debug("Removing existing FAISS index %s", index_path)
                index_path.unlink()
                meta_path = Path(str(index_path) + ".meta")
                if meta_path.exists():
                    meta_path.unlink()
            else:
                logging.info("FAISS index already exists: %s (skipping)", index_path)
                continue

        index_path.parent.mkdir(parents=True, exist_ok=True)
        index = FAISSVectorIndex(dimension=dimension)
        index.save(str(index_path))
        logging.info(
            "Initialized FAISS index: %s (dimension=%s)", index_path, dimension
        )


def _teardown_paths(paths: Iterable[Path]) -> None:
    for path in paths:
        if not path.exists():
            logging.info("Nothing to remove at %s", path)
            continue

        path.unlink()
        logging.info("Removed file: %s", path)


def _teardown_resources(databases: Iterable[Path], faiss_indices: Iterable[Path]) -> None:
    _teardown_paths(databases)

    for index_path in faiss_indices:
        if index_path.exists():
            index_path.unlink()
            logging.info("Removed FAISS index: %s", index_path)
        else:
            logging.info("FAISS index not found (skipping): %s", index_path)

        meta_path = Path(str(index_path) + ".meta")
        if meta_path.exists():
            meta_path.unlink()
            logging.debug("Removed FAISS metadata: %s", meta_path)


def handle_setup(args: argparse.Namespace) -> None:
    databases = _resolve_targets(args.db, DEFAULT_DATABASES, not args.no_defaults)
    faiss_indices = _resolve_targets(args.faiss, DEFAULT_FAISS, not args.no_defaults)

    if not databases and not faiss_indices:
        logging.info("No resources specified for setup. Nothing to do.")
        return

    _setup_databases(databases, overwrite=args.overwrite)
    _setup_faiss_indices(faiss_indices, dimension=args.dimension, overwrite=args.overwrite)


def handle_teardown(args: argparse.Namespace) -> None:
    databases = _resolve_targets(args.db, DEFAULT_DATABASES, not args.no_defaults)
    faiss_indices = _resolve_targets(args.faiss, DEFAULT_FAISS, not args.no_defaults)

    if not databases and not faiss_indices:
        logging.info("No resources specified for teardown. Nothing to do.")
        return

    _teardown_resources(databases, faiss_indices)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Set up or tear down SQLite and FAISS artifacts for ACE."
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    setup_parser = subparsers.add_parser(
        "setup",
        help="Create SQLite databases and FAISS indices.",
    )
    setup_parser.add_argument(
        "--db",
        action="append",
        help="Path to a SQLite database file to initialize. "
             "Can be provided multiple times.",
    )
    setup_parser.add_argument(
        "--faiss",
        action="append",
        help="Path to a FAISS index file to initialize. "
             "Can be provided multiple times.",
    )
    setup_parser.add_argument(
        "--dimension",
        type=int,
        default=1536,
        help="Vector dimension to use for new FAISS indices (default: 1536).",
    )
    setup_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing files instead of skipping them.",
    )
    setup_parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not operate on the default example resources when no paths are provided.",
    )
    setup_parser.set_defaults(func=handle_setup)

    teardown_parser = subparsers.add_parser(
        "teardown",
        help="Remove SQLite databases and FAISS indices.",
    )
    teardown_parser.add_argument(
        "--db",
        action="append",
        help="Path to a SQLite database file to remove. "
             "Can be provided multiple times.",
    )
    teardown_parser.add_argument(
        "--faiss",
        action="append",
        help="Path to a FAISS index file to remove. "
             "Can be provided multiple times.",
    )
    teardown_parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Do not operate on the default example resources when no paths are provided.",
    )
    teardown_parser.set_defaults(func=handle_teardown)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    _configure_logging(args.verbose)

    handler = getattr(args, "func", None)
    if handler is None:
        parser.error("No action specified.")

    handler(args)


if __name__ == "__main__":
    main()
