"""Shared DuckDB helpers for memory-bounded entity-resolution jobs."""
from __future__ import annotations

from pathlib import Path

import duckdb


def _sql_literal(value: str) -> str:
    """Return a single-quoted DuckDB string literal."""
    return "'" + value.replace("'", "''") + "'"


def connect(memory_limit: str = "2GB", threads: int = 4) -> duckdb.DuckDBPyConnection:
    """Create a DuckDB connection with bounded memory and thread usage."""
    if threads < 1:
        raise ValueError("threads must be at least 1")

    con = duckdb.connect()
    con.execute(f"SET memory_limit={_sql_literal(memory_limit)}")
    con.execute(f"SET threads={int(threads)}")
    return con


def read_tsv_sql(path: Path, columns: list[str] | None = None) -> str:
    """Build a DuckDB read_csv expression for an untouched official TSV."""
    source = _sql_literal(str(Path(path).resolve()).replace("\\", "/"))
    expression = (
        f"read_csv({source}, delim='\\t', quote='', header=true, "
        "all_varchar=true, null_padding=true)"
    )
    if not columns:
        return expression

    selected = ", ".join('"' + name.replace('"', '""') + '"' for name in columns)
    return f"(SELECT {selected} FROM {expression})"
