"""Listings inside archived observations, read by their shape, and their rows served verbatim.

A search that wants every row of a listing is answered by copying the rows, not
by asking a model to reproduce them: a model declines long tables, runs into its
completion limit on identifier-heavy rows, and paraphrases what it does return.

Three shapes are recognised, because workflows render listings differently:

* aligned  -- a column line of two or more names separated by runs of 2+ spaces,
              then rows split the same way;
* markdown -- ``| a | b |``, a ``|---|---|`` separator, then ``| .. | .. |`` rows;
* tabbed   -- a tab-separated column line, then rows with the same tab count.

A parse is only returned when it is PROVABLY the whole listing: the rows run to
the end of the text, a blank line, or a line that is plainly not a row. A line
that is row-shaped but does not fit the columns (an aligned row with more cells
than the header, a markdown or tabbed row with a different count) makes the
parse ambiguous, and an ambiguous parse is refused rather than truncated -- a
served listing that silently stopped early would claim to be complete. When the
text above the columns states a shown count (``shown=60``), the parse must match
it too.
"""
from __future__ import annotations

import re
from typing import Any, Optional

ROWS_SERVED_MARK = "[search_memory ROWS:"

_ALIGNED_SPLIT = re.compile(r" {2,}")
_COLUMN_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_ ./()-]{0,39}$")
_MARKDOWN_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_SHOWN = re.compile(r"\bshown\s*[=:]\s*(\d+)", re.IGNORECASE)


def _aligned_cells(line: str) -> list[str]:
    return _ALIGNED_SPLIT.split(line.strip())


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_column_line(cells: list[str]) -> bool:
    return len(cells) >= 2 and all(_COLUMN_NAME.match(cell) for cell in cells)


def _aligned_at(lines: list[str], index: int) -> Optional[tuple[int, list[str]]]:
    """``(end, rows)`` for an aligned listing whose column line is ``lines[index]``."""
    if "|" in lines[index] or "\t" in lines[index]:
        return None
    header = _aligned_cells(lines[index])
    if not _is_column_line(header):
        return None
    rows = []
    for line in lines[index + 1:]:
        if not line.strip():
            break
        cells = _aligned_cells(line)
        if len(cells) < 2:
            break
        if len(cells) > len(header):
            return None
        rows.append(line)
    return (index + 1 + len(rows), rows) if rows else None


def _markdown_at(lines: list[str], index: int) -> Optional[tuple[int, list[str]]]:
    line = lines[index].strip()
    if not (line.startswith("|") and index + 1 < len(lines)
            and _MARKDOWN_SEPARATOR.match(lines[index + 1].strip())):
        return None
    width = len(_markdown_cells(line))
    rows = []
    for row in lines[index + 2:]:
        if not row.strip().startswith("|"):
            break
        if len(_markdown_cells(row)) != width:
            return None
        rows.append(row)
    return (index + 2 + len(rows), rows) if rows and width >= 2 else None


def _tabbed_at(lines: list[str], index: int) -> Optional[tuple[int, list[str]]]:
    header = lines[index].split("\t")
    if len(header) < 2 or not _is_column_line([c.strip() for c in header]):
        return None
    rows = []
    for row in lines[index + 1:]:
        if "\t" not in row:
            break
        if len(row.split("\t")) != len(header):
            return None
        rows.append(row)
    return (index + 1 + len(rows), rows) if rows else None


_SHAPES = (("markdown", _markdown_at), ("tabbed", _tabbed_at), ("aligned", _aligned_at))


def parse_table(text: str) -> Optional[dict[str, Any]]:
    """``{"shape", "preamble", "columns", "rows"}`` for the first whole listing, or None."""
    lines = text.splitlines()
    for index in range(len(lines)):
        for shape, parse in _SHAPES:
            found = parse(lines, index)
            if found is None:
                continue
            end, rows = found
            columns = "\n".join(lines[index:index + (2 if shape == "markdown" else 1)])
            shown = _SHOWN.search("\n".join(lines[:index]))
            if shown is not None and int(shown.group(1)) != len(rows):
                return None
            return {"shape": shape, "preamble": lines[:index], "columns": columns,
                    "rows": rows, "end_line": end}
    return None


def served_rows(alias: str, table: dict[str, Any], max_bytes: int) -> tuple[str, int, int]:
    """The listing's own rows, copied, as many whole rows as fit *max_bytes*.

    ``(text, shown, total)``. Rows are never paraphrased or reordered, and the
    closing line says how many were shown, so a cut listing cannot read as the
    whole one.
    """
    head = "\n".join([*table["preamble"], table["columns"]])
    total = len(table["rows"])

    def closing(shown: int) -> str:
        rest = "" if shown == total else f"; rows {shown + 1}-{total} are NOT shown here"
        return (f"{ROWS_SERVED_MARK} rows 1-{shown} of the {total} rows in {alias}{rest}. "
                f"Every row of {alias} is restored in full when the final answer is written.]")

    # Reserve the longer of the two closings: the cut form names the rows left out.
    used = len(head.encode("utf-8")) + 2 + max(
        len(closing(0).encode("utf-8")), len(closing(total).encode("utf-8")))
    shown: list[str] = []
    for row in table["rows"]:
        size = len(row.encode("utf-8")) + 1
        if used + size > max_bytes:
            break
        shown.append(row)
        used += size
    return "\n".join([head, *shown, closing(len(shown))]), len(shown), total
