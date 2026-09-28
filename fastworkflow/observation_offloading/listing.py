"""Listings inside archived observations, read by their shape, and their rows served verbatim.

A search that wants every row of a listing is answered by copying the rows, not
by asking a model to reproduce them: a model declines long tables, runs into its
completion limit on identifier-heavy rows, and paraphrases what it does return.

Three shapes are recognised, because workflows render listings differently:

* aligned  -- a column line of two or more names separated by runs of 2+ spaces,
              then rows split the same way;
* markdown -- ``| a | b |``, a ``|---|---|`` separator, then ``| .. | .. |`` rows;
              also the compact ``|-|-|`` separator and the pipe-less
              ``a | b`` / ``--- | ---`` form, when the separator has exactly
              one cell per column (and, pipe-less, the column line reads as
              column names);
* tabbed   -- a tab-separated column line, then rows with the same tab count.

A parse is only returned when it is PROVABLY the whole listing: the rows run to
the end of the text, a blank line, or a line that is plainly not a row. A line
that is row-shaped but does not fit the columns (an aligned row with more cells
than the header, a markdown or tabbed row with a different count) makes the
parse ambiguous, and an ambiguous parse is refused rather than truncated -- a
served listing that silently stopped early would claim to be complete. So is a
one-cell line below aligned rows that reads as one of them (an empty trailing
cell, separators collapsed to one space, a wrapped label), and any row-shaped
line anywhere after the listing (a second group or table). When the text above
or below the rows states a shown count (``shown=60``), the parse must match it
too, and a text that says there is more -- ``remaining=57``, ``complete=false``,
``Showing 10 of 57``, ``Page 1 of 29``, ``re-run with page=2`` -- is refused.
(``parse_table(text, require_complete=False)`` skips these completeness checks
but not the structural ones; it reads labels and is never served.)

The first header candidate that accepts a row decides the parse: its listing is
returned or the whole text is refused. No later line is tried as a header, so a
header is never found inside another candidate's rows and a parse costs one
pass over the text however it ends.
"""
from __future__ import annotations

import re
from typing import Any, Optional

from fastworkflow.observation_offloading.labels import LABEL_RESTORE_MARK, quote_marker_lines

ROWS_SERVED_MARK = "[search_memory ROWS:"

_ALIGNED_SPLIT = re.compile(r" {2,}")
_ALIGNED_CELL = re.compile(r"\S+(?: \S+)*")
_COLUMN_NAME = re.compile(r"^[^\W\d][\w ./()-]{0,39}$")
_MARKDOWN_SEPARATOR = re.compile(r"^\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)*\|?\s*$")
_GFM_SEPARATOR = re.compile(r"^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
_SHOWN = re.compile(r"\bshown\s*[=:]\s*(\d+)", re.IGNORECASE)
# Generic ways a text says its listing is one page of a longer one. They are read
# only in the text around the rows, never in the rows themselves.
_REMAINING = re.compile(r"\bremaining\s*[=:]\s*(\d+)", re.IGNORECASE)
_TOTAL = re.compile(r"\btotal\s*[=:]\s*(\d+)", re.IGNORECASE)
_PAGES = re.compile(r"\b(?:pages|page_count|total_pages)\s*[=:]\s*(\d+)", re.IGNORECASE)
_NOT_COMPLETE = re.compile(
    r"\b(?:complete|is_complete)\s*[=:]\s*(?:false|no|0)\b"
    r"|\b(?:has_more|more|truncated)\s*[=:]\s*(?:true|yes|1)\b", re.IGNORECASE)
_MORE_PAGES = re.compile(
    r"\bpage\s*[=:]\s*(?:[2-9]|\d{2,})\b"
    r"|\bnext[ _-]?(?:page|cursor|token)\b|\bcursor\s*[=:]\s*\S"
    r"|(?<!no )\bmore\s+(?:rows|results|items|records|entries|lines)\b"
    r"|\btruncated\b|\bnot\s+(?:all\s+)?(?:rows\s+|results\s+)?(?:are\s+)?shown\b",
    re.IGNORECASE)
_N_OF_M = re.compile(r"\b(\d+)\s+of\s+(?:about\s+|approximately\s+)?(\d+)\b", re.IGNORECASE)
#: A footer written in the columns' own shape reads as a last row; one that
#: names a summary is not one of the listed rows.
_SUMMARY_ROW = re.compile(r"^\s*(?:grand\s+total|sub-?total|total|sum|count)\b", re.IGNORECASE)
#: The same for a footer that labels itself (``Note:  2 items``,
#: ``Legend: x  ...``): a last row whose first cell ends with a colon, or
#: starts with one word and a colon. Read by shape, not by a list of words.
_LABELLED_FOOTER_CELL = re.compile(r"^(?:.*:|[^\W\d_][\w-]*:\s.*)$")
_RANGE_OF_M = re.compile(r"\b(\d+)\s*(?:-|–|to)\s*(\d+)\s+of\s+(\d+)\b", re.IGNORECASE)

#: A header candidate that accepted rows and then met a row it cannot place.
#: The whole text is refused: a later header would sit inside this candidate's
#: rows, and retrying from every later line is what made refusals quadratic.
REFUSED = object()


def _aligned_cells(line: str) -> list[str]:
    return _ALIGNED_SPLIT.split(line.strip())


def _markdown_cells(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def _is_column_line(cells: list[str]) -> bool:
    return len(cells) >= 2 and all(_COLUMN_NAME.match(cell) for cell in cells)


def _cell_starts(line: str) -> list[int]:
    return [match.start() for match in _ALIGNED_CELL.finditer(line)]


def _indent(line: str) -> int:
    return len(line) - len(line.lstrip())


def _cut_row(line: str, header_line: str, rows: list[str]) -> bool:
    """Whether a one-cell *line* directly below aligned *rows* reads as one of them.

    Such a line is a row whose trailing cells are empty, a row whose separators
    collapsed to single spaces, or a wrapped continuation of the row above --
    ending the listing on it would drop it and every row after it.
    """
    if _indent(line) > _indent(header_line):
        return True
    tokens = line.split()
    above = rows[-1]
    if len(tokens) == len(above.split()) and len(tokens[0]) == len(_aligned_cells(above)[0]):
        return True
    widths = [len(_aligned_cells(row)[0]) for row in rows]
    if len(tokens) == 1 and min(widths) <= len(tokens[0]) <= max(widths):
        return True
    # Where the rows are padded to the column line's offsets, those offsets say
    # which columns a line's text occupies.
    offsets = _cell_starts(header_line)
    if all(set(_cell_starts(row)) <= set(offsets) for row in rows):
        if len(line.rstrip()) <= offsets[1]:
            return True
        if any(line[offset - 1:offset] == " " and line[offset:offset + 1].strip()
               for offset in offsets[1:]):
            return True
    return False


def _aligned_at(lines: list[str], index: int) -> Any:
    """``(end, rows)`` for an aligned listing whose column line is ``lines[index]``."""
    if "|" in lines[index] or "\t" in lines[index]:
        return None
    header = _aligned_cells(lines[index])
    if not _is_column_line(header):
        return None
    rows: list[str] = []
    for line in lines[index + 1:]:
        if not line.strip():
            break
        cells = _aligned_cells(line)
        if len(cells) < 2:
            if rows and _cut_row(line, lines[index], rows):
                return REFUSED
            break
        if len(cells) > len(header):
            return REFUSED if rows else None
        rows.append(line)
    return (index + 1 + len(rows), rows) if rows else None


def _markdown_at(lines: list[str], index: int) -> Any:
    line = lines[index].strip()
    if "|" not in line or index + 1 >= len(lines):
        return None
    width = len(_markdown_cells(line))
    separator = lines[index + 1].strip()
    # The compact ``|-|-|`` and pipe-less forms are only taken when the
    # separator has exactly one cell per column; a pipe-less column line must
    # also read as column names, since prose can contain a pipe.
    exact = (_GFM_SEPARATOR.match(separator) is not None
             and len(_markdown_cells(separator)) == width)
    if line.startswith("|"):
        if not (_MARKDOWN_SEPARATOR.match(separator) or exact):
            return None
    elif not (exact and _is_column_line(_markdown_cells(line))):
        return None
    if width < 2:
        return None
    piped = line.startswith("|")
    rows = []
    for row in lines[index + 2:]:
        if not (row.strip().startswith("|") if piped else "|" in row):
            break
        if len(_markdown_cells(row)) != width:
            return REFUSED if rows else None
        rows.append(row)
    return (index + 2 + len(rows), rows) if rows else None


def _tabbed_at(lines: list[str], index: int) -> Any:
    header = lines[index].split("\t")
    if len(header) < 2 or not _is_column_line([c.strip() for c in header]):
        return None
    rows = []
    for row in lines[index + 1:]:
        if "\t" not in row:
            break
        if len(row.split("\t")) != len(header):
            return REFUSED if rows else None
        rows.append(row)
    return (index + 1 + len(rows), rows) if rows else None


_SHAPES = (("markdown", _markdown_at), ("tabbed", _tabbed_at), ("aligned", _aligned_at))


def _first_cell(shape: str, row: str) -> str:
    if shape == "markdown":
        return _markdown_cells(row)[0]
    if shape == "tabbed":
        return row.split("\t")[0].strip()
    return _aligned_cells(row)[0]


def _is_footer_row(shape: str, row: str) -> bool:
    """Whether the last row *row* is a summary or a labelled footer, not a listed row."""
    return (_SUMMARY_ROW.match(row.strip().strip("|")) is not None
            or _LABELLED_FOOTER_CELL.match(_first_cell(shape, row)) is not None)


def _row_like(line: str) -> bool:
    """Whether *line* has the shape of a row of any listing shape."""
    stripped = line.strip()
    return (stripped.startswith("|")
            or ("|" in stripped and len([cell for cell in _markdown_cells(stripped) if cell]) >= 2)
            or len([cell for cell in stripped.split("\t") if cell.strip()]) >= 2
            or len(_aligned_cells(stripped)) >= 2)


def _states_incomplete(text: str, rows: int) -> bool:
    """Whether *text* around a listing says the listing is not all of it."""
    if any(int(match.group(1)) > 0 for match in _REMAINING.finditer(text)):
        return True
    if _NOT_COMPLETE.search(text) or _MORE_PAGES.search(text):
        return True
    if any(int(match.group(1)) != rows for match in _SHOWN.finditer(text)):
        return True
    if any(int(match.group(1)) != rows for match in _TOTAL.finditer(text)):
        return True
    if any(int(match.group(1)) > 1 for match in _PAGES.finditer(text)):
        return True
    if any(int(match.group(1)) != int(match.group(2)) for match in _N_OF_M.finditer(text)):
        return True
    return any(int(first) > 1 or int(last) != int(total)
               for first, last, total in _RANGE_OF_M.findall(text))


def parse_table(text: str, *, require_complete: bool = True) -> Optional[dict[str, Any]]:
    """``{"shape", "preamble", "columns", "rows"}`` for the first whole listing, or None.

    With ``require_complete=False`` the listing need not be the whole one: a
    second group or table below it, a trailing summary or labelled footer row
    (``_is_footer_row``; dropped) and a text
    saying there is more (pagination, ``shown``, ``total``, ``remaining``) no
    longer refuse it, and ``end_line`` says where a caller may look for the next
    one. The structural rules still hold: a row that does not fit the columns
    refuses the text. For reading labels only -- rows it returns may be a page
    of a longer listing, so they must never be served as the listing.
    """
    lines = text.splitlines()
    for index in range(len(lines)):
        for shape, parse in _SHAPES:
            found = parse(lines, index)
            if found is REFUSED:
                return None
            if found is None:
                continue
            end, rows = found
            summary = _is_footer_row(shape, rows[-1])
            if not require_complete:
                if summary:
                    rows = rows[:-1]
                if not rows:
                    return None
            # A second group or table below this one, or a notice around it
            # that more rows exist, means these rows are not the whole listing.
            elif any(_row_like(line) for line in lines[end:]) or summary:
                return None
            elif _states_incomplete("\n".join([*lines[:index], *lines[end:]]), len(rows)):
                return None
            columns = "\n".join(lines[index:index + (2 if shape == "markdown" else 1)])
            return {"shape": shape, "preamble": lines[:index], "columns": columns,
                    "rows": rows, "end_line": end}
    return None


def served_rows(alias: str, table: dict[str, Any],
                max_bytes: int) -> Optional[tuple[str, int, int]]:
    """The listing's own rows, copied, as many whole rows as fit *max_bytes*.

    ``(text, shown, total)``, or None when not one row fits beside the text
    above the rows and the closing line -- the search is then left to the
    model. Rows are never paraphrased or reordered, and the closing line says
    how many were shown, so a cut listing cannot read as the whole one.

    The copied text is the backend's, so a line of it shaped like a framework
    marker or handle line is printed quoted (``quote_marker_lines``); the
    closing line is then the only unquoted ``[search_memory`` marker. Rows
    are measured as printed.
    """
    head = quote_marker_lines("\n".join([*table["preamble"], table["columns"]]))
    rows = [quote_marker_lines(row) for row in table["rows"]]
    total = len(rows)

    def closing(shown: int) -> str:
        rest = "" if shown == total else f"; rows {shown + 1}-{total} are NOT shown here"
        return (f"{ROWS_SERVED_MARK} rows 1-{shown} of the {total} rows listed in "
                f"{alias}{rest}. {LABEL_RESTORE_MARK}]")

    # History: an earlier version said "Reserve the longer of the two closings:
    # the cut form names the rows left out." and reserved that width up front.
    # The closing line's width depends on how many rows it names, so each count
    # is measured with its own closing rather than a reserved maximum.
    used = len(head.encode("utf-8")) + 1
    fits = 0
    for shown in range(1, total + 1):
        used += len(rows[shown - 1].encode("utf-8")) + 1
        if used > max_bytes:
            break
        if used + len(closing(shown).encode("utf-8")) <= max_bytes:
            fits = shown
    if fits == 0:
        return None
    return "\n".join([head, *rows[:fits], closing(fits)]), fits, total
