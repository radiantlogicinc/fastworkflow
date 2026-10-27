"""Wording the library's planning commands share.

A file whose name starts with "_" is never a command: planning commands import it.
"""
from fastworkflow.planning import count


def copies(number):
    """How many copies the library owns, in words: "1 copy", "3 copies"."""
    return count(number, "copy", "copies")
