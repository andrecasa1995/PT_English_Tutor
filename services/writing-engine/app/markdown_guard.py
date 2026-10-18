"""Identify Markdown regions that must never receive automated suggestions.

Conservative protection for prototype. Full Markdown parsing is planned.
"""
import re

FENCED = re.compile(r"(?m)^[ \t]{0,3}(`{3,}|~{3,})[^\n]*\n[\s\S]*?^[ \t]{0,3}\1[ \t]*$", re.MULTILINE)
WIKILINK = re.compile(r"!?\[\[[^\]\n]+\]\]")
MD_LINK = re.compile(r"!?\[[^\]\n]*\]\([^\n)]*\)")
INLINE_CODE = re.compile(r"(`+)(?!`)[^\n]*?\1")
URL = re.compile(r"https?://[^\s<>]+")


def protected_ranges(text: str) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if text.startswith("---\n"):
        end = re.search(r"(?m)^---\s*$", text[4:])
        if end is not None:
            ranges.append((0, 4 + end.end()))
    for regex in (FENCED, WIKILINK, MD_LINK, INLINE_CODE, URL):
        ranges.extend((m.start(), m.end()) for m in regex.finditer(text))
    return ranges


def overlaps_protected(text: str, start: int, end: int) -> bool:
    return any(start < protected_end and end > protected_start
               for protected_start, protected_end in protected_ranges(text))
