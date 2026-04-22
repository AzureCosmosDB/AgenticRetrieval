"""
Generic XML element helper utilities.

Small accessor functions for pulling text, attribute values, and dates
out of :mod:`xml.etree.ElementTree` elements.
"""

from typing import Optional

XLINK_NS = "http://www.w3.org/1999/xlink"


def text(el, path: str, default: str = "") -> str:
    """Find a child element by *path* and return all inner text concatenated."""
    if el is None:
        return default
    node = el.find(path)
    if node is None:
        return default
    return "".join(node.itertext()).strip()


def all_text(el, path: str) -> list[str]:
    """Return text of every element matching *path*."""
    if el is None:
        return []
    texts: list[str] = []
    for n in el.findall(path):
        value = "".join(n.itertext()).strip()
        if value:
            texts.append(value)
    return texts


def attr(el, path: str, attr_name: str, ns_map: dict | None = None, default: str = "") -> str:
    """Find element by *path* and return one of its attributes."""
    if el is None:
        return default
    node = el.find(path)
    if node is None:
        return default
    if ns_map:
        # Build Clark notation key, e.g. {http://www.w3.org/1999/xlink}href
        for prefix, uri in ns_map.items():
            attr_name = attr_name.replace(f"{prefix}:", f"{{{uri}}}")
    return node.get(attr_name, default)


def parse_date(el) -> Optional[str]:
    """Convert a ``<date>``-style element with ``<year>``/``<month>``/``<day>`` children to ISO date."""
    if el is None:
        return None
    day   = text(el, "day")   or "01"
    month = text(el, "month") or "01"
    year  = text(el, "year")
    if year:
        return f"{year}-{month.zfill(2)}-{day.zfill(2)}"
    return None
