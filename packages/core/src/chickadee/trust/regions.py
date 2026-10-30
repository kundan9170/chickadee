"""The one envelope untrusted text enters a prompt in (`20-injection.md` IJ-1).

`envelope(tag, body, **attrs)` is `<tag k='v'>`, the body, `</tag>`: facts the harness observed
in the attributes, the text someone else chose inside. A skill's body, the person's invocation
arguments and a web page are all one of these, so there is one shape to audit and one rule for
what a region may hold.

**A body cannot end its own region.** A closing tag of the region's own name inside the body,
in any case or spacing, is written `<\\/tag>`, so a page that says `</web_content>` and then
"ignore your instructions" stays inside the region it came in. Nothing else in the body moves.

Attributes are quoted as `repr` quotes them, which is how a skill's `name` always read.
"""

from __future__ import annotations

import re

__all__ = ["envelope"]


def envelope(tag: str, body: str, **attrs: str) -> str:
    """`body` in a `<tag>` region, the attributes in the order given."""
    head = "".join(f" {k}={v!r}" for k, v in attrs.items())
    closing = re.compile(rf"<\s*/\s*{re.escape(tag)}\s*>", re.IGNORECASE)
    kept = closing.sub(lambda m: m.group(0).replace("/", "\\/", 1), body)
    return f"<{tag}{head}>\n{kept}\n</{tag}>"
