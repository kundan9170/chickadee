"""A skill's front matter: the open standard's fields and Claude Code's agent fields, and
nothing else.

**The fields.** The Agent Skills standard's `name`, `description`, `license`, `compatibility`,
`metadata` and `allowed-tools`, our `argument-hint`, and Claude Code's `context` and `agent`:

- `context: fork` makes an agent skill, one `spawn_agent` runs; without it a skill is knowledge,
  loaded whole by `load_skill`;
- `agent` names the profile an agent skill runs as;
- `allowed-tools` narrows an agent skill's tools and never widens them. It is
  never a pre-approval;
- `metadata` is a one-level map of strings. Keys prefixed `chickadee-` are ours, and only
  `chickadee-report` and `chickadee-keywords` exist.

**What one of ours may not say.** `model` is refused by name: rung choice has an owner.
So is any other field outside the set, and an agent field on a knowledge skill: an author who
wrote one believed it did something, and must be told. An external skill (`.claude/skills/`,
`.agents/skills/`) was written for another tool, so a field we do not read, or one that means
nothing on that skill, is ignored and listed, so the index can say so.

**The standard's bounds are not limits.** `NAME_MAX`, `DESCRIPTION_MAX` and `COMPATIBILITY_MAX`
define what a valid skill is (agentskills.io, "Specification"), so a skill written here stays a
skill everywhere. They are not rows of the limits register: they bound no spend, and a config
that raised them would accept skills no other tool can read (the owner's call). They
apply to ours; another tool's file is read as that tool left it.

**A strict reader, not YAML.** What is read: top-level `key: value`, a single- or double-quoted
value, a `>` or `|` block whose lines are indented under it, and `metadata`'s indented
`key: value` lines. An indented line always belongs to the key above it, so an external skill's
nested field we do not read is skipped whole with its key. A YAML library would be a dependency
that accepts far more than any skill may say.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from chickadee.errors import ConfigError

__all__ = [
    "COMPATIBILITY_MAX",
    "DESCRIPTION_MAX",
    "FIELDS",
    "METADATA_KEYS",
    "NAME_MAX",
    "FrontMatter",
    "Parsed",
    "Raw",
    "parse",
    "read_block",
    "scalar",
    "string_map",
    "tool_names",
    "valid_name",
]

FIELDS = (
    "name",
    "description",
    "argument-hint",
    "license",
    "compatibility",
    "metadata",
    "context",
    "agent",
    "allowed-tools",
)
"""The closed set, in the order a skill writes them."""

METADATA_KEYS = ("chickadee-report", "chickadee-keywords")
"""Our `metadata` keys. Another `chickadee-` key in one of ours is a typo."""

NAME_MAX = 64
"""The standard's longest name, in characters (agentskills.io, "Specification")."""
DESCRIPTION_MAX = 1024
"""The standard's longest description, in characters."""
COMPATIBILITY_MAX = 500
"""The standard's longest `compatibility`, in characters."""

_FENCE = "---"
_KEY = re.compile(r"^([A-Za-z][A-Za-z0-9_-]*)\s*:\s?(.*)$")
_NAME = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
"""The standard's rule: lower case letters, digits and single hyphens, none at either end. It
is also what a person types after `/`."""
_TOOL = re.compile(r"[A-Za-z_][A-Za-z0-9_-]*(?:\([^)]*\))?")
"""One `allowed-tools` entry: a name, with Claude Code's `(…)` pattern if it has one."""


def tool_names(text: str) -> tuple[str, ...]:
    """A tool list as Claude Code writes one, in any of its forms (`Read, Grep`, `Read Grep`,
    a YAML list, `Bash(git log:*)` with its pattern): `allowed-tools` and an agent file's
    `tools`, read by the one rule."""
    return tuple(_TOOL.findall(text))


@dataclass(frozen=True)
class FrontMatter:
    name: str
    description: str
    argument_hint: str | None = None
    license: str | None = None
    compatibility: str | None = None
    metadata: tuple[tuple[str, str], ...] = ()
    """`metadata`, in file order; a tuple so the record stays hashable."""
    context: Literal["fork"] | None = None
    agent: str | None = None
    """The profile an agent skill names, as written (`discover` maps another tool's names)."""
    allowed_tools: tuple[str, ...] | None = None
    """What narrows an agent skill's tools, as written; None narrows nothing."""

    @property
    def is_agent(self) -> bool:
        return self.context == "fork"

    def meta(self, key: str) -> str | None:
        return next((v for k, v in self.metadata if k == key), None)

    @property
    def report_sections(self) -> tuple[str, ...]:
        """`chickadee-report`: the extra sections an agent's report must carry."""
        return _listed(self.meta("chickadee-report"))

    @property
    def keywords(self) -> tuple[str, ...]:
        """`chickadee-keywords`: extra words `find_skill` matches."""
        return _listed(self.meta("chickadee-keywords"))


@dataclass(frozen=True)
class Parsed:
    front: FrontMatter
    body: str
    """Everything after the closing fence: what `load_skill` brings in, whole."""
    ignored: tuple[str, ...] = ()
    """An external skill's fields we did not read, in file order."""


def parse(path: Path, text: str, *, external: bool) -> Parsed:
    """`text`'s front matter and body. A missing or malformed block, a missing `name` or
    `description`, or an unusable name is `ConfigError` naming `path`; so, for one of ours, is
    any field it may not declare."""
    raw, body = read_block(path, text)
    outside = tuple(k for k in raw if k not in FIELDS)
    if outside and not external:
        named = ", ".join(f"`{k}`" for k in outside)
        why = " (a skill may not choose its model, SK-15)" if "model" in outside else ""
        raise ConfigError(f"{path}: front matter may not declare {named}{why}")
    read = _Fields(path, raw, external)
    name = read.scalar("name").strip()
    description = " ".join(read.scalar("description").split())
    if not name:
        raise ConfigError(f"{path}: a skill needs a `name`")
    if not valid_name(name):
        raise ConfigError(
            f"{path}: skill name {name!r} must be at most {NAME_MAX} lower case letters, "
            "digits and single hyphens, with no hyphen at either end"
        )
    if not description:
        raise ConfigError(
            f"{path}: a skill needs a `description` (the index shows nothing else)"
        )
    if not external:
        _ours_bounds(path, name, description)
    context = read.context()
    front = FrontMatter(
        name,
        description,
        argument_hint=read.scalar("argument-hint").strip() or None,
        license=read.scalar("license").strip() or None,
        compatibility=read.compatibility(),
        metadata=read.metadata(),
        context=context,
        agent=read.agent(context),
        allowed_tools=read.allowed_tools(context),
    )
    ignored = tuple(k for k in raw if k not in FIELDS or k in read.ignored)
    return Parsed(front, body, ignored)


def valid_name(name: str) -> bool:
    """The standard's name rule, which every skill and agent file's name meets."""
    return len(name) <= NAME_MAX and _NAME.match(name) is not None


def _ours_bounds(path: Path, name: str, description: str) -> None:
    if len(description) > DESCRIPTION_MAX:
        raise ConfigError(
            f"{path}: a description may be at most {DESCRIPTION_MAX} characters "
            f"(the Agent Skills standard); this one is {len(description)}"
        )
    folder = path.parent.name if path.name == "SKILL.md" else path.stem
    if folder != name:
        raise ConfigError(
            f"{path}: skill {name!r} must be in a file or folder of the same name "
            "(the Agent Skills standard)"
        )


Raw = dict[str, tuple[str, list[str]]]
"""Each key's text after the colon and the indented lines under it, in file order."""


def read_block(path: Path, text: str) -> tuple[Raw, str]:
    """`text`'s front-matter fields, uninterpreted, and its body: the strict reader every file
    of this layer is read with, a skill or another tool's agent file (`agent_files.py`)."""
    lines = text.replace("\r\n", "\n").split("\n")
    if not lines or lines[0].strip() != _FENCE:
        raise ConfigError(f"{path}: a skill starts with a `---` front-matter block")
    try:
        close = next(i for i in range(1, len(lines)) if lines[i].strip() == _FENCE)
    except StopIteration:
        raise ConfigError(
            f"{path}: the front-matter block is never closed with `---`"
        ) from None
    body = "\n".join(lines[close + 1 :]).lstrip("\n")
    return _fields(path, lines[1:close]), body


def scalar(path: Path, raw: Raw, key: str) -> str:
    """`key`'s value as one string, or `""` when it is absent."""
    if key not in raw:
        return ""
    text, nested = raw[key]
    return _value(path, key, text, nested)


class _Fields:
    """Reads each field `parse` keeps. For one of ours a field that means nothing where it is
    written is refused; for an external skill it is added to `ignored` instead."""

    def __init__(self, path: Path, raw: Raw, external: bool) -> None:
        self._path = path
        self._raw = raw
        self._external = external
        self.ignored: set[str] = set()

    def _refuse_or_ignore(self, key: str, why: str) -> None:
        if not self._external:
            raise ConfigError(f"{self._path}: `{key}` {why}")
        self.ignored.add(key)

    def scalar(self, key: str) -> str:
        return scalar(self._path, self._raw, key)

    def compatibility(self) -> str | None:
        text = " ".join(self.scalar("compatibility").split())
        if not text:
            return None
        if not self._external and len(text) > COMPATIBILITY_MAX:
            raise ConfigError(
                f"{self._path}: `compatibility` may be at most {COMPATIBILITY_MAX} characters "
                "(the Agent Skills standard)"
            )
        return text

    def context(self) -> Literal["fork"] | None:
        value = self.scalar("context").strip()
        if not value:
            return None
        if value != "fork":
            self._refuse_or_ignore("context", f"is {value!r}; the only context is `fork`")
            return None
        return "fork"

    def agent(self, context: Literal["fork"] | None) -> str | None:
        value = self.scalar("agent").strip()
        if not value:
            return None
        if context is None:
            self._refuse_or_ignore("agent", "names a profile, which only `context: fork` uses")
            return None
        return value

    def allowed_tools(self, context: Literal["fork"] | None) -> tuple[str, ...] | None:
        if "allowed-tools" not in self._raw:
            return None
        if context is None:
            self._refuse_or_ignore(
                "allowed-tools",
                "narrows an agent's tools, which only `context: fork` has; "
                "it never pre-approves a tool (SK-15)",
            )
            return None
        return tool_names(self.scalar("allowed-tools"))

    def metadata(self) -> tuple[tuple[str, str], ...]:
        if "metadata" not in self._raw:
            return ()
        raw, nested = self._raw["metadata"]
        pairs = string_map(nested) if raw.strip() in ("", "{}") else None
        if pairs is None:
            self._refuse_or_ignore("metadata", "must be a map of strings, one level deep")
            return ()
        if not self._external:
            for key, _ in pairs:
                if key.startswith("chickadee-") and key not in METADATA_KEYS:
                    known = ", ".join(METADATA_KEYS)
                    raise ConfigError(
                        f"{self._path}: metadata key `{key}` is not ours; ours are {known}"
                    )
        return pairs


def string_map(nested: list[str]) -> tuple[tuple[str, str], ...] | None:
    """A field's indented lines as pairs, or None when they are not one level of
    `key: value`."""
    lines = [n for n in nested if n.strip() and not n.lstrip().startswith("#")]
    if not lines:
        return ()
    indent = len(lines[0]) - len(lines[0].lstrip())
    pairs: list[tuple[str, str]] = []
    for line in lines:
        if len(line) - len(line.lstrip()) != indent:
            return None
        found = _KEY.match(line.strip())
        if found is None:
            return None
        value = _strip_comment(found.group(2)).strip()
        if value[:1] in ("|", ">", "[", "{", "-") or not value:
            return None
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        elif value[:1] in "'\"":
            return None
        pairs.append((found.group(1), value))
    return tuple(pairs)


def _listed(text: str | None) -> tuple[str, ...]:
    if not text:
        return ()
    return tuple(part.strip() for part in text.split(",") if part.strip())


def _fields(path: Path, block: list[str]) -> Raw:
    fields: Raw = {}
    i = 0
    while i < len(block):
        line = block[i]
        i += 1
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        if line[0].isspace():
            raise ConfigError(f"{path}: front-matter line {line.strip()!r} belongs to no key")
        found = _KEY.match(line)
        if found is None:
            raise ConfigError(f"{path}: front-matter line {line!r} is not `key: value`")
        key, raw = found.group(1), _strip_comment(found.group(2)).strip()
        if key in fields:
            raise ConfigError(f"{path}: front-matter key `{key}` is declared twice")
        nested: list[str] = []
        while i < len(block) and (not block[i].strip() or block[i][0].isspace()):
            nested.append(block[i])
            i += 1
        # Nothing is interpreted yet: a field outside the set is never read, since an external
        # skill's may be any YAML, and one of ours is refused by name whatever its value.
        fields[key] = (raw, nested)
    return fields


def _value(path: Path, key: str, raw: str, nested: list[str]) -> str:
    if raw[:1] == ">":
        return " ".join(n.strip() for n in nested if n.strip())
    if raw[:1] == "|":
        return "\n".join(n.strip() for n in nested).strip()
    if any(n.strip() for n in nested):
        # A plain scalar continued on indented lines: YAML folds it, and so does this.
        raw = " ".join([raw, *(n.strip() for n in nested if n.strip())])
    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in "'\"":
        inner = raw[1:-1]
        return inner.replace("''", "'") if raw[0] == "'" else inner.replace('\\"', '"')
    if raw[:1] and raw[0] in "'\"":
        raise ConfigError(f"{path}: front-matter `{key}` opens a quote it never closes")
    return raw


def _strip_comment(raw: str) -> str:
    """A ` #` outside quotes starts a comment, as in YAML."""
    if raw[:1] and raw[0] in "'\"":
        return raw
    at = raw.find(" #")
    return raw if at < 0 else raw[:at]
