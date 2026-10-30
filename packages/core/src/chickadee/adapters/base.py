"""What every role's composition reads of its model's prompts and sampling: the templates, the
tools' descriptions, the first call's and a retry's parameters, and the one parse of a JSON
rung's reply.

Never calls a model. The packet worker's turn parse, its reply formats' packet
rules and the family adapters retired with the packet loop: the box roles call
tools natively, and their edits are the edit tools'.
"""

from __future__ import annotations

import json
import re
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Final

from pydantic import BaseModel, ValidationError

from chickadee.adapters.frontier import TEMPLATES
from chickadee.edits.formats.fences import scan_fences
from chickadee.schema.adapter import ParseError, RetryParams

if TYPE_CHECKING:
    from chickadee.config import RungEndpoint
    from chickadee.tools.spec import ToolSpec


_ATTEMPT1_TEMP = 0.2
_RETRY_TEMP = 0.4


SHARED_TEMPLATES: Final = Path(__file__).resolve().parent / "shared" / "templates"
"""Prompts tied to no model family: the worker's and Athena's instructions, and the
shipped hand-off templates, since the box roles may run on any model."""


def load_template(directory: Path, name: str) -> str:
    """Versioned artifact. Missing is a harness bug, not a silent empty prompt."""
    path = directory / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        from chickadee.errors import HarnessBug

        raise HarnessBug(f"adapter template unreadable: {path}") from exc


def tool_descriptions(
    specs: Iterable[ToolSpec], *roots: Path, present: Collection[str] = ()
) -> dict[str, str]:
    """Each tool's description by its template name (`ToolSpec.template`):
    what a native schema carries, and whose first line is the stub. The one reader of a
    tool's description, for every role that shows tools natively.

    `roots` are searched in order and the first holding a tool's file wins, so a role reads its
    own description of a tool where it has one (`BOX_DESCRIPTIONS`) and
    the shared one otherwise: a role is never told about a tool as another role
    holds it. Each description's optional blocks are resolved by `present`, as the role's
    prompt is (`select_blocks`): the box shell's says when it runs unsealed."""
    if not roots:
        raise ValueError("tool_descriptions needs at least one template root")
    return {
        s.description: select_blocks(
            load_template(_first(roots, s.template), s.template), present
        )
        for s in specs
    }


def _first(roots: tuple[Path, ...], name: str) -> Path:
    return next((r for r in roots if (r / name).is_file()), roots[-1])


BOX_DESCRIPTIONS: Final = (SHARED_TEMPLATES, TEMPLATES)
"""Where the worker's and the lead engineer's tool descriptions are read: their own first
(`shared/templates/tools/`: the box shell, `atlas` without `coder`), then the frontier set for
the tools every role describes alike. The scout has no description of its own since `grep` and
`glob` replaced the tools it described apart, so it reads the frontier set."""


_BLOCK = re.compile(r"^\{\{([#^/])([a-z_]+)\}\}$")


def select_blocks(text: str, present: Collection[str]) -> str:
    """A role prompt with its optional blocks resolved: the lines between
    `{{#name}}` and `{{/name}}` kept only when `name` is in `present`, those between
    `{{^name}}` and `{{/name}}` only when it is not. Each marker is a line of its own, and
    blocks do not nest. Rendered once per composition, so the bytes a role is sent are the
    same for every request of a run and the provider's cache holds them."""
    out: list[str] = []
    open_: tuple[str, str] | None = None
    for line in text.splitlines(keepends=True):
        marker = _BLOCK.match(line.strip())
        if marker is None:
            if open_ is None or (open_[0] == "#") == (open_[1] in present):
                out.append(line)
            continue
        kind, name = marker.groups()
        if kind == "/":
            if open_ is None or open_[1] != name:
                raise ValueError(f"{{{{/{name}}}}} closes no open block")
            open_ = None
        elif open_ is not None:
            raise ValueError(f"{{{{{kind}{name}}}}} opens inside {{{{{open_[0]}{open_[1]}}}}}")
        else:
            open_ = (kind, name)
    if open_ is not None:
        raise ValueError(f"{{{{{open_[0]}{open_[1]}}}}} is never closed")
    return "".join(out)


def retry_params(attempt: int, cap: int) -> RetryParams:
    """Temperature never decreases. AD-4: no stop sequences. D195: every attempt may
    write up to the model's configured output cap; a cut answer is still `truncated`."""
    temp = _ATTEMPT1_TEMP if attempt <= 1 else _RETRY_TEMP
    return RetryParams(
        temperature=temp,
        max_tokens=cap,
        stop=(),
        reasoning=None,
    )


def first_attempt_params(endpoint: RungEndpoint) -> RetryParams:
    """A rung's first call: first-attempt sampling, the model's whole output cap, and its
    configured reasoning (per model, never a retry lever), with the model's own
    temperature where its catalog entry sets one. One rule, for every rung that makes a
    single call per request: Zeus's and the scout's."""
    base = retry_params(1, endpoint.max_output_tokens)
    update: dict[str, object] = {"reasoning": endpoint.reasoning}
    if endpoint.temperature is not None:
        update["temperature"] = endpoint.temperature
    return base.model_copy(update=update)


def parse_structured[M: BaseModel](text: str, model: type[M]) -> M | ParseError:
    """JSON rungs. Schema-valid is not semantically valid — caller still lints."""
    raw = text.strip()
    fences, truncated = scan_fences(raw)
    if truncated:
        return ParseError(
            kind="truncated",
            hint="close the JSON fence and finish the object",
        )
    if len(fences) == 1 and fences[0].closed:
        raw = fences[0].body.strip()
    elif fences:
        return ParseError(
            kind="multiple_blocks",
            hint="emit one JSON object, optionally inside a single fence",
        )
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return ParseError(
            kind="bad_format",
            hint="emit one JSON object matching the schema in the prompt",
        )
    try:
        return model.model_validate(payload)
    except ValidationError as exc:
        # The re-ask carries the validation error itself. The old hint said only "fix
        # the fields named in the prompt", so a model was re-asked without being told which
        # field, and a plan's cycle went back unnamed though `PlanDAG` spells its path out.
        return ParseError(
            kind="bad_format",
            hint=f"JSON did not match the schema: {validation_summary(exc)}",
        )


VALIDATION_ERRORS = 5
"""How many of a validation's errors a re-ask names: enough to fix a plan, few enough that one
bad list does not flood the turn."""


def validation_summary(exc: ValidationError) -> str:
    """A validation error as a model can act on it: each field's location and message, the
    first `VALIDATION_ERRORS` of them, and how many more there were."""
    errors = exc.errors(include_url=False, include_input=False)
    shown = [
        f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {' '.join(e['msg'].split())[:300]}"
        for e in errors[:VALIDATION_ERRORS]
    ]
    more = len(errors) - len(shown)
    return "; ".join(shown) + (f"; and {more} more" if more > 0 else "")
