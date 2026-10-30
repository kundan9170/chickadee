"""The merged configuration, its provenance, and `config_sha`.

`run_start.config_sha` is in **built, frozen** code (`schema/journal.py`) and nothing said
what it covers. One file? The merge? The `RunConfig` that resulted? They differ, and only the
last makes a run reproducible from its own record.

ST-9 settles it: the sha is taken over the **fully merged, resolved** configuration,
canonically serialised, and the layer that supplied each winning value is recorded beside it.

**One vocabulary.** A key exists exactly when `EngineConfig`, or a model nested in it,
declares it. This module used to keep a closed key set of its own, in names nothing wrote: it
refused every `rungs.<name>` table and the `[spend]` table `chickadee init` seeds. Now each
layer is checked against the model, and the merge is validated into it.

**And who may write it.** The engine config is the person's; the repository's
`config.toml` holds repository facts. A section the model does not mark `REPOSITORY` is refused
in the project layer, naming the file it belongs in.

`EngineConfig` is imported where it is used, not here: every module that asks
`chickadee.state` for its roots imports this package, and none of them should pay for the
engine's config models to learn where a directory is.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import types
import typing
from collections.abc import Callable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

from chickadee.errors import ConfigError
from chickadee.state.layers import Layer
from chickadee.state.merge import (
    LayerProvenance,
    MergeError,
    UnknownKey,
    is_table,
    merge,
    provenance,
)
from chickadee.state.trust import TRUST_FILE, is_active

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from chickadee.config import EngineConfig
    from chickadee.secrets.denylist import Denylist

PROJECT_LAYER = "project"
"""`layers.LAYER_ORDER`'s name for `<repo>/.chickadee/config.toml` — the repository's file."""

LOCAL_LAYER = "local"
"""`layers.LAYER_ORDER`'s name for `<repo-local>/work.toml`, the person's choice for this clone
(`state.local`): it may hold the work folder and nothing else."""


class LocalHoldsOnlyTheWorkFolder(MergeError):
    """`work.toml` holding anything but `dir`: the file is the harness's record of one answer,
    never a second engine config, so anything else in it refuses the load, naming it."""


def check_local(layer: Layer) -> None:
    """`LocalHoldsOnlyTheWorkFolder` unless the `local` layer holds `work.dir` alone."""
    if layer.name != LOCAL_LAYER or not layer.body:
        return
    work: object = layer.body.get("work")
    keys: list[str] = []
    if isinstance(work, Mapping):
        keys = sorted(str(k) for k in typing.cast("Mapping[object, object]", work))
    if set(layer.body) != {"work"} or keys != ["dir"]:
        found = ", ".join(f"`{k}`" for k in keys) or "nothing"
        raise LocalHoldsOnlyTheWorkFolder(
            f"{_where(layer)}: holds `dir`, the work folder chosen for this clone, and nothing "
            f"else; found {found} (layer: {layer.name}). Delete the file to choose again."
        )


class EngineOnlyKey(MergeError):
    """A key only the engine config may set, found in the repository's file.

    Not `UnknownKey`: the key is real, and the fix is to move it, not to spell it. A clone
    whose file could set a rung's endpoint could send the person's key to its own host, and
    one that could set the cap would be spending the person's money.
    """


@dataclasses.dataclass(frozen=True)
class TrustDecision:
    """Whether a clone accepted its repository's executable config, and **for which config**.

    Made only by `Resolved.trust`, and it names the `EngineConfig` it was taken for:
    `PacketEngine` refuses one taken for another, so an answer computed over one load cannot
    be handed to an engine built from a second.
    """

    engine: EngineConfig
    accepted: bool


@dataclasses.dataclass(frozen=True)
class Resolved:
    config: Mapping[str, Any]
    config_sha: str
    provenance: tuple[LayerProvenance, ...]
    sources: tuple[str, ...]
    engine: EngineConfig
    """The merge, validated: what the engine reads. `config` is what the layers said."""
    project: Mapping[str, Any] = dataclasses.field(default_factory=dict[str, Any])
    """The repository's own layer, as spelled — what `31-state.md` §6 digests. The person's
    engine config is not in it: their own `allow_secret_read` is their decision, not one a
    clone makes on their behalf."""

    def declared(self, overrides: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        """What the repository declares, as §6 digests it: `project` with `overrides` merged
        in. The one composition: `trust` digests it and `chickadee trust` shows and accepts
        it (`state/load.executable`), so an accepted config never reads as unaccepted."""
        return {**self.project, **(overrides or {})}

    def trust(
        self,
        repo_local: pathlib.Path,
        *,
        overrides: Mapping[str, Any] | None = None,
        root: pathlib.Path | None = None,
    ) -> TrustDecision:
        """Whether this clone's owner accepted the repository's executable subset.

        §6 step 1 digests the **project** config: what a repository ships. The layers as
        merged would put the person's own settings under a digest meant for the repository,
        and a validated dump would lose any executable key the model does not hold. So the
        digest is `project` with `overrides` merged in — the project profile's
        `toolchain.bootstrap`, as `profiles.loader.executable_overrides(repo)` returns it —
        which must be exactly what the acceptance surface digests, or an accepted config
        reads as unaccepted.

        A `trusted.toml` that cannot be read is `ConfigError` naming it: guessing either way
        would decide, without saying so, whether a repository's removals take effect. With
        `root`, the repository, the hook scripts' bytes are digested with it.
        """
        declared = self.declared(overrides)
        try:
            accepted = is_active(declared, repo_local, root)
        except (OSError, ValueError, TypeError) as exc:
            # ValueError covers tomllib.TOMLDecodeError; TypeError an `accepted` that is not
            # a list.
            raise ConfigError(
                f"{repo_local / TRUST_FILE} could not be read to tell whether this clone "
                f"accepted its repository's trust digest: {exc}"
            ) from exc
        return TrustDecision(engine=self.engine, accepted=accepted)


def _present(annotation: object) -> object:
    """`X | None` is X: an optional table is still a table when a layer sets it."""
    if typing.get_origin(annotation) in (typing.Union, types.UnionType):
        options = [a for a in typing.get_args(annotation) if a is not type(None)]
        if len(options) == 1:
            return options[0]
    return annotation


def _table_model(annotation: object) -> type[BaseModel] | None:
    """A nested model: a table whose keys are the model's fields."""
    from pydantic import BaseModel

    shape = _present(annotation)
    if isinstance(shape, type) and issubclass(shape, BaseModel):
        return shape
    return None


def _entry_model(annotation: object) -> type[BaseModel] | None:
    """`dict[str, Model]` (`rungs`, `prices`): a table of names, each entry a `Model`."""
    shape = _present(annotation)
    if typing.get_origin(shape) in (dict, Mapping):
        return _table_model(typing.get_args(shape)[1])
    return None


def _where(layer: Layer) -> str:
    return str(layer.source) if layer.source else f"the {layer.name} layer"


def _one_spelling(layer: Layer) -> Layer:
    """`[limits.worker] max_tool_calls = 50` and `[limits]
    "worker.max_tool_calls" = 50` are one key. Each layer is brought to the dotted spelling
    before anything else reads it, so a higher layer overrides a lower one's value however
    either spelled it, and a key spelled both ways in ONE file is refused naming that file."""
    limits = layer.body.get("limits")
    if not is_table(limits):
        return layer
    from chickadee.config import flatten_limits

    try:
        flat = flatten_limits(limits)
    except ConfigError as exc:
        raise ConfigError(f"{_where(layer)}: {exc}") from exc
    dotted = {key.removeprefix("limits."): value for key, value in flat.items()}
    return Layer(layer.name, layer.source, {**layer.body, "limits": dotted})


def check_layer(layer: Layer, schema: type[BaseModel] | None = None) -> None:
    """ST-6 for one layer: every key it sets must be one `schema` declares, at every depth
    the model gives a table shape, and every entry of a section with closed names must be
    one of them. In the repository's layer, every section must be one the model lets a
    repository set. Anything else is a leaf, and its value is the model's to
    validate once the layers are merged.

    Per layer and before the merge, because after it the file and the layer a key came from
    are gone, and the refusal has to name both. `schema` is a parameter so that the claim
    this function exists for can be tested: a section a later task adds to the model is
    checked here with no edit here. It defaults to `EngineConfig`.
    """
    if schema is None:
        from chickadee.config import EngineConfig

        schema = EngineConfig
    _check(_one_spelling(layer).body, schema, layer, "")


def _marked(field: FieldInfo, kind: type[Any]) -> Any:
    return next((m for m in field.metadata if isinstance(m, kind)), None)


def _check(
    body: Mapping[str, Any],
    schema: type[BaseModel],
    layer: Layer,
    path: str,
) -> None:
    from chickadee.config import ENGINE_CONFIG_PATH, Names, RepositoryMaySet, retired_key

    fields = schema.model_fields
    for key, value in body.items():
        here = f"{path}.{key}" if path else key
        if key not in fields:
            # A key the config stopped reading is not a typo: it is refused naming where its
            # number lives now, not with a list of keys that does not contain it.
            reason = retired_key(here)
            if reason is not None:
                raise UnknownKey(f"{_where(layer)}: {reason} (layer: {layer.name})")
            raise UnknownKey(
                f"{_where(layer)}: unknown key `{here}` (layer: {layer.name}). "
                f"Known keys here: {sorted(fields)}"
            )
        field = fields[key]
        if not path and layer.name == PROJECT_LAYER and not _marked(field, RepositoryMaySet):
            repository = sorted(k for k, f in fields.items() if _marked(f, RepositoryMaySet))
            raise EngineOnlyKey(
                f"{_where(layer)}: `{here}` is the engine config's, not the repository's "
                f"(layer: {layer.name}). Set it in {ENGINE_CONFIG_PATH} or the environment; "
                f"a repository's config sets only {repository} (D127)"
            )
        if not is_table(value):
            continue
        names: Any = _marked(field, Names)
        if names is not None:
            # A retired name is not a typo: it passes here, is never offered as a known name,
            # and the model refuses it naming the decision that retired it.
            candidates = sorted(k for k in value if k not in names.retired)
            if names.rule is not None:
                # The open form: any name the rule accepts declares an entry.
                for k in candidates:
                    why = names.rule(k)
                    if why is not None:
                        raise UnknownKey(
                            f"{_where(layer)}: invalid {names.what} name `{here}.{k}` "
                            f"(layer: {layer.name}): {why}"
                        )
            stray = [k for k in candidates if names.rule is None and k not in names.allowed]
            if stray:
                raise UnknownKey(
                    f"{_where(layer)}: unknown {names.what} `{here}.{stray[0]}` "
                    f"(layer: {layer.name}). Known here: {sorted(names.allowed)}"
                )
        if (table := _table_model(field.annotation)) is not None:
            _check(value, table, layer, here)
        elif (entry := _entry_model(field.annotation)) is not None:
            retired = names.retired if names is not None else frozenset[str]()
            for name, sub in value.items():
                # A retired entry is refused whole by the model, naming its decision; what it
                # holds is not checked here, or a key inside it would be refused first, for
                # the wrong reason.
                if is_table(sub) and name not in retired:
                    _check(sub, entry, layer, f"{here}.{name}")


DENY_ADD = "secrets.deny_add"
"""The one key whose layers accumulate rather than override."""


def _deny_adds(layers: Sequence[Layer]) -> list[tuple[Layer, Any]]:
    """Each layer that sets `secrets.deny_add`, with the value it sets, in layer order."""
    setters: list[tuple[Layer, Any]] = []
    for layer in layers:
        secrets = layer.body.get("secrets")
        if is_table(secrets) and "deny_add" in secrets:
            setters.append((layer, secrets["deny_add"]))
    return setters


def _accumulated_additions(layers: Sequence[Layer]) -> list[Any] | None:
    """`31-state.md` §4 rule 6: every layer's `secrets.deny_add`, in layer order, each
    once. An addition only tightens, so the merge's higher-wins would let a clone erase the
    person's additions and make readable a file its owner had denied. `None` when no layer
    sets it.

    A value that is not a list is refused here, naming its layer, not left for the model:
    under a higher layer's list it would never reach the model, and an addition its author
    believes they made would be dropped without a word."""
    setters = _deny_adds(layers)
    if not setters:
        return None
    additions: list[Any] = []
    for layer, added in setters:
        if not isinstance(added, list | tuple):
            raise ConfigError(
                f"{_where(layer)}: `{DENY_ADD}` must be a list of patterns, got "
                f"{type(added).__name__} {added!r} (layer: {layer.name})"
            )
        for pattern in typing.cast("Sequence[Any]", added):
            if pattern not in additions:
                additions.append(pattern)
    return additions


COMMANDS = "commands"
"""`[[commands]]`: every layer's rows are kept, each with its layer, rather than the
highest layer's list replacing the rest: the person's rows and the repository's are merged by
prefix in `EngineConfig.command_table`, where an unaccepted repository leaves the person's
whole."""

_ROW_LAYER = "layer"


HOOKS = "hooks"
"""`[[hooks]]`: merged by `(event, name)` like any array of tables,
each row that defines a hook (`use` or `command`) first marked with its layer, so a higher
layer that only switches a hook off leaves it the layer that wrote it."""


def _marked_rows(
    layer: Layer, key: str, given: object, *, marks: Callable[[Mapping[str, Any]], bool]
) -> list[Any]:
    """`given`, a layer's array of tables at `key`, each row `marks` takes marked with the
    layer's name. A row that names its own layer is refused: where a row came from is the
    loader's fact, never the file's claim, or a repository could mark its rows the person's and
    skip the acceptance they need. The one check, for `[[commands]]` and
    `[[hooks]]` alike."""
    if not isinstance(given, list | tuple):
        raise ConfigError(
            f"{_where(layer)}: `[[{key}]]` must be an array of tables, got "
            f"{type(given).__name__} {given!r} (layer: {layer.name})"
        )
    rows: list[Any] = []
    for row in typing.cast("Sequence[Any]", given):
        if not is_table(row):
            raise ConfigError(
                f"{_where(layer)}: each `[[{key}]]` row is a table, got {row!r} "
                f"(layer: {layer.name})"
            )
        if _ROW_LAYER in row:
            raise ConfigError(
                f"{_where(layer)}: a `[[{key}]]` row may not set `{_ROW_LAYER}`; the loader "
                f"records the file each row came from (layer: {layer.name})"
            )
        rows.append({**row, _ROW_LAYER: layer.name} if marks(row) else dict(row))
    return rows


def _marked_hooks(layer: Layer) -> Layer:
    """`layer` with each of its `[[hooks]]` rows that defines a hook (`use` or `command`) marked
    with the layer's name; a row that only switches one off keeps the layer that defined it."""
    given = layer.body.get(HOOKS)
    if given is None:
        return layer
    rows = _marked_rows(layer, HOOKS, given, marks=lambda row: "use" in row or "command" in row)
    return dataclasses.replace(layer, body={**layer.body, HOOKS: rows})


def _command_layers(layers: Sequence[Layer]) -> list[tuple[Layer, Any]]:
    """Each layer that sets `[[commands]]`, with the rows it sets, in layer order."""
    return [(layer, layer.body[COMMANDS]) for layer in layers if COMMANDS in layer.body]


def _command_rows(layers: Sequence[Layer]) -> list[Any] | None:
    """Every layer's `[[commands]]` rows, in layer order, each marked with the layer it came
    from (`_marked_rows`); `None` when no layer sets any."""
    setters = _command_layers(layers)
    if not setters:
        return None
    return [
        row
        for layer, given in setters
        for row in _marked_rows(layer, COMMANDS, given, marks=lambda _row: True)
    ]


def unaccepted_denylist(layers: Sequence[Layer]) -> Denylist:
    """The deny list before any engine config is resolved: the built-ins, and every layer's
    additions accumulated (rule 6), with a repository's removals inert. That is
    `SecretsPolicy.denylist(accepted=False)`, the safe reading of a trust question nobody has
    asked, and what `init` indexes under so a file the repository marks secret is
    never parsed."""
    from chickadee.config import SecretsPolicy

    spelled = [_one_spelling(layer) for layer in layers]
    additions = _accumulated_additions(spelled) or []
    policy = SecretsPolicy.model_validate({"deny_add": additions})
    return policy.denylist(accepted=False)


def _provenance(layers: Sequence[Layer]) -> tuple[LayerProvenance, ...]:
    """The record of which layer supplied each value. An accumulated key has no one winning
    layer: every layer that set `secrets.deny_add` supplied part of it, so its row names
    them all, in layer order — `user+project`. Naming only the last would credit the
    repository with the person's additions, and say nothing moved when theirs did."""
    rows = provenance([(layer.name, layer.body) for layer in layers])
    joined = {
        DENY_ADD: "+".join(layer.name for layer, _ in _deny_adds(layers)),
        COMMANDS: "+".join(layer.name for layer, _ in _command_layers(layers)),
    }
    return tuple(
        dataclasses.replace(row, layer=joined[row.key]) if row.key in joined else row
        for row in rows
    )


def canonical(config: Mapping[str, Any]) -> str:
    """Stable bytes for hashing: sorted keys, no incidental whitespace.

    Two runs whose merged config is equal must hash equal **however the layers were split**,
    which is what makes the sha a statement about the configuration rather than about the
    filesystem that happened to produce it.
    """
    return json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)


def _described(layers: Sequence[Layer]) -> str:
    """What was validated, for a refusal: the files, in merge order, and any other layer
    that set something. Empty layers are left out; they cannot have caused it."""
    parts = [_where(layer) for layer in layers if layer.body]
    return " + ".join(parts) or "an empty configuration"


def resolve(layers: Sequence[Layer]) -> Resolved:
    """Check each layer, merge, validate, hash, and record where each value came from."""
    from chickadee.config import EngineConfig

    spelled = [_one_spelling(layer) for layer in layers]
    for layer in spelled:
        check_local(layer)
        _check(layer.body, EngineConfig, layer, "")
    spelled = [_marked_hooks(layer) for layer in spelled]
    config: Mapping[str, Any] = {}
    for layer in spelled:
        # `[[commands]]` is kept per layer below, never merged: its rows have no identity key,
        # and the person's and the repository's are both kept, each with its layer.
        config = merge(config, {k: v for k, v in layer.body.items() if k != COMMANDS})
    if (additions := _accumulated_additions(spelled)) is not None:
        config = merge(config, {"secrets": {"deny_add": additions}})
    if (commands := _command_rows(spelled)) is not None:
        config = {**config, COMMANDS: commands}
    return Resolved(
        config=config,
        config_sha=hashlib.sha256(canonical(config).encode("utf-8")).hexdigest(),
        provenance=_provenance(spelled),
        sources=tuple(str(layer.source) for layer in spelled if layer.source is not None),
        engine=EngineConfig.with_defaults(config, source=_described(spelled)),
        project=_project_layer(spelled),
    )


def _project_layer(layers: Sequence[Layer]) -> Mapping[str, Any]:
    """The repository's layer, or an empty one when the repository has no config file."""
    for layer in layers:
        if layer.name == PROJECT_LAYER:
            return layer.body
    return {}
