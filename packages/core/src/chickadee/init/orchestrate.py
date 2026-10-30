"""Init orchestration, reached through `PacketEngine.init_repo()`.

Idempotent and safe to re-run: it rebuilds what is derived and never overwrites what a
human may have edited. The index is derived and always reconstructible; `config.toml`
is not, so a re-run reports drift and changes nothing. `--force` rewrites the config after
saying what it would overwrite, and uses the identical staging/fsync/rename/marker protocol
— force is not a less careful path, only a more destructive one.

No model calls, no network, no dependency installation, and no repository mutation outside
`.chickadee/`, the resolved Git exclude line, and their same-directory temps.
"""

from __future__ import annotations

import datetime
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path

from chickadee.box.spawn import HarnessSpawn, index_resolver
from chickadee.errors import RepoNotIndexed
from chickadee.graph.indexer import Indexer, IndexProgress, IndexReport
from chickadee.graph.lsp import CheckerResolver
from chickadee.graph.store import index_exists
from chickadee.init import detect, lease, prepare, publish, toolchain
from chickadee.isolation.base import SandboxBackend, select
from chickadee.paths import within
from chickadee.process.supervisor import ProcessSupervisor
from chickadee.profiles.loader import ProfileSet
from chickadee.secrets.denylist import Denylist
from chickadee.secrets.denylist import resolve as resolve_denylist
from chickadee.secrets.walk_memo import WalkMemo
from chickadee.state.cap import engine_cap
from chickadee.state.layers import CONFIG_NAME, Layer, LayerReadError, load_layers
from chickadee.state.resolve import unaccepted_denylist
from chickadee.state.roots import Roots, roots


class InitError(Exception):
    """Init cannot proceed. Carries the fix where there is one."""


@dataclass(frozen=True)
class InitReport:
    root: Path
    repository_id: str
    git_common_dir: Path
    index: IndexReport
    """The code graph init built: files, symbols, edge classes, and every
    file it could not read as a gap with its reason."""
    created: tuple[str, ...] = ()
    profiles: tuple[detect.ProfileEvidence, ...] = ()
    gaps: tuple[str, ...] = ()
    """Detection's gaps: files no installed profile claims."""
    toolchains: dict[str, dict[str, str | None]] = field(default_factory=lambda: {})
    hard_cost_cap_usd: Decimal | None = None
    """The cap in force: the engine config's `[spend] hard_cost_usd`. `None` when
    there is none to read — the report's warnings then say why and where — so a cap the
    person never set is never printed as one they did."""
    degraded: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    generation: str = ""
    config_preserved: bool = False


def _generation(now: datetime.datetime | None = None) -> str:
    ts = now or datetime.datetime.now()
    return ts.strftime("%Y%m%d-%H%M%S-") + f"{ts.microsecond:06d}"


def render_config(evidence: tuple[detect.ProfileEvidence, ...]) -> str:
    """The seeded `.chickadee/config.toml`: facts about this repository and nothing else.
    Models, the spend cap and the engine's own settings are the person's engine
    config — a repository's file that set them would refuse to load — so none appear here,
    not even commented."""
    names = ", ".join(f'"{e.name}"' for e in evidence)
    return f"""# Written by `chickadee init`. Yours to edit — a re-run preserves this file.
# Facts about this repository only (D127). Models, the spend cap and the engine's own
# settings are yours, not the repository's: they live in your engine config,
# ~/.config/chickadee/config.toml, and setting them here refuses the load.
profiles = "auto"   # detected: [{names}]

# Patterns this repository keeps secret, on top of the built-in deny list (D118):
# [secrets]
# deny_add = ["*.kdbx"]
"""


def _read_layers(scopes: Roots) -> list[Layer] | LayerReadError:
    """The config layers, read once for both the cap and the deny list, which answer an
    unreadable file differently: the cap becomes a note, the deny list a refusal."""
    try:
        return load_layers(scopes)
    except LayerReadError as exc:
        return exc


@dataclass(frozen=True)
class IndexSandbox:
    """How an index's language servers run: the harness's wrap over `backend`, network
    off and the repository read-only. `require_sandbox` refuses to run one unwrapped, so with no
    backend here no server starts and the edges fall back; the default, since `chickadee init`
    runs before any config says otherwise."""

    backend: SandboxBackend | None = None
    """None is the best backend here (`isolation.select`)."""
    require_sandbox: bool = True

    def resolver(self, root: Path, denylist: Denylist, profiles: ProfileSet) -> CheckerResolver:
        spawn = HarnessSpawn(
            self.backend or select(),
            ProcessSupervisor(),
            require_sandbox=self.require_sandbox,
            deny_read=resolve_denylist(denylist, [root, Path.home()], memo=WalkMemo.shared()),
        )
        return index_resolver(root, spawn, profiles)


def _index_denylist(layers: list[Layer] | LayerReadError) -> Denylist:
    """What the index build may not parse when no engine passed its own list: the
    built-ins and every layer's additions, a repository's removals inert. A layer that
    cannot be read refuses init: indexing without it could parse a file it marks secret."""
    if isinstance(layers, LayerReadError):
        raise InitError(
            f"the deny list could not be read, so nothing was indexed: {layers}; fix the "
            "file and re-run"
        )
    return unaccepted_denylist(layers)


def init(
    root: Path,
    *,
    force: bool = False,
    denylist: Denylist | None = None,
    builtin_dir: Path | None = None,
    hook: publish.FaultHook = None,
    now: datetime.datetime | None = None,
    sandbox: IndexSandbox | None = None,
    say: Callable[[str], None] = lambda _line: None,
    progress: Callable[[IndexProgress], None] | None = None,
) -> InitReport:
    """`denylist` is the engine's resolved list when an engine calls; `None` reads the
    repository's own, as `chickadee init` does before any engine config exists. `sandbox` is
    how the index's language servers run: the engine's backend and
    `require_sandbox`; `None` is the best backend here, refusing to run one unwrapped. A
    folder that is not a git repository is made one first (`prepare.make_repository`), what
    it did told to `say` as it goes. `progress` is told how far the index's pass has got.
    """
    made: tuple[str, ...] | None = None
    try:
        identity = detect.resolve(root)
    except detect.NotAGitRepository:
        if not root.is_dir():
            raise
        try:
            made = prepare.make_repository(root, denylist, say=say)
        except prepare.NotHere as exc:
            raise InitError(str(exc)) from exc
        identity = detect.resolve(root)
    root = identity.root
    scopes = roots(root)
    layers = _read_layers(scopes)
    if denylist is None:
        denylist = _index_denylist(layers)

    profiles = ProfileSet.load(repo=root, builtin_dir=builtin_dir)
    evidence, gaps = detect.detect_profiles(root, profiles, builtin_dir=builtin_dir)
    names = tuple(e.name for e in evidence)

    missing = toolchain.missing_grammars(names, profiles)
    if missing:
        # Fatal, unlike a missing linter: with no grammar the code cannot be parsed, so the
        # index would omit it silently and every later gate would answer about a file it
        # never read.
        raise InitError(
            f"no tree-sitter grammar for profile(s) {', '.join(missing)}; "
            "install the grammar before initialising"
        )

    toolchains, degraded = toolchain.probe(names, profiles, root)
    generation = _generation(now)
    created: list[str] = (
        [] if made is None else [f"git repository, first commit {prepare.COMMIT_MESSAGE!r}"]
    )
    warnings: list[str] = (
        [f"left out of git history as secret: {', '.join(made)}"] if made else []
    )

    with lease.RepoLease.acquire(identity.git_common_dir, identity.repository_id):
        try:
            staging = publish.stage(root, generation, hook)
            publish.fault(hook, "index_build")
            index = _build_index(root, denylist, profiles, sandbox or IndexSandbox(), progress)

            config = publish.chickadee_dir(root) / publish.CONFIG_NAME
            config_preserved = config.exists() and not force
            if config_preserved:
                warnings.append(
                    f"{publish.CONFIG_NAME} preserved; detected profiles are "
                    f"[{', '.join(names)}] — reconcile by hand or re-run with --force"
                )
            else:
                if config.exists():
                    warnings.append(f"--force rewrote {publish.CONFIG_NAME}")
                publish.write_file(config, render_config(evidence), hook, step="config")
                created.append(f"{publish.CHICKADEE_DIR}/{publish.CONFIG_NAME}")

            # Exclude first, then the generation marker: the line is idempotent
            # and harmless if init later fails, whereas a marker published early would
            # advertise a generation that does not exist.
            if publish.ensure_git_excluded(detect.exclude_path(root), hook):
                created.append("git exclude: .chickadee/")

            publish.publish(root, staging, generation, hook)
            created.append(f"{publish.CHICKADEE_DIR}/generations/{generation}")
        except BaseException:
            # The previously published generation is untouched byte-for-byte: nothing above
            # writes outside staging until the final rename.
            publish.discard_staging(root, generation)
            raise

    cap, note = engine_cap(scopes.user / CONFIG_NAME, layers)
    if note is not None:
        warnings.append(note)
    return InitReport(
        root=root,
        repository_id=identity.repository_id,
        git_common_dir=identity.git_common_dir,
        index=index,
        created=tuple(created),
        profiles=evidence,
        gaps=gaps,
        toolchains=toolchains,
        hard_cost_cap_usd=cap,
        degraded=degraded,
        warnings=tuple(warnings),
        generation=generation,
        config_preserved=config_preserved,
    )


def _build_index(
    root: Path,
    denylist: Denylist,
    profiles: ProfileSet,
    sandbox: IndexSandbox,
    progress: Callable[[IndexProgress], None] | None = None,
) -> IndexReport:
    """The code graph, rebuilt from scratch (on every run, `--force` or not). Its home
    is the checkout's `graph.db`, not the staged generation: one index, one place.
    The build is one SQLite transaction, so a crash part-way rolls back and a reader sees
    the previous index until the commit. The commit precedes the generation marker; a crash
    between leaves a fresh repository uninitialised, or an initialised one with a newer
    complete index, which is derived and checked per file against its sha. It reads
    each file by `profiles`, the set `init` detected with, so the report and the index agree."""
    indexer = Indexer(
        root,
        sandbox.resolver(root, denylist, profiles),
        denylist=denylist,
        profiles=profiles,
        progress=progress,
    )
    try:
        return indexer.full_index()
    finally:
        indexer.close()


def reindex(
    root: Path,
    changed: Sequence[Path] | None = None,
    *,
    denylist: Denylist,
    sandbox: IndexSandbox | None = None,
) -> IndexReport:
    """Refresh this checkout's graph: `changed` reparsed, or every file
    whose sha moved when it is `None`. A relative path is the checkout's; each is resolved
    as the indexer names files, under the checkout's real root, so a path spelled through a
    link reaches the same row, and one that lands outside the checkout is not indexed.

    Under the repository WRITE lease, so it never interleaves with a run's packet
    commits or an init. A checkout with no index is `RepoNotIndexed`: an incremental pass
    over nothing would silently become the slow full build `init` exists to announce."""
    with lease.take_write_lease(detect.resolve(root)):
        return reindex_held(root, changed, denylist=denylist, sandbox=sandbox)


def reindex_held(
    root: Path,
    changed: Sequence[Path] | None = None,
    *,
    denylist: Denylist,
    sandbox: IndexSandbox | None = None,
) -> IndexReport:
    """`reindex`, for a caller that already holds the repository WRITE lease: a handoff that
    lands refreshes the checkout's graph in the same hold (`session_handoff.finalize_session`),
    so no run starts between the branch moving and its index catching up."""
    identity = detect.resolve(root)
    # Real, as `within` resolves each changed path: the indexer names a file relative to it.
    root = Path(os.path.realpath(identity.root))
    if not index_exists(root):
        raise RepoNotIndexed(f"{root} has no code index; run init_repo() first")
    paths = None if changed is None else [p for c in changed if (p := within(root, c))]
    profiles = ProfileSet.load(repo=root)
    resolver = (sandbox or IndexSandbox()).resolver(root, denylist, profiles)
    indexer = Indexer(root, resolver, denylist=denylist, profiles=profiles)
    try:
        return indexer.reindex(paths)
    finally:
        indexer.close()


def require_initialised(root: Path) -> None:
    """Guard for every command that assumes a prepared repository.

    A published, complete generation or nothing. A partially published one reads as
    uninitialised, which is the whole point of the marker being written last.
    """
    if not publish.is_initialised(root):
        raise InitError(
            f"{root} is not initialised — run `chickadee init` first "
            "(no complete generation is published)"
        )
