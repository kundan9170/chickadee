"""Connection management, WAL, and rebuild-on-mismatch.

The store holds paths and spans, never file contents. A cache of bytes has to be
invalidated; a cache of *locations* is checked against the file's own sha256 at index time
and cannot silently disagree with what is on disk.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path

from chickadee.errors import InfraError
from chickadee.state.roots import NotARepository, Roots
from chickadee.state.roots import roots as resolve_roots

SCHEMA_VERSION = 6
"""6 since T-3.220: `analyzed` one row a file and source, and each problem its verdict's server,
a linter's beside a language server's. 5 since T-3.220: a task's start kept,
`baseline_files` and `baseline_diags`. 4 since T-3.170: the `diagnostics` and
`analyzed` tables. 3 since T-3.164: a file's `lang` is its profile's name
(`tsx` became `typescript`), and a file no profile owns has no rows. An index from before either
is rebuilt rather than read."""
DB_NAME = "graph.db"
_DDL = Path(__file__).with_name("schema.sql")


def db_path_for(roots: Roots) -> Path:
    """The index is **checkout-local**.

    It describes *these* bytes, so it belongs to this working tree and not to the clone. That
    is the half of an earlier security review which actually bites: under the default
    isolation mode a run happens in a linked worktree, and an index keyed to the repository
    would mean every reindex there overwrote the index the person is working against.

    Journals go the other way — repo-local — because history belongs to the clone. Two roots,
    two questions, and answering them the same way is what the old single `.chickadee/` did.
    """
    return roots.checkout_local / DB_NAME


STANDALONE_DIR = ".chickadee"
"""Where the index goes for a directory that is **not** a checkout — see `db_path`."""


def db_path(repo: Path) -> Path:
    """Convenience for callers that hold a directory rather than `Roots`.

    Inside a repository this resolves through `db_path_for`, so there is one decision about
    where a checkout's index lives and two ways to ask it — never two answers.

    **Outside a repository it falls back**, and that is not a compromise: "checkout-local"
    presupposes a checkout. A plain directory has none, so there is no checkout-local place
    for its index to go. Indexing one is a real use — unit tests build a small tree to
    exercise traversal without the cost of `git init` — and refusing it would buy nothing,
    because such a directory has no worktree for a stray index to pollute, which is the
    entire hazard ST-3 exists to remove.

    `graph/queries.py` and `graph/indexer.py` still pass a path; when the engine resolves
    `Roots` once and passes it down, this function can go.
    """
    try:
        return db_path_for(resolve_roots(repo))
    except NotARepository:
        return repo / STANDALONE_DIR / DB_NAME


def index_exists(repo: Path) -> bool:
    """Whether this checkout already has an index: a database at the current schema
    that at least one index pass has committed to. A file `connect` created and nothing
    filled, or a first build that crashed before its commit, is not an index.

    A fresh linked worktree has none — it is a new directory, and the index describes bytes
    rather than history, so nothing carries over. `connect` will build one, and building it
    is the slow step `24-init.md` §3 warns about, so a surface must be able to **say that is
    happening** rather than leaving the user with an unexplained pause.

    A predicate rather than a flag on `connect`, because the announcement belongs to whoever
    is talking to the user and `connect` is called from places that are not.
    """
    path = db_path(repo)
    return (
        path.exists()
        and _meta_int(path, "schema_version") == SCHEMA_VERSION
        and (_meta_int(path, "generation") or 0) >= 1
    )


def connect(repo: Path) -> sqlite3.Connection:
    """Open the graph, creating or rebuilding it as needed.

    **Rebuild on mismatch rather than migrate.** The graph is a derived index: every row can
    be recomputed from the repository, so a schema change costs one reindex and a migration
    path costs a permanent branch in the code for every version ever shipped. This is the
    opposite of the journal's rule, and deliberately — the journal is the
    source of truth and cannot be regenerated, so it migrates.

    **A current database is opened, never written.** Creating the schema takes the write
    lock, so a connect that always did would wait on, and after sqlite's timeout fail
    against, an index build in progress: the reader WAL exists to serve.
    """
    path = db_path(repo)
    path.parent.mkdir(parents=True, exist_ok=True)
    current = path.exists() and _meta_int(path, "schema_version") == SCHEMA_VERSION
    if path.exists() and not current:
        path.unlink()
        for suffix in ("-wal", "-shm"):
            path.with_name(path.name + suffix).unlink(missing_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")  # ON DELETE CASCADE is load-bearing for GR-2
    if current:
        return conn
    conn.execute("PRAGMA journal_mode=WAL")  # persistent: concurrent reads during a run
    conn.executescript(_DDL.read_text(encoding="utf-8"))
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('schema_version', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(SCHEMA_VERSION),),
    )
    conn.commit()
    return conn


def _meta_int(path: Path, key: str) -> int | None:
    """An integer from the `meta` table, or None when the file cannot answer.

    A corrupt or unreadable database returns None, which reads as a schema mismatch and
    triggers a rebuild (the index always completes), and as no completed pass.
    Crashing here would make an unrelated corruption fatal to a run that could simply
    reindex.
    """
    try:
        conn = sqlite3.connect(path)
    except sqlite3.Error:
        return None
    try:
        row = conn.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        return int(row[0]) if row else None
    except (sqlite3.Error, ValueError, TypeError):
        return None
    finally:
        conn.close()


def generation(conn: sqlite3.Connection) -> int:
    """Monotonic counter bumped by every index pass, so callers can detect a stale read."""
    row = conn.execute("SELECT value FROM meta WHERE key='generation'").fetchone()
    return int(row[0]) if row else 0


def indexed_sha(conn: sqlite3.Connection, path: str) -> str | None:
    """The sha256 of the bytes the index last parsed `path` from, or None when it holds no row
    for it: what tells a reader whether the index's rows still describe the file on disk."""
    row = conn.execute("SELECT sha256 FROM files WHERE path = ?", (path,)).fetchone()
    return None if row is None else str(row[0])


def bump_generation(conn: sqlite3.Connection) -> int:
    nxt = generation(conn) + 1
    conn.execute(
        "INSERT INTO meta(key, value) VALUES('generation', ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (str(nxt),),
    )
    return nxt


class GraphUnavailable(InfraError):
    """A graph query or write failed. Escalated as infra; never read as an
    empty answer."""


@contextmanager
def graph_failures_are_infra(what: str | None = None) -> Generator[None]:
    """The graph's own failures surface as `sqlite3.Error`/`OSError`; `18-engine` §5 says a
    required tool failing is `InfraError`, never the model's fault. The one translation, for
    every writer of the index: a step's and a commit's reindex (`box/commit.Reindexer`). `what`
    prefixes the message with what was being done. Moved from the packer.

    Everything raised deliberately inside (`HarnessBug`, `GraphUnavailable`) is neither of
    those and passes straight through.
    """
    try:
        yield
    except (sqlite3.Error, OSError) as exc:
        raise GraphUnavailable(f"{what}: {exc}" if what else str(exc)) from exc
