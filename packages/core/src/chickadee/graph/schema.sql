-- Graph store DDL (`16-graph.md` §3). Paths and spans only, never file contents (GR-6); the
-- one table that can go stale, `diagnostics`, carries the sha256 it was judged against.

CREATE TABLE IF NOT EXISTS files (
  path TEXT PRIMARY KEY, sha256 TEXT NOT NULL, lang TEXT,
  parse_ok INTEGER NOT NULL, indexed_at TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS symbols (
  id INTEGER PRIMARY KEY, name TEXT NOT NULL, qualified_name TEXT NOT NULL,
  file TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  kind TEXT NOT NULL,                      -- function|class|method|const|module
  line_start INTEGER, line_end INTEGER,
  sig TEXT, doc_first_line TEXT,
  parent_id INTEGER REFERENCES symbols(id) ON DELETE SET NULL,
  exported INTEGER NOT NULL DEFAULT 0);

CREATE TABLE IF NOT EXISTS edges (
  src TEXT NOT NULL, dst TEXT NOT NULL, type TEXT NOT NULL,
  edge_class TEXT NOT NULL,                -- resolved | fallback | unknown
  file TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  line INTEGER, stale INTEGER NOT NULL DEFAULT 0);

CREATE TABLE IF NOT EXISTS imports (
  file TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  module TEXT NOT NULL, alias TEXT, line INTEGER);

CREATE TABLE IF NOT EXISTS globals (
  qualified_name TEXT PRIMARY KEY, name TEXT NOT NULL,
  file TEXT NOT NULL REFERENCES files(path) ON DELETE CASCADE,
  line INTEGER NOT NULL, kind TEXT NOT NULL, type_text TEXT);

CREATE VIRTUAL TABLE IF NOT EXISTS symbols_fts USING fts5(
  qualified_name, tokens, content='');     -- `tokens` = subword expansion, GR-3

-- What a language server said about a file's bytes (§4.1, GR-12): verdicts about bytes, never
-- the bytes, each stamped with the sha256 it judged. No key over (path, line, col, code): a
-- server may report two problems at one place under one code, and a key would drop one.
CREATE TABLE IF NOT EXISTS diagnostics (
  path TEXT NOT NULL, line INTEGER NOT NULL, col INTEGER,
  severity TEXT NOT NULL,                  -- error | warning | hint
  code TEXT, message TEXT NOT NULL,
  source TEXT NOT NULL,                    -- the label the problem came with
  sha256 TEXT NOT NULL,
  server TEXT NOT NULL DEFAULT '');        -- the verdict it belongs to: `analyzed.source`

-- A row means "a server finished judging these bytes" (GR-13): without it an empty
-- `diagnostics` result is indistinguishable from a clean file. `stale_by` is the file whose
-- change has made the verdict stale since (D205 decision 2), cleared by a fresh verdict.
-- One row a file and source (D237 B2): a language server's verdict and a linter's sit side by
-- side, neither erasing the other. `local` is a source that judges one file alone (a linter):
-- another file's change never outdates it.
CREATE TABLE IF NOT EXISTS analyzed (
  path TEXT NOT NULL, sha256 TEXT NOT NULL, source TEXT NOT NULL, stale_by TEXT,
  stale_gen INTEGER,                       -- the index generation that marked it (D205 A2)
  local INTEGER NOT NULL DEFAULT 0,
  PRIMARY KEY (path, source));

-- A task's start, kept (D237 B1): each file's verdict as it stood the first time one of the
-- task's passes changed it or marked it stale, before that pass or its re-check replaced it.
-- `judged` is 0 where there was no fresh verdict then: "no record", never "clean".
CREATE TABLE IF NOT EXISTS baseline_files (
  task TEXT NOT NULL, path TEXT NOT NULL, judged INTEGER NOT NULL,
  PRIMARY KEY (task, path));

CREATE TABLE IF NOT EXISTS baseline_diags (
  task TEXT NOT NULL, path TEXT NOT NULL, line INTEGER NOT NULL,
  severity TEXT NOT NULL, code TEXT, message TEXT NOT NULL, source TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

-- `edges(dst, type)` is non-negotiable (§1.10 principle 5): callers() and blast_radius()
-- both walk edges backwards, and without it every hop is a full scan.
CREATE INDEX IF NOT EXISTS idx_edges_dst  ON edges(dst, type);
CREATE INDEX IF NOT EXISTS idx_edges_src  ON edges(src, type);
CREATE INDEX IF NOT EXISTS idx_sym_name   ON symbols(name);
CREATE INDEX IF NOT EXISTS idx_sym_file   ON symbols(file);
CREATE INDEX IF NOT EXISTS idx_diag_path  ON diagnostics(path, line);
CREATE INDEX IF NOT EXISTS idx_base_diag  ON baseline_diags(task, path);
