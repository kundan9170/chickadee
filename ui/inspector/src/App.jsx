import { useEffect, useMemo, useRef, useState } from "react";
import { buildRun, buildSession, isProblem, itemText } from "./model.js";
import { Item } from "./cards.jsx";
import { Badge, ExpandAll } from "./parts.jsx";

const POLL_MS = 1500;

// The page's address is `#/run/<id>` or `#/session/<id>`, so a view survives a reload.
function useRoute() {
  const read = () => {
    const [, kind, id] = window.location.hash.split("/");
    return kind && id ? { kind, id: decodeURIComponent(id) } : null;
  };
  const [route, setRoute] = useState(read);
  useEffect(() => {
    const on = () => setRoute(read());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return [route, (kind, id) => (window.location.hash = `/${kind}/${encodeURIComponent(id)}`)];
}

function usePoll(fn, deps, ms) {
  useEffect(() => {
    let live = true;
    let timer;
    const tick = async () => {
      try {
        await fn(() => live);
      } catch (e) {
        console.error(e);
      }
      if (live) timer = setTimeout(tick, ms);
    };
    tick();
    return () => {
      live = false;
      clearTimeout(timer);
    };
  }, deps); // eslint-disable-line react-hooks/exhaustive-deps
}

// Every line of a run or session, fetched once and then only what was appended since.
function useEvents(route) {
  const empty = { events: [], error: null, damage: null, tables: { messages: {}, tools: {} } };
  const [state, setState] = useState(empty);
  const last = useRef(0);
  useEffect(() => {
    last.current = 0;
    setState(empty);
  }, [route?.kind, route?.id]); // eslint-disable-line react-hooks/exhaustive-deps
  usePoll(
    async (alive) => {
      if (!route) return;
      const path = route.kind === "run" ? "runs" : "sessions";
      const res = await fetch(`/api/${path}/${encodeURIComponent(route.id)}?after=${last.current}`);
      const body = await res.json();
      if (!alive()) return;
      if (body.error) return setState((s) => ({ ...s, error: body.error }));
      if (body.events.length) {
        last.current = body.events[body.events.length - 1].seq;
        setState((s) => ({
          events: [...s.events, ...body.events],
          error: null,
          damage: body.damage,
          tables: {
            messages: { ...s.tables.messages, ...(body.messages || {}) },
            tools: { ...s.tables.tools, ...(body.tools || {}) },
          },
        }));
      }
    },
    [route?.kind, route?.id],
    POLL_MS
  );
  return state;
}

export default function App() {
  const [route, go] = useRoute();
  const [index, setIndex] = useState(null);
  usePoll(
    async (alive) => {
      const body = await (await fetch("/api/index")).json();
      if (alive()) setIndex(body);
    },
    [],
    3000
  );
  return (
    <div className="app">
      <Sidebar index={index} route={route} go={go} />
      <main className="main">{route ? <View route={route} go={go} index={index} /> : <Welcome index={index} />}</main>
    </div>
  );
}

function ago(mtime) {
  if (!mtime) return "";
  const s = Date.now() / 1000 - mtime;
  if (s < 60) return "just now";
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

function Sidebar({ index, route, go }) {
  if (!index) return <aside className="side">loading…</aside>;
  const runs = Object.fromEntries(index.runs.map((r) => [r.id, r]));
  const inSession = new Set(index.sessions.flatMap((s) => s.runs));
  const loose = index.runs.filter((r) => !inSession.has(r.id));
  const active = (kind, id) => (route?.kind === kind && route?.id === id ? "active" : "");
  const RunLink = ({ r, nested }) => (
    <button className={`side-item ${nested ? "nested" : ""} ${active("run", r.id)}`} onClick={() => go("run", r.id)}>
      <div className="side-title">
        {nested ? "↳ " : ""}
        {r.goal || r.id}
      </div>
      <div className="side-meta">
        {r.id} · {ago(r.mtime)}{" "}
        {r.outcome ? <Badge tone={r.outcome === "COMPLETED" ? "green" : "warn"}>{r.outcome}</Badge> : <Badge tone="live">running?</Badge>}
      </div>
    </button>
  );
  return (
    <aside className="side">
      <div className="brand">Chickadee Inspector</div>
      <div className="muted small state" title={index.state}>
        {index.state}
      </div>
      <div className="side-h">Sessions (Zeus)</div>
      {index.sessions.length === 0 && <div className="muted small">none</div>}
      {index.sessions.map((s) => (
        <div key={s.id}>
          <button className={`side-item ${active("session", s.id)}`} onClick={() => go("session", s.id)}>
            <div className="side-title">{s.title || s.id}</div>
            <div className="side-meta">
              {s.id} · {ago(s.mtime)}
            </div>
          </button>
          {s.runs.map((id) => (runs[id] ? <RunLink key={id} r={runs[id]} nested /> : null))}
        </div>
      ))}
      <div className="side-h">Runs without a session</div>
      {loose.length === 0 && <div className="muted small">none</div>}
      {loose.map((r) => (
        <RunLink key={r.id} r={r} />
      ))}
    </aside>
  );
}

function Welcome({ index }) {
  return (
    <div className="welcome">
      <h1>What did the model see, and what did it say?</h1>
      <p>
        Pick a session on the left to read Zeus's turns, or a run to read Athena's and each worker's conversation: the system
        prompt, every message sent, each reply, each tool call's arguments and the result the model got back.
      </p>
      <p className="muted">
        Reading {index?.state || "…"}. The page refreshes by itself while a run is going.
      </p>
    </div>
  );
}

function View({ route, go }) {
  const { events, error, damage, tables } = useEvents(route);
  const built = useMemo(() => {
    if (route.kind === "run") return buildRun(events, tables);
    const c = buildSession(events);
    return { convs: [c], timeline: events };
  }, [events, tables, route.kind]);
  const [tab, setTab] = useState(null);
  const [problems, setProblems] = useState(false);
  const [showEvents, setShowEvents] = useState(true);
  const [query, setQuery] = useState("");
  const [expand, setExpand] = useState(0);
  const [follow, setFollow] = useState(false);
  const bottom = useRef(null);

  useEffect(() => setTab(null), [route.kind, route.id]);
  const conv = built.convs.find((c) => c.key === tab) || built.convs[0];

  const items = useMemo(() => {
    if (!conv) return [];
    const q = query.trim().toLowerCase();
    return conv.items.filter((it) => {
      if (problems && !isProblem(it)) return false;
      if (!showEvents && (it.type === "event" || it.type === "compaction")) return false;
      if (q && !itemText(it).includes(q)) return false;
      return true;
    });
  }, [conv, problems, showEvents, query]);

  useEffect(() => {
    if (follow) bottom.current?.scrollIntoView({ behavior: "smooth" });
  }, [items.length, follow]);

  const totals = built.convs.reduce(
    (t, c) => ({ calls: t.calls + c.stats.calls, cost: t.cost + c.stats.cost }),
    { calls: 0, cost: 0 }
  );
  const ended = events.some((e) => e.kind === "run_end" || e.kind === "session_end");

  return (
    <div>
      <div className="view-head">
        <div className="view-title">
          <b>{route.kind === "run" ? "Run" : "Session"}</b> {route.id}
          {route.kind === "run" && !ended && <Badge tone="live">live</Badge>}
          <Badge>{events.length} lines</Badge>
          <Badge>{totals.calls} model calls</Badge>
          <Badge>${totals.cost.toFixed(4)}</Badge>
        </div>
        {error && <div className="alert">{error}</div>}
        {damage && <div className="alert">Journal damage after the last line shown: {damage}</div>}
        <div className="tabs">
          {built.convs.map((c) => (
            <button key={c.key} className={`tab ${conv?.key === c.key ? "on" : ""}`} onClick={() => setTab(c.key)}>
              <b>{c.label}</b>
              <span className="muted small">
                {c.stats.calls} calls · {(c.stats.tokensIn / 1000).toFixed(1)}k in · {(c.stats.tokensOut / 1000).toFixed(1)}k out
              </span>
              {(c.stats.refusals > 0 || c.stats.denied > 0) && (
                <Badge tone="red">
                  {c.stats.refusals + c.stats.denied} problem{c.stats.refusals + c.stats.denied > 1 ? "s" : ""}
                </Badge>
              )}
            </button>
          ))}
        </div>
        <div className="controls">
          <input placeholder="search this conversation…" value={query} onChange={(e) => setQuery(e.target.value)} />
          <label>
            <input type="checkbox" checked={problems} onChange={(e) => setProblems(e.target.checked)} /> problems only
          </label>
          <label>
            <input type="checkbox" checked={showEvents} onChange={(e) => setShowEvents(e.target.checked)} /> harness events
          </label>
          <label>
            <input type="checkbox" checked={follow} onChange={(e) => setFollow(e.target.checked)} /> follow
          </label>
          <button className="btn" onClick={() => setExpand(Math.abs(expand) + 1)}>
            expand all
          </button>
          <button className="btn" onClick={() => setExpand(-(Math.abs(expand) + 1))}>
            collapse all
          </button>
        </div>
      </div>
      <ExpandAll.Provider value={expand}>
        <div className="items">
          {items.length === 0 && <div className="muted">nothing to show</div>}
          {items.map((it) => (
            <Item key={`${it.type}-${it.ev.seq}`} item={it} convLabel={conv.label} onOpenRun={(id) => go("run", id)} />
          ))}
          <div ref={bottom} />
        </div>
      </ExpandAll.Provider>
    </div>
  );
}
