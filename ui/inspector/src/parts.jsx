import { createContext, useContext, useEffect, useState } from "react";
import { contentText } from "./model.js";

// "Expand all" reaches every collapsible through this, so one switch opens the page.
export const ExpandAll = createContext(0);

export function Collapsible({ title, badges, defaultOpen = false, tone, children }) {
  const signal = useContext(ExpandAll);
  const [open, setOpen] = useState(defaultOpen);
  useEffect(() => {
    if (signal > 0) setOpen(true);
    if (signal < 0) setOpen(defaultOpen);
  }, [signal, defaultOpen]);
  return (
    <div className={`coll ${tone || ""}`}>
      <button className="coll-head" onClick={() => setOpen(!open)}>
        <span className="caret">{open ? "▾" : "▸"}</span>
        <span className="coll-title">{title}</span>
        {badges}
      </button>
      {open && <div className="coll-body">{children}</div>}
    </div>
  );
}

export function Badge({ children, tone }) {
  return <span className={`badge ${tone || ""}`}>{children}</span>;
}

export function TextBlock({ text, maxLines = 14 }) {
  const signal = useContext(ExpandAll);
  const [all, setAll] = useState(false);
  useEffect(() => {
    if (signal > 0) setAll(true);
    if (signal < 0) setAll(false);
  }, [signal]);
  if (text == null || text === "") return <div className="empty">(empty)</div>;
  const lines = text.split("\n");
  const long = lines.length > maxLines;
  const shown = long && !all ? lines.slice(0, maxLines).join("\n") : text;
  return (
    <div className="text-wrap">
      <div className="text-tools">
        <span className="muted">
          {lines.length} lines · {text.length.toLocaleString()} chars · ~{Math.ceil(text.length / 4).toLocaleString()} tok
        </span>
        <button className="link" onClick={() => navigator.clipboard?.writeText(text)}>
          copy
        </button>
      </div>
      <pre className="text">{shown}</pre>
      {long && (
        <button className="link more" onClick={() => setAll(!all)}>
          {all ? "show less" : `show all ${lines.length} lines`}
        </button>
      )}
    </div>
  );
}

export function Json({ value }) {
  return <pre className="text json">{JSON.stringify(value, null, 2)}</pre>;
}

export function Content({ content, maxLines }) {
  return <TextBlock text={contentText(content)} maxLines={maxLines} />;
}

export function argsSummary(args) {
  if (!args || typeof args !== "object") return "";
  const s = Object.entries(args)
    .map(([k, v]) => `${k}=${typeof v === "string" ? JSON.stringify(v.length > 40 ? v.slice(0, 40) + "…" : v) : JSON.stringify(v)}`)
    .join(", ");
  return s.length > 140 ? s.slice(0, 140) + "…" : s;
}

export function ToolCallView({ tc }) {
  if (tc.name === "python") return <PythonCallView tc={tc} />;
  return (
    <Collapsible
      title={
        <span>
          <span className="tool-name">{tc.name}</span>
          <span className="muted">({argsSummary(tc.arguments)})</span>
        </span>
      }
      badges={<Badge>id {tc.id}</Badge>}
    >
      <div className="label">arguments, exactly as the model sent them</div>
      <Json value={tc.arguments} />
    </Collapsible>
  );
}

// PR 641's Python calls: the reply's one block is journaled as a `python` call with the code
// as its argument, or with the error the block was refused on.
function PythonCallView({ tc }) {
  const a = tc.arguments || {};
  return (
    <Collapsible
      defaultOpen
      tone={a.error ? "bad" : ""}
      title={
        <span>
          <span className="tool-name">python</span>{" "}
          <span className="muted">{a.error ? "block refused" : `${(a.code || "").split("\n").length} lines`}</span>
        </span>
      }
      badges={<Badge>id {tc.id}</Badge>}
    >
      {a.error && <div className="alert">{a.error}</div>}
      {a.code != null && <TextBlock text={a.code} maxLines={40} />}
      {a.reply != null && (
        <>
          <div className="label">the reply it came from</div>
          <TextBlock text={a.reply} />
        </>
      )}
    </Collapsible>
  );
}
