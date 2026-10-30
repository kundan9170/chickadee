import { useEffect, useState } from "react";
import { Badge, Collapsible, Content, Json, TextBlock, ToolCallView } from "./parts.jsx";
import { contentText } from "./model.js";

const ROLE_TONE = { system: "sys", user: "user", harness: "harness" };

export function Item({ item, convLabel, onOpenRun }) {
  switch (item.type) {
    case "message":
      return <MessageCard item={item} />;
    case "call":
      return <CallCard item={item} convLabel={convLabel} />;
    case "assistant":
      return <AssistantCard item={item} />;
    case "tool":
      return <ToolCard item={item} />;
    case "toolmeta":
      return <ToolMetaRow item={item} />;
    case "compaction":
      return <EventRow item={item} tone="warn" />;
    case "prefix":
      return <PrefixCard item={item} />;
    case "episode":
      return <EpisodeCard item={item} onOpenRun={onOpenRun} />;
    default:
      return <EventRow item={item} />;
  }
}

function Where({ ev }) {
  return (
    <span className="where">
      #{ev.seq}
      {ev.task_id ? ` · ${ev.task_id}` : ""}
      {ev.round ? ` r${ev.round}` : ""}
      {" · "}
      {(ev.ts || ev.at || "").replace("T", " ").replace("Z", "")}
    </span>
  );
}

function MessageCard({ item }) {
  const side = item.side;
  const text = contentText(item.msg.content);
  if (side === "system") {
    return (
      <div className="card sys">
        <Collapsible
          title={<b>System prompt</b>}
          badges={
            <>
              <Badge>~{Math.ceil(text.length / 4).toLocaleString()} tok</Badge>
              <Where ev={item.ev} />
            </>
          }
        >
          <TextBlock text={text} maxLines={40} />
        </Collapsible>
      </div>
    );
  }
  const label = side === "user" ? "User message" : side === "harness" ? "Harness notice" : side;
  return (
    <div className={`card ${ROLE_TONE[side] || ""}`}>
      <div className="card-head">
        <b>{label}</b>
        <Where ev={item.ev} />
      </div>
      <TextBlock text={text} maxLines={side === "user" ? 30 : 10} />
    </div>
  );
}

function CallCard({ item, convLabel }) {
  const c = item.call;
  const r = item.response;
  const bad = c && (c.refusal || c.stop_reason === "max_tokens");
  return (
    <div className={`card call ${bad ? "bad" : ""}`}>
      <div className="card-head">
        <b>
          {convLabel} · model call #{item.n}
        </b>
        {c?.rung && <Badge>{c.rung}</Badge>}
        {c?.model && <Badge>{c.model}</Badge>}
        {c && (
          <Badge>
            {c.tokens_in.toLocaleString()} in → {c.tokens_out.toLocaleString()} out
          </Badge>
        )}
        {c && <Badge>${c.cost_usd.toFixed(4)}</Badge>}
        {c?.stop_reason && <Badge tone={c.stop_reason === "max_tokens" ? "red" : ""}>stop: {c.stop_reason}</Badge>}
        {c?.refusal && <Badge tone="red">answer refused</Badge>}
        <Where ev={item.ev} />
      </div>
      {c?.refusal && (
        <div className="alert">
          <b>Refused and asked again:</b> {c.refusal}
        </div>
      )}
      {item.sent ? (
        <SentRequest sent={item.sent} />
      ) : (
        <Collapsible title={`Request, rebuilt from the conversation: ${item.request.length} messages`} tone="req">
          {item.compacted && (
            <div className="alert soft">
              A compaction ran before this call: some older tool results were sent as short receipts, not as shown here.
            </div>
          )}
          <div className="muted small">
            This call's exact request was not recorded (a run from before it was), so its tool schemas and settings are not shown.
          </div>
          {item.request.map((m, i) => (
            <RequestMessage key={i} m={m} i={i} />
          ))}
        </Collapsible>
      )}
      <div className="label">reply</div>
      {r ? (
        <>
          {contentText(r.content) ? <Content content={r.content} maxLines={30} /> : <div className="empty">(no text)</div>}
          {(r.tool_calls || []).length > 0 && (
            <div className="calls">
              <div className="label">tool calls ({r.tool_calls.length})</div>
              {r.tool_calls.map((tc) => (
                <ToolCallView key={tc.id} tc={tc} />
              ))}
            </div>
          )}
        </>
      ) : c?.text_preview ? (
        <>
          <div className="muted small">The answer was not added to the conversation; this is the journal's bounded preview.</div>
          <TextBlock text={c.text_preview} />
          {c.tools?.length > 0 && <div className="muted">asked for: {c.tools.join(", ")}</div>}
        </>
      ) : (
        <div className="empty">{c?.tools?.length ? `asked for: ${c.tools.join(", ")}` : "(no reply recorded yet)"}</div>
      )}
      {item.findings && Object.keys(item.findings).length > 0 && (
        <Collapsible title="findings">
          <Json value={item.findings} />
        </Collapsible>
      )}
    </div>
  );
}

function SentRequest({ sent }) {
  return (
    <Collapsible
      tone="req"
      title={`Exact request sent: ${sent.messages.length} messages, ${sent.tools.length} tools`}
      badges={
        <>
          {sent.rewritten && <Badge tone="warn">differs from the conversation</Badge>}
          {sent.forceTool && <Badge>tool call forced</Badge>}
        </>
      }
    >
      {sent.rewritten && (
        <div className="alert soft">
          What the provider got is not the role's conversation as journaled: a client rewrote it before sending (Python calls do),
          or a compaction replaced older results with receipts. Below is what was actually sent.
        </div>
      )}
      <Collapsible title={`tools (${sent.tools.length})`}>
        {sent.tools.length === 0 && <div className="empty">no native tools were sent</div>}
        {sent.tools.map((t) => (
          <Collapsible key={t.name} title={<span className="tool-name">{t.name}</span>}>
            <TextBlock text={t.description} maxLines={20} />
            <div className="label">parameters</div>
            <Json value={t.parameters} />
          </Collapsible>
        ))}
      </Collapsible>
      <Collapsible title="settings">
        <Json value={{ ...sent.params, force_tool: sent.forceTool }} />
      </Collapsible>
      {sent.messages.map((m, i) => (
        <RequestMessage key={i} m={m} i={i} />
      ))}
    </Collapsible>
  );
}

function RequestMessage({ m, i }) {
  const text = contentText(m.content);
  const role = m.role || "?";
  return (
    <Collapsible
      title={
        <span>
          <span className="muted">{i + 1}.</span> <b className={`role-${role}`}>{role}</b>{" "}
          <span className="muted">{text.slice(0, 90).replace(/\n/g, " ")}</span>
        </span>
      }
      badges={m.tool_call_id ? <Badge>{m.tool_call_id}</Badge> : null}
    >
      <TextBlock text={text} maxLines={20} />
      {(m.tool_calls || []).map((tc) => (
        <ToolCallView key={tc.id} tc={tc} />
      ))}
    </Collapsible>
  );
}

function AssistantCard({ item }) {
  const r = item.msg;
  return (
    <div className="card call">
      <div className="card-head">
        <b>Assistant message</b>
        <Where ev={item.ev} />
      </div>
      <Content content={r.content} maxLines={30} />
      {(r.tool_calls || []).map((tc) => (
        <ToolCallView key={tc.id} tc={tc} />
      ))}
    </div>
  );
}

function ToolCard({ item }) {
  const m = item.meta;
  const denied = m?.denied_reason;
  return (
    <div className={`card tool ${denied ? "bad" : ""}`}>
      <div className="card-head">
        <b>
          ← result of <span className="tool-name">{item.call?.name || m?.tool || "?"}</span>
        </b>
        {denied && <Badge tone="red">refused</Badge>}
        {m?.outcome && m.outcome !== "served" && <Badge tone="warn">{m.outcome}</Badge>}
        {m?.dedup_hit && <Badge tone="warn">duplicate call</Badge>}
        {m?.form && m.form !== "native" && <Badge tone="warn">written as {m.form}</Badge>}
        {m?.result_tokens != null && <Badge>{m.result_tokens} tok</Badge>}
        <Where ev={item.ev} />
      </div>
      {denied && (
        <div className="alert">
          <b>Refusal:</b> {denied}
        </div>
      )}
      {item.call && item.call.name !== "python" && (
        <Collapsible title={<span className="muted">arguments: {JSON.stringify(item.call.arguments).slice(0, 120)}</span>}>
          <Json value={item.call.arguments} />
        </Collapsible>
      )}
      {item.inner?.length > 0 && (
        <>
          <div className="label">tool calls the block made ({item.inner.length})</div>
          {item.inner.map((m, i) => (
            <InnerCall key={i} m={m} />
          ))}
        </>
      )}
      <div className="label">what the model was given back</div>
      <Content content={item.msg.content} maxLines={20} />
    </div>
  );
}

function InnerCall({ m }) {
  return (
    <Collapsible
      tone={m.denied_reason ? "bad" : ""}
      title={
        <span>
          <span className="tool-name">{m.tool}</span> <span className="muted">{(m.args_preview || "").slice(0, 110)}</span>
        </span>
      }
      badges={
        <>
          {m.denied_reason && <Badge tone="red">refused</Badge>}
          <Badge>{m.result_tokens} tok</Badge>
        </>
      }
    >
      {m.denied_reason && <div className="alert">{m.denied_reason}</div>}
      <div className="label">arguments (as journaled, bounded)</div>
      <TextBlock text={m.args_preview} />
      <div className="label">result (as journaled, bounded)</div>
      <TextBlock text={m.result_preview} />
    </Collapsible>
  );
}

function ToolMetaRow({ item }) {
  const m = item.meta;
  return (
    <div className={`card tool ${m.denied_reason ? "bad" : ""}`}>
      <div className="card-head">
        <b>
          tool call <span className="tool-name">{m.tool}</span> by {m.role}
        </b>
        {m.denied_reason && <Badge tone="red">refused</Badge>}
        <Where ev={item.ev} />
      </div>
      {m.denied_reason && <div className="alert">{m.denied_reason}</div>}
      {m.args_preview && (
        <>
          <div className="label">arguments (preview)</div>
          <TextBlock text={m.args_preview} />
        </>
      )}
      {m.result_preview && (
        <>
          <div className="label">result (preview)</div>
          <TextBlock text={m.result_preview} />
        </>
      )}
    </div>
  );
}

function PrefixCard({ item }) {
  const prompt = item.ev.prompt;
  return (
    <div className="card sys">
      <div className="card-head">
        <b>Zeus system prompt</b>
        <Badge>{item.ev.payload.model_id}</Badge>
        {prompt && <Badge>~{Math.ceil(prompt.system.length / 4).toLocaleString()} tok</Badge>}
        {prompt && <Badge>{prompt.tools.length} tools</Badge>}
        <Where ev={item.ev} />
      </div>
      {prompt ? (
        <>
          <Collapsible title="system message, exactly as sent">
            <TextBlock text={prompt.system} maxLines={40} />
          </Collapsible>
          <Collapsible title={`tools sent with every request (${prompt.tools.length})`}>
            {prompt.tools.map((t) => (
              <Collapsible key={t.name} title={<span className="tool-name">{t.name}</span>}>
                <TextBlock text={t.description} maxLines={20} />
                <div className="label">parameters</div>
                <Json value={t.parameters} />
              </Collapsible>
            ))}
          </Collapsible>
        </>
      ) : (
        <>
          <div className="alert soft">
            This session began before the prompt's text was recorded; its next turn records it. Below is the template file as it
            is on disk now; the tool list, skills index and environment block that were also sent are not shown.
          </div>
          <ZeusTemplate />
        </>
      )}
      <Collapsible title="part hashes">
        <Json value={item.ev.payload} />
      </Collapsible>
    </div>
  );
}

function ZeusTemplate() {
  const [t, setT] = useState(null);
  useEffect(() => {
    fetch("/api/zeus-template")
      .then((r) => r.json())
      .then(setT)
      .catch(() => setT({ text: "(could not load)" }));
  }, []);
  return (
    <Collapsible title={<span>template: {t?.path?.split("/").slice(-3).join("/") || "zeus.md"}</span>}>
      <TextBlock text={t?.text} maxLines={40} />
    </Collapsible>
  );
}

function EpisodeCard({ item, onOpenRun }) {
  const p = item.ev.payload;
  return (
    <div className="card episode">
      <div className="card-head">
        <b>coder episode started</b>
        <Badge>{p.path}</Badge>
        <Where ev={item.ev} />
      </div>
      <button className="btn" onClick={() => onOpenRun(p.run_id)}>
        open run {p.run_id} (Athena + workers) →
      </button>
    </div>
  );
}

const EVENT_TEXT = {
  task_delegated: (p) => `delegated ${p.task_id}: ${p.title}`,
  task_reported: (p) => `${p.task_id} round ${p.round} ended: ${p.stopped}`,
  task_reviewed: (p) => `${p.task_id} reviewed: ${p.answer}`,
  plan_written: (p) => `plan v${p.version} by ${p.by}: ${p.outcome}`,
  run_start: (p) => `run started: ${p.goal}`,
  run_end: (p) => `run ended: ${p.outcome}`,
  effect_completed: (p) => `${p.operation_id}: ${p.outcome}`,
  exec: (p) => `${p.label} → exit ${p.exit_code}${p.timed_out ? " (timed out)" : ""}`,
  box_compaction: (p) => `compaction (${p.role}): ${p.tokens_before} → ${p.tokens_after} tok`,
};

function EventRow({ item, tone }) {
  const ev = item.ev;
  const p = ev.payload || {};
  const text = EVENT_TEXT[ev.kind]?.(p) || ev.kind;
  const blobs = item.ev.blobs || {};
  return (
    <div className={`event ${tone || ""}`}>
      <Collapsible
        title={
          <span>
            <span className="kind">{ev.kind}</span> {text !== ev.kind ? text : ""}
          </span>
        }
        badges={<Where ev={ev} />}
      >
        {Object.entries(blobs).map(([k, v]) => (
          <div key={k}>
            <div className="label">{k} (blob)</div>
            <TextBlock text={v} maxLines={30} />
          </div>
        ))}
        <Json value={p} />
      </Collapsible>
    </div>
  );
}
