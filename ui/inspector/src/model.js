// Turns journal lines into what the page shows: one conversation per role, in order, with
// each model call joined to the answer it produced and each tool result to the call that
// asked for it. Pure functions, so the views only render.

const TASK_KINDS = new Set(["task_delegated", "task_reported", "task_reviewed"]);

export function contentText(content) {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .map((p) => (p.text != null ? p.text : `[image ${p.image}, ${p.bytes} bytes]`))
    .join("\n");
}

function newConv(key, label, role, task) {
  return {
    key,
    label,
    role,
    task,
    items: [],
    messages: [],
    shas: [],
    callsById: {},
    pending: null,
    stats: { calls: 0, tokensIn: 0, tokensOut: 0, cost: 0, refusals: 0, denied: 0 },
  };
}

// --- a run: Athena and each worker task -------------------------------------------------

export function buildRun(events, tables = { messages: {}, tools: {} }) {
  const convs = new Map();
  const timeline = [];
  const conv = (role, task) => {
    const key = role === "worker" ? `worker:${task ?? "?"}` : role;
    if (!convs.has(key)) {
      const label = role === "worker" ? `Worker ${task ?? ""}` : cap(role);
      convs.set(key, newConv(key, label, role, task));
    }
    return convs.get(key);
  };
  conv("athena", null); // always first

  for (const ev of events) {
    timeline.push(ev);
    const p = ev.payload || {};
    const task = ev.task_id;
    switch (ev.kind) {
      case "box_message": {
        const c = conv(p.role, p.role === "worker" ? task : null);
        const msg = ev.message || { role: p.side, content: "[message blob missing]" };
        if (p.side === "assistant") {
          for (const tc of msg.tool_calls || []) c.callsById[tc.id] = tc;
          if (c.pending && !c.pending.response) {
            c.pending.response = msg;
          } else {
            c.items.push({ type: "assistant", ev, msg });
          }
          c.pending = null;
        } else if (p.side === "tool") {
          const call = c.callsById[msg.tool_call_id];
          const meta = takeMeta(c, call?.name);
          // A Python block's calls (PR 641) are journaled one by one before its result.
          const inner = call?.name === "python" ? takeInner(c) : null;
          c.items.push({ type: "tool", ev, msg, call, meta, inner });
        } else {
          c.items.push({ type: "message", side: p.side, ev, msg });
        }
        c.messages.push(msg);
        c.shas.push(p.sha);
        break;
      }
      case "model_call": {
        const c = conv(p.role, p.role === "worker" ? task : null);
        const s = c.stats;
        s.calls += 1;
        s.tokensIn += p.tokens_in;
        s.tokensOut += p.tokens_out;
        s.cost += p.cost_usd;
        if (p.refusal) s.refusals += 1;
        const item = {
          type: "call",
          ev,
          call: p,
          n: s.calls,
          response: null,
          request: [...c.messages],
          compacted: c.items.some((i) => i.type === "compaction"),
          sent: ev.sent
            ? {
                messages: ev.sent.messages.map((sha) => tables.messages[sha] || { role: "?", content: `[message ${sha.slice(0, 12)} missing]` }),
                tools: tables.tools[ev.sent.tools] || [],
                params: ev.sent.params,
                forceTool: ev.sent.force_tool,
                rewritten: !sameList(ev.sent.messages, c.shas),
              }
            : null,
        };
        c.items.push(item);
        c.pending = item;
        break;
      }
      case "tool_call": {
        const c = conv(p.role, p.role === "worker" ? task : null);
        if (p.denied_reason) c.stats.denied += 1;
        // Held until its result message arrives, which then carries it.
        c.items.push({ type: "toolmeta", ev, meta: p, merged: false });
        break;
      }
      case "box_compaction": {
        conv(p.role, p.role === "worker" ? task : null).items.push({ type: "compaction", ev });
        break;
      }
      default: {
        const item = { type: "event", ev };
        if (TASK_KINDS.has(ev.kind) && task) {
          conv("worker", task).items.push(item);
          conv("athena", null).items.push(item);
        } else if (task) {
          conv("worker", task).items.push(item);
        } else {
          conv("athena", null).items.push(item);
        }
      }
    }
  }
  for (const c of convs.values()) c.items = c.items.filter((i) => !(i.type === "toolmeta" && i.merged));
  return { convs: [...convs.values()].filter((c) => c.items.length), timeline };
}

function sameList(a, b) {
  return a.length === b.length && a.every((x, i) => x === b[i]);
}

function takeInner(c) {
  const inner = [];
  for (let i = c.items.length - 1; i >= 0; i--) {
    const it = c.items[i];
    if (it.type === "call") break;
    if (it.type === "toolmeta" && !it.merged && it.meta.tool !== "python") {
      it.merged = true;
      inner.unshift(it.meta);
    }
  }
  return inner;
}

function takeMeta(c, toolName) {
  for (let i = c.items.length - 1; i >= 0; i--) {
    const it = c.items[i];
    if (it.type === "toolmeta" && !it.merged && (!toolName || it.meta.tool === toolName)) {
      it.merged = true;
      return it.meta;
    }
  }
  return null;
}

// --- a session: Zeus ---------------------------------------------------------------------

export function buildSession(events) {
  const c = newConv("zeus", "Zeus", "zeus", null);
  for (const ev of events) {
    const p = ev.payload || {};
    switch (ev.kind) {
      case "prefix":
        c.items.push({ type: "prefix", ev });
        break;
      case "turn":
        if (p.role === "user") {
          const msg = { role: "user", content: p.text };
          c.items.push({ type: "message", side: "user", ev, msg });
          c.messages.push(msg);
        } else if (p.role === "assistant") {
          const msg = {
            role: "assistant",
            content: p.text,
            tool_calls: (p.calls || []).map((x) => ({ id: x.id, name: x.name, arguments: x.arguments })),
          };
          for (const tc of msg.tool_calls) c.callsById[tc.id] = tc;
          c.stats.calls += 1;
          c.items.push({
            type: "call",
            ev,
            n: c.stats.calls,
            call: null,
            response: msg,
            findings: p.findings,
            request: [...c.messages],
          });
          c.messages.push(msg);
        } else {
          const msg = { role: "harness", content: p.text };
          c.items.push({ type: "message", side: "harness", ev, msg });
          c.messages.push(msg);
        }
        break;
      case "tool": {
        const msg = { role: "tool", content: p.text, tool_call_id: p.call_id };
        c.items.push({ type: "tool", ev, msg, call: c.callsById[p.call_id], meta: p.ran === false ? { denied_reason: "not run" } : null });
        c.messages.push(msg);
        break;
      }
      case "spend": {
        const s = c.stats;
        s.tokensIn += p.tokens_in;
        s.tokensOut += p.tokens_out;
        s.cost += p.cost_usd;
        // A spend line charges the call just made: put it on that card.
        const last = [...c.items].reverse().find((i) => i.type === "call");
        if (last && !last.call) last.call = { ...p, role: "zeus", stop_reason: null };
        else c.items.push({ type: "event", ev });
        break;
      }
      case "frame":
        c.items.push({ type: "message", side: "harness", ev, msg: { content: p.text } });
        break;
      case "episode":
        c.items.push({ type: "episode", ev });
        break;
      default:
        c.items.push({ type: "event", ev });
    }
  }
  return c;
}

// --- filtering ---------------------------------------------------------------------------

export function isProblem(item) {
  const call = item.call;
  if (item.type === "call" && call && (call.refusal || call.stop_reason === "max_tokens")) return true;
  if ((item.type === "tool" || item.type === "toolmeta") && item.meta?.denied_reason) return true;
  if (item.type === "tool" && item.inner?.some((m) => m.denied_reason)) return true;
  if (item.type === "call" && item.response?.tool_calls?.some((t) => t.name === "python" && t.arguments?.error)) return true;
  if (item.type === "event") {
    const p = item.ev.payload || {};
    if (item.ev.kind === "task_reported" && p.stopped !== "replied") return true;
    if (item.ev.kind === "task_reviewed" && p.answer !== "accept") return true;
    if (item.ev.kind === "effect_completed" && p.outcome !== "succeeded") return true;
    if (item.ev.kind === "exec" && (p.exit_code !== 0 || p.timed_out)) return true;
    if (item.ev.kind === "callback_error" || item.ev.kind === "budget") return true;
    if (item.ev.kind === "run_end" && p.outcome !== "COMPLETED") return true;
  }
  return false;
}

export function itemText(item) {
  const parts = [item.ev?.kind, JSON.stringify(item.ev?.payload)];
  if (item.msg) parts.push(contentText(item.msg.content));
  if (item.response) {
    parts.push(contentText(item.response.content));
    parts.push(JSON.stringify(item.response.tool_calls || []));
  }
  if (item.call) parts.push(JSON.stringify(item.call));
  return parts.join("\n").toLowerCase();
}

function cap(s) {
  return s ? s[0].toUpperCase() + s.slice(1) : s;
}
