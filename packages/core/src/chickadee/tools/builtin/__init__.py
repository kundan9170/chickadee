"""The built-in tools: one module each, one line each in `BUILTINS`.

Adding a tool is a module here, a description template under
`adapters/<family>/templates/tools/`, and a line below; it passes `test_conformance.py`, and a
config may then enable it for any role whose ceiling admits it (decision 15). No dispatcher,
name list, renderer, parser or journal kind is edited.

Zeus's seventeen register here (`grep` and `glob` where `find` was), the reads
among them the ones the scout and the box roles share (one implementation,
not three forks); then the scout's own `request_exec`; then the edit tools,
which write the task's worktree through the `edits` port, the
line tools and the symbol tools among them. The packet worker's
four reads retired with the packet loop, and the edit tools lost their `packet_` prefix.
"""

from __future__ import annotations

from chickadee.tools.builtin import (
    apply_patch,
    ask_user,
    atlas,
    coder,
    create,
    delegate,
    delete,
    doc,
    edit,
    find_skill,
    finish,
    follow_up,
    glob,
    grep,
    grep_in,
    impact,
    insert_after_symbol,
    insert_lines,
    job,
    load_skill,
    locate,
    move,
    neighbors,
    note,
    note_concern,
    note_rejected,
    outgrown,
    pause,
    plan,
    read,
    recall,
    replace_lines,
    replace_symbol,
    reply,
    request_exec,
    review,
    shell,
    spawn_agent,
    symbol,
    tell,
    tests_for,
    tools,
    web_fetch,
    web_search,
    worktree,
    write,
)
from chickadee.tools.registry import Registry

__all__ = ["BUILTINS"]

BUILTINS = Registry(
    (
        atlas.TOOL,
        grep.TOOL,
        glob.TOOL,
        symbol.TOOL,
        neighbors.TOOL,
        impact.TOOL,
        recall.TOOL,
        locate.TOOL,
        spawn_agent.TOOL,
        follow_up.TOOL,
        tools.TOOL,
        read.TOOL,
        grep_in.TOOL,
        load_skill.TOOL,
        find_skill.TOOL,
        shell.TOOL,
        job.TOOL,
        coder.TOOL,
        ask_user.TOOL,
        note_concern.TOOL,
        note.TOOL,
        note_rejected.TOOL,
        request_exec.TOOL,
        write.TOOL,
        edit.TOOL,
        finish.TOOL,
        apply_patch.TOOL,
        replace_lines.TOOL,
        insert_lines.TOOL,
        replace_symbol.TOOL,
        insert_after_symbol.TOOL,
        create.TOOL,
        delete.TOOL,
        move.TOOL,
        tests_for.TOOL,
        plan.TOOL,
        delegate.TOOL,
        review.TOOL,
        outgrown.TOOL,
        reply.TOOL,
        doc.TOOL,
        worktree.TOOL,
        web_fetch.TOOL,
        web_search.TOOL,
        pause.TOOL,
        tell.TOOL,
    )
)
"""In their registered order, Zeus's sixteen (`SURFACE`), then the scout's own two,
then the edit tools: three by text, two by line, two by
symbol; then the task worker's: `create`, `delete`, `move` and
`tests_for`; then Athena's task tools: `plan`, `delegate` and `review`,
`outgrown` and `reply`, over the `desk`; then `doc`, Zeus's and Athena's documents over `docs`;
then `worktree`,
Zeus's moves between the checkout and a worktree over `worktree`; then the researcher's
`web_fetch` and `web_search`, over `web`; then `pause` and `tell`,
Zeus's over `team`."""
