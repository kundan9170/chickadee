Hand one job to a fresh agent and get its report back. `task` is required, and `skill` or `profile`.

The agent is a cheaper model in a context of its own: it sees your `task` and the results whose
`handles` you pass, whole, and nothing else of this conversation. Name a `skill` to run an agent
skill (its own method, profile and tools), or a `profile` to run one with no skill: `reader`
reads and searches the repository; `reviewer` also finds the tests that cover a file and
runs read-only commands; `researcher` searches and reads the web, and nothing of the repository:
give it no `handles`, and put what it must know in the task in your own words. These three are
profiles, not skills: send `profile: "researcher"`, never `skill: "researcher"`, and do not look
for them with `find_skill`. No agent edits a file, and one that would need to cannot.

Write the task as four things: what to find out, what to return, where to look, and what is out
of scope. A vague task wastes the agent's calls. The report has a summary, findings each citing
`file:lines`, and what was not done; each citation was checked against the file, and the report
says how many passed and flags the rest. Its claims are the agent's: weigh them as you would a
colleague's notes. A report marked `partial` ran out of calls, time or spend; one marked `failed`
says why. Several `spawn_agent` calls in one response run at once, up to a cap: hand
independent jobs out together, and the reports come back in the order you asked. To ask an
agent for more once it has reported, use `follow_up`, which keeps its conversation.
