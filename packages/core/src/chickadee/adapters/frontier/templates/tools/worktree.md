Move this conversation's work between the user's checkout and a separate worktree, when the user asks. `action` is enter, leave, land or discard.

The work runs in the user's checkout by default. A worktree is a separate copy of the repository on a branch of its own, for work the user wants kept apart from their checkout until they choose to bring it in.

- `enter`: work in this conversation's worktree from now on, a new one or the one it already holds. Call it only when the user asks for separate work ("in a worktree", "on the side", "try it separately"). When their words did not say worktree, the user is asked first.
- `leave`: back to the user's checkout. The worktree and its work are kept; `enter` returns to them, and a later conversation can continue them.
- `land`: bring the worktree's work into the user's checkout, when the user asks for it ("bring it back", "merge it in"). `how` is `bring`, its changes written into the checkout uncommitted for the user to review and commit (the default); `apply`, its commits put on the user's branch as they are, when the user asks to keep the commits; or `squash`, one commit, when the user asks for one. The user approves it first.
- `discard`: drop the worktree and its work, when the user asks. The user approves it first, and a snapshot is kept.

A request for a branch in the checkout ("make a branch", "merge it into main") is git through `shell(`, not this tool. Never land or discard on your own judgment, and never enter a worktree the user did not ask for.
