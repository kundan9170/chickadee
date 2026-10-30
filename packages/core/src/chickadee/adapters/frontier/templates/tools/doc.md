Write a document of the work: `op` is write (a whole file), edit (exact text) or remove, and `file` names it from your folder.

`file` is named from your folder: `README.md`, or `tasks/T1-add-retry.md`. A path that leaves it
is refused, so a document is never written over the code. Your folder's `README.md` is its
entry point and is required; other files are detail it links to. The first call of a design
gives the work its `title`, which names the folder; every later call, and every call that writes
a plan, writes into the same one and gives no title. To add lines, use op edit with `old` the line
they follow and `new` that line followed by them.
