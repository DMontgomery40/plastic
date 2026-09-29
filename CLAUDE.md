# plastic: shared repository instructions

@AGENTS.md

Use `AGENTS.md` as the shared project guidance for Claude and Codex. Read the latest
`SHARED_SCRATCHPAD.md` before concurrent work and honor the ownership recorded there.

Before substantive work, complete the mandatory research briefing and readiness
note in `AGENTS.md`. Apply its design and completion gates; familiar methods and
remembered literature are not substitutes for checked evidence.

For checkpoint publication and Space access, follow the **GitHub and Hugging Face
are one maintained project** section of `AGENTS.md`: upload actual release weights
to the Hub even though local artifacts are gitignored, verify their download and
preserve them through source sync. The user has enabled the Space's Dev Mode for
SSH/development access; use its current connection details and bring live source
changes back into GitHub. These rules apply equally to Claude and Codex.
When the user requests Space validation, run the tests inside that Space over SSH,
not on the local Mac.
