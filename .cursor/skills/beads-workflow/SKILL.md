---
name: beads-workflow
description: Uses bd (beads) for issue tracking. Use when finding work, claiming or closing issues, syncing with git, or when the user mentions bd, beads, issues, or tasks.
---

# Beads (bd) Issue Tracking

This project uses [Beads (bd)](https://github.com/steveyegge/beads) for issue tracking.

## Core rules

- Track ALL work in bd (never use markdown TODOs or comment-based task lists).
- Use `bd ready` to find available work.
- Use `bd create` to track new issues/tasks/bugs.
- Use `bd sync` at end of session to sync with git remote.
- Git hooks auto-sync on commit/merge.

## Quick reference

```bash
bd onboard                              # Get started / load workflow context
bd prime                                # Load complete workflow docs (AI-optimized)
bd ready                                # Show issues ready to work (no blockers)
bd list --status=open                   # List all open issues
bd create --title="..." --type=task     # Create new issue
bd show <id>                            # View issue details
bd update <id> --status=in_progress     # Claim work
bd close <id>                           # Mark complete
bd dep add <issue> <depends-on>         # Add dependency
bd export -o .beads/issues.jsonl        # Write the JSONL git tracks (see below)
bd sync                                 # Sync with git remote
```

## Writes are not visible in git until you export — with `-o`

`bd` keeps issues in a local Dolt database. `.beads/issues.jsonl` is a passive
export of it, and that file is what git tracks and what other tools read. A
`bd close` updates the database immediately; **it does not update the JSONL.**

In bd 1.1.2 the export API changed: plain `bd export` writes to **stdout**.
Writing the file needs `-o`:

```bash
bd export -o .beads/issues.jsonl     # correct
bd export                            # dumps JSONL to your terminal, changes nothing
```

Get this wrong and the symptom is not an error — it is `bd close` reporting
success while `git status` shows nothing and the JSONL still says `open`. That
looked exactly like bd silently losing ~50% of writes, which is how it came to
be filed twice as a bug (`fix-46c`, `fix-fyw`) before the cause was found.

**So: run `bd export -o .beads/issues.jsonl` after any batch of writes, and
verify by reading the file rather than trusting the command's output.**

```bash
python3 -c "
import json
for l in open('.beads/issues.jsonl'):
    d = json.loads(l)
    if d['id'] == 'fix-abc': print(d['status'])
"
```

Two related traps:

- The default export **excludes memories** (`bd remember`) and infra beads.
  That matches how `.beads/issues.jsonl` was generated, so plain
  `bd export -o` is the right call — do **not** add `--all` unless you intend
  to start tracking those, because it will add records the file never had.
- `bd export` reads the database, so it cannot rescue a write that never
  reached it. If a close is missing from the JSONL *after* an export, check
  `bd show <id>` before assuming the export failed.

## Workflow

1. Check for ready work: `bd ready`
2. Claim an issue: `bd update <id> --status=in_progress`
3. Do the work
4. Mark complete: `bd close <id>`
5. **Export: `bd export -o .beads/issues.jsonl`** (see above — this is the step
   whose omission looks like data loss)
6. Sync: `bd sync` (or let git hooks handle it)

## Context loading

- **New to project:** run `bd onboard`
- **Full workflow docs:** run `bd prime` for AI-optimized context (~1–2k tokens)

For more: see AGENTS.md or `bd --help`.
