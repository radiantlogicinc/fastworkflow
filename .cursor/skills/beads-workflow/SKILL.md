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
bd sync                                 # Sync with git remote
```

## Workflow

1. Check for ready work: `bd ready`
2. Claim an issue: `bd update <id> --status=in_progress`
3. Do the work
4. Mark complete: `bd close <id>`
5. Sync: `bd sync` (or let git hooks handle it)

## Context loading

- **New to project:** run `bd onboard`
- **Full workflow docs:** run `bd prime` for AI-optimized context (~1–2k tokens)

For more: see AGENTS.md or `bd --help`.
