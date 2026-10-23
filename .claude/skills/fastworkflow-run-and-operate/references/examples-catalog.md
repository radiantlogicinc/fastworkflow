# Bundled examples catalog (verified 2026-07-09, v2.22.2, commit c33b9a5)

Eight examples ship inside the package at `fastworkflow/examples/` (verified via
`fastworkflow examples list` and `ls fastworkflow/examples/`). `fastworkflow examples fetch
<name>` copies one to `./examples/<name>` (the repo-root `examples/` dir is gitignored) and drops
`fastworkflow.env` + `fastworkflow.passwords.env` templates into `./examples/`.

Universal loop per example:

```sh
fastworkflow examples fetch <name>
fastworkflow train ./examples/<name> ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
fastworkflow run   ./examples/<name> ./examples/fastworkflow.env ./examples/fastworkflow.passwords.env
```

Read them in this order — each introduces exactly one new concept.

## 1. hello_world — the minimum viable workflow
- App: `application/add_two_numbers.py` (one function).
- Commands: `_commands/add_two_numbers.py` — a single "global" command (no context class).
- Teaches: the command-file anatomy (Signature, validate_extracted_parameters, ResponseGenerator)
  and the train→run loop. `_commands/README.md` inside it explains the layout.

## 2. messaging_app_1 — smallest hand-written workflow
- App: `application/send_message.py` (plain function).
- Commands: `_commands/send_message.py` (global) + `context_inheritance_model.json`.
- Teaches: writing a command by hand instead of `fastworkflow build`.

## 3. messaging_app_2 — class-based command context
- Commands: `_commands/User/send_message.py` + `_commands/startup.py`.
- Teaches: contexts = your application classes. A command under `_commands/User/` is only
  available when the current context is a `User`; `startup.py` binds a `User` instance as the
  root context at session start.

## 4. messaging_app_3 — context inheritance
- Commands: `_commands/User/send_message.py`, `_commands/PremiumUser/send_priority_message.py`,
  global `_commands/initialize_user.py`; `context_inheritance_model.json` declares
  `PremiumUser` extends `User`.
- Teaches: a `PremiumUser` context sees BOTH `send_message` (inherited) and
  `send_priority_message`.

## 5. messaging_app_4 — container context + navigation
- Commands: `_commands/ChatRoom/` (`add_user.py`, `broadcast_message.py`, `list_users.py`,
  `set_current_user.py`, `get_current_user.py`) plus per-context callback modules
  `_ChatRoom.py` / `_User.py` / `_PremiumUser.py`, global `set_root_context.py`, and a
  `startup_action.json` consumed via `--startup_action`.
- Also the only bundled example with a workflow-root `context_hierarchy_model.json` — the
  SECOND context-model file (parent/containment ancestry), distinct from
  `_commands/context_inheritance_model.json`. Largely undocumented elsewhere; the other in-tree
  user is `tests/todo_list_workflow/`.
- Teaches: navigating between contexts (ChatRoom contains Users; `_<ContextName>.py` classes
  expose e.g. `get_parent` for navigation).

## 6. retail_workflow — the tau-bench retail domain
- 15 global commands in `_commands/`: calculate, cancel_pending_order,
  exchange_delivered_order_items, find_user_id_by_email, find_user_id_by_name_zip,
  get_order_details, get_product_details, get_user_details, list_all_product_types,
  modify_pending_order_address, modify_pending_order_items, modify_pending_order_payment,
  modify_user_address, return_delivered_order_items, transfer_to_human_agents.
- `tools/` holds the underlying tau-bench tool implementations (plus `tool.py` base and
  `think.py`, which is a tau-bench tool not exposed as a command here); `retail_data/` holds the
  JSON fixtures; `workflow_description.txt` the domain description.
- Teaches: a realistic multi-command domain, and it IS the in-repo half of the tau-bench
  integration (the harness/adapter lives in the tau-bench fork — see
  `fastworkflow-taubench-reference`).
- **Parity rule:** never modify these tools/tasks for benchmark runs; disclose any nonstandard
  trade. Changes here are change-control matters (`fastworkflow-change-control`).

## 7. simple_workflow_template — JSON-driven workflow
- `simple_workflow_template.json` defines a WorkItem type hierarchy over
  `application/workitem.py`; `_commands/WorkItem/` has the full navigation/manipulation command
  set (add/remove child, go_to, move_to_next/previous/first/last, mark_as_complete, get_status,
  show_schema) plus `_WorkItem.py`; `startup_action.json` included.
- Teaches: template workflows driven by a schema file rather than bespoke classes.

## 8. extended_workflow_example — workflow inheritance
- `workflow_inheritance_model.json`: `{"base": ["fastworkflow.examples.simple_workflow_template"]}`.
- Demonstrates all three extension moves (its README.md documents them):
  1. **Override**: `_commands/startup.py` replaces the base startup while calling into it.
  2. **New command**: `_commands/generate_report.py`.
  3. **Wrapper**: `_commands/WorkItem/get_status.py` wraps the base command.
- Precedence is last-wins along the inheritance chain.

## Not an example but referenced constantly
- `tests/hello_world_workflow/` and `tests/todo_list_workflow/` are git-tracked TEST
  workflows; `tests/example_workflow/` is mostly GITIGNORED (`.gitignore:7` — only 6 files
  tracked, so fresh clones lack its `_commands/` tree; see `fastworkflow-validation-and-qa`
  §3) (CLAUDE.md omits `todo_list_workflow` — doc rot). There is NO
  `fastworkflow/examples/todo_list` anymore (moved to tests in commit d679c38); a stale root
  `examples/todo_list` fetch copy may exist locally.
