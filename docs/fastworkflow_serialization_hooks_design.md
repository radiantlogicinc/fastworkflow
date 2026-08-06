# Workflow serialization hooks

**Status:** **Approved 2026-08-05.** All five §12 decisions settled. Not implemented.
**Scope:** How a workflow author tells fastWorkflow how to persist and restore application state
**Tracking:** `fix-qyz` (this design), `fix-gbh` (reference implementations), `fix-g03.13` (the decision that produced it)
**Deferred from here:** `fix-cj4` (redaction of secrets and PII)
**Blocks:** `fix-g03.17` and all Release B persistence work
**Verified against:** fastWorkflow `2.25.0`, commit `a4ba82d`

> This design exists because decision 25 of the memory-bounds design was reversed on 2026-08-05. That
> document is stale in §11.1, §11.2 and decision-log entries 18 and 25, and gets a revision 4 once this
> mechanism is approved.

---

## 1. What this has to do

Two jobs, and the second one is easy to overlook.

**Job 1 — let an author extend or override serialization.** The accepted model is that fastWorkflow
persists all JSON-native workflow context by default, and the author can extend or override that.

**Job 2 — make a workflow evictable at all.** This is the larger job. A session is pinned — never
evicted, so never bounded — when its workflow holds a live command-context object. Those objects are
ordinary Python instances (`TodoListManager`, a `WorkItem` tree), so persist-all-JSON never reaches
them. The hook is the only thing that can. Five bundled workflows are pinned today, including
`simple_workflow_template`, the scaffold every new workflow is copied from.

The stakes are higher than "unbounded memory". Measured on `tests/todo_list_workflow` (`fix-3vy`): a
session that is evicted today comes back with `root_command_context = None` and current context `*`,
silently. Commands then resolve against the global context. It is rare only because it needs more than
2,000 live sessions; lowering that cap is exactly what Release C does.

### 1.1 Requirements

From the adjudication (`fix-g03.13`):

1. Hooks let the author **extend or override** serialization of their own state.
2. **Warn** when a workflow that needs a hook has not implemented one.
3. An explicit **no-op does not warn**. A no-op is the author saying "I know, and I choose nothing."
   Silence is consent; absence is not.
4. Follow **well-known, established patterns** rather than inventing a bespoke one.
5. **Approval before implementation.**

Requirement 6 — a default redaction pass over persisted context — was proposed here and **deferred**
(`fix-cj4`). Until it lands, the boundary is documentation plus one framework-owned exclusion; see §6.2.

---

## 2. What the framework already gives us

fastWorkflow already has an author-supplied-hook idiom, and this design should use it rather than
introduce a second one.

A context class lives at `_commands/<ContextName>/_<ContextName>.py` and is always named `Context`.
It is discovered by `CommandContextModel.get_context_class(name, ModuleType.CONTEXT_CLASS)`
(`command_context_model.py:272-297`), keyed on the **live object's class name**
(`Workflow.get_command_context_name`, `workflow.py:248-252`). A missing file returns `None` silently.

Two optional hooks exist today, and they differ in a way worth copying:

| Hook | Guard | Absent behaviour |
|---|---|---|
| `get_displayname` | `hasattr` (`workflow.py:193`) | falls back to the class name |
| `get_parent` | none (`workflow.py:230`) | file missing → `None`; file present but method missing → `AttributeError` |

`get_displayname` is the better precedent: optional, `hasattr`-guarded, graceful. `get_parent`'s
unguarded call is a latent `AttributeError` and should not be imitated.

A real one, from the scaffold (`simple_workflow_template/_commands/WorkItem/_WorkItem.py`):

```python
class Context:
    @classmethod
    def get_parent(cls, command_context_object: WorkItem) -> Optional[WorkItem]:
        return command_context_object.parent or command_context_object

    @classmethod
    def get_displayname(cls, command_context_object: WorkItem) -> str:
        return f'{command_context_object.__class__.__name__}: {command_context_object.get_absolute_path()}'
```

This is what a fastWorkflow author already writes. The serialization hook should look like it belongs
next to these.

### 2.1 The shapes we actually have to support

Surveyed across the five pinned workflows, because the shape determines the design:

| Workflow | Anchor | Does `current` move off the anchor? | Callback file today |
|---|---|---|---|
| `simple_workflow_template` | `root` = `WorkItem` tree | **yes**, to a sub-workitem | `_WorkItem.py` |
| `tests/todo_list_workflow` | `root` = `TodoListManager` | **yes**, to `TodoList` / `TodoItem` | all three present |
| `messaging_app_4` | `root` = `ChatRoom` | **yes**, to a `User` | all three present |
| `messaging_app_2` | `root` = `User` | no | **none** |
| `messaging_app_3` | **no root**; sets `current` only | n/a | **none** |

Three of five point `current_command_context` at a node *inside* a graph the framework did not build.
So identity inside a restored graph is the normal case, not a corner. And `messaging_app_3` has no root
at all, so the design cannot assume one.

---

## 3. Prior art, and what we take from it

| Pattern | Shape | Why not wholesale |
|---|---|---|
| `__getstate__` / `__setstate__` (pickle protocol) | instance dump, in-place restore onto an uninitialised instance | Dunders on the **application** class change `pickle` and `copy` behaviour for the author's own class — a side effect outside our remit. Restore also assumes framework-controlled allocation. |
| `__reduce__` / `copyreg` | callable + args | Same side-effect problem, and expresses "how to call the constructor", which is more than we need. |
| cattrs `register_structure_hook` / `register_unstructure_hook` | external converter registry | No side effects on the class — good — but needs a registration bootstrap fastWorkflow does not have. |
| pydantic `model_serializer` / `model_validator` | decorators on the model | Only applies if the state is a pydantic model. Ours is arbitrary. |
| Django `natural_key()` + `get_by_natural_key()` | instance dump + **manager-side factory** | Closest structural match: restore needs a factory, not an instance method. |
| Java `Externalizable`, .NET `ISerializable` | explicit write/read pair, class-side reconstruction | Same conclusion: a pair, with construction on the class side. |

Every one of them converges on the same two-part shape: **a dump, and a class-side factory** — because
restoring has to create the object. What differs is *where* the pair lives. We take cattrs' answer
(off the application class, so no side effects) and Django's (a class-side factory), and express both
through the `Context` class fastWorkflow already uses for exactly this kind of author callback.

---

## 4. The mechanism

### 4.1 The state pair

On the `Context` class the author already writes:

```python
class Context:
    state_version = 1

    @classmethod
    def get_state(cls, command_context_object) -> dict[str, Any] | fastworkflow.Ephemeral:
        """Return a JSON-native snapshot of this object, or fastworkflow.EPHEMERAL to decline."""

    @classmethod
    def from_state(
        cls,
        state: dict[str, Any],
        workflow: fastworkflow.Workflow,
        *,
        state_version: int,
    ) -> object:
        """Rebuild the object this Context describes, from get_state's output.

        state_version is the version the snapshot was WRITTEN with, which may be
        older than cls.state_version. Migrating it is the author's job (§5).
        """
```

`get_state` / `from_state` echoes `__getstate__` / `__setstate__` so the naming is familiar, fits the
repo's `get_*` convention, and avoids the dunder side effects. `from_state` is a classmethod factory
for the Django/cattrs reason: restoring constructs.

`workflow` is passed to `from_state` because reconstruction usually needs it —
`todo_list_workflow` builds `TodoListManager(filepath)` from `workflow.folderpath`, and
`simple_workflow_template` loads its schema the same way.

### 4.2 The trichotomy — how "no-op" differs from "absent"

This is requirement 3, and it needs a positive signal, since a method that returns nothing is
indistinguishable from a method that isn't there.

| Author wrote | Framework does | Warns? |
|---|---|---|
| **Nothing** — no `_<Ctx>.py`, or no `get_state` on it | session **pins**; never evicted | **yes**, once per context class per process |
| `get_state` returning **`fastworkflow.EPHEMERAL`** | session **pins**; never evicted | **no** — the author declared it |
| `get_state` returning a dict | serialize; session is **evictable** | no |

`EPHEMERAL` is a module-level sentinel (`fastworkflow.EPHEMERAL`), not `None`, so a hook that falls off
the end of a function and implicitly returns `None` is a bug the framework can catch rather than
silently read as consent. Returning `None` is an error: it pins **and** warns, with a message saying to
return `EPHEMERAL` if declining was intended.

Presence of the method is the consent signal. That is the whole trick, and it is why the hook is a
method rather than a return-value convention on an existing one.

### 4.3 Identity inside a restored graph

The framework serializes exactly one object — the **anchor**: `root_command_context` if set, otherwise
`current_command_context` (which is what `messaging_app_3` needs). `current_command_context` and
`command_context_for_response_generation` are then stored as **locators relative to the anchor**, never
as second snapshots. Two independent snapshots of the same node would restore as two objects, and
`workflow.current_command_context` would no longer be a node inside `workflow.root_command_context` —
the identity failure §11.1 of the memory-bounds design warns about.

When a slot equals the anchor, its locator is a reserved `"."` and no hook is involved. When it differs:

```python
class Context:
    @classmethod
    def get_locator(cls, command_context_object) -> str:
        """A stable path identifying this object within its anchor."""

    @classmethod
    def find_by_locator(cls, anchor, locator: str) -> object | None:
        """Resolve a locator produced by get_locator, against a restored anchor."""
```

The locator pair is needed **only** when a slot differs from the anchor. If it does and the pair is
absent, the session pins and warns — same rule, applied to a smaller question.

This is nearly free for the workflows that need it. `simple_workflow_template` already has
`get_absolute_path()` (its `get_displayname` uses it), and `todo_list_workflow` has a parent chain.

**Self-check at serialize time.** Before trusting a locator, the framework asserts

```
find_by_locator(anchor, get_locator(obj)) is obj
```

If that fails, the session pins and logs an ERROR naming the context class. A locator that doesn't
round-trip is a bug that would otherwise surface much later as a silently wrong current context, so it
is worth one cheap identity check per eviction.

### 4.4 The workflow-context projection

Everything above is about command-context **objects**. The `workflow.context` **dict** persists by
default. An author extends or overrides that through one optional module, `_commands/_serialization.py`
— a `_`-prefixed file, which the command scanner already ignores
(`command_directory.py:218-270`), so it introduces no collision:

```python
class Serialization:
    state_version = 1

    @classmethod
    def get_workflow_state(cls, workflow) -> dict[str, Any] | fastworkflow.Ephemeral:
        """Override the default projection of workflow.context entirely."""

    @classmethod
    def redact(cls, workflow_context: dict[str, Any]) -> dict[str, Any]:
        """Extend the framework's default redaction (see §6)."""
```

Both are optional and neither warns when absent, because the default (persist-all, redacted) is the
accepted behaviour rather than a gap. This is the "extend or override" half of requirement 1; the
`Context` pair is the half that unpins the session.

---

## 5. Versioning and migration

**The author owns migration; the framework's job is to hand over enough information to do it.**

An earlier draft of this design had the framework refuse to call `from_state` on a version mismatch.
That was wrong: it makes migration impossible by construction, since the only code that understands an
old state shape is the code the author would write. Every pattern in §3 puts migration on the author —
`__setstate__` receives whatever `__getstate__` produced and inspects it; Django migrations are
authored; Avro and protobuf define reader/writer rules the application resolves.

So the stored version is passed through:

```python
@classmethod
def from_state(cls, state: dict[str, Any], workflow, *, state_version: int) -> object:
    if state_version < 2:
        state = cls._migrate_v1_to_v2(state)
    return WorkItem.from_state_dict(state, ...)
```

An author who cannot migrate raises. The framework treats a raised exception as "this snapshot is not
restorable" (§9) rather than guessing.

### 5.1 Fail fast, and surface it to the caller

Failing fast is right in the sense that matters: state that cannot be interpreted must **never** be
partially applied, and the failure must be loud. When `from_state` raises:

1. the snapshot is **quarantined** — moved aside, not deleted, so it survives for debugging and cannot
   re-trigger on the next attempt;
2. an ERROR names the context class and both versions;
3. the request **fails with HTTP 500**.

The caller sees the failure rather than a session that quietly forgot where it was. A context reset is
not a neutral outcome — it is the same loss `fix-3vy` describes, and returning 200 with an amnesiac
session is how that defect stayed invisible. Because step 1 quarantines before step 3, the client's
retry does not hit the same snapshot: it proceeds cold, and the loss becomes explicit at a moment the
caller can attribute.

**Not a process crash, deliberately.** Restore runs on a live request path during cold rehydration, so
a process-level failure would turn one unreadable snapshot into an outage for every unrelated session
on that pod, and — since the snapshot outlives the restart — into a crash loop. The blast radius of a
bad snapshot is the session that owns it, made visible to that session's caller.

---

## 6. Authoring guidance

Requirement: this mechanism ships with documentation on standard practice, not just an API. This
section is the source for it; it graduates into author-facing docs when the mechanism is implemented.

### 6.1 Versioning and serialization practice

- **Version from day one.** Ship `state_version = 1` with the first hook, not when the first breaking
  change arrives. Retrofitting a version onto unversioned state means guessing.
- **Bump on any incompatible shape change**; a field removed, renamed, retyped, or given new meaning.
- **Prefer tolerant readers.** A purely additive field that `from_state` can default does not need a
  bump. Read defensively (`state.get("x", default)`) so additive change stays cheap.
- **Migrate forward in `from_state`**, never in `get_state`. `get_state` always writes the current
  shape; `from_state` accepts every shape still in the field.
- **Do not serialize what you can recompute.** Derived values, caches and indexes are rebuild-on-load,
  not state. Less to version, less to get wrong.
- **Do not serialize framework objects.** Never put a `Workflow`, another context object, or a file
  handle in state. Take those from the `workflow` argument on restore.
- **Keep state JSON-native.** No sets, tuples as dict keys, datetimes without an explicit format, or
  class instances. The strict serializer rejects them (`fix-g03.16`), and lossy coercion is the defect
  `default=str` caused elsewhere in this codebase.
- **Rebuild identity explicitly.** Parent back-references and identity-keyed maps do not survive a
  round trip on their own; reconstruct them top-down (§7).
- **Round-trip in a test, including the locator.** `from_state(get_state(x))` must be equivalent *and*
  `find_by_locator(anchor, get_locator(obj)) is obj` must hold. Equality alone passes while identity is
  broken.

### 6.2 Secrets and PII

**Redaction is out of scope for this design** and is tracked separately (`fix-cj4`). Persist-all-by-
default means whatever an author puts in `workflow.context` reaches disk, so until that work lands the
boundary is documentation:

> **Do not attach secrets or PII to command contexts or workflow context, and do not inject them into
> agent trajectories, without strong encryption.** Persisted session state is written in the clear.

#### The norm to teach first: re-supply, don't store

`http_bearer_token` is the worked example, and it teaches the opposite of "encrypt what you store".

fastWorkflow injects the caller's JWT into application context itself — `_merge_workflow_context`
(`run_fastapi_mcp/utils.py:253`) and `_update_http_bearer_token` (`:260`) both set
`workflow.context["http_bearer_token"]`. It is **request-scoped data living in a session-scoped
container**: every authenticated request re-supplies it, because `get_session_and_ensure_runtime`
passes that request's own token through (`__main__.py:250`) and `ensure_user_runtime_exists` refreshes
it on the existing runtime before returning.

So persisting it — encrypted or not — is not merely risky, it is **wrong three times over**:

- **Unnecessary.** The next authenticated request supplies a fresh token regardless.
- **Actively harmful.** A restored snapshot would reinstate the token that was current when the
  snapshot was written. That value is stale by construction and may be expired, and depending on
  ordering it can overwrite the fresh token the current request just installed.
- **Already forbidden.** Non-negotiable rule 2 of the memory-bounds design (§1, line 83) states that
  request-scoped credentials are never written to durable channel state and never installed into
  shared workflow state outside an accepted turn's lifecycle.

That last point matters for scope: excluding `http_bearer_token` is **not** a carve-out from the
decision to defer redaction. It is implementing a rule that predates it. Redaction is a heuristic over
values an author chose to store; this is a framework-owned key the framework must not store at all.

**The norm, stated for authors:** a credential that arrives with a request belongs to that request.
Re-supply it per request; never checkpoint it. Reach for encryption only for state that is both
genuinely long-lived and genuinely secret.

#### On encrypting secrets that must persist

When state really must survive and really is sensitive — a long-lived third-party refresh token, say —
encryption is the right tool, with one condition that decides whether it is protection or theatre:
**the key must not travel with the ciphertext.** A framework-managed key sitting beside the data it
protects is the "encrypted at rest, key on the same disk" anti-pattern; it changes the audit checkbox
and not the exposure. Real protection means envelope encryption with a key from the deployment's KMS or
secret manager, which is infrastructure fastWorkflow should integrate with rather than mandate. Reusing
the JWT signing key would also be wrong — it couples two concerns and widens that key's blast radius.

An optional author-facing `encrypt` / `decrypt` extension point on `Serialization` is a reasonable way
to offer this without owning key management. It belongs with the redaction work in `fix-cj4`, not here,
because both answer the same question — what to do about sensitive values an author has chosen to
persist — and should be designed together rather than in two passes.

---

## 7. What an author writes

The scaffold, end to end. This is the worked example every new workflow would be copied from
(`simple_workflow_template/_commands/WorkItem/_WorkItem.py`, additions only):

```python
class Context:
    state_version = 1

    # ... existing get_parent / get_displayname ...

    @classmethod
    def get_state(cls, command_context_object: WorkItem) -> dict[str, Any]:
        return command_context_object.to_state_dict()      # author-owned, on WorkItem

    @classmethod
    def from_state(cls, state, workflow, *, state_version: int) -> WorkItem:
        if state_version != cls.state_version:
            raise fastworkflow.UnsupportedStateVersion(state_version)
        schema = WorkItem.WorkflowSchema.from_json_file(
            f'{workflow.folderpath}/simple_workflow_template.json'
        )
        return WorkItem.from_state_dict(state, workflow_schema=schema)

    @classmethod
    def get_locator(cls, command_context_object: WorkItem) -> str:
        return command_context_object.get_absolute_path()

    @classmethod
    def find_by_locator(cls, anchor: WorkItem, locator: str) -> Optional[WorkItem]:
        return anchor.find_by_absolute_path(locator)
```

The framework-facing part is small. The real work is `to_state_dict` / `from_state_dict` on `WorkItem`
itself, and it is not a one-liner: the tree has `_parent` back-references (so the graph is cyclic) and
`_child_pos`, a `Dict[WorkItem, int]` keyed by object **identity**. A correct restore rebuilds the tree
top-down and re-establishes both. `WorkflowSchema` already has `to_dict`/`from_dict`, so the schema half
exists; the live-tree half does not. That is `fix-gbh`, and it is the reason this design puts the
identity self-check in §4.3.

---

## 8. Warnings, and where they fire

Warning at workflow load would fire for contexts a session never touches. The warning fires instead the
first time a workflow **assigns a command-context object whose class has no `get_state`** — the moment
the session actually becomes pinned:

```
WARNING  workflow <name>: context class 'WorkItem' has no get_state hook, so sessions holding it
         are pinned and will never be evicted. Add get_state/from_state to
         _commands/WorkItem/_WorkItem.py, or return fastworkflow.EPHEMERAL to declare this
         deliberate and silence this warning.
```

Rate-limited to once per `(workflow folderpath, context class)` per process, per §17 of the
memory-bounds design. Alongside it, `pinned_sessions` becomes a metric, because for undeclared
workflows pinned is the steady state and a rate-limited warning is suppressed exactly when it matters
most. Never log state, context values or credentials.

---

## 9. Failure modes

| Situation | Behaviour |
|---|---|
| `get_state` raises | do not evict; ERROR naming the context class; rate-limited |
| `get_state` returns a non-JSON-native value | rejected by the strict serializer (`fix-g03.16`); pin; ERROR |
| `get_state` returns `None` | pin **and** warn — `EPHEMERAL` is how you decline |
| `from_state` raises | snapshot not applied; session treated as cold; ERROR |
| `from_state` returns an object whose class name ≠ the recorded context | not applied; quarantine; ERROR |
| Locator self-check fails | pin; ERROR |
| `state_version` mismatch | `from_state` is called **with** the stored version so the author can migrate (§5) |
| author declines to migrate (raises) | snapshot quarantined, then the request fails with **500**; ERROR naming both versions. The retry proceeds cold. |

The bias throughout is: **refuse to evict rather than lose state**, which is the memory-bounds design's
non-negotiable rule 1. Every failure above degrades to "this session stays in memory", which is a
visible, metered condition, never a silent one.

---

## 10. Verification

- Round-trip per bundled workflow: evict, drop the reference, rehydrate, assert `root_command_context`
  and `current_command_context` are restored **and that current is a node inside root**, not an equal
  copy. Shaped like the `fix-3vy` reproduction, which already drives `todo_list_workflow`.
- The trichotomy: absent hook pins and warns exactly once; `EPHEMERAL` pins and does not warn;
  a returned dict evicts and restores.
- `None` return pins and warns.
- Locator self-check failure pins rather than restoring a wrong current context.
- Redaction: a context carrying each deny-listed key persists without it, restores without it, and the
  key names appear only in DEBUG.
- Version mismatch quarantines and does not call `from_state`.
- Integration tests only, real workflows, no mocks; never train in or delete
  `fastworkflow/examples/*/___command_info`.

---

## 11. What this design does not do

- It does not make an undeclared workflow evictable. That is the point of the trichotomy.
- It does not persist the logical-turn accumulator or the CME continuation state — `fix-g03.25`.
- It does not redact secrets or PII from persisted state, and does not offer an encryption extension
  point — both deferred to `fix-cj4`. Until that lands, §6.2 is the boundary, with `http_bearer_token`
  excluded because non-negotiable rule 2 requires it.
- It does not change eviction mechanics; leases and the union predicate are `fix-g03.14`.

---

## 12. Decisions

Adjudicated 2026-08-05. Recorded here so implementation does not relitigate them.

| # | Decision | Outcome |
|---|---|---|
| 1 | Hook naming and placement | **`get_state` / `from_state` on `class Context`**, not dunders on the application class. |
| 2 | Locator handling | **Separate `get_locator` / `find_by_locator` pair**, with the serialize-time identity self-check. |
| 3 | Version mismatch | **Author owns migration.** `from_state` receives the stored version. On refusal: quarantine the snapshot, ERROR, and **fail the request with 500** so the caller sees the loss instead of an amnesiac session; the retry proceeds cold. Not a process crash (§5.1). Ships with the authoring guidance in §6.1. |
| 4 | Redaction | **Out of scope**, deferred to `fix-cj4`, together with an optional `encrypt`/`decrypt` extension point. Author responsibility documented (§6.2). `http_bearer_token` is **excluded** — not as an exception to this decision, but because non-negotiable rule 2 already forbids persisting request-scoped credentials, and it is re-supplied on every authenticated request. Encrypting it instead was considered and rejected: it would preserve a value that is stale by construction. |
| 5 | Workflow projection | **New optional module type** `_commands/_serialization.py`. |

The three norms in §6.2 are approved as author-facing best practice and are published in `README.md`
under *Production deployment → Handling secrets and PII*, alongside a factual statement of what reaches
disk today. Keep the two in step: the README is the normative copy for authors, this section is the
rationale.
