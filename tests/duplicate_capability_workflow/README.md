# duplicate_capability_workflow

A positive control for the near-duplicate capability detector
(`fastworkflow/train/duplicate_detection.py`, spec R9b / finding F14).

Every other workflow in this repository is a *negative* control: none of them contains a
genuine duplicate capability, so running the detector over them can only ever show that it
stays quiet. That proves it does not cry wolf; it proves nothing about whether it can see
a wolf. This workflow supplies the wolf.

It reproduces the case F14 was written from. On a large multi-context workflow,
`ControlsMonitor/list_findings` and `Directory/search_control_findings` answer the same
question, and no amount of utterance engineering separates them — they present as permanent
benchmark failures and send developers chasing a defect that is not in the model.

| Command | Role |
|---|---|
| `list_findings` | duplicate — half of the pair |
| `search_control_findings` | duplicate — the other half, phrased by a different author |
| `acknowledge_finding` | hard negative: same subject matter, different capability |
| `create_user` | easy negative: unrelated capability |

The two duplicates are written the way the real pair was written: two developers, working in
different parts of an application, independently exposing "show me the control findings"
and reaching for the same vocabulary. Nothing is copied verbatim between them — an
exact-duplicate detector would be trivial and useless.

The commands are flat (global context) rather than split across `ControlsMonitor` and
`Directory` contexts. The real pair was cross-context; the detector compares every command
pair in the workflow regardless of context and reports which contexts, if any, the pair
shares, so the flat arrangement exercises the same code path with less machinery.

Nothing here is trained or executed. The command bodies exist only because
`CommandDirectory` requires a `ResponseGenerator`; the detector reads seed utterances,
`Signature` docstrings and `Signature.Input` field names.
