# ADR 0003: Controlled workflow production contract

## Decision

`AssistantRuntime` uses the bounded controlled workflow by default. The
`run_text_turn` and `stream_text_turn` entrypoints are facades over that
controller; `turn_workflow="shadow"` is retained only for explicit
characterization and offline comparison. There is no production legacy
workflow selector.

The controller owns goal, capability, action, observation, revision,
verification and terminal state. A planner can schedule a bounded action set;
after a retrieval observation, the controller may request one typed refinement
or evidence-completion action within the shared budget. Side effects require
authorization, retries use one ledger, and every run emits exactly one typed
terminal outcome.

The runtime synthesizes the answer only after the scheduled actions finish.
The facade returns a factual answer only when the controller terminal is
`achieved`. A typed `insufficient_evidence` terminal may return the model's
explicit abstention when the evidence prompt had no usable evidence; that
abstention is not an achieved answer, is never added to session memory, and is
observable in the trace. Failed verification, exhausted evidence budget or
typed backend failure is surfaced as a blocked runtime result and is not
converted into a successful answer. Public updates are progress events, never
answer text, and only a successful terminal answer is appended to session
memory.

Event protocol v2 carries `session_id`, `run_id`, `goal_id`, monotonic sequence,
surface, timestamp, node and status. Terminal taxonomy remains explicit:
`achieved`, `needs_clarification`, `insufficient_evidence`, `safe_failure`,
`budget_exhausted`, `cancelled`, and `system_failure`.

## Rollback and characterization

Rollback is an operator action to a release tag before this cutover, with the
documented checkpoint/index migration procedure. It is not an automatic
runtime fallback. The shadow selector remains available only to tests and
offline characterization, and must not be used as a hidden production
downgrade.
