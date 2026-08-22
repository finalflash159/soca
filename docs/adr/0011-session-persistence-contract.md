# ADR 0011: Engine-owned resumable session contract

Date: 2026-08-22

## Context

The desktop client currently has one process-lifetime transcript. Its “new
conversation” action only returns to the chat page and stops voice capture; it
does not create a new engine context. The engine can opt into JSON working-memory
and goal checkpoints through CLI flags, but the desktop launcher does not use
them. Those checkpoints intentionally contain a bounded, compacted prompt view,
not a user-visible transcript or a session catalog.

The product needs local, opt-in sessions that a person can create, list, open,
continue, rename and permanently delete. Chat and voice must remain one ordered
conversation. A restart may restore settled work, but must never automatically
replay a model call, tool call, write action or microphone capture that was in
flight when the engine stopped.

## Decision

The Python engine owns the canonical session aggregate. The desktop WebView is a
protocol client: it requests operations, receives typed receipts and hydrates a
snapshot. It never writes a second transcript to browser storage, Tauri storage
or a side database.

```text
React desktop ── NDJSON v3 ──> Python engine ──> sessions.sqlite3
      │                               │
      └──── session snapshot <────────┴── transcript, checkpoints, receipts
```

### Persistence and privacy defaults

- `ram_only` remains the default. It creates no durable session row or file.
- `local_resumable` requires explicit user consent in a future desktop setting.
  Enabling it does not retroactively persist the active RAM session unless the
  user explicitly requests that conversion.
- Auto-opening the last persisted session is a separate preference and defaults
  to off.
- Persisted data is final user/assistant text, repair/interruption state,
  structured citation/provenance/usage records, working-memory state and goal
  state. Raw audio, ASR partials, token deltas, hidden reasoning, API keys,
  raw tool results, vectors and retrieved snippets are not persisted.
- Durable session history is not memory evidence and is never automatically
  indexed or promoted to episodic memory.

### Repository and transaction model

`SessionRepository` will use Python's standard-library `sqlite3` at the current
XDG state root. It is the only production writer. The schema has versioned
migrations and contains, at minimum:

| Table                 | Responsibility                                                                    |
| --------------------- | --------------------------------------------------------------------------------- |
| `sessions`            | UUID, title, timestamps, revision, state, sequence and list preview               |
| `turns`               | Full settled chat/voice turn text, terminal semantics, citations, route and usage |
| `turn_steps`          | Public workflow receipts only; never chain-of-thought                             |
| `working_checkpoints` | Typed compacted working context and revision/digest                               |
| `goal_checkpoints`    | Active goal and last terminal run diagnostics                                     |
| `session_settings`    | Persistence consent, auto-open preference and last active session                 |
| `schema_migrations`   | Applied migration identity and timestamp                                          |

A coordinator commits the durable session aggregate. Beginning a user turn
inserts the pending turn and its working checkpoint in one transaction. A
terminal transition commits the settled response, provenance, usage, working
checkpoint, goal checkpoint and session list metadata together. The engine must
not emit a successful persisted terminal outcome if that commit failed.

The initial database configuration is:

```text
journal_mode=DELETE
synchronous=FULL
foreign_keys=ON
trusted_schema=OFF
secure_delete=ON
bounded busy_timeout
```

The engine has a single writable owner, so WAL is not needed for the initial
release. In particular, the development runtime reports SQLite 3.51.1 while the
SQLite WAL documentation records a WAL-reset fix in 3.51.3 and selected
backports. WAL can only be reconsidered after the packaged runtime pins a fixed
SQLite version, a benchmark proves the benefit, and multi-connection crash tests
pass. `SQLITE_BUSY`, lease conflict, corruption and write failures are typed,
observable states; none may silently fall back to RAM persistence.

### Session lifecycle

Each persisted session has a UUID. A fresh persisted session must not reuse the
legacy `default` ID. The engine owns these lifecycle operations:

| Operation | Required outcome                                                                                                                                                              |
| --------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Create    | Create a fresh aggregate atomically; the desktop clears only after receipt                                                                                                    |
| List      | Return newest-first bounded pages with stable cursor and metadata only                                                                                                        |
| Open      | Flush current state, recover pending work, load target snapshot, then switch ownership                                                                                        |
| Rename    | Validate normalized title and use an expected revision compare-and-swap                                                                                                       |
| Delete    | Permanently remove the aggregate after explicit confirmation; deleting the active session also creates a valid fresh active session in the same operation or fails atomically |
| Resume    | Restore settled state for the next user input; never execute unfinished work automatically                                                                                    |

On restart, a pending turn/run becomes `interrupted` or `system_failure` with the
typed reason `process_terminated`. The snapshot visibly distinguishes that
interruption from a completed answer. A user must explicitly issue the next
instruction.

### Protocol versioning

Session ownership changes the meaning of every durable transcript frame. The
engine and clients will move together from protocol v2 to **v3**. V3 requires
`session_id`, stable `turn_id`/`run_id`, and monotonic sequence on every frame
that can alter transcript state. A client must reject an incompatible engine
before it can send normal commands.

The v3 command family is:

```text
sessions_list { request_id, cursor?, limit? }
session_create { request_id }
session_open { request_id, session_id }
session_turns { request_id, before_sequence?, limit? }
session_rename { request_id, session_id, title, expected_revision }
session_delete { request_id, session_id, expected_revision }
session_preferences_get
session_preferences_set { request_id, ... }
session_status
```

Mutating operations emit `session_operation` with `request_id`, action, target
session, effective revision and one of `started`, `completed`, `rejected` or
`failed`. A timeout/toast is never evidence of success. `session_snapshot`
contains the active metadata, a bounded latest-turn page and cursors for older
history. Historical turns are hydrated by a dedicated reducer action, not by
replaying live protocol events.

While a turn, voice capture or controlled action is active, session switch is
rejected with a typed busy reason. Delayed frames from an old session are dropped
by the desktop reducer and recorded as protocol drift telemetry.

### Migration and recovery

The existing JSON working and goal checkpoints are migration input only. Before
activation, the migration acquires an exclusive store lease and makes a private,
consistent backup with a manifest containing schema, digest, SQLite version and
timestamp. It imports each old checkpoint in a transaction and validates foreign
keys, integrity and semantic round-trip before atomically activating the new
database.

Legacy checkpoints do not contain full transcript history. Their imported
sessions are marked `checkpoint_only`; the UI must say that only context state
was restored and must not invent missing history. Any corrupt or unknown-schema
source stops migration with a typed report. It must not skip the record or reset
to RAM mode silently.

After migration, production JSON readers and writers are removed in the same
implementation boundary. Read-only migration fixtures may remain for test data.

## Consequences

- The sidecar process, engine protocol and desktop reducer change together; a
  UI-only implementation is explicitly disallowed.
- Existing `SessionMemory` and `ActiveGoalStore` must hand persistence control to
  the coordinator instead of independently swallowing checkpoint conflicts.
- The user receives a truthful local-data consent and failure/recovery surface.
- Search across all session history, cloud sync, branch/regenerate, encryption at
  rest and exact historical source snippets remain out of scope until separately
  decided.

## Verification and release boundary

The feature is not complete on unit tests alone. Required evidence includes:

- repository, migration, two-process conflict, permissions, corruption,
  no-write-in-RAM and permanent-delete tests;
- engine process flow for create → turn → kill/restart → list/open → continue;
- adversarial delayed-frame and no-auto-replay tests;
- desktop keyboard, focus, live-region and failure-preserves-old-session tests;
- real trajectory evidence for grounded, empty-evidence, provider/tool failure,
  correction/follow-up and voice/ASR noise cases;
- packaged-app migration and persistence evidence on every platform claimed as
  supported.

The benchmark harness records OS, filesystem, Python/SQLite version, data
revision, raw local logs, median/P95/P99 and failures. The initial release
hypotheses are 150 ms P95 persistence cold-open overhead, 100 ms P95 recent-list
and snapshot query, and 25/75 ms P95 terminal persistence for 1 KB/20 KB turns.

## Rollback

Rollback is an explicit operator action to a known compatible binary and the
private backup made before migration. An older binary encountering a newer
schema must fail with recovery instructions; it must not open and overwrite the
store. Restoring a backup is verified in a temporary location and atomically
activated only after integrity checks pass.

## Sources

- [SQLite locking and concurrency](https://www.sqlite.org/lockingv3.html)
- [SQLite atomic commit](https://www.sqlite.org/atomiccommit.html)
- [SQLite write-ahead logging](https://www.sqlite.org/wal.html)
- [SQLite PRAGMA reference](https://www.sqlite.org/pragma.html)
- [SQLite backup API](https://www.sqlite.org/backup.html)
- [`docs/09-hybrid-rag-memory.md`](../09-hybrid-rag-memory.md)
- [`docs/18-engine-protocol.md`](../18-engine-protocol.md)
- [`soca/memory/session_store.py`](../../soca/memory/session_store.py)
