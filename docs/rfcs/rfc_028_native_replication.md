# RFC 028 — Native Replication (YRP): purpose-built clustering for cognitive memory

Status: **Draft v2** (2026-07-18) — revised after external red-team review (sol / GPT-5.6, 2 rounds; session 0913322c). v1's election and recovery designs contained safety holes; this revision adopts the review's corrections wholesale.
Author: Pranab Sarkar + Claude (yantrikdb-server), reviewed by sol (GPT-5.6)
Decision: Pranab, 2026-07-18 — *"focus on replication, leader selection, clustering… I opt for having our own solution because others might be too generic and not suitable for our needs."*
Constraint: **no shortcuts** (RFC first, then code), **cluster-mode validation is a required release gate**, **process-never-wedges** (the CT 141 lesson, precisely scoped below), **deterministic apply** (#104/Item 3 contract).
Supersedes: the openraft evaluation track of epic 60 (saga task 232).

## 1. What we are building — and what we are not

**We are rejecting openraft's operational posture, not Raft's safety mathematics.**

The evidence against the generic library is operational: wedge-on-confusion boot (took CT 141 down 10 days and needed manual state surgery), a library-owned snapshot format that broke under mixed versions, a generic membership API, and ~4,200 lines of integration glue — versus ~2,000 lines of our own raft-lite built around the engine's oplog. Every determinism guarantee we actually shipped this year (embedding bytes on the wire #52, `WriteAdmission`, origin-actor preservation, capability exchange #53) came from our own layers; the library contributed the wedge.

But the red-team review established — and we accept — that the election/commit *math* cannot be improvised. v1 of this RFC proposed scalar-watermark elections and auto-resumed voting after state loss; the review showed both permit split-brain (§8, scenarios R1–R3). The published, proven mechanisms (log matching, last-`(term,index)` voting, persist-vote-before-respond, quorum intersection, single-voter membership changes) are free to adopt.

**YRP = a Raft-shaped safety core — small, by the book, model-checked — plus purpose-built everything else:** memory-native oplog payloads (embedding bytes, provenance, HLC metadata), engine-checkpoint snapshots instead of a library format, quarantine-not-wedge recovery posture, honest ack tiers, and homelab-aware identity fencing. Owning the implementation is the point; inventing new consensus math never was. *"Do not invent a midpoint"* (sol) applies to the whole protocol.

### Non-goals
Multi-writer/multi-leader; Byzantine tolerance; geo-replication; linearizable follower reads by default; clusters beyond ~5 nodes.

## 2. The safety core (Raft-shaped, adopted verbatim)

1. **Canonical prefix log.** Short, compactable, immutable `(term, index)` positions, previous-entry continuity check, prefix hash for fork/corruption detection. The engine's CRDT-style idempotence (UUIDv7 `op_id` dedupe) remains an *application* property that heals duplicate delivery — it carries **zero weight** in the consensus argument.
2. **Election freshness = Raft's rule.** A voter grants its vote only to a candidate whose last `(term, index)` is at least as up-to-date as its own — protecting **possibly-committed suffixes** (entries quorum-replicated but not yet locally known-committed), not merely the locally-known committed frontier. (v1's watermark rule is dead: two divergent histories can share a watermark, and a committed-frontier comparison discards quorum-accepted writes — review scenario R3.)
3. **Vote persistence ordering.** A voter durably persists `(current_term, voted_for)` **before** sending a granted-vote response. Named invariant; crash-injection test at exactly that boundary.
4. **Write safety = per-write quorum confirmation + fencing.** A write is `quorum-durable` only after: leader durably appends under its term; a write quorum durably accepts; each acceptor validates term / membership epoch / prefix continuity / checksum / node incarnation; acceptors that have observed a higher term reject lower-term leaders. No wall-clock assumption is load-bearing. **Leases are demoted** to an optional read-latency optimization with explicitly documented partial-synchrony assumptions — never part of the write-safety proof.
5. **Stale-leader behavior.** An isolated leader may persist *tentative-local* writes (visibly tentative, capped with backpressure — see §4) but can never return `quorum-durable` success; it stops even tentative acceptance after bounded failed quorum renewal so orphan branches stay small.

## 3. Identity, membership, capabilities

- **Cluster identity:** immutable cluster ID; ops and votes carry it; mismatch → quarantine.
- **Node incarnations are quorum-managed.** A local counter rolls back with a VM/LXC snapshot (Proxmox cloning is a one-command operation here — this is *the* homelab-specific threat). The surviving cluster holds each node's authoritative incarnation; a node must have its incarnation authorized before voting; a new incarnation fences all older ones; two live connections claiming one identity is a hard alarm; old TLS keys alone never restore voting rights.
- **Membership: formally-specified single-voter-change protocol** (not joint consensus, not an invented midpoint), with the review's nine restrictions: one voter added/removed per committed transition (replace = two transitions); joiners enter as non-voting learners; learners catch up to a safe frontier before promotion; the config op commits under the *old* config's quorum; no next change until the prior one is provably committed (not merely appended); elections/replication derive config from the log, never local files; a removed leader is fenced by the new epoch; no automatic removal of unreachable voters (symmetric partition-repair is how clusters destroy themselves); a floor on voter count; a defined crash-recovery rule for selecting the governing config when a config entry's commit status is unknown. Removal tombstones and force-recovery epoch fencing are permanent protocol state.
- **Capabilities as a replicated state machine:** observe → confirm all electable voters compatible → quorum-commit activation → new encoding begins. Capability gates voter/candidate/ack eligibility: a node that cannot losslessly carry an active field (e.g. embedding bytes) must not acknowledge entries containing it nor be electable while the feature is active. Unknown fields round-trip losslessly or the entry is rejected — decode-and-drop is corruption. (Extends #53's capability exchange from warn-on-downgrade to quorum-safe activation.)

## 4. Acknowledgment & read contract (honest tiers)

| Tier | Meaning | Loss semantics |
|---|---|---|
| `tentative-local` | fsynced on the current leader only | may vanish on crash / partition / election / branch discard — **visibly tentative in the API response** |
| `replicated-N` | fsynced on N data-bearing nodes | survives only under a stated placement model; **no independence or election-intersection implied**; witnesses never count |
| `quorum-durable` | accepted by a write quorum with full validation | survives every legal subsequent election and recovery |

- v1's "loss window is ~heartbeat-sized" claim is **deleted**: in an asynchronous model no time bound exists. What is promised: quorum-durable is never lost under the fault model; tentative loss is explicit, and tentative branches are **capped by configured size/count with backpressure**.
- Failure-domain honesty: replicas co-located on one Proxmox host are not independent copies; `replicated-N` surfaces placement.
- **Read modes:** `local-stale` (labeled) · `session` (RYW) · `quorum-confirmed` (read-index round). RYW tokens identify **history**, not a scalar: {membership epoch, leader term, frontier identity}, with four API outcomes: *satisfied / catching-up / operation-orphaned / history-changed*. (A scalar `min_seq` cannot distinguish branches after failover; today's `?min_seq` becomes the degenerate single-history case.)
- A quarantined node's health surface separates "process up / diagnostics available" from "data servable"; it may serve explicitly-labeled stale reads only if data files independently verify.

## 5. Recovery — "never-wedge," precisely scoped

**The process always starts; consensus metadata fails closed.** The CT 141 outage was a *process* that crash-looped for 10 days — diagnostics dead, operator blind. That must never recur. But v1's cure (auto-reconstruct replication state and resume) enables double-voting (review R2). The correct split:

- **Data plane never wedges:** the process boots, serves diagnostics and (if data verifies) labeled stale reads, and continuously attempts authorized rejoin.
- **Consensus metadata fails closed:** on any safety-critical uncertainty — torn `(term, vote)`, cluster-ID mismatch, incarnation or epoch regression, log gap, invalid snapshot manifest, checksum/hash-chain failure, frontier beyond verifiable data, unsupported active capability, evidence of VM rollback — the node enters **non-voting quarantine**. Refusing to vote is not a wedge; it is the mechanism that prevents split-brain.
- **Rejoin is quorum-authorized, never self-directed:** a current quorum authorizes the node's incarnation, role, recovery source, and target frontier; promotion back to voter is explicit. A node never picks "the most advanced reachable peer" itself — a stale partition can be internally consistent and obsolete (review R5).
- **Recoverable incompleteness vs corruption evidence:** incompleteness → authorized snapshot resync (old state quarantined timestamped, forensics preserved — automated CT 141 procedure). Corruption evidence → preserve, alarm, replace only from a verified source. Auto-resync must not silence a corruption signal.
- Operator command `yantrikdb cluster rejoin --from-leader` drives the same authorized path manually.

## 6. Snapshots, GC, generations

- **Snapshot = engine checkpoint + manifest**, not a library format. Manifest binds: content hash, exact log frontier, membership epoch, schema/capability/**generation** state, dedupe + tombstone metadata. All per-namespace checkpoints are captured against **one global frontier under an apply-quiesce barrier**; a manifest referencing mixed frontiers is invalid.
- **Install is crash-safe:** stage to new location → fsync files+dirs → verify integrity + continuity → atomic switch → retain old state until the node has rejoined and caught up. Disk-full leaves old-complete or new-complete, never a bootable-looking hybrid.
- **Source selection requires authority AND completeness:** a quorum-backed leadership certificate proves authority, not byte completeness — the source must also prove the target frontier, per-namespace completeness, and referenced-blob presence.
- **GC frontier:** an op (and its dedupe/tombstone record) is collected only behind a quorum-stable frontier tied to membership + snapshots; a node behind the frontier is forced through snapshot rebootstrap and **cannot upload pre-GC ops** (fenced incarnations make this enforceable). Every recovery path has either retained log coverage or a verified snapshot.
- **Generation cutover (re-embedding) is a multi-phase replicated state machine**, not an O(1) marker alone: declare/build → leader re-embeds and streams per-row vector-carrying correct-ops (generation-ID + source-content-version stamped; followers never embed) → deterministic completeness manifest → quorum-durable completion → **atomic activation** → retire old generation behind a safe frontier. Queries use one coherent generation or explicitly defined mixed-generation behavior; snapshots carry generation state so a snapshot cannot capture the marker without the vectors.

## 7. Fork handling — discard-with-export, constrained

No automatic merge, ever. A stale leader's tentative tail becomes an **orphan branch**: quarantined, inspectable, exportable. Constraints (review-hardened):

1. **Admission + idempotency claim + op are one atomic replicated unit** — the claim is quorum-durable before any durable-success ack, so exactly-once holds across failover. (This also answers core's open 3b design question: claims are origin-ingress *and* commit-coupled; Admitted appliers never consult the claims table.)
2. Tentative visibility is API-explicit — a client can never mistake "you may read this now" for "this survives failover."
3. Export preserves **dependency closure** (intra-branch entity refs, supersession edges, generation context) and identifies prerequisites missing from canonical history.
4. Re-admission mints a **new canonical `op_id`** with provenance pointing at the orphan original — never reuses the orphan id. Ops carry branch/leadership identity.

Workload note (why discard-with-export fits): writers are agents with idempotency keys and retry loops (a lost tentative ack is typically re-driven within seconds, exactly-once via claims), and the memory model already treats competing truths as first-class conflicts rather than corruption.

## 8. Failure scenarios this design must defeat (from review; each becomes a test)

- **R1** stale leader acknowledging during partition → §2.4/§2.5
- **R2** double-vote after auto-recovery of torn vote state → §2.3/§5
- **R3** divergent-equal-watermark; possibly-committed suffix discarded by committed-frontier election → §2.1/§2.2
- **R4** sync-ack set not intersecting a later election quorum; witness counted as durability → §4
- **R5** rebootstrap from a reachable-but-stale "leader"; corruption amplified by auto-resync → §5/§6
- **R6** Proxmox clone/VM rollback resurrecting an old voter identity → §3
- **R7** one node's uncommitted local tail blocking every election forever → §2.2 (tentative-suffix distinction)
- **R8** GC'd op re-uploaded by a long-offline node → §6
- **R9** future-skewed HLC poisoning LWW for years → §9
- **R10** mixed-version voter acknowledging entries it cannot represent → §3 (capabilities)

## 9. HLC — metadata only, non-destructive

HLC plays **no role** in election, commit, or frontier semantics (all `(term,index)`-based). It remains provenance/ordering metadata and an LWW *input* in the engine's conflict layer (evidence, not truth). Safeguards: persisted across restart; monotone after clock rollback; remote physical components clamped to a plausibility bound; never advanced to arbitrary unauthenticated remote values; deterministic non-time tiebreak; HLC alone never triggers irreversible deletion/compaction/conflict-erasure — the evidence-not-truth principle extends into retention code; repair tooling for future-dated records. Election timers use monotonic clocks only.

## 10. Determinism claim, stated precisely

**Logical-state determinism + exact embedding-byte preservation** — identical op sets yield identical logical state and identical stored vectors (appliers never invoke the embedder; vectors travel in ops). *Not* byte-identical SQLite files (page layout, WAL timing, and library versions legitimately differ). Replicas skip application *policy* admission (done once at origin ingress) but always perform deterministic *protocol* validation: cluster ID, epoch, term, incarnation, continuity, checksum, schema/generation validity, capability support.

## 11. Validation stack (proof before deployment)

Chaos testing is a **detector, not a proof**. In order:
1. **Deterministic simulation** (primary tool): seeded schedule exploration, crash injection at every metadata-persistence boundary (vote persist, config commit, snapshot switch), packet delay/duplication/reorder/asymmetric partition, VM pause + clock rollback, node clone.
2. **Small formal model** (TLA+/PlusCal or exhaustive checker) of the three interleaving-dominated components: election safety, single-voter reconfiguration, incarnation fencing.
3. **Chaos CI** as the regression net (kill-leader under load, partition+heal, torn-state boot, stale-rejoin beyond GC, disk-full during snapshot install, mixed-version pairing) — the permanent cluster-mode release gate.

**Release-gate invariants** (each a mechanical test):

Gate **A — before any two nodes replicate, even in dev:**
1. Vote safety — no node votes twice per term/incarnation, including after any crash or recovery.
2. Authority safety — no two nodes return durable-success under overlapping leadership authority.
3. Possibly-committed-suffix protection in elections.
4. Quarantine fails closed on the §5 trigger list.
5. Ack-tier honesty in the API (tentative is visibly tentative).

Gate **C — before production deployment:**
6. Commit preservation across every legal election.
7. Membership safety — old configs, removed nodes, clones cannot form a competing quorum.
8. Snapshot atomicity + manifest completeness.
9. GC safety — no pre-GC op reintroduction.
10. Version/capability safety.
11. History-aware RYW statuses.
12. Explicit loss semantics in metrics (durable-ack-based lag, not send-based).

## 12. Migration

| Phase | Content |
|---|---|
| **A** | Safety core (§2) + quarantine/recovery (§5) + identity/incarnation (§3), built against the simulator; Gate A green |
| **B** | Data plane: push path, capability activation, snapshot/GC (§6), generation cutover, fork tooling (§7); formal model + chaos suite; Gate C green in sim |
| **C** | 3-node homelab deployment (always-on LXC voters node1/node2/node4 + witness CT 142; desktop Docker node ReadReplica at most), soak, failover-RTO benchmark (closes epic 59 task 226) |
| **D** | Deprecate `raft_mode = "openraft"` (warn → refuse-with-hint), delete `raft/` (~4,200 lines) + openraft dep + SplitRuntime complexity |

Existing deployments are standalone (`raft_mode = "disabled"`) — untouched through all phases; nothing replicates in anger before Gate A.

## 13. Honest scope note

The review roughly doubled the phase A–B correctness surface relative to v1 — correctly. Mitigations: the safety core is **copied from proven algorithms rather than designed** (the novelty budget — where risk actually lives — shrinks to the memory-native layers); production is all-standalone with no timeline pressure, so phases sequence strictly; and the election/recovery/membership machinery is provable in simulation long before any live cluster exists.
