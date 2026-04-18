# The Rashomon Engine

**Reconstructing truth from conflicting testimony — and proving it with real queries, not hardcoded narratives.**

Feed YantrikDB five witness statements about a data breach, plus badge logs and git logs. Some witnesses are lying. The engine reconstructs what actually happened using only its claims ledger and recall engine.

Every lie surfaced in the output is *computed*, not scripted. The synthesis at the end is driven entirely by API queries against the claims table.

---

## Why no other memory system can do this

Before YantrikDB v0.6.1, every cognitive memory system had one of these problems:

| System | Failure mode |
|---|---|
| Vector DB (Pinecone, Weaviate, Qdrant) | Returns all 5 witness statements as "similar." No concept of contradiction. No source attribution. No polarity. |
| Full-text search (Elastic) | Finds keyword matches. Can't tell that "David was home by 10" contradicts "David left at 11:31". |
| File-based memory (CLAUDE.md) | Stuffs all 5 statements into context, lets the LLM figure it out. Doesn't scale, no provenance chain. |
| Graph DB (Neo4j) | Can model entities + relations, but no temporal validity or polarity on edges. Can't distinguish "David claims X" from "X is true." |
| **YantrikDB v0.6.1+** | **Scoped claims with polarity + validity + source attribution. Multi-source assertions coexist. Polarity contradiction detection is automatic.** |

---

## The scenario

**2026-03-15, 19:00–00:00 UTC.** Helios Labs, Cambridge MA.

Source code for the flagship product leaks to a public repo at 23:15. Five people had badge access that night. Each tells their version.

### The cast

| Person | Role | Their story |
|---|---|---|
| **Maya Chen** | Senior engineer | "Left at 10pm. David's light was on." (truthful, partial) |
| **David Park** | CTO | "Home by 10. Did **NOT** touch production repo." (**lying**) |
| **Alex Rivera** | Night janitor | "David typed in his office at 11pm. Stressed exit at 11:30." (truthful, high reliability) |
| **Sarah Kim** | Receptionist | "WFH, have badge alerts on phone." (corroborative only) |
| **Jamie Torres** | Junior engineer | "Worked from home all night. Pushed a PR at 10:45pm." (**partial lie**) |

### The two authoritative log sources

- **system.badge** — every card swipe on the building doors
- **system.git** — every commit, push, and visibility change on the code repos

### The ground truth (only the designers know)

David exfiltrated production code at 23:14, made the fork public at 23:15. He lies about both his departure time and repo access. Jamie briefly came to the office to cover their tracks with an innocuous commit. Badge and git logs are authoritative.

---

## What the engine does

The showcase runs 8 phases against the YantrikDB HTTP cluster:

1. **Seed** — ingest 5 witness narratives as memories + structured claims with polarity/validity/source
2. **Think** — run the conflict scanner
3. **Polarity contradiction detection** — walk the claims ledger, find same `(src, rel, dst)` tuples asserted with opposite polarity by different sources
4. **Temporal contradiction detection** — compare `was_at` validity windows across sources; flag >15 min discrepancies
5. **Presence denial detection** — catch claims like "Jamie was NOT at office" (polarity=-1) against `system.badge` records (polarity=+1)
6. **Evidence chain via recall** — "what did David do that night" — show what the memory system surfaces first
7. **Scoped conflicts** — the new `GET /v1/conflicts?namespace=X` from v0.7.2
8. **Verdict** — rank suspects by contradiction score, name the primary suspect, cite the damning claims

---

## What made this possible (the v0.6.1 fix)

The V17 schema had `UNIQUE(src, dst, rel_type)` on the claims table. That silently overwrote any previous source's claim whenever another source asserted the same fact. This single constraint made multi-witness investigation impossible — David's denial would overwrite Maya's testimony, or vice versa.

**V18 schema (yantrikdb 0.6.1):**
```sql
UNIQUE(src, dst, rel_type, extractor, polarity, namespace)
```

Now David's "accessed=NO" (polarity=-1) and `system.git`'s "accessed=YES" (polarity=+1) are **both stored as distinct rows** in the claims table. The contradiction detector sees them. The showcase can surface them.

Before this fix: the Rashomon pattern was theoretically expressible but practically broken.
After this fix: it's a 300-line Python script against the HTTP API.

---

## Running it

```bash
python docs/showcase/rashomon_engine.py ydb_your_token http://your-cluster:7438
```

---

## Applications beyond fictional crimes

This exact pattern applies to:

- **Legal discovery** — conflicting depositions, timelines, documentary evidence
- **Incident response** — logs from multiple systems + human bug reports + postmortems
- **Investigative journalism** — source statements, official records, timeline reconstruction
- **Medical diagnosis** — patient self-report, test results, family history, symptom timeline
- **Financial forensics** — transaction logs, interviews, stated vs actual activity
- **Historical research** — primary sources that contradict each other across time
- **Multi-agent AI systems** — sub-agents reporting observations, some stale, some buggy, some biased

Any domain where **truth must be reconstructed from partial, biased, or deceptive sources** is a domain for this pattern.

---

*Memory as a reasoning substrate, not a search index.*
