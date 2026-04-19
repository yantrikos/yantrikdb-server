# CORRECTIONS

Methodology errors discovered, what they affected, and where the
corrections live. Preserved publicly because the audit trail matters
more than a clean-looking repo.

---

## 2026-04-19 — Phase 3 "structured memory" condition was a simulator, not yantrikdb

### What was wrong

Across Phase 3C and Phase 3D (including L1 oracle, L3 longmemeval_s, L4
scaling), the condition labeled `C_structured` / "structured memory"
was implemented by [`docs/phase3c/memory_sim.py`](docs/phase3c/memory_sim.py) —
a ~120-line Python module that stored memories as a list of
`(key, value, session)` tuples with Dice word-overlap retrieval at
query time.

The simulator had **none** of yantrikdb's actual features:

- No embeddings (no HNSW vector index)
- No `think()` loop (no consolidation, no conflict scan)
- No multi-signal scoring (no temporal decay, no importance, no graph)
- No knowledge graph / entity extraction
- No conflict detection or resolution
- No temporal validity / supersession logic

Treating this simulator as a fair proxy for yantrikdb meant the
benchmarks measured "stripped-down key/value dict vs markdown" — not
"yantrikdb vs markdown."

### Where it was cited

Conclusions that rested on simulator results, now withdrawn:

- **"Plain structured memory loses to markdown"** (Phase 3C)
- **"40% stale-rate shows structured memory can't handle supersession"** (Phase 3C)
- **"Null result on RFC 006 temporal substrate"** (Phase 3D L3)
- **"Tiered memory thesis — structured memory only pays off at very high scale"** (Phase 3D L4)

All of these were drawn from experiments against the simulator, not yantrikdb.

### How it was caught

The project maintainer asked during a consultation: *"so then Yantrik
functionality, the core functionality did not run at all. we just ran
it as a vector db [or less]."*

Correct observation. No defense.

### What the correction looks like

Phase 3E ([`docs/phase3e/`](docs/phase3e/)) is the rerun with real
yantrikdb via HTTP endpoints:

- Client-side MiniLM-L6-v2 embeddings (384-dim) passed in every
  `/v1/remember` and `/v1/recall` call
- `/v1/think` called after each session for consolidation + conflict scan
- Namespace-scoped recall for per-experiment isolation

Preliminary Phase 3C rerun (n=2, same scenario):

```
                        simulator    real yantrikdb    Δ
overall_score             0.584   →    0.850        +26.6 pts
answer_accuracy           0.600   →    0.867        +26.7 pts
supersession_accuracy     0.500   →    0.800        +30.0 pts
stale_error_rate          0.400   →    0.200        -20.0 pts (halved)
alias_disambiguation      0.500   →    0.750        +25.0 pts
direct_accuracy           0.600   →    0.900        +30.0 pts
```

In the 4-condition matrix on the same scenario:
```
A cold:        0.000
B self-note:   0.383
D markdown:    0.667
C real ydb:    0.850  ← was 0.584 with simulator; NOW beats markdown
```

### What's pending

Before any public findings post:

- [ ] L1 LongMemEval rerun with real yantrikdb (in progress)
- [ ] Fresh-DB 3C audit (wipe data/, rerun to rule out cross-run contamination)
- [ ] `think(on)` vs `think(off)` ablation on 3C (mechanism attribution)
- [ ] Bump 3C to n=5 per cell
- [ ] L3 longmemeval_s rerun with real yantrikdb (gated on L1 outcome)

Pre-registered publication rule: POST a correction + findings post
ONLY IF fresh-DB gap holds AND think-off ablation shows think() is
load-bearing by ≥10 pts AND L1 is non-embarrassing. If any check fails,
the correction becomes the public artifact without the positive claims.

### Deprecation notices

Added prominent banners at the top of:
- [docs/phase3c/README.md](docs/phase3c/README.md)
- [docs/phase3d/README.md](docs/phase3d/README.md)

Both now route readers to Phase 3E for the corrected numbers.

### What's unchanged and still valid

- **Phase 3A and 3B** — tested LLM note-writing behavior only; never
  involved the simulator or yantrikdb. Findings still stand.
- **Phase 3D L4 scaling methodology** (pad haystacks, measure
  recall@k vs answer accuracy divergence) — the method is valid; the
  numbers need rerunning with real yantrikdb.
- **Phase 3D L3 finding that LongMemEval grading accepts hedging** — the
  observation about the benchmark's grading rubric is correct and
  independent of the memory substrate being tested.

### Lesson

Before committing any benchmark, verify the condition labeled as
"product X" actually invokes product X's features. The failure here
was predictable and should have been caught on day one with a
single test: "does the simulator call any code path in the real
yantrikdb binary?" No. Case closed, stop the benchmark, build the
real integration. Instead the simulator was allowed to stand in as a
proxy through 5 phases of experiments.

Future benchmark commits should include a feature-invocation
assertion at the top of the harness output — "this run made N calls
to [endpoint], used [features X, Y, Z]" — so that a reviewer can
see what the condition actually exercised.
