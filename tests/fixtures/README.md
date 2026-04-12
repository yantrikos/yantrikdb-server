# Test Fixtures

## data-dir-v1/

A minimal YantrikDB data directory created with the current schema version.
Contains:
- `control.db` — control database with one "test" database + one token
- `test/yantrik.db` — tenant database with 3 memories

Used by `tests/compat_test.rs` to verify that the current build can open
and query data directories from older schema versions.

### Regenerating

Run from repo root:

```bash
cargo run -p yantrikdb-server -- db --data-dir tests/fixtures/data-dir-v1 create test
cargo run -p yantrikdb-server -- token --data-dir tests/fixtures/data-dir-v1 create --db test --label compat-test
```

Then seed memories via the test itself or the HTTP API.
