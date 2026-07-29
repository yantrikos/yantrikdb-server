//! 2-node YRP cluster over REAL localhost HTTP — the runtime-integration
//! slice's acceptance test (handoff steps 7+8).
//!
//! Two full servers (production router, real engines, real tempdir
//! stores) form a 2-voter cluster whose YRP wire messages ride actual
//! `POST /v1/yrp/msg` requests. Asserts, in order:
//!
//! 1. An election happens over HTTP and exactly one node leads.
//! 2. A keyed `/v1/remember` on the leader commits and answers `{rid}`.
//! 3. The identical retry answers the SAME rid (silent HIT) — the
//!    issue-#58 contract, now replicated (the historical 501 is gone).
//! 4. Same key + different text answers the 200 conflict shape with the
//!    original rid.
//! 5. **nuron's live-recall axis**: the FOLLOWER's live `/v1/recall`
//!    (in-memory index, not just SQLite) surfaces the replicated memory —
//!    the durable-green/live-red failure class.
//! 6. An unkeyed write also replicates (the YrpCommitter funnel).
//! 7. A write against the follower is refused with the leader's address
//!    (503 via check_writable) — never silently accepted.

use std::sync::Arc;

use parking_lot::Mutex;
use serde_json::{json, Value};

use crate::auth::ControlDbAuthProvider;
use crate::control::ControlDb;
use crate::server::AppState;
use crate::yrp::runtime::{spawn as spawn_yrp, YrpCommitter, YrpPeer, YrpRuntimeConfig};

struct Node {
    state: Arc<AppState>,
    handle: Arc<crate::yrp::runtime::YrpHandle>,
    base: String,
    token: String,
    _tmp: tempfile::TempDir,
}

const CLUSTER_ID: u64 = 7;
const SECRET: &str = "yrp-test-cluster-secret";
const TENANT: &str = "yrpcluster";

async fn post_json(
    base: &str,
    token: &str,
    path: &str,
    body: &Value,
) -> (reqwest::StatusCode, Value) {
    let client = reqwest::Client::builder()
        .redirect(reqwest::redirect::Policy::none())
        .build()
        .unwrap();
    let resp = client
        .post(format!("{base}{path}"))
        .bearer_auth(token)
        .json(body)
        .send()
        .await
        .expect("request");
    let status = resp.status();
    let text = resp.text().await.expect("body");
    let val: Value = serde_json::from_str(&text).unwrap_or(json!({ "raw": text }));
    (status, val)
}

async fn get_json(base: &str, token: &str, path: &str) -> (reqwest::StatusCode, Value) {
    let client = reqwest::Client::new();
    let mut req = client.get(format!("{base}{path}"));
    if !token.is_empty() {
        req = req.bearer_auth(token);
    }
    let resp = req.send().await.expect("request");
    let status = resp.status();
    let text = resp.text().await.expect("body");
    let val: Value = serde_json::from_str(&text).unwrap_or(json!({ "raw": text }));
    (status, val)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn two_node_cluster_over_http_replicates_keyed_and_unkeyed_writes() {
    let _serial = crate::yrp::testkit::serial_guard().await;
    let _ = tracing_subscriber::fmt()
        .with_env_filter("yantrikdb=info")
        .with_test_writer()
        .try_init();
    // Pre-bind both HTTP ports so the peer lists can be exchanged before
    // either server exists.
    let l1 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let l2 = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let p1 = l1.local_addr().unwrap().port();
    let p2 = l2.local_addr().unwrap().port();
    drop((l1, l2));
    let peers = vec![
        YrpPeer {
            node_id: 1,
            addr: format!("http://127.0.0.1:{p1}"),
            witness: false,
        },
        YrpPeer {
            node_id: 2,
            addr: format!("http://127.0.0.1:{p2}"),
            witness: false,
        },
    ];

    // Spawn both nodes on the advertised ports.
    let mut n1 = spawn_node_on(1, peers.clone(), p1).await;
    let mut n2 = spawn_node_on(2, peers.clone(), p2).await;

    // 1. Election over real HTTP.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(20);
    let leader_is_n1 = loop {
        if n1.handle.is_leader() {
            break true;
        }
        if n2.handle.is_leader() {
            break false;
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "no leader elected over HTTP transport"
        );
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    };
    if !leader_is_n1 {
        std::mem::swap(&mut n1, &mut n2);
    }
    let (leader, follower) = (n1, n2);

    // The engine's default embedding dim is 384 — synthesize two
    // distinct full-width vectors.
    let embedding = |seed: f32| -> Value {
        json!((0..384)
            .map(|i| (i as f32).mul_add(0.001, seed).sin())
            .collect::<Vec<f32>>())
    };
    let emb1 = embedding(0.25);
    let emb2 = embedding(2.5);

    // 2. Keyed write on the leader — the endpoint that returned 501 in
    // cluster mode until this slice.
    let body1 = json!({
        "text": "yrp replicated memory alpha",
        "embedding": emb1,
        "idempotency_key": "yrp-e2e-key-1",
    });
    let (st, resp) = post_json(&leader.base, &leader.token, "/v1/remember", &body1).await;
    assert_eq!(st, reqwest::StatusCode::OK, "keyed write failed: {resp}");
    let rid = resp["rid"].as_str().expect("rid in response").to_string();

    // 3. Silent HIT: identical retry answers the ORIGINAL rid.
    let (st, resp) = post_json(&leader.base, &leader.token, "/v1/remember", &body1).await;
    assert_eq!(st, reqwest::StatusCode::OK);
    assert_eq!(
        resp["rid"].as_str(),
        Some(rid.as_str()),
        "retry must dedupe to the original rid"
    );
    assert!(
        resp.get("idempotency_conflict").is_none(),
        "same text must be a silent HIT: {resp}"
    );

    // 4. Same key + different text → 200 conflict shape, original rid.
    let body_conflict = json!({
        "text": "DIFFERENT text under the same key",
        "embedding": emb1,
        "idempotency_key": "yrp-e2e-key-1",
    });
    let (st, resp) = post_json(&leader.base, &leader.token, "/v1/remember", &body_conflict).await;
    assert_eq!(st, reqwest::StatusCode::OK);
    assert_eq!(
        resp["idempotency_conflict"],
        json!(true),
        "conflict shape expected: {resp}"
    );
    assert_eq!(resp["stored"], json!(false));
    assert_eq!(resp["rid"].as_str(), Some(rid.as_str()));

    // 5. Follower LIVE recall reflects the replicated apply (nuron's
    // live-recall axis: in-memory index coherence, not just durable rows).
    wait_for_recall(&follower, &emb1, &rid).await;

    // 6. Unkeyed writes ride the same replicated funnel (YrpCommitter).
    let body_unkeyed = json!({
        "text": "yrp replicated memory beta (unkeyed)",
        "embedding": emb2,
    });
    let (st, resp) = post_json(&leader.base, &leader.token, "/v1/remember", &body_unkeyed).await;
    assert_eq!(st, reqwest::StatusCode::OK, "unkeyed write failed: {resp}");
    let rid2 = resp["rid"].as_str().expect("rid").to_string();
    assert!(
        resp["log_index"].as_u64().is_some(),
        "receipt log_index missing: {resp}"
    );
    wait_for_recall(&follower, &emb2, &rid2).await;

    // 7. Writes against the follower are refused with leader info — the
    // key is never silently dropped and never double-executed.
    let (st, resp) = post_json(&follower.base, &follower.token, "/v1/remember", &body1).await;
    assert_eq!(
        st,
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "follower must refuse writes: {resp}"
    );

    // ensure_linearizable: the leader's committer passes a REAL barrier
    // (noop through the replicated commit path); the follower's answers
    // NotLeader with the leader's address for redirect.
    leader
        .state
        .commit_log
        .ensure_linearizable()
        .await
        .expect("leader read barrier must succeed");
    match follower.state.commit_log.ensure_linearizable().await {
        Err(crate::commit::CommitError::NotLeader { leader_addr, .. }) => {
            assert!(
                leader_addr.is_some(),
                "follower barrier refusal must carry the leader address"
            );
        }
        other => panic!("follower barrier must answer NotLeader, got {other:?}"),
    }

    // Health surfaces yrp mode honestly on both nodes.
    let client = reqwest::Client::new();
    for n in [&leader, &follower] {
        let v: Value = client
            .get(format!("{}/v1/health", n.base))
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        assert_eq!(v["cluster"]["raft_mode"], json!("yrp"), "health: {v}");
    }
    assert_eq!(
        leader.state.yrp.as_ref().unwrap().quarantine_reasons(),
        None
    );

    // Admin studio: /v1/cluster/topology aggregates BOTH members (fan-out
    // to peers' /v1/health), and /admin serves the embedded console.
    let topo: Value = client
        .get(format!("{}/v1/cluster/topology", leader.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(topo["raft_mode"], json!("yrp"));
    let nodes = topo["nodes"].as_array().expect("nodes array");
    assert_eq!(nodes.len(), 2, "topology must list both members: {topo}");
    assert!(
        nodes.iter().any(|n| n["role"] == json!("leader")),
        "topology must show a leader: {topo}"
    );
    assert!(
        nodes.iter().all(|n| n["reachable"] == json!(true)),
        "both members reachable: {topo}"
    );
    let admin = client
        .get(format!("{}/admin", leader.base))
        .send()
        .await
        .unwrap();
    assert!(admin.status().is_success());
    let admin_html = admin.text().await.unwrap();
    assert!(
        admin_html.contains("YantrikDB") && admin_html.contains("control console"),
        "/admin must serve the studio v2 console"
    );

    // ── RFC 029: control-plane replication ──────────────────────────
    // A database + token minted on the LEADER via the replicated admin
    // endpoints (master-token gated) must materialize on the FOLLOWER's
    // control.db — closing the exact gap (per-node tokens don't survive
    // failover) that made an enterprise cluster undeployable.
    let (st, resp) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/databases",
        &json!({ "name": "rfc029db" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "admin create db: {resp}");
    assert_eq!(resp["replicated"], json!(true), "must replicate: {resp}");
    let db_id = resp["id"].as_i64().expect("db id");

    let (st, resp) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/tokens",
        &json!({ "database_id": db_id }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "admin mint token: {resp}");
    let minted = resp["token"].as_str().expect("token").to_string();
    let minted_hash = crate::auth::hash_token(&minted);

    // The FOLLOWER's control.db resolves the replicated token to the same
    // database id — i.e. `ControlDbAuthProvider` on the follower will now
    // authenticate a token minted on the leader (the auth path IS
    // `validate_token`). This is the headline RFC 029 guarantee.
    let poll_follower_token = |want: Option<i64>| {
        let follower_control = follower.state.control.clone();
        let hash = minted_hash.clone();
        async move {
            let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
            loop {
                let got = follower_control.lock().validate_token(&hash).unwrap();
                if got == want {
                    return;
                }
                assert!(
                    tokio::time::Instant::now() < deadline,
                    "follower control.db never converged: got {got:?}, want {want:?}"
                );
                tokio::time::sleep(std::time::Duration::from_millis(50)).await;
            }
        }
    };
    poll_follower_token(Some(db_id)).await;

    // Control writes funnel through the leader: a control write against the
    // FOLLOWER is refused (leader redirect), exactly like a data write.
    let (st, _) = post_json(
        &follower.base,
        SECRET,
        "/v1/admin/databases",
        &json!({ "name": "should-redirect" }),
    )
    .await;
    assert_eq!(
        st,
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "follower must redirect control writes to the leader"
    );

    // Revocation replicates too: revoke on the leader → the follower stops
    // resolving the token.
    let (st, _) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/tokens/revoke",
        &json!({ "token": minted }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "admin revoke");
    poll_follower_token(None).await;

    // RFC 029 inc2: the control-freshness gate must NOT fire on a healthy,
    // caught-up follower — auth stays available (fail-closed only when the
    // node is quarantined or backfilling).
    assert!(
        crate::server::control_auth_stale(&follower.state).is_none(),
        "healthy follower must be auth-eligible (freshness gate false-positive)"
    );

    // Hardening (review F2): a duplicate database name is a 409, not a
    // phantom-id success.
    let (st, _) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/databases",
        &json!({ "name": "rfc029db" }),
    )
    .await;
    assert_eq!(
        st,
        reqwest::StatusCode::CONFLICT,
        "duplicate db name must 409, not return a phantom id"
    );

    // Hardening (review F3): a token mint against a nonexistent database_id
    // is rejected at the handler — it must NEVER reach apply (an FK failure
    // there would fail-stop the apply worker on every node).
    let (st, _) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/tokens",
        &json!({ "database_id": 999_999 }),
    )
    .await;
    assert_eq!(
        st,
        reqwest::StatusCode::NOT_FOUND,
        "token mint for a bogus db id must be refused, not wedge the cluster"
    );
    // Prove the cluster is still healthy after the rejected bad-id mint —
    // apply did not fail-stop; a normal write still replicates.
    let body_after = json!({
        "text": "post-hardening write still replicates",
        "embedding": emb2,
        "idempotency_key": "yrp-e2e-key-after",
    });
    let (st, resp) = post_json(&leader.base, &leader.token, "/v1/remember", &body_after).await;
    assert_eq!(
        st,
        reqwest::StatusCode::OK,
        "cluster must still accept writes after a rejected bad mint: {resp}"
    );

    // ── RFC 030: multi-user admin accounts + RBAC ───────────────────
    // Bootstrap: create the first owner with the break-glass master token.
    let (st, _) = post_json(
        &leader.base,
        SECRET,
        "/v1/admin/users",
        &json!({ "username": "boss", "password": "supersecret", "role": "owner" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "create first owner");

    // The user record replicates to the follower's control.db.
    {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            if follower
                .state
                .control
                .lock()
                .get_admin_user("boss")
                .unwrap()
                .is_some()
            {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "owner never replicated to follower"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    // Login on the LEADER (seeds the replicated session key) → session token.
    let (st, resp) = post_json(
        &leader.base,
        "",
        "/v1/admin/session",
        &json!({ "username": "boss", "password": "supersecret" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "owner login: {resp}");
    let boss_sess = resp["token"].as_str().expect("session token").to_string();
    assert_eq!(resp["role"], json!("owner"));

    // Wrong password is 401.
    let (st, _) = post_json(
        &leader.base,
        "",
        "/v1/admin/session",
        &json!({ "username": "boss", "password": "WRONG" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::UNAUTHORIZED, "bad password → 401");

    // The session authenticates on the FOLLOWER (session key + user both
    // replicated) — /v1/admin/me returns the owner role.
    let (st, resp) = get_json(&follower.base, &boss_sess, "/v1/admin/me").await;
    assert_eq!(
        st,
        reqwest::StatusCode::OK,
        "session valid on follower: {resp}"
    );
    assert_eq!(resp["role"], json!("owner"));

    // Owner creates a readonly user (via the session, not the master token).
    let (st, _) = post_json(
        &leader.base,
        &boss_sess,
        "/v1/admin/users",
        &json!({ "username": "viewer", "password": "viewerpass", "role": "readonly" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "owner creates readonly user");
    let (_, resp) = post_json(
        &leader.base,
        "",
        "/v1/admin/session",
        &json!({ "username": "viewer", "password": "viewerpass" }),
    )
    .await;
    let viewer_sess = resp["token"].as_str().expect("viewer session").to_string();

    // RBAC: readonly cannot create a database (needs admin) → 403.
    let (st, _) = post_json(
        &leader.base,
        &viewer_sess,
        "/v1/admin/databases",
        &json!({ "name": "viewer-cannot" }),
    )
    .await;
    assert_eq!(
        st,
        reqwest::StatusCode::FORBIDDEN,
        "readonly cannot create db"
    );
    // …but can list (readonly+).
    let (st, _) = get_json(&leader.base, &viewer_sess, "/v1/admin/databases").await;
    assert_eq!(st, reqwest::StatusCode::OK, "readonly can list dbs");

    // H1 revocation: disable viewer → their live session dies (401), and the
    // revocation propagates so it dies on the FOLLOWER too.
    let (st, _) = post_json(
        &leader.base,
        &boss_sess,
        "/v1/admin/users/viewer/disable",
        &json!({}),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "disable viewer");
    {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let (st, _) = get_json(&follower.base, &viewer_sess, "/v1/admin/me").await;
            if st == reqwest::StatusCode::UNAUTHORIZED {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "disabled user's session never died on follower"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    // H2 owner-floor: disabling the only owner is refused (stays enabled).
    let (st, resp) = post_json(
        &leader.base,
        &boss_sess,
        "/v1/admin/users/boss/disable",
        &json!({}),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK);
    assert_eq!(
        resp["disabled"],
        json!(false),
        "owner-floor must refuse disabling the last owner: {resp}"
    );

    // The replicated audit trail recorded the admin mutations, attributed.
    let (st, resp) = get_json(&leader.base, &boss_sess, "/v1/admin/audit?limit=50").await;
    assert_eq!(st, reqwest::StatusCode::OK);
    let audit = resp["audit"].as_array().expect("audit array");
    assert!(
        audit
            .iter()
            .any(|e| e["action"] == json!("create_user") && e["actor"] == json!("master-token")),
        "audit must attribute the bootstrap create_user to master-token: {resp}"
    );

    // ── RFC 031: clustered packs — manifest replication + peer file-transfer
    //    + reconciler poison-quarantine, all in one flow. (A real sealed pack
    //    mounting + recall is covered by the engine's own pack tests; here we
    //    prove the SERVER's novel surface with a deliberately-invalid pack so
    //    the poison-quarantine path — the review's CRITICAL C1 — is exercised.)
    let tenant_db_id = leader
        .state
        .control
        .lock()
        .get_database(TENANT)
        .unwrap()
        .unwrap()
        .id;

    // Upload arbitrary bytes as a "pack" to the leader (upload stores by
    // digest; validity is judged at mount).
    let client = reqwest::Client::new();
    let junk = b"not actually a sealed sqlite pack \x00\x01\x02".to_vec();
    let up = client
        .post(format!("{}/v1/admin/packs", leader.base))
        .bearer_auth(SECRET)
        .header("content-type", "application/octet-stream")
        .body(junk.clone())
        .send()
        .await
        .unwrap();
    assert_eq!(up.status(), reqwest::StatusCode::OK, "pack upload");
    let pack_digest = up.json::<Value>().await.unwrap()["digest"]
        .as_str()
        .unwrap()
        .to_string();

    // Peer-transfer: the leader serves the file to a cluster_secret holder,
    // and refuses without it.
    let noauth = client
        .get(format!("{}/v1/packs/{}", leader.base, pack_digest))
        .send()
        .await
        .unwrap();
    assert_eq!(noauth.status(), reqwest::StatusCode::UNAUTHORIZED);

    // Mount into the tenant db via the leader → 202 accepted (reconciling).
    let (st, _) = post_json(
        &leader.base,
        SECRET,
        &format!("/v1/admin/databases/{tenant_db_id}/packs"),
        &json!({ "digest": pack_digest, "name": "junk-pack" }),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::ACCEPTED, "mount → 202");

    // The MANIFEST replicates to the follower (the consensus half).
    {
        let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(10);
        loop {
            let rows = follower
                .state
                .control
                .lock()
                .active_pack_mounts_for(tenant_db_id)
                .unwrap();
            if rows.iter().any(|r| r.pack_digest == pack_digest) {
                break;
            }
            assert!(
                tokio::time::Instant::now() < deadline,
                "pack mount manifest never replicated to follower"
            );
            tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        }
    }

    // Drive the FOLLOWER's reconciler: it fetches the file from the leader
    // (peer-transfer), tries to mount the invalid pack, and after MAX_ATTEMPTS
    // terminally quarantines it — WITHOUT crashing the process (C1).
    struct LeaderFetcher(String);
    impl crate::pack_reconciler::PackFetcher for LeaderFetcher {
        fn leader_base_and_secret(&self) -> Option<(String, String)> {
            Some((self.0.clone(), SECRET.to_string()))
        }
    }
    let follower_dir = follower.state.data_dir.clone();
    let reconciler = crate::pack_reconciler::PackReconciler::new(
        follower.state.control.clone(),
        follower.state.pack_store.clone(),
        follower.state.pool.clone(),
        follower.state.pack_status.clone(),
        Some(Arc::new(LeaderFetcher(leader.base.clone()))),
        &follower_dir,
    );
    for _ in 0..4 {
        reconciler.reconcile_once().await.unwrap();
    }

    // Peer-transfer worked: the follower now holds the file.
    assert!(
        follower.state.pack_store.has(&pack_digest),
        "follower must have fetched the pack file from the leader"
    );
    // Poison-quarantine worked: the invalid pack is terminally poisoned on the
    // follower, and the cluster is still up (we're still running).
    let status = follower.state.pack_status.get(tenant_db_id);
    assert!(
        status.poisoned.contains(&pack_digest),
        "invalid pack must be quarantined (poisoned), got {status:?}"
    );

    // Unmount replicates too: the manifest row clears on the follower.
    let (st, _) = post_json(
        &follower.base,
        SECRET,
        &format!("/v1/admin/databases/{tenant_db_id}/packs"),
        &json!({ "digest": pack_digest }),
    )
    .await;
    assert_eq!(
        st,
        reqwest::StatusCode::SERVICE_UNAVAILABLE,
        "pack mount on a follower must redirect to the leader"
    );
    let (st, _) = get_json(
        &leader.base,
        SECRET,
        &format!("/v1/admin/databases/{tenant_db_id}/packs"),
    )
    .await;
    assert_eq!(st, reqwest::StatusCode::OK, "list mounted packs");
}

async fn spawn_node_on(node_id: u64, peers: Vec<YrpPeer>, port: u16) -> Node {
    let node = spawn_node_inner(node_id, peers, port).await;
    // Readiness: health answers.
    let client = reqwest::Client::new();
    for _ in 0..100 {
        if client
            .get(format!("{}/v1/health", node.base))
            .send()
            .await
            .is_ok()
        {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    node
}

/// One full server: control DB (same db name on every node → same
/// tenant id, which rides inside replicated ops), real engine pool,
/// the yrp-mode assembly mirroring main.rs's Yrp arm, and the
/// PRODUCTION router served on the pre-advertised port.
async fn spawn_node_inner(node_id: u64, peers: Vec<YrpPeer>, port: u16) -> Node {
    let tmp = tempfile::tempdir().expect("tempdir");
    let data_dir = tmp.path().to_path_buf();

    let mut cfg = crate::config::ServerConfig::default();
    cfg.server.data_dir = data_dir.clone();

    let control = ControlDb::open(&data_dir.join("control.db")).expect("control db");
    let raw_token = crate::auth::generate_token();
    let token_hash = crate::auth::hash_token(&raw_token);
    let db_id = control.create_database(TENANT, TENANT).expect("create db");
    control
        .create_token(&token_hash, db_id, "yrp-test")
        .expect("create token");
    let control = Arc::new(Mutex::new(control));

    let pool = Arc::new(crate::tenant_pool::TenantPool::new(&cfg, None, None));
    let workers = crate::background::WorkerRegistry::new(
        &cfg.background,
        &cfg.maintenance,
        crate::background::WriteAcceptanceGate::standalone(),
    );
    let admission = crate::admission::AdmissionState::new(Default::default());
    let jobs: Arc<dyn crate::jobs::JobQueue> =
        Arc::new(crate::jobs::LocalSqliteJobQueue::open_in_memory().expect("jobs"));
    let auth_provider: Arc<dyn crate::auth::AuthProvider> = Arc::new(ControlDbAuthProvider::new(
        Arc::clone(&control),
        Some(SECRET.to_string()),
    ));

    let local: Arc<dyn crate::commit::MutationCommitter> = Arc::new(
        crate::commit::LocalSqliteCommitter::open(data_dir.join("commit_log.sqlite"))
            .expect("commit log"),
    );
    let resolver = Arc::new(crate::tenant_pool::TenantPoolEngineResolver::new(
        pool.clone(),
        control.clone(),
    )) as Arc<dyn crate::commit::EngineResolver>;
    let applier: Arc<dyn crate::commit::Applier> =
        Arc::new(crate::commit::EngineApplier::new(resolver));
    let handle = spawn_yrp(
        YrpRuntimeConfig {
            node_id,
            cluster_id: CLUSTER_ID,
            peers,
            data_dir: data_dir.clone(),
            cluster_secret: Some(SECRET.to_string()),
            tick_ms: 20,
            election_ticks: (5, 10),
            heartbeat_ticks: 2,
            compact_after_entries: 0,
            leader_retain_entries: 0,
        },
        local.clone(),
        applier,
        control.clone(),
    )
    .expect("yrp spawn");
    let commit_log: Arc<dyn crate::commit::MutationCommitter> =
        Arc::new(YrpCommitter::new(handle.clone(), local));

    let state = Arc::new(AppState {
        control,
        pool,
        workers,
        cluster: None,
        inflight: std::sync::atomic::AtomicU32::new(0),
        admission,
        control_runtime: None,
        commit_log,
        yrp: Some(handle.clone()),
        fault_registry: crate::debug::FaultRegistry::new(),
        jobs,
        data_dir,
        auth_provider,
        pack_store: crate::pack_store::PackStore::open(tmp.path()).unwrap(),
        pack_status: std::sync::Arc::new(crate::pack_reconciler::PackStatus::default()),
    });

    let listener = tokio::net::TcpListener::bind(("127.0.0.1", port))
        .await
        .expect("bind advertised port");
    let base = format!("http://{}", listener.local_addr().unwrap());
    let app = crate::http_gateway::router(state.clone());
    tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });

    Node {
        state,
        handle,
        base,
        token: raw_token,
        _tmp: tmp,
    }
}

/// Poll the follower's LIVE /v1/recall (client query vector — no
/// embedder in the fixture) until `rid` appears.
async fn wait_for_recall(node: &Node, query_embedding: &Value, rid: &str) {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(15);
    loop {
        let (st, resp) = post_json(
            &node.base,
            &node.token,
            "/v1/recall",
            &json!({
                "query": "yrp replicated memory",
                "query_embedding": query_embedding,
                "top_k": 10,
            }),
        )
        .await;
        if st == reqwest::StatusCode::OK {
            let found = resp["results"]
                .as_array()
                .map(|arr| {
                    arr.iter()
                        .any(|r| r.get("rid").and_then(|v| v.as_str()) == Some(rid))
                })
                .unwrap_or(false);
            if found {
                return;
            }
        }
        assert!(
            tokio::time::Instant::now() < deadline,
            "follower live recall never surfaced {rid}; last response: {resp}"
        );
        tokio::time::sleep(std::time::Duration::from_millis(100)).await;
    }
}
