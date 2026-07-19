//! Chaos gate (saga 237): mechanical proof, not vibes — the cluster-mode
//! release gate the v0.8.18 regression made mandatory.
//!
//! Every scenario runs REAL servers over real localhost HTTP with real
//! persistence, then breaks something and asserts the two properties the
//! whole protocol exists for:
//! - **never-double-write**: a key acked with rid R answers R forever;
//! - **never-lost-ack**: a 200 survives kills, partitions, and rejoins.
//!
//! Faults are injected at the receive seam (`FaultRegistry::verdict` in
//! the `/v1/yrp/msg` route) — the codex-D1 design: externally
//! controllable, restart-surviving, exercising the production HTTP path.

use serde_json::json;

use super::testkit::*;
use crate::debug::FaultKind;

/// Retry a keyed write (same key + same text) and assert it answers the
/// ORIGINAL rid as a silent HIT — on whichever node currently leads.
async fn assert_dedupes_everywhere(
    nodes: &[&TestNode],
    key: &str,
    text: &str,
    emb: &serde_json::Value,
    expected_rid: &str,
) {
    let (rid, _) = keyed_write_until_accepted(nodes, key, text, emb).await;
    assert_eq!(
        rid, expected_rid,
        "key {key} answered a DIFFERENT rid after chaos — double-write"
    );
}

/// Kill the leader mid-stream under keyed write load. Acked writes must
/// dedupe to their original rid on the new leader; nothing double-writes;
/// the restarted node catches up live.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn kill_leader_under_keyed_load_never_double_writes() {
    let (nodes, _tmps) = spawn_cluster(3, ClusterSpec::default()).await;
    let mut live: Vec<TestNode> = nodes;
    let emb = embedding(0.3);

    // Phase 1: writes while healthy.
    let all: Vec<&TestNode> = live.iter().collect();
    let mut rids = Vec::new();
    for i in 0..5 {
        let (rid, _) = keyed_write_until_accepted(
            &all,
            &format!("chaos-k{i}"),
            &format!("chaos replicated memory {i}"),
            &emb,
        )
        .await;
        rids.push(rid);
    }

    // Kill the current leader.
    let leader_idx = wait_leader(&live.iter().collect::<Vec<_>>()).await;
    let dead = live.remove(leader_idx).kill();

    // Phase 2: writes against the survivors (forces re-election).
    let survivors: Vec<&TestNode> = live.iter().collect();
    for i in 5..10 {
        let (rid, _) = keyed_write_until_accepted(
            &survivors,
            &format!("chaos-k{i}"),
            &format!("chaos replicated memory {i}"),
            &emb,
        )
        .await;
        rids.push(rid);
    }

    // EVERY acked key — from before and after the kill — dedupes to its
    // original rid on the surviving cluster.
    for i in 0..10 {
        assert_dedupes_everywhere(
            &survivors,
            &format!("chaos-k{i}"),
            &format!("chaos replicated memory {i}"),
            &emb,
            &rids[i],
        )
        .await;
    }

    // The killed ex-leader restarts from disk and converges live.
    let revived = dead.restart().await;
    wait_for_recall(&revived, &emb, &rids[9]).await;
}

/// Partition the leader from the majority: its writes stall (never
/// commit), the majority elects, and on heal the stale leader is fenced,
/// truncates, and adopts canonical history — retries dedupe, nothing
/// leaks.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn partition_and_heal_fences_stale_leader() {
    let (nodes, _tmps) = spawn_cluster(3, ClusterSpec::default()).await;
    let emb = embedding(1.1);
    let all: Vec<&TestNode> = nodes.iter().collect();
    let leader_idx = wait_leader(&all).await;
    let leader_id = nodes[leader_idx].node_id as u32;
    let others: Vec<u32> = nodes
        .iter()
        .filter(|n| n.node_id as u32 != leader_id)
        .map(|n| n.node_id as u32)
        .collect();

    // Full cut, injected at every RECEIVER's registry.
    for n in &nodes {
        n.state.fault_registry.inject(
            FaultKind::Partition {
                side_a: vec![leader_id],
                side_b: others.clone(),
            },
            None,
        );
    }

    // A write fired at the partitioned leader must never report success —
    // fire-and-forget with a short client timeout; the claim may sit
    // tentative and is expected to be truncated on heal OR re-executed
    // cleanly on retry (both are correct; double-success is not).
    let doomed = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()
        .unwrap()
        .post(format!("{}/v1/remember", nodes[leader_idx].base))
        .bearer_auth(&nodes[leader_idx].token)
        .json(&json!({
            "text": "doomed partition write",
            "embedding": emb,
            "idempotency_key": "chaos-doomed",
        }))
        .send()
        .await;
    assert!(
        doomed.is_err() || !doomed.unwrap().status().is_success(),
        "a partitioned leader must not ack a write"
    );

    // Majority side elects and commits.
    let majority: Vec<&TestNode> = nodes
        .iter()
        .filter(|n| n.node_id as u32 != leader_id)
        .collect();
    let (rid_k, _) =
        keyed_write_until_accepted(&majority, "chaos-part-k", "canonical partition write", &emb)
            .await;

    // Heal.
    for n in &nodes {
        n.state.fault_registry.clear();
    }

    // The canonical write dedupes cluster-wide; the ex-leader converges
    // to it live (its tentative suffix truncated in favor of canon).
    assert_dedupes_everywhere(
        &all,
        "chaos-part-k",
        "canonical partition write",
        &emb,
        &rid_k,
    )
    .await;
    wait_for_recall(&nodes[leader_idx], &emb, &rid_k).await;

    // The doomed key resolves EXACTLY-ONCE post-heal: first retry gets
    // some rid, second retry gets the same one.
    let (rid_d1, _) =
        keyed_write_until_accepted(&all, "chaos-doomed", "doomed partition write", &emb).await;
    assert_dedupes_everywhere(
        &all,
        "chaos-doomed",
        "doomed partition write",
        &emb,
        &rid_d1,
    )
    .await;
}

/// Torn replication state boots QUARANTINED (fail closed, health says
/// so), then rejoins via a quorum-authorized grant and resumes as a
/// follower — the CT 141 posture, live end to end.
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn torn_state_boots_quarantined_then_rejoins_via_grant() {
    let (nodes, _tmps) = spawn_cluster(3, ClusterSpec::default()).await;
    let emb = embedding(2.2);
    let all: Vec<&TestNode> = nodes.iter().collect();
    let (_rid_a, _) =
        keyed_write_until_accepted(&all, "chaos-torn-a", "pre-tear write", &emb).await;

    // Kill node 3 and tear its replication state (NOT its engine data).
    let mut live = nodes;
    let victim_idx = live
        .iter()
        .position(|n| n.node_id == 3 && !n.handle.is_leader())
        .unwrap_or_else(|| {
            live.iter()
                .position(|n| !n.handle.is_leader())
                .expect("some follower")
        });
    let dead = live.remove(victim_idx).kill();
    std::fs::write(dead.data_dir.join("yrp.state"), b"CORRUPT GARBAGE BYTES").unwrap();

    // Meanwhile the cluster keeps committing.
    let survivors: Vec<&TestNode> = live.iter().collect();
    let (rid_b, _) =
        keyed_write_until_accepted(&survivors, "chaos-torn-b", "post-tear write", &emb).await;

    // Restart: boot inspection must quarantine, honestly, on /v1/health.
    let revived = dead.restart().await;
    let client = reqwest::Client::new();
    let health: serde_json::Value = client
        .get(format!("{}/v1/health", revived.base))
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(
        health["status"],
        json!("quarantined"),
        "torn state must surface as quarantine: {health}"
    );

    // Rejoin loop → leader grant → adoption → quarantine clears.
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(30);
    while revived.handle.quarantine_reasons().is_some() {
        assert!(
            tokio::time::Instant::now() < deadline,
            "quarantined node never rejoined"
        );
        tokio::time::sleep(std::time::Duration::from_millis(200)).await;
    }
    // Forensics preserved before resync (the automated CT 141 backup).
    let preserved = std::fs::read_dir(&revived.data_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .any(|e| {
            e.file_name()
                .to_string_lossy()
                .starts_with("yrp.preserved-")
        });
    assert!(preserved, "old state must be preserved before resync");

    // Post-rejoin the node replicates live again.
    wait_for_recall(&revived, &emb, &rid_b).await;
}

/// A straggler that falls below the leader's compaction base catches up
/// via InstallSnapshot — and claims RIDE the snapshot: a keyed retry of
/// a COMPACTED entry still dedupes (P1-9, live on real HTTP).
#[tokio::test(flavor = "multi_thread", worker_threads = 4)]
async fn straggler_beyond_gc_catches_up_and_compacted_claims_survive() {
    let (nodes, _tmps) = spawn_cluster(
        3,
        ClusterSpec {
            compact_after_entries: 8,
            leader_retain_entries: 2,
        },
    )
    .await;
    let emb = embedding(3.3);
    let all: Vec<&TestNode> = nodes.iter().collect();

    let (rid_first, _) =
        keyed_write_until_accepted(&all, "chaos-gc-first", "the earliest keyed write", &emb).await;

    // Kill a follower, then push the cluster far past the compaction
    // threshold so the victim's next_index falls below the leader's base.
    let mut live = nodes;
    let victim_idx = live
        .iter()
        .position(|n| !n.handle.is_leader())
        .expect("some follower");
    let dead = live.remove(victim_idx).kill();

    let survivors: Vec<&TestNode> = live.iter().collect();
    let mut last_rid = String::new();
    for i in 0..30 {
        let (rid, _) = keyed_write_until_accepted(
            &survivors,
            &format!("chaos-gc-{i}"),
            &format!("bulk write {i}"),
            &emb,
        )
        .await;
        last_rid = rid;
    }

    // Restart the straggler: it must converge (snapshot install + tail
    // replication) to the newest write, LIVE.
    let revived = dead.restart().await;
    wait_for_recall(&revived, &emb, &last_rid).await;

    // The claim of the FIRST write — whose log entry is long compacted on
    // the leader — still dedupes to the original rid. Claims ride the
    // snapshot; compaction may never open a replay window.
    let with_revived: Vec<&TestNode> = survivors
        .iter()
        .copied()
        .chain(std::iter::once(&revived))
        .collect();
    assert_dedupes_everywhere(
        &with_revived,
        "chaos-gc-first",
        "the earliest keyed write",
        &emb,
        &rid_first,
    )
    .await;
}
