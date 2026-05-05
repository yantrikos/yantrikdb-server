//! Wire-format conformance tests for RFC 010 PR-3.
//!
//! ## What this gates
//!
//! Every `MemoryMutation` variant has a pinned JSON wire format at v1.0.
//! This test serializes a representative instance of each variant and
//! compares the output byte-for-byte to a frozen string. Any change to
//! field order, naming, type, or shape that would break replay trips
//! one of these tests with a clear diff.
//!
//! Once landed, this contract is **inviolable**: an entry written today
//! at wire 1.0 must be readable by every future v1.x build forever.
//! Breaking it requires bumping to wire 2.0 (a major upgrade).
//!
//! ## Why golden tests
//!
//! Property-based tests (round-trip serialize-deserialize) prove
//! reflexive consistency but NOT format stability. A subtle change like
//! reordering struct fields or renaming a serde alias still round-trips
//! — but breaks any other build that hasn't been recompiled.
//!
//! Golden tests pin the actual byte sequence. If a field order changes,
//! the test fails with `assertion `left == right` failed: expected ...
//! got ...`. The diff tells the reviewer exactly what changed.
//!
//! ## When to update a golden
//!
//! Almost never. Updating a golden = breaking the wire format. The only
//! valid reason is bumping the major wire version (e.g. v2.0). When
//! that happens, the v1.0 goldens stay as witnesses of historical
//! compat — they're imported by replay-from-old-data tests.
//!
//! Adding a new variant in v1.1+? Add a new golden test alongside, do
//! NOT modify v1.0 ones. The whole point is that v1.0 stays frozen.

#[path = "../src/cache/mod.rs"]
mod cache;

#[path = "../src/commit/mod.rs"]
mod commit;

#[path = "../src/forget/mod.rs"]
mod forget;

#[path = "../src/key_provider/mod.rs"]
mod key_provider;

#[path = "../src/index/mod.rs"]
mod index;

#[path = "../src/jobs/mod.rs"]
mod jobs;

#[path = "../src/migrations/mod.rs"]
mod migrations;

#[path = "../src/version/mod.rs"]
mod version;

use commit::MemoryMutation;

#[test]
fn upsert_memory_v1_1_wire_format() {
    // RFC 010 PR-6.2 — UpsertMemory at wire 1.1 carries three additional
    // materialized-state fields beyond the v1.0 shape: extracted_entities,
    // created_at_unix_micros, embedding_model. All three are #[serde(default)]
    // so a v1.0 payload still deserializes cleanly (covered by
    // historical_v1_0_payload_round_trips_into_current_build).
    //
    // This test pins the v1.1 wire output. If you add another field, this
    // golden string must change AND the wire minor must bump (1.1 → 1.2).
    // If you change a field's name or type, you must bump the wire major.
    let m = MemoryMutation::UpsertMemory {
        rid: "mem_test_1".into(),
        text: "the cat sat on the mat".into(),
        memory_type: "semantic".into(),
        importance: 0.5,
        valence: 0.0,
        half_life: 168.0,
        namespace: "default".into(),
        certainty: 1.0,
        domain: "general".into(),
        source: "user".into(),
        emotional_state: None,
        embedding: None,
        metadata: serde_json::json!({}),
        extracted_entities: vec![],
        created_at_unix_micros: None,
        embedding_model: None,
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"UpsertMemory","rid":"mem_test_1","text":"the cat sat on the mat","memory_type":"semantic","importance":0.5,"valence":0.0,"half_life":168.0,"namespace":"default","certainty":1.0,"domain":"general","source":"user","emotional_state":null,"embedding":null,"metadata":{},"extracted_entities":[],"created_at_unix_micros":null,"embedding_model":null}"#;
    assert_eq!(
        actual, expected,
        "WIRE FORMAT DRIFT — UpsertMemory v1.1 has changed. \
         Bumping wire minor (to 1.2) required if adding a new field, \
         OR bumping wire major (to 2.x) required for a field rename or \
         type change. Existing commit logs in the field will fail to \
         deserialize otherwise."
    );
}

#[test]
fn update_memory_patch_v1_0_wire_format() {
    let m = MemoryMutation::UpdateMemoryPatch {
        rid: "mem_test_2".into(),
        text: Some("updated".into()),
        importance: Some(0.7),
        valence: None,
        certainty: None,
        metadata_patch: Some(serde_json::json!({"tag": "x"})),
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"UpdateMemoryPatch","rid":"mem_test_2","text":"updated","importance":0.7,"valence":null,"certainty":null,"metadata_patch":{"tag":"x"}}"#;
    assert_eq!(actual, expected, "UpdateMemoryPatch v1.0 wire drift");
}

#[test]
fn tombstone_memory_v1_0_wire_format() {
    let m = MemoryMutation::TombstoneMemory {
        rid: "mem_test_3".into(),
        reason: Some("user requested".into()),
        requested_at_unix_micros: 1_700_000_000_000_000,
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"TombstoneMemory","rid":"mem_test_3","reason":"user requested","requested_at_unix_micros":1700000000000000}"#;
    assert_eq!(actual, expected, "TombstoneMemory v1.0 wire drift");
}

#[test]
fn purge_memory_v1_0_wire_format() {
    let m = MemoryMutation::PurgeMemory {
        rid: "mem_test_4".into(),
        purge_epoch: 42,
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"PurgeMemory","rid":"mem_test_4","purge_epoch":42}"#;
    assert_eq!(actual, expected, "PurgeMemory v1.0 wire drift");
}

#[test]
fn upsert_entity_edge_v1_0_wire_format() {
    let m = MemoryMutation::UpsertEntityEdge {
        edge_id: "edge_test_1".into(),
        src: "alice".into(),
        dst: "bob".into(),
        rel_type: "knows".into(),
        weight: 0.9,
        namespace: "default".into(),
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"UpsertEntityEdge","edge_id":"edge_test_1","src":"alice","dst":"bob","rel_type":"knows","weight":0.9,"namespace":"default"}"#;
    assert_eq!(actual, expected, "UpsertEntityEdge v1.0 wire drift");
}

#[test]
fn delete_entity_edge_v1_0_wire_format() {
    let m = MemoryMutation::DeleteEntityEdge {
        edge_id: "edge_test_1".into(),
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"DeleteEntityEdge","edge_id":"edge_test_1"}"#;
    assert_eq!(actual, expected, "DeleteEntityEdge v1.0 wire drift");
}

#[test]
fn tenant_config_patch_v1_0_wire_format() {
    let m = MemoryMutation::TenantConfigPatch {
        key: "admission.max_concurrent_expanded_recall".into(),
        value: serde_json::json!(8),
    };
    let actual = serde_json::to_string(&m).expect("serialize");
    let expected = r#"{"kind":"TenantConfigPatch","key":"admission.max_concurrent_expanded_recall","value":8}"#;
    assert_eq!(actual, expected, "TenantConfigPatch v1.0 wire drift");
}

#[test]
fn historical_v1_0_payload_round_trips_into_current_build() {
    // The strongest contract: a JSON payload literally written by a
    // hypothetical v1.0 build today must deserialize cleanly into the
    // current build (and any future v1.x build) and round-trip back to
    // the same shape. This catches:
    // - field renames (would break deserialize)
    // - field type changes (would break deserialize)
    // - removed fields (might break if not optional)
    let v1_0_payload = r#"{
        "kind": "UpsertMemory",
        "rid": "historical_payload_test",
        "text": "this was written by a v1.0 build",
        "memory_type": "episodic",
        "importance": 0.42,
        "valence": -0.1,
        "half_life": 240.0,
        "namespace": "tenant_x",
        "certainty": 0.85,
        "domain": "specialized",
        "source": "extractor-v1",
        "emotional_state": "concerned",
        "embedding": [0.1, 0.2, 0.3],
        "metadata": {"audit_id": 7, "tag": "verified"}
    }"#;
    let m: MemoryMutation =
        serde_json::from_str(v1_0_payload).expect("v1.0 payload must deserialize cleanly forever");
    match &m {
        MemoryMutation::UpsertMemory {
            rid,
            importance,
            emotional_state,
            embedding,
            extracted_entities,
            created_at_unix_micros,
            embedding_model,
            ..
        } => {
            assert_eq!(rid, "historical_payload_test");
            assert!((importance - 0.42).abs() < 1e-9);
            assert_eq!(emotional_state.as_deref(), Some("concerned"));
            assert_eq!(embedding.as_ref().map(|v| v.len()), Some(3));
            // v1.1 fields default-empty when deserializing a v1.0 payload —
            // the cross-version compat property that lets old commit logs
            // replay through a v1.1 binary.
            assert!(extracted_entities.is_empty());
            assert!(created_at_unix_micros.is_none());
            assert!(embedding_model.is_none());
        }
        other => panic!("wrong variant after round-trip: {other:?}"),
    }
    // Round-trip back; should still be UpsertMemory.
    let reserialized = serde_json::to_string(&m).expect("re-serialize");
    let _re_deserialized: MemoryMutation =
        serde_json::from_str(&reserialized).expect("re-deserialize");
}

#[test]
fn variant_feature_flags_match_registry() {
    // Every variant in MemoryMutation MUST have a matching entry in
    // crate::version::gate::FEATURE_FLOORS. If you add a variant and
    // forget to register the floor, this test fails — preventing the
    // "writer emits a variant some peers can't accept" footgun.
    let variants = vec![
        MemoryMutation::UpsertMemory {
            rid: "x".into(),
            text: String::new(),
            memory_type: String::new(),
            importance: 0.0,
            valence: 0.0,
            half_life: 0.0,
            namespace: String::new(),
            certainty: 0.0,
            domain: String::new(),
            source: String::new(),
            emotional_state: None,
            embedding: None,
            metadata: serde_json::Value::Null,
            extracted_entities: vec![],
            created_at_unix_micros: None,
            embedding_model: None,
        },
        MemoryMutation::UpdateMemoryPatch {
            rid: "x".into(),
            text: None,
            importance: None,
            valence: None,
            certainty: None,
            metadata_patch: None,
        },
        MemoryMutation::TombstoneMemory {
            rid: "x".into(),
            reason: None,
            requested_at_unix_micros: 0,
        },
        MemoryMutation::PurgeMemory {
            rid: "x".into(),
            purge_epoch: 0,
        },
        MemoryMutation::UpsertEntityEdge {
            edge_id: "x".into(),
            src: String::new(),
            dst: String::new(),
            rel_type: String::new(),
            weight: 0.0,
            namespace: String::new(),
        },
        MemoryMutation::DeleteEntityEdge {
            edge_id: "x".into(),
        },
        MemoryMutation::TenantConfigPatch {
            key: "x".into(),
            value: serde_json::Value::Null,
        },
    ];

    let registered: std::collections::HashSet<&str> = version::gate::FEATURE_FLOORS
        .iter()
        .map(|(name, _)| *name)
        .collect();

    for v in &variants {
        let flag = v.feature_flag();
        assert!(
            registered.contains(flag),
            "variant `{}` has feature_flag `{}` but no entry in FEATURE_FLOORS — \
             writers can't gate emitting this variant on cluster_min",
            v.variant_name(),
            flag
        );
    }
}

#[test]
fn all_initial_variants_introduced_at_1_0() {
    // The whole grammar shipped at v1.0. Future variants will return
    // higher minor versions; pin the initial set as a regression guard.
    use commit::MemoryMutation::*;
    let cases: Vec<(&str, MemoryMutation)> = vec![
        (
            "UpsertMemory",
            UpsertMemory {
                rid: "x".into(),
                text: String::new(),
                memory_type: String::new(),
                importance: 0.0,
                valence: 0.0,
                half_life: 0.0,
                namespace: String::new(),
                certainty: 0.0,
                domain: String::new(),
                source: String::new(),
                emotional_state: None,
                embedding: None,
                metadata: serde_json::Value::Null,
                extracted_entities: vec![],
                created_at_unix_micros: None,
                embedding_model: None,
            },
        ),
        (
            "TombstoneMemory",
            TombstoneMemory {
                rid: "x".into(),
                reason: None,
                requested_at_unix_micros: 0,
            },
        ),
        (
            "TenantConfigPatch",
            TenantConfigPatch {
                key: "x".into(),
                value: serde_json::Value::Null,
            },
        ),
    ];

    for (name, m) in &cases {
        let wv = m.wire_introduced_at();
        assert_eq!(wv.major, 1, "{name} should be wire 1.x");
        assert_eq!(wv.minor, 0, "{name} should be introduced at minor 0");
    }
}
