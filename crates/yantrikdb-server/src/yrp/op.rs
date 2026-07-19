//! The replicated-op envelope: what actually rides inside
//! [`super::replica::Payload::Op`] bytes (runtime-integration slice).
//!
//! ## Encoding choice — serde_json, deliberately
//!
//! [`crate::commit::MemoryMutation`] is an internally-tagged serde enum
//! (`#[serde(tag = "kind")]`). Internally-tagged enums require a
//! self-describing format on deserialize (`deserialize_any`), which
//! bincode is not — bincode round-trips of this envelope FAIL at decode
//! time. serde_json is the same encoding the commit-log `payload` column
//! and the openraft log already use for mutations, so the YRP log stays
//! byte-compatible with the rest of the substrate's durable formats.
//! (The outer wire envelope [`super::driver::WireMsg`] IS bincode — it
//! contains only externally-tagged protocol types plus these opaque
//! bytes, which bincode treats as a plain byte vector.)
//!
//! ## Why the rid lives in here (codex F3)
//!
//! The gateway allocates the client-visible rid BEFORE proposing and
//! bakes it into the mutation. Every replica therefore applies the same
//! rid deterministically, and a keyed retry answered from the claims
//! table can recover its outcome from the entry (or the sink's outcome
//! table) — no volatile per-connection state required.

use serde::{Deserialize, Serialize};

use crate::commit::{MemoryMutation, OpId, TenantId};

/// One replicated engine operation. Everything a follower needs to apply
/// deterministically: the tenant, the commit-log idempotency op_id, the
/// full materialized mutation (rid + embedding + entities + timestamps
/// inside), and — for keyed writes — the client's idempotency key string
/// (the entry's `key` field carries only its 64-bit digest; the full
/// string rides here so the apply sink can store it for collision
/// verification).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct YrpOp {
    pub tenant_id: TenantId,
    pub op_id: OpId,
    pub mutation: MemoryMutation,
    /// The client idempotency key for keyed writes; `None` for unkeyed.
    pub idempotency_key: Option<String>,
}

impl YrpOp {
    /// Serialize for `Payload::Op`. Infallible in practice (the mutation
    /// grammar contains no non-string-keyed maps), but surfaced as a
    /// Result so a grammar change that breaks encoding fails loudly at
    /// the propose site, not silently on a follower.
    pub fn encode(&self) -> Result<Vec<u8>, String> {
        serde_json::to_vec(self).map_err(|e| format!("encode YrpOp: {e}"))
    }

    pub fn decode(bytes: &[u8]) -> Result<Self, String> {
        serde_json::from_slice(bytes).map_err(|e| format!("decode YrpOp: {e}"))
    }
}

/// FNV-1a 64-bit — the stable digest used to derive the protocol-level
/// claim key (`LogEntry.key: u64`) from string identities. Chosen because
/// it is trivially portable and has NO platform/process variation (std's
/// `DefaultHasher` is explicitly unstable across releases — unacceptable
/// for a value that is replicated and persisted).
///
/// Collisions: two distinct identities hashing to the same u64 would make
/// the second write dedupe against the first. The apply sink stores the
/// FULL key string with each outcome, and the gateway verifies it on
/// every dedupe answer — a collision is detected and refused (409), never
/// silently mis-deduped. At ~2^-64 per pair it is a documented
/// astronomically-unlikely refusal, not a correctness hole.
pub fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

/// Claim key for a client-supplied idempotency key, scoped per tenant so
/// identical key strings from different tenants never collide by design.
pub fn claim_key_for_idempotency(tenant: TenantId, key: &str) -> u64 {
    let mut buf = Vec::with_capacity(key.len() + 12);
    buf.extend_from_slice(b"idem:");
    buf.extend_from_slice(&tenant.0.to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    fnv1a64(&buf)
}

/// Claim key for an unkeyed write: derived from the op_id, giving every
/// replicated mutation retry-idempotency at the protocol layer with the
/// same semantics the commit log's `(tenant, op_id)` unique index
/// provides at the storage layer.
pub fn claim_key_for_op(tenant: TenantId, op_id: &OpId) -> u64 {
    let mut buf = Vec::with_capacity(64);
    buf.extend_from_slice(b"op:");
    buf.extend_from_slice(&tenant.0.to_le_bytes());
    buf.extend_from_slice(op_id.to_string().as_bytes());
    fnv1a64(&buf)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_op(key: Option<&str>) -> YrpOp {
        YrpOp {
            tenant_id: TenantId::new(7),
            op_id: OpId::new_random(),
            mutation: MemoryMutation::UpsertMemory {
                rid: "0198-test-rid".into(),
                text: "hello".into(),
                memory_type: "semantic".into(),
                importance: 0.5,
                valence: 0.0,
                half_life: 168.0,
                metadata: serde_json::json!({}),
                namespace: "ns".into(),
                certainty: 1.0,
                domain: "work".into(),
                source: "user".into(),
                emotional_state: None,
                embedding: Some(vec![0.25, -0.5]),
                extracted_entities: vec!["hello".into()],
                created_at_unix_micros: Some(1_784_000_000_000_000),
                embedding_model: Some("default".into()),
            },
            idempotency_key: key.map(String::from),
        }
    }

    /// The load-bearing property: the envelope round-trips through the
    /// bytes that ride in `Payload::Op` — including the internally-tagged
    /// mutation enum that bincode cannot carry.
    #[test]
    fn yrp_op_round_trips_through_payload_bytes() {
        for key in [None, Some("client-key-1")] {
            let op = sample_op(key);
            let bytes = op.encode().expect("encode");
            let back = YrpOp::decode(&bytes).expect("decode");
            assert_eq!(op, back);
        }
    }

    /// The whole point of the digest: stable across processes/platforms.
    /// These constants are the contract — if this test ever fails, the
    /// hash changed and every persisted claim key is invalidated.
    #[test]
    fn claim_key_digest_is_pinned() {
        assert_eq!(fnv1a64(b""), 0xcbf29ce484222325);
        assert_eq!(fnv1a64(b"a"), 0xaf63dc4c8601ec8c);
        let k1 = claim_key_for_idempotency(TenantId::new(1), "same-key");
        let k2 = claim_key_for_idempotency(TenantId::new(2), "same-key");
        assert_ne!(k1, k2, "tenant scoping must separate identical keys");
        assert_eq!(
            k1,
            claim_key_for_idempotency(TenantId::new(1), "same-key"),
            "digest must be deterministic"
        );
    }
}
