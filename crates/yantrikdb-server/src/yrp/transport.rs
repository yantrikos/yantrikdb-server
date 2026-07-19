//! HTTP transport adapter — YRP wire messages over the cluster's HTTP
//! plane (runtime-integration slice; codex F4 queue discipline).
//!
//! ## Shape
//!
//! - **Outbound**: [`HttpTransport`] implements [`Transport`]. One client
//!   task per peer; the owner's `send` classifies and enqueues, never
//!   blocks, never observes delivery failures (the protocol's timers and
//!   retransmits are the recovery mechanism — silence is the contract).
//! - **Inbound**: the axum route `POST /v1/yrp/msg` (mounted by the
//!   gateway) decodes the bincode envelope and funnels
//!   `DriverEvent::Inbound` to the owner.
//!
//! ## Queue discipline (codex F4)
//!
//! Replication traffic (AppendEntries / InstallSnapshot) is CUMULATIVE:
//! a newer message strictly supersedes an older one to the same peer. So
//! each peer gets a single latest-message slot — under backpressure we
//! coalesce to the newest instead of queueing a backlog. Control traffic
//! (votes, rejoin) is small and non-cumulative: a bounded queue; overflow
//! drops the NEWEST (the protocol re-times-out and re-sends).
//!
//! Codex F4 also suggested term-generation tags to cancel stale sends on
//! step-down. With a latest-only slot the stale window is exactly ONE
//! in-flight message, which the receiver term-fences — the tag machinery
//! buys nothing here, so it is deliberately omitted (documented deviation
//! for the codex review pass).

use std::collections::BTreeMap;
use std::sync::Arc;
use std::time::Duration;

use parking_lot::Mutex;
use tokio::sync::{mpsc, Notify};

use super::driver::{DriverEvent, Transport, WireMsg};
use super::replica::Message;
use super::types::NodeId;

/// Wire envelope for `POST /v1/yrp/msg`: (sender node id, message),
/// bincode-encoded. Everything inside is externally-tagged serde —
/// bincode-safe (the engine-mutation bytes inside `Payload::Op` are an
/// opaque byte vector at this layer; see `yrp::op` for THEIR encoding).
pub type WireEnvelope = (u64, WireMsg);

/// Bounded control-queue depth per peer (votes, rejoin traffic).
const CONTROL_QUEUE_DEPTH: usize = 64;

/// Per-request send timeout. Losing a message is always safe; wedging a
/// peer task behind a black-holed connection is not.
const SEND_TIMEOUT: Duration = Duration::from_secs(2);

struct PeerQueues {
    /// Latest replication message only — newer supersedes older.
    latest: Mutex<Option<WireMsg>>,
    wake: Notify,
    control: mpsc::Sender<WireMsg>,
}

/// Outbound HTTP transport: one sender task per peer.
pub struct HttpTransport {
    me: NodeId,
    peers: BTreeMap<NodeId, Arc<PeerQueues>>,
}

impl HttpTransport {
    /// `peer_urls`: peer node id → HTTP base url (e.g. `http://10.0.0.2:7438`).
    /// `secret`: shared cluster secret sent as a bearer token (peers refuse
    /// envelopes without it when configured).
    pub fn new(me: NodeId, peer_urls: BTreeMap<NodeId, String>, secret: Option<String>) -> Self {
        let client = reqwest::Client::builder()
            .timeout(SEND_TIMEOUT)
            .build()
            .expect("reqwest client");
        let mut peers = BTreeMap::new();
        for (peer, base) in peer_urls {
            let (control_tx, control_rx) = mpsc::channel(CONTROL_QUEUE_DEPTH);
            let q = Arc::new(PeerQueues {
                latest: Mutex::new(None),
                wake: Notify::new(),
                control: control_tx,
            });
            tokio::spawn(run_peer_sender(
                me,
                peer,
                base,
                client.clone(),
                secret.clone(),
                q.clone(),
                control_rx,
            ));
            peers.insert(peer, q);
        }
        Self { me, peers }
    }
}

impl Transport for HttpTransport {
    fn send(&self, to: NodeId, msg: WireMsg) {
        let Some(q) = self.peers.get(&to) else {
            tracing::warn!(?to, "YRP send to unknown peer dropped");
            return;
        };
        let cumulative = matches!(
            msg,
            WireMsg::Replica(Message::AppendEntries { .. })
                | WireMsg::Replica(Message::InstallSnapshot { .. })
        );
        if cumulative {
            *q.latest.lock() = Some(msg);
            q.wake.notify_one();
        } else if q.control.try_send(msg).is_err() {
            // Full queue: drop the newest — the sender's timers re-drive.
            tracing::debug!(?to, "YRP control queue full; message dropped");
        }
    }
}

async fn run_peer_sender(
    me: NodeId,
    peer: NodeId,
    base: String,
    client: reqwest::Client,
    secret: Option<String>,
    q: Arc<PeerQueues>,
    mut control_rx: mpsc::Receiver<WireMsg>,
) {
    let url = format!("{}/v1/yrp/msg", base.trim_end_matches('/'));
    loop {
        // Control first (election traffic must not starve behind bulky
        // appends), then the latest replication message.
        let msg = tokio::select! {
            biased;
            ctrl = control_rx.recv() => match ctrl {
                Some(m) => m,
                None => return, // transport dropped
            },
            _ = q.wake.notified() => match q.latest.lock().take() {
                Some(m) => m,
                None => continue, // already taken by a prior wake
            },
        };
        let envelope: WireEnvelope = (me.0, msg);
        let body = match bincode::serialize(&envelope) {
            Ok(b) => b,
            Err(e) => {
                tracing::error!(error = %e, "YRP envelope serialize failed; dropped");
                continue;
            }
        };
        let mut req = client.post(&url).body(body);
        if let Some(s) = &secret {
            req = req.bearer_auth(s);
        }
        if let Err(e) = req.send().await {
            // Silence IS the failure signal — the protocol retransmits.
            tracing::debug!(?peer, error = %e, "YRP send failed (will retransmit)");
        }
    }
}

/// Decode an inbound `POST /v1/yrp/msg` body and forward it to the owner
/// funnel. Returns Err on malformed bytes (HTTP 400 at the route).
pub fn deliver_inbound(
    owner: &mpsc::UnboundedSender<DriverEvent>,
    body: &[u8],
) -> Result<(), String> {
    let (from, msg): WireEnvelope =
        bincode::deserialize(body).map_err(|e| format!("malformed YRP envelope: {e}"))?;
    owner
        .send(DriverEvent::Inbound {
            from: NodeId(from),
            msg,
        })
        .map_err(|_| "YRP driver not running".to_string())
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::replica::Payload;
    use super::super::types::{LogPosition, Term};

    /// The wire envelope must round-trip through bincode — including an
    /// AppendEntries carrying real Op bytes (the exact production shape).
    #[test]
    fn wire_envelope_bincode_round_trip() {
        let entry = super::super::replica::LogEntry {
            term: Term(3),
            payload: Payload::Op(vec![1, 2, 3, 255]),
            key: Some(42),
            activate: None,
        };
        let msg = WireMsg::Replica(Message::AppendEntries {
            term: Term(3),
            leader: NodeId(1),
            prev: LogPosition { term: 2, index: 9 },
            entries: vec![entry],
            commit: 9,
        });
        let env: WireEnvelope = (1, msg);
        let bytes = bincode::serialize(&env).unwrap();
        let (from, back): WireEnvelope = bincode::deserialize(&bytes).unwrap();
        assert_eq!(from, 1);
        match back {
            WireMsg::Replica(Message::AppendEntries { entries, .. }) => {
                assert_eq!(entries[0].payload, Payload::Op(vec![1, 2, 3, 255]));
                assert_eq!(entries[0].key, Some(42));
            }
            other => panic!("wrong decode: {other:?}"),
        }
    }

    /// Coalescing: two replication sends before the peer task drains
    /// leave only the LATEST in the slot.
    #[tokio::test]
    async fn replication_messages_coalesce_to_latest() {
        // Build the queues directly (no network) to observe the slot.
        let (control_tx, _control_rx) = mpsc::channel(CONTROL_QUEUE_DEPTH);
        let q = Arc::new(PeerQueues {
            latest: Mutex::new(None),
            wake: Notify::new(),
            control: control_tx,
        });
        let transport = HttpTransport {
            me: NodeId(1),
            peers: [(NodeId(2), q.clone())].into_iter().collect(),
        };
        let hb = |commit| {
            WireMsg::Replica(Message::AppendEntries {
                term: Term(1),
                leader: NodeId(1),
                prev: LogPosition::ZERO,
                entries: vec![],
                commit,
            })
        };
        transport.send(NodeId(2), hb(1));
        transport.send(NodeId(2), hb(2));
        let latest = q.latest.lock().take().expect("slot filled");
        match latest {
            WireMsg::Replica(Message::AppendEntries { commit, .. }) => assert_eq!(commit, 2),
            other => panic!("wrong slot content: {other:?}"),
        }
    }
}
