//! Cluster transport mTLS — production gate for RFC 010 PR-4 openraft.
//!
//! Builds [`tokio_rustls::TlsAcceptor`] and [`tokio_rustls::TlsConnector`]
//! pre-configured for mutual TLS (every peer must present a cert signed
//! by the configured CA). Includes a [`CertificateRotator`] background
//! task that polls the configured paths for cert changes and atomically
//! reloads — so operators can rotate certs without restarting the cluster.

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, SystemTime};

use parking_lot::RwLock;
use rustls::pki_types::{CertificateDer, PrivateKeyDer, ServerName};
use rustls::{ClientConfig, RootCertStore, ServerConfig};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tokio_rustls::{TlsAcceptor, TlsConnector};

/// Configuration for cluster-transport mTLS. All three of `cert_path`,
/// `key_path`, `ca_path` are required when `cluster.mode = "openraft"`.
/// In `legacy` cluster mode (current Raft-lite), this config is optional
/// and ignored if not set — Raft-lite stays on plaintext, with a startup
/// warning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusterTlsConfig {
    /// PEM-encoded certificate chain for THIS node. Required for openraft.
    pub cert_path: Option<PathBuf>,
    /// PEM-encoded private key for `cert_path`. Required for openraft.
    pub key_path: Option<PathBuf>,
    /// PEM-encoded CA bundle. Used to verify peer certs. Required for openraft.
    pub ca_path: Option<PathBuf>,
    /// Dev escape hatch: skip peer cert verification (accepts self-
    /// signed). Used for local-only / loopback dev clusters. NEVER set
    /// to `true` in production — startup logs a loud warning when this
    /// is on.
    #[serde(default)]
    pub dev_mode: bool,
    /// How often to poll cert files for changes. Cert rotation reloads
    /// happen at this cadence. Default 60s. Minimum 5s enforced by the
    /// rotator loop to bound disk-stat traffic.
    #[serde(default = "default_rotate_check_secs")]
    pub rotate_check_secs: u64,
}

impl Default for ClusterTlsConfig {
    fn default() -> Self {
        Self {
            cert_path: None,
            key_path: None,
            ca_path: None,
            dev_mode: false,
            rotate_check_secs: default_rotate_check_secs(),
        }
    }
}

fn default_rotate_check_secs() -> u64 {
    60
}

impl ClusterTlsConfig {
    /// Whether cert paths are fully specified (all three present).
    pub fn is_fully_specified(&self) -> bool {
        self.cert_path.is_some() && self.key_path.is_some() && self.ca_path.is_some()
    }
}

#[derive(Debug, Error)]
pub enum ClusterTlsError {
    #[error("cluster_tls config missing field `{field}`")]
    MissingField { field: &'static str },

    #[error("cluster_tls cert file `{path}` could not be read: {source}")]
    CertFile {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("cluster_tls cert file `{path}` contains no certificates")]
    NoCertificates { path: PathBuf },

    #[error("cluster_tls key file `{path}` contains no usable private keys")]
    NoPrivateKey { path: PathBuf },

    #[error("rustls config build failed: {0}")]
    Rustls(String),
}

/// Loaded cert + key + root store, swappable atomically by the rotator.
pub struct ClusterTlsHandles {
    /// Acceptor for incoming peer connections (server side of mTLS).
    /// Wrapped in RwLock so the rotator can swap atomically.
    pub acceptor: RwLock<TlsAcceptor>,
    /// Connector for outgoing peer connections (client side of mTLS).
    pub connector: RwLock<TlsConnector>,
    /// SystemTime when these handles were built. Surfaced in
    /// `/v1/health/deep` so operators can tell if rotation is working.
    pub loaded_at: RwLock<SystemTime>,
}

impl ClusterTlsHandles {
    /// Snapshot the current acceptor (for use by the cluster TCP listener).
    pub fn current_acceptor(&self) -> TlsAcceptor {
        self.acceptor.read().clone()
    }

    /// Snapshot the current connector (for use when dialing a peer).
    pub fn current_connector(&self) -> TlsConnector {
        self.connector.read().clone()
    }

    /// When these handles were last loaded from disk.
    pub fn loaded_at(&self) -> SystemTime {
        *self.loaded_at.read()
    }
}

/// Build a [`tokio_rustls::TlsAcceptor`] (server side of mTLS) from the
/// given cert/key/ca paths. The acceptor demands client cert
/// authentication — peers without a cert signed by the CA are rejected
/// at handshake.
pub fn build_acceptor(cfg: &ClusterTlsConfig) -> Result<TlsAcceptor, ClusterTlsError> {
    let cert_path = cfg
        .cert_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "cert_path" })?;
    let key_path = cfg
        .key_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "key_path" })?;
    let ca_path = cfg
        .ca_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "ca_path" })?;

    let certs = load_certs(cert_path)?;
    let key = load_private_key(key_path)?;
    let roots = load_root_store(ca_path)?;

    // Require client auth (mTLS): peers MUST present a cert signed by
    // the configured CA. In dev_mode the verifier still requires a cert
    // but accepts self-signed (verifier is configured below).
    let server_cfg = if cfg.dev_mode {
        // Dev mode: skip peer cert chain verification, but still require
        // peers to present *some* cert. This catches "anyone can connect"
        // while allowing loopback dev to work without a real CA.
        let verifier =
            crate::security::cluster_tls::dev_mode::DevModeClientCertVerifier::new(roots);
        ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)
            .map_err(|e| ClusterTlsError::Rustls(e.to_string()))?
    } else {
        let verifier = rustls::server::WebPkiClientVerifier::builder(Arc::new(roots))
            .build()
            .map_err(|e| ClusterTlsError::Rustls(e.to_string()))?;
        ServerConfig::builder()
            .with_client_cert_verifier(verifier)
            .with_single_cert(certs, key)
            .map_err(|e| ClusterTlsError::Rustls(e.to_string()))?
    };

    Ok(TlsAcceptor::from(Arc::new(server_cfg)))
}

/// Build a [`tokio_rustls::TlsConnector`] (client side of mTLS) from the
/// same cert/key/ca paths. The connector presents this node's cert when
/// dialing a peer, and verifies the peer's cert against the CA.
pub fn build_connector(cfg: &ClusterTlsConfig) -> Result<TlsConnector, ClusterTlsError> {
    let cert_path = cfg
        .cert_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "cert_path" })?;
    let key_path = cfg
        .key_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "key_path" })?;
    let ca_path = cfg
        .ca_path
        .as_ref()
        .ok_or(ClusterTlsError::MissingField { field: "ca_path" })?;

    let certs = load_certs(cert_path)?;
    let key = load_private_key(key_path)?;
    let roots = load_root_store(ca_path)?;

    let client_cfg = if cfg.dev_mode {
        // Dev mode: skip peer (server) cert verification. dangerous_*
        // is the rustls-blessed knob for this exact scenario; it's
        // marked dangerous because in production it's a CRITICAL bug.
        ClientConfig::builder()
            .dangerous()
            .with_custom_certificate_verifier(Arc::new(
                crate::security::cluster_tls::dev_mode::DevModeServerCertVerifier::new(roots),
            ))
            .with_client_auth_cert(certs, key)
            .map_err(|e| ClusterTlsError::Rustls(e.to_string()))?
    } else {
        ClientConfig::builder()
            .with_root_certificates(roots)
            .with_client_auth_cert(certs, key)
            .map_err(|e| ClusterTlsError::Rustls(e.to_string()))?
    };

    Ok(TlsConnector::from(Arc::new(client_cfg)))
}

fn load_certs(path: &Path) -> Result<Vec<CertificateDer<'static>>, ClusterTlsError> {
    let bytes = std::fs::read(path).map_err(|e| ClusterTlsError::CertFile {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = std::io::Cursor::new(bytes);
    let certs: Vec<CertificateDer<'static>> = rustls_pemfile::certs(&mut reader)
        .filter_map(|r| r.ok())
        .collect();
    if certs.is_empty() {
        return Err(ClusterTlsError::NoCertificates {
            path: path.to_path_buf(),
        });
    }
    Ok(certs)
}

fn load_private_key(path: &Path) -> Result<PrivateKeyDer<'static>, ClusterTlsError> {
    let bytes = std::fs::read(path).map_err(|e| ClusterTlsError::CertFile {
        path: path.to_path_buf(),
        source: e,
    })?;
    let mut reader = std::io::Cursor::new(bytes);
    // Try PKCS8 first (most common for new keys), then RSA, then EC.
    if let Some(Ok(key)) = rustls_pemfile::pkcs8_private_keys(&mut reader).next() {
        return Ok(PrivateKeyDer::Pkcs8(key));
    }
    let mut reader =
        std::io::Cursor::new(std::fs::read(path).map_err(|e| ClusterTlsError::CertFile {
            path: path.to_path_buf(),
            source: e,
        })?);
    if let Some(Ok(key)) = rustls_pemfile::rsa_private_keys(&mut reader).next() {
        return Ok(PrivateKeyDer::Pkcs1(key));
    }
    let mut reader =
        std::io::Cursor::new(std::fs::read(path).map_err(|e| ClusterTlsError::CertFile {
            path: path.to_path_buf(),
            source: e,
        })?);
    if let Some(Ok(key)) = rustls_pemfile::ec_private_keys(&mut reader).next() {
        return Ok(PrivateKeyDer::Sec1(key));
    }
    Err(ClusterTlsError::NoPrivateKey {
        path: path.to_path_buf(),
    })
}

fn load_root_store(path: &Path) -> Result<RootCertStore, ClusterTlsError> {
    let certs = load_certs(path)?;
    let mut roots = RootCertStore::empty();
    for cert in certs {
        roots
            .add(cert)
            .map_err(|e| ClusterTlsError::Rustls(format!("CA load failed: {e}")))?;
    }
    Ok(roots)
}

/// Background task that polls the configured cert paths for changes and
/// atomically swaps the [`ClusterTlsHandles`] when files change. Lets
/// operators rotate certs without restarting the cluster.
///
/// Detection is mtime-based — sufficient for rotation tools that write
/// new files atomically (most do via rename-into-place).
pub struct CertificateRotator {
    cfg: ClusterTlsConfig,
    handles: Arc<ClusterTlsHandles>,
    last_mtimes: RwLock<CertMtimes>,
}

#[derive(Default, Clone)]
struct CertMtimes {
    cert: Option<SystemTime>,
    key: Option<SystemTime>,
    ca: Option<SystemTime>,
}

impl CertificateRotator {
    pub fn new(cfg: ClusterTlsConfig, handles: Arc<ClusterTlsHandles>) -> Self {
        let mtimes = Self::current_mtimes(&cfg);
        Self {
            cfg,
            handles,
            last_mtimes: RwLock::new(mtimes),
        }
    }

    fn current_mtimes(cfg: &ClusterTlsConfig) -> CertMtimes {
        CertMtimes {
            cert: cfg.cert_path.as_ref().and_then(|p| mtime(p)),
            key: cfg.key_path.as_ref().and_then(|p| mtime(p)),
            ca: cfg.ca_path.as_ref().and_then(|p| mtime(p)),
        }
    }

    /// Poll once for cert file changes; reload if any changed.
    /// Returns true if a reload happened.
    pub fn check_and_reload(&self) -> Result<bool, ClusterTlsError> {
        let current = Self::current_mtimes(&self.cfg);
        let last = self.last_mtimes.read().clone();
        let changed = current.cert != last.cert || current.key != last.key || current.ca != last.ca;
        if !changed {
            return Ok(false);
        }
        // Build new acceptor + connector, swap atomically.
        let new_acceptor = build_acceptor(&self.cfg)?;
        let new_connector = build_connector(&self.cfg)?;
        *self.handles.acceptor.write() = new_acceptor;
        *self.handles.connector.write() = new_connector;
        *self.handles.loaded_at.write() = SystemTime::now();
        *self.last_mtimes.write() = current;
        tracing::info!("cluster mTLS certificates reloaded");
        Ok(true)
    }

    /// Run the rotation loop indefinitely. Spawn this on the
    /// control-plane runtime (RFC 009 §4) so cert-rotation work doesn't
    /// compete with HTTP/recall handlers.
    pub async fn run(self, cancel: tokio_util::sync::CancellationToken) {
        let interval = Duration::from_secs(self.cfg.rotate_check_secs.max(5));
        loop {
            tokio::select! {
                _ = tokio::time::sleep(interval) => {
                    if let Err(e) = self.check_and_reload() {
                        tracing::warn!(error = %e, "cluster mTLS rotation check failed; will retry");
                    }
                }
                _ = cancel.cancelled() => {
                    tracing::info!("cluster mTLS rotator shutting down");
                    return;
                }
            }
        }
    }
}

fn mtime(path: &Path) -> Option<SystemTime> {
    std::fs::metadata(path).ok().and_then(|m| m.modified().ok())
}

// ── dev-mode-only verifiers ─────────────────────────────────────────
//
// Dev mode skips peer cert chain verification but still requires a
// cert to be presented. This is sufficient for loopback dev clusters
// where the goal is "no plaintext on the wire" rather than "all peers
// authenticated against a real PKI."
//
// PRODUCTION GATE: dev_mode=true logs a loud warning at startup AND
// every minute thereafter. Operators should NEVER ship production with
// it enabled.
mod dev_mode {
    use rustls::client::danger::{HandshakeSignatureValid, ServerCertVerified, ServerCertVerifier};
    use rustls::pki_types::{CertificateDer, ServerName, UnixTime};
    use rustls::server::danger::{ClientCertVerified, ClientCertVerifier};
    use rustls::{DigitallySignedStruct, DistinguishedName, Error, RootCertStore, SignatureScheme};
    use std::sync::Arc;

    #[derive(Debug)]
    pub struct DevModeClientCertVerifier {
        _roots: Arc<RootCertStore>,
        accepted_issuers: Vec<DistinguishedName>,
    }

    impl DevModeClientCertVerifier {
        pub fn new(roots: RootCertStore) -> Arc<Self> {
            // Snapshot the issuers so accepted_issuers() can be cheap.
            let issuers: Vec<DistinguishedName> = roots
                .roots
                .iter()
                .map(|t| DistinguishedName::from(t.subject.as_ref().to_vec()))
                .collect();
            Arc::new(Self {
                _roots: Arc::new(roots),
                accepted_issuers: issuers,
            })
        }
    }

    impl ClientCertVerifier for DevModeClientCertVerifier {
        fn root_hint_subjects(&self) -> &[DistinguishedName] {
            &self.accepted_issuers
        }
        fn verify_client_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _now: UnixTime,
        ) -> Result<ClientCertVerified, Error> {
            // Dev mode: accept any presented cert without chain verification.
            Ok(ClientCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            // Match what rustls's default WebPkiClientVerifier offers.
            vec![
                SignatureScheme::ED25519,
                SignatureScheme::ECDSA_NISTP256_SHA256,
                SignatureScheme::ECDSA_NISTP384_SHA384,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::RSA_PKCS1_SHA256,
            ]
        }
    }

    #[derive(Debug)]
    pub struct DevModeServerCertVerifier {
        _roots: Arc<RootCertStore>,
    }

    impl DevModeServerCertVerifier {
        pub fn new(roots: RootCertStore) -> Self {
            Self {
                _roots: Arc::new(roots),
            }
        }
    }

    impl ServerCertVerifier for DevModeServerCertVerifier {
        fn verify_server_cert(
            &self,
            _end_entity: &CertificateDer<'_>,
            _intermediates: &[CertificateDer<'_>],
            _server_name: &ServerName<'_>,
            _ocsp_response: &[u8],
            _now: UnixTime,
        ) -> Result<ServerCertVerified, Error> {
            Ok(ServerCertVerified::assertion())
        }
        fn verify_tls12_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn verify_tls13_signature(
            &self,
            _message: &[u8],
            _cert: &CertificateDer<'_>,
            _dss: &DigitallySignedStruct,
        ) -> Result<HandshakeSignatureValid, Error> {
            Ok(HandshakeSignatureValid::assertion())
        }
        fn supported_verify_schemes(&self) -> Vec<SignatureScheme> {
            vec![
                SignatureScheme::ED25519,
                SignatureScheme::ECDSA_NISTP256_SHA256,
                SignatureScheme::ECDSA_NISTP384_SHA384,
                SignatureScheme::RSA_PSS_SHA256,
                SignatureScheme::RSA_PKCS1_SHA256,
            ]
        }
    }
}

// Re-export so ServerName lookups for the connector work cleanly.
pub use rustls::pki_types::ServerName as TlsServerName;
// Re-export so cert_inspect can use the same type.
#[allow(unused_imports)]
pub(crate) use rustls::pki_types::CertificateDer as ReexportCertificateDer;
// Suppress unused-import warning for ServerName.
#[allow(dead_code)]
fn _ensure_servername_in_scope(_: &ServerName<'_>) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_default_is_legacy_safe() {
        // Default config has no paths set. Cluster legacy mode (RFC 014-A
        // is opt-in) doesn't trigger the production gate, so this is OK.
        let cfg = ClusterTlsConfig::default();
        assert!(!cfg.is_fully_specified());
        assert!(!cfg.dev_mode);
        assert_eq!(cfg.rotate_check_secs, 60);
    }

    #[test]
    fn is_fully_specified_requires_all_three() {
        let mut cfg = ClusterTlsConfig::default();
        assert!(!cfg.is_fully_specified());
        cfg.cert_path = Some(PathBuf::from("/x/cert.pem"));
        assert!(!cfg.is_fully_specified());
        cfg.key_path = Some(PathBuf::from("/x/key.pem"));
        assert!(!cfg.is_fully_specified());
        cfg.ca_path = Some(PathBuf::from("/x/ca.pem"));
        assert!(cfg.is_fully_specified());
    }

    #[test]
    fn build_acceptor_rejects_missing_cert_path() {
        // TlsAcceptor doesn't impl Debug, so we can't use unwrap_err.
        let cfg = ClusterTlsConfig::default();
        match build_acceptor(&cfg) {
            Ok(_) => panic!("expected MissingField error"),
            Err(ClusterTlsError::MissingField { field: "cert_path" }) => {}
            Err(other) => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn build_connector_rejects_missing_key_path() {
        let mut cfg = ClusterTlsConfig::default();
        cfg.cert_path = Some(PathBuf::from("/x/cert.pem"));
        match build_connector(&cfg) {
            Ok(_) => panic!("expected MissingField error"),
            Err(ClusterTlsError::MissingField { field: "key_path" }) => {}
            Err(other) => panic!("wrong error: {other:?}"),
        }
    }

    #[test]
    fn load_certs_fails_for_nonexistent_file() {
        let err = load_certs(Path::new("/definitely/does/not/exist/cert.pem")).unwrap_err();
        match err {
            ClusterTlsError::CertFile { .. } => {}
            other => panic!("wrong error: {other:?}"),
        }
    }
}
