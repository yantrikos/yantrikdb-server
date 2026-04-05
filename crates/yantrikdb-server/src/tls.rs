//! TLS setup for wire protocol and HTTP gateway.

use std::io::BufReader;
use std::path::Path;
use std::sync::Arc;

use rustls::ServerConfig;
use tokio_rustls::TlsAcceptor;

use crate::config::TlsSection;

/// Build a TLS acceptor from cert and key files.
pub fn build_tls_acceptor(tls_config: &TlsSection) -> anyhow::Result<TlsAcceptor> {
    let cert_path = tls_config
        .cert_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("TLS cert_path not configured"))?;
    let key_path = tls_config
        .key_path
        .as_ref()
        .ok_or_else(|| anyhow::anyhow!("TLS key_path not configured"))?;

    let certs = load_certs(cert_path)?;
    let key = load_key(key_path)?;

    let config = ServerConfig::builder()
        .with_no_client_auth()
        .with_single_cert(certs, key)?;

    Ok(TlsAcceptor::from(Arc::new(config)))
}

fn load_certs(path: &Path) -> anyhow::Result<Vec<rustls::pki_types::CertificateDer<'static>>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let certs = rustls_pemfile::certs(&mut reader).collect::<Result<Vec<_>, _>>()?;
    if certs.is_empty() {
        anyhow::bail!("no certificates found in {}", path.display());
    }
    Ok(certs)
}

fn load_key(path: &Path) -> anyhow::Result<rustls::pki_types::PrivateKeyDer<'static>> {
    let file = std::fs::File::open(path)?;
    let mut reader = BufReader::new(file);
    let key = rustls_pemfile::private_key(&mut reader)?
        .ok_or_else(|| anyhow::anyhow!("no private key found in {}", path.display()))?;
    Ok(key)
}
