//! Minimal cert-file verifier — `yantrikdb tls verify-cluster` uses this.
//!
//! Reports counts (certs in chain, keys in key file, roots in CA) and
//! basic load success. Detailed cert metadata (subject, expiry,
//! fingerprint) requires an x509 parser; for PR-5 we intentionally keep
//! this lightweight. RFC 014-B (auth/RBAC) brings in heavier cert
//! inspection if needed.

use std::path::Path;

use rustls_pemfile;

use super::cluster_tls::ClusterTlsConfig;

#[derive(Debug, serde::Serialize)]
pub struct InspectionReport {
    pub cert_path: Option<String>,
    pub cert_count: usize,
    pub key_path: Option<String>,
    pub key_count: usize,
    pub ca_path: Option<String>,
    pub ca_root_count: usize,
    pub fully_specified: bool,
    pub dev_mode: bool,
}

impl InspectionReport {
    /// Render as a human-readable string for `yantrikdb tls verify-cluster`.
    pub fn render_human(&self) -> String {
        let mut out = String::new();
        out.push_str("Cluster mTLS configuration\n");
        out.push_str("──────────────────────────\n");
        out.push_str(&format!(
            "  cert file       : {} ({} cert{})\n",
            self.cert_path.as_deref().unwrap_or("(not set)"),
            self.cert_count,
            if self.cert_count == 1 { "" } else { "s" }
        ));
        out.push_str(&format!(
            "  key file        : {} ({} key{})\n",
            self.key_path.as_deref().unwrap_or("(not set)"),
            self.key_count,
            if self.key_count == 1 { "" } else { "s" }
        ));
        out.push_str(&format!(
            "  CA file         : {} ({} root{})\n",
            self.ca_path.as_deref().unwrap_or("(not set)"),
            self.ca_root_count,
            if self.ca_root_count == 1 { "" } else { "s" }
        ));
        out.push_str(&format!("  fully specified : {}\n", self.fully_specified));
        if self.dev_mode {
            out.push_str("  dev_mode        : ENABLED  ⚠  NEVER set in production\n");
        } else {
            out.push_str("  dev_mode        : disabled\n");
        }
        out
    }
}

pub fn inspect(cfg: &ClusterTlsConfig) -> InspectionReport {
    let cert_count = cfg
        .cert_path
        .as_ref()
        .map(|p| count_pem_blocks(p, "CERTIFICATE"))
        .unwrap_or(0);
    let key_count = cfg.key_path.as_ref().map(count_keys).unwrap_or(0);
    let ca_root_count = cfg
        .ca_path
        .as_ref()
        .map(|p| count_pem_blocks(p, "CERTIFICATE"))
        .unwrap_or(0);

    InspectionReport {
        cert_path: cfg.cert_path.as_ref().map(|p| p.display().to_string()),
        cert_count,
        key_path: cfg.key_path.as_ref().map(|p| p.display().to_string()),
        key_count,
        ca_path: cfg.ca_path.as_ref().map(|p| p.display().to_string()),
        ca_root_count,
        fully_specified: cfg.is_fully_specified(),
        dev_mode: cfg.dev_mode,
    }
}

fn count_pem_blocks(path: &Path, tag: &str) -> usize {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    let needle = format!("-----BEGIN {tag}-----");
    String::from_utf8_lossy(&bytes).matches(&needle).count()
}

fn count_keys(path: &std::path::PathBuf) -> usize {
    let bytes = match std::fs::read(path) {
        Ok(b) => b,
        Err(_) => return 0,
    };
    let mut total = 0;
    {
        let mut reader = std::io::Cursor::new(&bytes);
        total += rustls_pemfile::pkcs8_private_keys(&mut reader)
            .filter_map(|r| r.ok())
            .count();
    }
    {
        let mut reader = std::io::Cursor::new(&bytes);
        total += rustls_pemfile::rsa_private_keys(&mut reader)
            .filter_map(|r| r.ok())
            .count();
    }
    {
        let mut reader = std::io::Cursor::new(&bytes);
        total += rustls_pemfile::ec_private_keys(&mut reader)
            .filter_map(|r| r.ok())
            .count();
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn empty_config_renders_zero_counts() {
        let cfg = ClusterTlsConfig::default();
        let report = inspect(&cfg);
        assert_eq!(report.cert_count, 0);
        assert_eq!(report.key_count, 0);
        assert_eq!(report.ca_root_count, 0);
        assert!(!report.fully_specified);
        assert!(!report.dev_mode);
    }

    #[test]
    fn render_human_includes_all_sections() {
        let cfg = ClusterTlsConfig {
            cert_path: Some(PathBuf::from("/etc/yantrikdb/cluster.crt")),
            key_path: Some(PathBuf::from("/etc/yantrikdb/cluster.key")),
            ca_path: Some(PathBuf::from("/etc/yantrikdb/ca.pem")),
            dev_mode: true,
            rotate_check_secs: 60,
        };
        let report = inspect(&cfg);
        let s = report.render_human();
        assert!(s.contains("cert file"));
        assert!(s.contains("key file"));
        assert!(s.contains("CA file"));
        assert!(s.contains("dev_mode"));
        assert!(s.contains("ENABLED"), "dev_mode flag should be loud: {s}");
    }

    #[test]
    fn missing_files_count_as_zero_not_panic() {
        let cfg = ClusterTlsConfig {
            cert_path: Some(PathBuf::from("/definitely/missing/cert.pem")),
            key_path: Some(PathBuf::from("/definitely/missing/key.pem")),
            ca_path: Some(PathBuf::from("/definitely/missing/ca.pem")),
            dev_mode: false,
            rotate_check_secs: 60,
        };
        let report = inspect(&cfg);
        // Missing files are reported as zero counts, not errors. The
        // validate-on-startup gate is what fails loudly; this is a
        // soft inspector for `yantrikdb tls verify-cluster`.
        assert_eq!(report.cert_count, 0);
        assert!(report.fully_specified, "paths set, even if files missing");
    }
}
