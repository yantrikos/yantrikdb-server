//! HTTP client wrapper around the YantrikDB HTTP gateway.

use anyhow::{Context, Result};
use reqwest::blocking::Client as HttpClient;
use serde_json::Value;

pub struct Client {
    http: HttpClient,
    base_url: String,
    token: String,
}

impl Client {
    pub fn new(base_url: String, token: String) -> Result<Self> {
        let http = HttpClient::builder()
            .timeout(std::time::Duration::from_secs(30))
            .build()?;
        Ok(Self {
            http,
            base_url,
            token,
        })
    }

    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url, path)
    }

    fn auth(&self) -> String {
        format!("Bearer {}", self.token)
    }

    pub fn get(&self, path: &str) -> Result<Value> {
        let resp = self
            .http
            .get(self.url(path))
            .header("Authorization", self.auth())
            .send()
            .with_context(|| format!("GET {}", path))?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            anyhow::bail!("{} {}: {}", status.as_u16(), path, text);
        }
        Ok(serde_json::from_str(&text)?)
    }

    pub fn post(&self, path: &str, body: &Value) -> Result<Value> {
        let resp = self
            .http
            .post(self.url(path))
            .header("Authorization", self.auth())
            .json(body)
            .send()
            .with_context(|| format!("POST {}", path))?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            anyhow::bail!("{} {}: {}", status.as_u16(), path, text);
        }
        Ok(serde_json::from_str(&text)?)
    }

    pub fn delete(&self, path: &str) -> Result<Value> {
        let resp = self
            .http
            .delete(self.url(path))
            .header("Authorization", self.auth())
            .send()
            .with_context(|| format!("DELETE {}", path))?;
        let status = resp.status();
        let text = resp.text()?;
        if !status.is_success() {
            anyhow::bail!("{} {}: {}", status.as_u16(), path, text);
        }
        Ok(serde_json::from_str(&text).unwrap_or(serde_json::json!({})))
    }

    pub fn health(&self) -> Result<Value> {
        // Health doesn't need auth, but we'll send it anyway
        let resp = self
            .http
            .get(self.url("/v1/health"))
            .send()
            .context("connecting to server")?;
        Ok(serde_json::from_str(&resp.text()?)?)
    }
}
