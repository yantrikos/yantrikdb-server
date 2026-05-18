//! Public API surface conventions for the `/v1/*` HTTP gateway.
//!
//! This module gathers cross-cutting types — error envelopes, pagination
//! envelopes, scope/permission types — that get applied uniformly across
//! every endpoint. Handlers live in `http_gateway.rs`; the conventions
//! they emit + consume live here.
//!
//! Established by issue #39 brainstorm (May 2026). See:
//! - `docs/error-codes.md` — stable error-code registry
//! - `errors` — structured error envelope (Option B)

pub mod access;
pub mod errors;
