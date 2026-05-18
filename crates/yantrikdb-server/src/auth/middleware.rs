//! Axum middleware: authenticate the request and attach a [`Principal`].
//!
//! Per RFC 014-B this layer sits between the router and any handler
//! that wants typed scope/namespace authorization. It:
//!
//! 1. Reads `Authorization: Bearer <tok>` from the request.
//! 2. Calls [`AuthProvider::authenticate`] (configured on [`AppState`]).
//! 3. On [`AuthOutcome::Authenticated`], inserts the [`Principal`] into
//!    request extensions and forwards. Handlers extract it via
//!    `Extension(principal): Extension<Principal>`.
//! 4. On any other outcome (missing header, malformed, unknown token,
//!    revoked, expired), short-circuits with `401 unauthenticated`.
//! 5. On a provider-side infrastructure error, returns `500 internal_error`.
//!
//! ## Scoping
//!
//! Apply selectively via [`axum::Router::route_layer`] to routes that
//! require an authenticated [`Principal`] — not blanket on `/v1/*`,
//! because legacy endpoints already do inline `resolve_engine`
//! authentication and `/v1/health{,/deep}` plus `/metrics` are
//! deliberately open. Issue #39 Phase 1 wires this layer onto new
//! routes only; migrating legacy routes is a follow-up.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::Response,
};

use crate::api::errors::{api_error, ApiErrorCode};
use crate::auth::{AuthOutcome, AuthProvider};
use crate::server::AppState;

/// Middleware: require a valid Bearer token, inject [`Principal`].
///
/// Wired via `axum::middleware::from_fn_with_state(state, require_authenticated_principal)`
/// — see [`crate::http_gateway::build_router`].
pub async fn require_authenticated_principal(
    State(state): State<Arc<AppState>>,
    mut req: Request,
    next: Next,
) -> Result<Response, (StatusCode, axum::Json<serde_json::Value>)> {
    let token = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|h| h.strip_prefix("Bearer "))
        .map(str::to_owned)
        .ok_or_else(|| {
            api_error(
                StatusCode::UNAUTHORIZED,
                ApiErrorCode::Unauthenticated,
                "missing Bearer token",
            )
        })?;

    let outcome = state
        .auth_provider
        .authenticate(&token)
        .await
        .map_err(|e| {
            api_error(
                StatusCode::INTERNAL_SERVER_ERROR,
                ApiErrorCode::InternalError,
                format!("auth backend error: {e}"),
            )
        })?;

    let principal = match outcome {
        AuthOutcome::Authenticated(p) => p,
        AuthOutcome::Unauthenticated => {
            return Err(api_error(
                StatusCode::UNAUTHORIZED,
                ApiErrorCode::Unauthenticated,
                "invalid or unknown token",
            ));
        }
        AuthOutcome::Revoked { .. } => {
            return Err(api_error(
                StatusCode::UNAUTHORIZED,
                ApiErrorCode::Unauthenticated,
                "token revoked",
            ));
        }
        AuthOutcome::Expired { .. } => {
            return Err(api_error(
                StatusCode::UNAUTHORIZED,
                ApiErrorCode::Unauthenticated,
                "token expired",
            ));
        }
    };

    req.extensions_mut().insert(principal);
    Ok(next.run(req).await)
}
