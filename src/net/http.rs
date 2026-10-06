//! Axum asynchronous HTTP API server.

use crate::config::EngineConfig;
use axum::{
    extract::Json,
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
    Router,
};
use serde::{Deserialize, Serialize};
use std::net::SocketAddr;

/// Errors arising from the HTTP server lifecycle.
#[derive(Debug, thiserror::Error)]
pub enum HttpServerError {
    /// Failed to parse socket address from host and port.
    #[error("invalid listen address '{address}': {source}")]
    AddressParse {
        /// Formatted host:port string.
        address: String,
        /// Underlying parse error.
        #[source]
        source: std::net::AddrParseError,
    },
    /// I/O failure when binding TCP listener.
    #[error("failed to bind TCP listener on '{address}': {source}")]
    Bind {
        /// Target listen address.
        address: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// Error during HTTP server execution.
    #[error("HTTP server error: {source}")]
    Server {
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
}

/// Completion request payload for the HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HttpCompletionRequest {
    /// Input prompt text.
    pub prompt: String,
    /// Maximum new tokens to generate (optional, defaults to 128).
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
}

const fn default_max_tokens() -> usize {
    128
}

/// Completion response payload for the HTTP API.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HttpCompletionResponse {
    /// Generated output text.
    pub text: String,
    /// Generated token IDs.
    pub tokens: Vec<u32>,
}

use crate::net::sse::create_token_stream;

/// Constructs the Axum HTTP router with all engine endpoints configured.
pub fn create_router() -> Router {
    Router::new()
        .route("/health", get(health_handler))
        .route("/v1/completions", post(handle_completion))
        .route(
            "/v1/completions/stream",
            post(handle_streaming_completion_post).get(handle_streaming_completion_get),
        )
}

async fn health_handler() -> &'static str {
    "OK"
}

async fn handle_completion(Json(payload): Json<HttpCompletionRequest>) -> impl IntoResponse {
    let response = HttpCompletionResponse {
        text: format!("Echo: {}", payload.prompt),
        tokens: vec![100, 101],
    };
    (StatusCode::OK, Json(response))
}

async fn handle_streaming_completion_post(
    Json(payload): Json<HttpCompletionRequest>,
) -> impl IntoResponse {
    let tokens = vec!["Echo: ".to_string(), payload.prompt, " [DONE]".to_string()];
    let stream = tokio_stream::iter(tokens);
    create_token_stream(stream)
}

async fn handle_streaming_completion_get() -> impl IntoResponse {
    let tokens = vec![
        "streaming ".to_string(),
        "response ".to_string(),
        "active".to_string(),
    ];
    let stream = tokio_stream::iter(tokens);
    create_token_stream(stream)
}

/// Starts the Axum HTTP REST server bound to the address specified in `config`.
///
/// # Errors
/// Returns [`HttpServerError::AddressParse`] if the host/port address fails to parse,
/// [`HttpServerError::Bind`] if binding the TCP socket fails, or [`HttpServerError::Server`]
/// if the server fails during execution.
pub async fn start_http_server(config: &EngineConfig) -> Result<(), HttpServerError> {
    let app = create_router();
    let addr_str = format!("{}:{}", config.host, config.port);

    let addr: SocketAddr = addr_str
        .parse()
        .map_err(|source| HttpServerError::AddressParse {
            address: addr_str.clone(),
            source,
        })?;

    tracing::info!("HTTP server listening on {}", addr);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .map_err(|source| HttpServerError::Bind {
            address: addr_str,
            source,
        })?;

    axum::serve(listener, app)
        .await
        .map_err(|source| HttpServerError::Server { source })?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tower::ServiceExt;

    #[tokio::test]
    async fn test_health_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .uri("/health")
                    .body(Body::empty())
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        assert_eq!(&body[..], b"OK");
    }

    #[tokio::test]
    async fn test_completions_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Hello"}"#))
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        let parsed: HttpCompletionResponse =
            serde_json::from_slice(&body).expect("valid completion response JSON");
        assert_eq!(parsed.text, "Echo: Hello");
        assert_eq!(parsed.tokens, vec![100, 101]);
    }

    #[tokio::test]
    async fn test_streaming_completions_post_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/completions/stream")
                    .header("content-type", "application/json")
                    .body(Body::from(r#"{"prompt": "Rust"}"#))
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .expect("content-type header missing"),
            "text/event-stream"
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        let body_str = String::from_utf8(body.to_vec()).expect("valid utf-8");
        assert!(body_str.contains(r#"data: {"token":"Echo: "}"#));
        assert!(body_str.contains(r#"data: {"token":"Rust"}"#));
        assert!(body_str.contains(r#"data: {"token":" [DONE]"}"#));
    }

    #[tokio::test]
    async fn test_streaming_completions_get_endpoint() {
        let app = create_router();

        let response = app
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/completions/stream")
                    .body(Body::empty())
                    .expect("failed to build HTTP request"),
            )
            .await
            .expect("service execution failed");

        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .expect("content-type header missing"),
            "text/event-stream"
        );

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .expect("failed to read response body");
        let body_str = String::from_utf8(body.to_vec()).expect("valid utf-8");
        assert!(body_str.contains(r#"data: {"token":"streaming "}"#));
        assert!(body_str.contains(r#"data: {"token":"response "}"#));
        assert!(body_str.contains(r#"data: {"token":"active"}"#));
    }
}
