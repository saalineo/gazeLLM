//! Server-Sent Events (SSE) streaming response utilities.

use axum::response::sse::{Event, KeepAlive, Sse};
use serde::{Deserialize, Serialize};
use std::convert::Infallible;
use std::time::Duration;
use tokio_stream::{Stream, StreamExt};

/// Structured payload for a single token SSE event.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SseTokenPayload {
    /// Generated token string.
    pub token: String,
}

impl SseTokenPayload {
    /// Creates a new [`SseTokenPayload`].
    #[must_use]
    pub fn new(token: impl Into<String>) -> Self {
        Self {
            token: token.into(),
        }
    }
}

/// Creates an Axum [`Sse`] response wrapper from a stream of token strings.
///
/// Each token emitted by the input stream is serialized into a JSON object:
/// `{"token": "<token_text>"}` and formatted as an SSE `data` event.
///
/// # Examples
///
/// ```rust
/// use gazellm::net::sse::create_token_stream;
/// use tokio_stream::iter;
///
/// let tokens = vec!["Hello".to_string(), " world".to_string()];
/// let _sse_response = create_token_stream(iter(tokens));
/// ```
pub fn create_token_stream<S>(stream: S) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = String> + Send + 'static,
{
    let event_stream = stream.map(|token_text| {
        let json_payload = serde_json::json!({ "token": token_text });
        Ok(Event::default().data(json_payload.to_string()))
    });
    Sse::new(event_stream)
}

/// Creates an Axum [`Sse`] response wrapper with keep-alive heartbeat enabled.
///
/// # Arguments
///
/// * `stream` - Stream yielding token text chunks.
/// * `interval` - Interval between keep-alive heartbeat comments.
pub fn create_token_stream_with_keepalive<S>(
    stream: S,
    interval: Duration,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>>
where
    S: Stream<Item = String> + Send + 'static,
{
    create_token_stream(stream).keep_alive(KeepAlive::new().interval(interval).text("keep-alive"))
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::response::IntoResponse;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_create_token_stream_payloads() {
        let tokens = vec![
            "Hello".to_string(),
            ",".to_string(),
            " world".to_string(),
            "!".to_string(),
        ];
        let stream = tokio_stream::iter(tokens);
        let sse = create_token_stream(stream);

        // Convert Sse into its inner stream by turning into response
        let response = sse.into_response();
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .expect("content-type missing"),
            "text/event-stream"
        );
    }

    #[tokio::test]
    async fn test_sse_token_payload_serde() {
        let payload = SseTokenPayload::new("token_test");
        let serialized = serde_json::to_string(&payload).expect("serialization failed");
        assert_eq!(serialized, r#"{"token":"token_test"}"#);

        let deserialized: SseTokenPayload =
            serde_json::from_str(&serialized).expect("deserialization failed");
        assert_eq!(deserialized, payload);
    }

    #[tokio::test]
    async fn test_event_stream_mapping() {
        let tokens = vec!["token_1".to_string(), "token_2".to_string()];
        let stream = tokio_stream::iter(tokens);

        let sse = create_token_stream(stream);
        let mut inner_stream = Box::pin(sse.into_response().into_body().into_data_stream());

        let mut received = Vec::new();
        while let Some(chunk_result) = inner_stream.next().await {
            let chunk = chunk_result.expect("chunk error");
            let text = String::from_utf8(chunk.to_vec()).expect("utf-8 conversion");
            received.push(text);
        }

        let full_output = received.join("");
        assert!(full_output.contains(r#"data: {"token":"token_1"}"#));
        assert!(full_output.contains(r#"data: {"token":"token_2"}"#));
    }

    #[tokio::test]
    async fn test_create_token_stream_with_keepalive() {
        let tokens = vec!["chunk".to_string()];
        let stream = tokio_stream::iter(tokens);
        let sse = create_token_stream_with_keepalive(stream, Duration::from_millis(50));
        let response = sse.into_response();
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .expect("content-type header missing"),
            "text/event-stream"
        );
    }
}
