//! gRPC bidirectional streaming implementation for token delivery.

use std::fmt::Write;
use std::net::SocketAddr;
use tokio::sync::mpsc;
use tokio_stream::wrappers::ReceiverStream;
use tonic::{Request, Response, Status, Streaming};

use super::grpc::{GenerateRequest, GenerateResponse, InferenceService, InferenceServiceServer};

/// Type alias for incoming client streaming request wrapper.
pub type ClientStream<T> = Streaming<T>;

/// Service implementation for streaming inference generation.
///
/// # Examples
///
/// ```
/// use gazellm::net::grpc_stream::InferenceServiceImpl;
///
/// let service = InferenceServiceImpl::new();
/// assert_eq!(service, InferenceServiceImpl::default());
/// ```
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct InferenceServiceImpl;

impl InferenceServiceImpl {
    /// Creates a new [`InferenceServiceImpl`] instance.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }

    /// Asynchronously streams generated token packets for a given prompt string.
    ///
    /// Streams 5 sequential token chunks (`token_0 ` through `token_4 `) through a
    /// bounded Tokio `mpsc` channel with a buffer capacity of 128 items.
    ///
    /// # Errors
    /// Returns [`Status`] if stream creation fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use gazellm::net::grpc_stream::InferenceServiceImpl;
    /// use tokio_stream::StreamExt;
    ///
    /// let response = InferenceServiceImpl::stream_generate_impl("Hello".to_string()).await?;
    /// let mut stream = response.into_inner();
    /// if let Some(token_res) = stream.next().await {
    ///     let token = token_res?;
    ///     assert_eq!(token.token_id, 0);
    /// }
    /// # Ok(())
    /// # }
    /// ```
    #[allow(clippy::unused_async)]
    pub async fn stream_generate_impl(
        prompt: String,
    ) -> Result<Response<ReceiverStream<Result<GenerateResponse, Status>>>, Status> {
        Self::stream_generate_custom("req_1".to_string(), prompt, 5, 128).await
    }

    /// Streams token generation responses based on a structured [`GenerateRequest`].
    ///
    /// Respects `request.request_id` and `request.max_new_tokens` (falling back to 5 tokens if unspecified).
    ///
    /// # Errors
    /// Returns [`Status`] if stream initialization fails.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use gazellm::net::grpc::GenerateRequest;
    /// use gazellm::net::grpc_stream::InferenceServiceImpl;
    /// use tokio_stream::StreamExt;
    ///
    /// let request = GenerateRequest {
    ///     request_id: "req_demo".to_string(),
    ///     prompt: "Explain Rust".to_string(),
    ///     max_new_tokens: 2,
    ///     ..Default::default()
    /// };
    ///
    /// let response = InferenceServiceImpl::stream_generate_with_request(request).await?;
    /// let mut stream = response.into_inner();
    /// let first = stream.next().await.unwrap()?;
    /// assert_eq!(first.request_id, "req_demo");
    /// # Ok(())
    /// # }
    /// ```
    pub async fn stream_generate_with_request(
        request: GenerateRequest,
    ) -> Result<Response<ReceiverStream<Result<GenerateResponse, Status>>>, Status> {
        let request_id = if request.request_id.is_empty() {
            "req_1".to_string()
        } else {
            request.request_id
        };

        let total_tokens = if request.max_new_tokens > 0 {
            usize::try_from(request.max_new_tokens).unwrap_or(5)
        } else {
            5
        };

        Self::stream_generate_custom(request_id, request.prompt, total_tokens, 128).await
    }

    /// Configurable stream generator with explicit request metadata, token limit, and channel buffer capacity.
    ///
    /// Pre-allocates token chunk text buffers to minimize heap reallocations on the generation hot path.
    ///
    /// # Errors
    /// Returns [`Status`] if channel creation or request parameters are invalid.
    ///
    /// # Examples
    ///
    /// ```
    /// # #[tokio::main]
    /// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// use gazellm::net::grpc_stream::InferenceServiceImpl;
    /// use tokio_stream::StreamExt;
    ///
    /// let response = InferenceServiceImpl::stream_generate_custom(
    ///     "custom_id".to_string(),
    ///     "prompt text".to_string(),
    ///     3,
    ///     64,
    /// ).await?;
    /// let mut stream = response.into_inner();
    /// assert!(stream.next().await.is_some());
    /// # Ok(())
    /// # }
    /// ```
    #[allow(
        clippy::unused_async,
        clippy::cast_possible_truncation,
        clippy::cast_possible_wrap
    )]
    pub async fn stream_generate_custom(
        request_id: String,
        prompt: String,
        max_tokens: usize,
        buffer_capacity: usize,
    ) -> Result<Response<ReceiverStream<Result<GenerateResponse, Status>>>, Status> {
        let buffer_size = buffer_capacity.max(1);
        let (tx, rx) = mpsc::channel(buffer_size);

        tokio::spawn(async move {
            let _ = prompt;
            let total = if max_tokens == 0 { 5 } else { max_tokens };

            tracing::debug!(
                request_id = %request_id,
                total_tokens = total,
                buffer_size,
                "initiating token stream generation"
            );

            for i in 0..total {
                let is_finished = i + 1 == total;
                let token_id = i32::try_from(i).unwrap_or(0);

                let mut token_text = String::with_capacity(16);
                let _ = write!(token_text, "token_{i} ");

                let resp = GenerateResponse {
                    request_id: request_id.clone(),
                    token_text,
                    token_id,
                    is_finished,
                    finish_reason: if is_finished {
                        "stop".to_string()
                    } else {
                        String::new()
                    },
                };

                if tx.send(Ok(resp)).await.is_err() {
                    tracing::debug!(
                        request_id = %request_id,
                        "client disconnected early; terminating token stream"
                    );
                    break;
                }
            }
        });

        Ok(Response::new(ReceiverStream::new(rx)))
    }
}

#[tonic::async_trait]
impl InferenceService for InferenceServiceImpl {
    type StreamGenerateStream = ReceiverStream<Result<GenerateResponse, Status>>;

    async fn stream_generate(
        &self,
        request: Request<GenerateRequest>,
    ) -> Result<Response<Self::StreamGenerateStream>, Status> {
        let inner = request.into_inner();
        Self::stream_generate_with_request(inner).await
    }
}

/// Starts the gRPC inference service on the specified socket address.
///
/// # Errors
/// Returns [`tonic::transport::Error`] if the server fails to bind or run.
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use gazellm::net::grpc_stream::start_grpc_server;
/// use std::net::SocketAddr;
///
/// let addr: SocketAddr = "127.0.0.1:50051".parse()?;
/// start_grpc_server(addr).await?;
/// # Ok(())
/// # }
/// ```
pub async fn start_grpc_server(addr: SocketAddr) -> Result<(), tonic::transport::Error> {
    let service = InferenceServiceServer::new(InferenceServiceImpl);
    tonic::transport::Server::builder()
        .add_service(service)
        .serve(addr)
        .await
}

/// Starts the gRPC inference service with a graceful shutdown signal.
///
/// # Errors
/// Returns [`tonic::transport::Error`] if the server fails during transport execution.
///
/// # Examples
///
/// ```no_run
/// # #[tokio::main]
/// # async fn main() -> Result<(), Box<dyn std::error::Error>> {
/// use gazellm::net::grpc_stream::start_grpc_server_with_shutdown;
/// use std::net::SocketAddr;
///
/// let addr: SocketAddr = "127.0.0.1:50051".parse()?;
/// let (tx, rx) = tokio::sync::oneshot::channel::<()>();
/// start_grpc_server_with_shutdown(addr, async { let _ = rx.await; }).await?;
/// # Ok(())
/// # }
/// ```
pub async fn start_grpc_server_with_shutdown<F>(
    addr: SocketAddr,
    signal: F,
) -> Result<(), tonic::transport::Error>
where
    F: std::future::Future<Output = ()> + Send + 'static,
{
    let service = InferenceServiceServer::new(InferenceServiceImpl);
    tonic::transport::Server::builder()
        .add_service(service)
        .serve_with_shutdown(addr, signal)
        .await
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_stream::StreamExt;

    #[tokio::test]
    async fn test_stream_generate_impl_emits_sequential_tokens() {
        let response = InferenceServiceImpl::stream_generate_impl("test prompt".to_string())
            .await
            .expect("stream_generate_impl should succeed");

        let mut stream = response.into_inner();
        let mut tokens_received = Vec::new();

        while let Some(item) = stream.next().await {
            let resp = item.expect("stream item should be Ok");
            tokens_received.push(resp);
        }

        assert_eq!(tokens_received.len(), 5);
        for (idx, resp) in tokens_received.iter().enumerate() {
            assert_eq!(resp.request_id, "req_1");
            assert_eq!(
                resp.token_id,
                i32::try_from(idx).expect("valid token index")
            );
            assert_eq!(resp.token_text, format!("token_{idx} "));
            if idx == 4 {
                assert!(resp.is_finished);
                assert_eq!(resp.finish_reason, "stop");
            } else {
                assert!(!resp.is_finished);
                assert_eq!(resp.finish_reason, "");
            }
        }
    }

    #[tokio::test]
    async fn test_stream_generate_with_custom_request() {
        let request = GenerateRequest {
            request_id: "custom-req-42".to_string(),
            prompt: "Describe the cosmos".to_string(),
            max_new_tokens: 3,
            temperature: 0.7,
            top_p: 0.9,
            top_k: 50,
        };

        let response = InferenceServiceImpl::stream_generate_with_request(request)
            .await
            .expect("custom request should succeed");

        let mut stream = response.into_inner();
        let mut collected = Vec::new();

        while let Some(item) = stream.next().await {
            collected.push(item.expect("valid token"));
        }

        assert_eq!(collected.len(), 3);
        assert_eq!(collected[0].request_id, "custom-req-42");
        assert_eq!(collected[2].token_id, 2);
        assert!(collected[2].is_finished);
        assert_eq!(collected[2].finish_reason, "stop");
    }

    #[tokio::test]
    async fn test_inference_service_trait_implementation() {
        let service = InferenceServiceImpl::new();
        let request = Request::new(GenerateRequest {
            request_id: "trait-test".to_string(),
            prompt: "testing trait invocation".to_string(),
            max_new_tokens: 2,
            temperature: 0.0,
            top_p: 1.0,
            top_k: 1,
        });

        let response = service
            .stream_generate(request)
            .await
            .expect("trait stream_generate should succeed");

        let mut stream = response.into_inner();

        let first = stream.next().await.expect("first token exists").unwrap();
        assert_eq!(first.request_id, "trait-test");
        assert_eq!(first.token_id, 0);
        assert!(!first.is_finished);

        let second = stream.next().await.expect("second token exists").unwrap();
        assert_eq!(second.request_id, "trait-test");
        assert_eq!(second.token_id, 1);
        assert!(second.is_finished);

        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn test_stream_cancellation_on_receiver_drop() {
        let response = InferenceServiceImpl::stream_generate_impl("cancel early".to_string())
            .await
            .expect("stream should open");

        let mut stream = response.into_inner();
        let first = stream.next().await;
        assert!(first.is_some());

        drop(stream);
    }

    #[tokio::test]
    async fn test_start_grpc_server_graceful_shutdown() {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("ephemeral port should bind");
        let local_addr = listener.local_addr().expect("local addr");
        drop(listener);

        let (shutdown_tx, shutdown_rx) = tokio::sync::oneshot::channel::<()>();

        let server_handle = tokio::spawn(async move {
            start_grpc_server_with_shutdown(local_addr, async {
                let _ = shutdown_rx.await;
            })
            .await
        });

        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let _ = shutdown_tx.send(());

        let server_shutdown_result = server_handle.await.expect("server handle joined");
        assert!(server_shutdown_result.is_ok());
    }
}
