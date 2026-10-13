//! Network server infrastructure (gRPC and HTTP REST frontend).

pub mod grpc;
pub mod grpc_stream;
pub mod http;
pub mod request_processor;
pub mod sse;

pub use axum;
pub use grpc::EngineInferenceService;
pub use grpc_stream::{
    ClientStream, InferenceServiceImpl, start_grpc_server, start_grpc_server_with_shutdown,
};
pub use http::{
    HttpCompletionRequest, HttpCompletionResponse, HttpServerError, create_router,
    start_http_server,
};
pub use prost;
pub use request_processor::{
    InferenceRequest, InferenceRequestBuilder, RequestValidationError, validate_request,
};
pub use sse::{SseTokenPayload, create_token_stream, create_token_stream_with_keepalive};
pub use tonic;

/// Network frontend server stub.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct NetworkServer;

impl NetworkServer {
    /// Creates a new [`NetworkServer`] instance.
    #[must_use]
    #[inline]
    pub const fn new() -> Self {
        Self
    }
}
