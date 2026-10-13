//! gazeLLM Core-Engine Library
//!
//! Zero-Python high-throughput LLM serving infrastructure.

#![warn(missing_docs)]
#![warn(clippy::all, clippy::pedantic, clippy::nursery)]
#![allow(clippy::module_name_repetitions)]

pub mod config;
pub mod ffi;
pub mod memory;
pub mod net;
pub mod scheduler;
pub mod tokenizer;
pub mod weights;

// Re-exported dependencies whose types appear in public APIs (Item 24).
pub use axum;
pub use libloading;
pub use prost;
pub use tokenizers;
pub use tonic;

/// Returns the engine version string.
#[must_use]
pub const fn engine_version() -> &'static str {
    "0.1.0-alpha"
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_engine_version() {
        assert_eq!(engine_version(), "0.1.0-alpha");
    }
}
