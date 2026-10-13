//! Engine configuration management module.

use serde::{Deserialize, Serialize};
use std::fmt;
use std::fs;
use std::path::Path;

/// High-performance engine configuration settings.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EngineConfig {
    /// Host IP address to bind network listeners.
    pub host: String,
    /// Port number for HTTP/REST and gRPC services.
    pub port: u16,
    /// Maximum number of concurrent requests in a batch.
    pub max_batch_size: usize,
    /// Number of tokens per `PagedAttention` memory block.
    pub block_size: usize,
    /// Total number of memory blocks allocated in host RAM.
    pub max_num_blocks: usize,
    /// Total number of memory blocks allocated on GPU VRAM.
    pub num_gpu_blocks: usize,
    /// File path to the compiled Mojo shared library (`.so` / `.dylib`).
    pub mojo_library_path: String,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            host: "127.0.0.1".to_string(),
            port: 8080,
            max_batch_size: 32,
            block_size: 16,
            max_num_blocks: 1024,
            num_gpu_blocks: 512,
            mojo_library_path: "target/libmojo_core.so".to_string(),
        }
    }
}

impl fmt::Display for EngineConfig {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "EngineConfig {{ host: {}, port: {}, max_batch_size: {}, block_size: {}, max_num_blocks: {}, num_gpu_blocks: {}, mojo_library_path: '{}' }}",
            self.host,
            self.port,
            self.max_batch_size,
            self.block_size,
            self.max_num_blocks,
            self.num_gpu_blocks,
            self.mojo_library_path
        )
    }
}

/// Errors encountered when loading or validating engine configuration.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    /// I/O failure when reading the configuration file.
    #[error("failed to read configuration file at '{path}': {source}")]
    Io {
        /// Target file path.
        path: String,
        /// Underlying I/O error.
        #[source]
        source: std::io::Error,
    },
    /// JSON deserialization failure.
    #[error("failed to parse JSON configuration in '{path}': {source}")]
    Json {
        /// Target file path.
        path: String,
        /// Underlying serde JSON error.
        #[source]
        source: serde_json::Error,
    },
    /// Invalid configuration value.
    #[error("invalid configuration for '{field}': {reason}")]
    InvalidValue {
        /// Configuration field name.
        field: &'static str,
        /// Explanation of why the value is invalid.
        reason: String,
    },
}

impl EngineConfig {
    /// Returns a new builder to construct an [`EngineConfig`].
    #[must_use]
    pub fn builder() -> EngineConfigBuilder {
        EngineConfigBuilder::default()
    }

    /// Loads configuration from a JSON file and validates its parameters.
    ///
    /// # Errors
    /// Returns [`ConfigError::Io`] if the file cannot be read, [`ConfigError::Json`]
    /// if the file contains invalid JSON structure, or [`ConfigError::InvalidValue`]
    /// if any configuration values violate invariant constraints.
    pub fn load_from_file<P: AsRef<Path>>(path: P) -> Result<Self, ConfigError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();
        let content = fs::read_to_string(path_ref).map_err(|source| ConfigError::Io {
            path: path_str.clone(),
            source,
        })?;
        let config: Self = serde_json::from_str(&content).map_err(|source| ConfigError::Json {
            path: path_str,
            source,
        })?;
        config.validate()?;
        Ok(config)
    }

    /// Validates configuration parameters against runtime constraints.
    ///
    /// # Errors
    /// Returns [`ConfigError::InvalidValue`] if any field violates invariant constraints.
    pub fn validate(&self) -> Result<(), ConfigError> {
        if self.host.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "host",
                reason: "host address cannot be empty".to_string(),
            });
        }
        if self.max_batch_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_batch_size",
                reason: "max_batch_size must be greater than 0".to_string(),
            });
        }
        if self.block_size == 0 {
            return Err(ConfigError::InvalidValue {
                field: "block_size",
                reason: "block_size must be greater than 0".to_string(),
            });
        }
        if self.max_num_blocks == 0 {
            return Err(ConfigError::InvalidValue {
                field: "max_num_blocks",
                reason: "max_num_blocks must be greater than 0".to_string(),
            });
        }
        if self.num_gpu_blocks == 0 {
            return Err(ConfigError::InvalidValue {
                field: "num_gpu_blocks",
                reason: "num_gpu_blocks must be greater than 0".to_string(),
            });
        }
        if self.mojo_library_path.is_empty() {
            return Err(ConfigError::InvalidValue {
                field: "mojo_library_path",
                reason: "mojo_library_path cannot be empty".to_string(),
            });
        }
        Ok(())
    }

    /// Returns the host IP address.
    #[must_use]
    #[inline]
    pub fn host(&self) -> &str {
        &self.host
    }

    /// Returns the port number.
    #[must_use]
    #[inline]
    pub const fn port(&self) -> u16 {
        self.port
    }

    /// Returns the maximum batch size.
    #[must_use]
    #[inline]
    pub const fn max_batch_size(&self) -> usize {
        self.max_batch_size
    }

    /// Returns the block size (tokens per block).
    #[must_use]
    #[inline]
    pub const fn block_size(&self) -> usize {
        self.block_size
    }

    /// Returns the maximum number of blocks allocated in host RAM.
    #[must_use]
    #[inline]
    pub const fn max_num_blocks(&self) -> usize {
        self.max_num_blocks
    }

    /// Returns the number of blocks allocated in GPU VRAM.
    #[must_use]
    #[inline]
    pub const fn num_gpu_blocks(&self) -> usize {
        self.num_gpu_blocks
    }

    /// Returns the Mojo shared library file path.
    #[must_use]
    #[inline]
    pub fn mojo_library_path(&self) -> &str {
        &self.mojo_library_path
    }
}

/// Builder for [`EngineConfig`].
#[derive(Debug, Clone, Default)]
pub struct EngineConfigBuilder {
    config: EngineConfig,
}

impl EngineConfigBuilder {
    /// Creates a new [`EngineConfigBuilder`] initialized with default engine settings.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the host IP address.
    #[must_use]
    pub fn host(mut self, host: impl Into<String>) -> Self {
        self.config.host = host.into();
        self
    }

    /// Sets the port number.
    #[must_use]
    pub const fn port(mut self, port: u16) -> Self {
        self.config.port = port;
        self
    }

    /// Sets the max batch size.
    #[must_use]
    pub const fn max_batch_size(mut self, size: usize) -> Self {
        self.config.max_batch_size = size;
        self
    }

    /// Sets the block size.
    #[must_use]
    pub const fn block_size(mut self, size: usize) -> Self {
        self.config.block_size = size;
        self
    }

    /// Sets the max number of memory blocks in host RAM.
    #[must_use]
    pub const fn max_num_blocks(mut self, num: usize) -> Self {
        self.config.max_num_blocks = num;
        self
    }

    /// Sets the number of GPU blocks.
    #[must_use]
    pub const fn num_gpu_blocks(mut self, num: usize) -> Self {
        self.config.num_gpu_blocks = num;
        self
    }

    /// Sets the Mojo library file path.
    #[must_use]
    pub fn mojo_library_path(mut self, path: impl Into<String>) -> Self {
        self.config.mojo_library_path = path.into();
        self
    }

    /// Builds and validates the [`EngineConfig`].
    ///
    /// # Errors
    /// Returns [`ConfigError::InvalidValue`] if configuration validation fails.
    pub fn build(self) -> Result<EngineConfig, ConfigError> {
        self.config.validate()?;
        Ok(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_load_valid_config() {
        let json = r#"{
            "host": "127.0.0.1",
            "port": 8080,
            "max_batch_size": 32,
            "block_size": 16,
            "max_num_blocks": 1024,
            "num_gpu_blocks": 512,
            "mojo_library_path": "target/libmojo_core.so"
        }"#;

        let config: EngineConfig =
            serde_json::from_str(json).expect("valid JSON should deserialize");
        assert_eq!(config, EngineConfig::default());
        assert!(config.validate().is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let config = EngineConfig::builder()
            .host("0.0.0.0")
            .port(9090)
            .max_batch_size(64)
            .block_size(32)
            .max_num_blocks(2048)
            .num_gpu_blocks(1024)
            .mojo_library_path("target/libmojo_kernel.so")
            .build()
            .expect("valid builder configuration");

        assert_eq!(config.host, "0.0.0.0");
        assert_eq!(config.host(), "0.0.0.0");
        assert_eq!(config.port, 9090);
        assert_eq!(config.port(), 9090);
        assert_eq!(config.max_batch_size, 64);
        assert_eq!(config.max_batch_size(), 64);
        assert_eq!(config.block_size, 32);
        assert_eq!(config.block_size(), 32);
        assert_eq!(config.max_num_blocks, 2048);
        assert_eq!(config.max_num_blocks(), 2048);
        assert_eq!(config.num_gpu_blocks, 1024);
        assert_eq!(config.num_gpu_blocks(), 1024);
        assert_eq!(config.mojo_library_path, "target/libmojo_kernel.so");
        assert_eq!(config.mojo_library_path(), "target/libmojo_kernel.so");

        let default_via_builder = EngineConfigBuilder::new().build().expect("default valid");
        assert_eq!(default_via_builder, EngineConfig::default());
    }

    #[test]
    fn test_validation_errors() {
        let mut config = EngineConfig::default();
        config.host.clear();
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue { field: "host", .. }
        ));

        config = EngineConfig::default();
        config.max_batch_size = 0;
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue {
                field: "max_batch_size",
                ..
            }
        ));

        config = EngineConfig::default();
        config.block_size = 0;
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue {
                field: "block_size",
                ..
            }
        ));

        config = EngineConfig::default();
        config.max_num_blocks = 0;
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue {
                field: "max_num_blocks",
                ..
            }
        ));

        config = EngineConfig::default();
        config.num_gpu_blocks = 0;
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue {
                field: "num_gpu_blocks",
                ..
            }
        ));

        config = EngineConfig::default();
        config.mojo_library_path.clear();
        assert!(matches!(
            config.validate().unwrap_err(),
            ConfigError::InvalidValue {
                field: "mojo_library_path",
                ..
            }
        ));
    }

    #[test]
    fn test_display_formatting() {
        let config = EngineConfig::default();
        let display_str = format!("{config}");
        assert!(display_str.contains("127.0.0.1"));
        assert!(display_str.contains("8080"));
    }

    #[test]
    fn test_load_nonexistent_file() {
        let load_result = EngineConfig::load_from_file("nonexistent_config_path_12345.json");
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(matches!(err, ConfigError::Io { .. }));
        assert!(
            err.to_string()
                .contains("failed to read configuration file")
        );
    }

    #[test]
    fn test_load_invalid_json_file() {
        use std::io::Write;
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("invalid_config_test.json");
        let mut file = fs::File::create(&file_path).expect("failed to create temporary file");
        writeln!(file, "{{ invalid json ").expect("failed to write invalid json");

        let load_result = EngineConfig::load_from_file(&file_path);
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(matches!(err, ConfigError::Json { .. }));
        assert!(
            err.to_string()
                .contains("failed to parse JSON configuration")
        );
        let _ = fs::remove_file(file_path);
    }
}
