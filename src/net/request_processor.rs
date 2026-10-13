//! Inference request validation and processing logic.

use serde::{Deserialize, Serialize};
use std::fmt;

/// Errors encountered during inference request payload validation.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum RequestValidationError {
    /// The prompt string is empty.
    #[error("prompt cannot be empty")]
    EmptyPrompt,
    /// `max_tokens` is 0 or exceeds the maximum configured sequence length.
    #[error("max_tokens out of range: {max_tokens} (allowed: 1..={max_seq_len})")]
    MaxTokensOutOfRange {
        /// Provided token generation limit.
        max_tokens: usize,
        /// Maximum allowed sequence length for the engine.
        max_seq_len: usize,
    },
    /// Sampling temperature is negative, exceeds 2.0, or is NaN.
    #[error("temperature must be between 0.0 and 2.0, got {0}")]
    InvalidTemperature(f32),
    /// Nucleus sampling `top_p` is not within `(0.0, 1.0]` or is NaN.
    #[error("top_p must be between 0.0 (exclusive) and 1.0 (inclusive), got {0}")]
    InvalidTopP(f32),
}

/// High-level client inference request payload.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct InferenceRequest {
    /// Unique identifier for the client request.
    pub id: String,
    /// Text prompt to generate completion for.
    pub prompt: String,
    /// Maximum new tokens to produce.
    pub max_tokens: usize,
    /// Sampling temperature (higher means more random).
    pub temperature: f32,
    /// Nucleus sampling cutoff threshold.
    pub top_p: f32,
}

impl fmt::Display for InferenceRequest {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "InferenceRequest {{ id: '{}', max_tokens: {}, temperature: {:.2}, top_p: {:.2} }}",
            self.id, self.max_tokens, self.temperature, self.top_p
        )
    }
}

impl Default for InferenceRequest {
    fn default() -> Self {
        Self {
            id: String::new(),
            prompt: String::new(),
            max_tokens: 128,
            temperature: 0.7,
            top_p: 0.9,
        }
    }
}

impl InferenceRequest {
    /// Constructs a new [`InferenceRequest`] with default temperature (0.7) and `top_p` (0.9).
    #[must_use]
    pub fn new(id: impl Into<String>, prompt: impl Into<String>, max_tokens: usize) -> Self {
        Self {
            id: id.into(),
            prompt: prompt.into(),
            max_tokens,
            temperature: 0.7,
            top_p: 0.9,
        }
    }

    /// Returns a new builder for [`InferenceRequest`].
    #[must_use]
    pub fn builder() -> InferenceRequestBuilder {
        InferenceRequestBuilder::default()
    }

    /// Validates the request fields against engine constraints.
    ///
    /// # Errors
    /// Returns [`RequestValidationError`] if the prompt is empty, `max_tokens` is out of bounds,
    /// or sampling parameters (`temperature`, `top_p`) are out of their allowed ranges.
    pub fn validate(&self, max_seq_len: usize) -> Result<(), RequestValidationError> {
        if self.prompt.is_empty() {
            return Err(RequestValidationError::EmptyPrompt);
        }
        if self.max_tokens == 0 || self.max_tokens > max_seq_len {
            return Err(RequestValidationError::MaxTokensOutOfRange {
                max_tokens: self.max_tokens,
                max_seq_len,
            });
        }
        if self.temperature.is_nan() || !(0.0..=2.0).contains(&self.temperature) {
            return Err(RequestValidationError::InvalidTemperature(self.temperature));
        }
        if self.top_p.is_nan() || !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(RequestValidationError::InvalidTopP(self.top_p));
        }
        Ok(())
    }
}

/// Builder for constructing [`InferenceRequest`].
#[derive(Debug, Default, Clone)]
pub struct InferenceRequestBuilder {
    id: Option<String>,
    prompt: Option<String>,
    max_tokens: usize,
    temperature: Option<f32>,
    top_p: Option<f32>,
}

impl InferenceRequestBuilder {
    /// Creates a new [`InferenceRequestBuilder`].
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the request ID.
    #[must_use]
    pub fn id(mut self, id: impl Into<String>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Sets the prompt string.
    #[must_use]
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = Some(prompt.into());
        self
    }

    /// Sets the maximum new tokens.
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: usize) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Sets the sampling temperature.
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Sets the nucleus sampling `top_p` threshold.
    #[must_use]
    pub const fn top_p(mut self, top_p: f32) -> Self {
        self.top_p = Some(top_p);
        self
    }

    /// Builds the [`InferenceRequest`].
    #[must_use]
    pub fn build(self) -> InferenceRequest {
        InferenceRequest {
            id: self.id.unwrap_or_default(),
            prompt: self.prompt.unwrap_or_default(),
            max_tokens: self.max_tokens,
            temperature: self.temperature.unwrap_or(0.7),
            top_p: self.top_p.unwrap_or(0.9),
        }
    }

    /// Builds and validates the [`InferenceRequest`] against the engine sequence limit.
    ///
    /// # Errors
    /// Returns [`RequestValidationError`] if validation constraints are violated.
    pub fn build_and_validate(
        self,
        max_seq_len: usize,
    ) -> Result<InferenceRequest, RequestValidationError> {
        let request = self.build();
        request.validate(max_seq_len)?;
        Ok(request)
    }
}

/// Convenience standalone function to validate an [`InferenceRequest`].
///
/// # Errors
/// Returns [`RequestValidationError`] if the request parameters fail validation.
#[inline]
pub fn validate_request(
    request: &InferenceRequest,
    max_seq_len: usize,
) -> Result<(), RequestValidationError> {
    request.validate(max_seq_len)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_request() -> InferenceRequest {
        InferenceRequest::new("req-1", "Hello world", 128)
    }

    #[test]
    fn test_valid_request() {
        assert!(sample_request().validate(2048).is_ok());
    }

    #[test]
    fn test_builder() {
        let req = InferenceRequest::builder()
            .id("req-builder")
            .prompt("Test prompt")
            .max_tokens(256)
            .temperature(0.5)
            .top_p(0.95)
            .build();

        assert_eq!(req.id, "req-builder");
        assert_eq!(req.prompt, "Test prompt");
        assert_eq!(req.max_tokens, 256);
        assert!((req.temperature - 0.5).abs() < f32::EPSILON);
        assert!((req.top_p - 0.95).abs() < f32::EPSILON);

        let validated_req = InferenceRequestBuilder::new()
            .id("req-valid")
            .prompt("Valid prompt")
            .max_tokens(100)
            .build_and_validate(2048)
            .expect("validation succeeds");
        assert_eq!(validated_req.id, "req-valid");

        let default_req = InferenceRequest::default();
        assert_eq!(default_req.max_tokens, 128);
        assert!((default_req.temperature - 0.7).abs() < f32::EPSILON);
    }

    #[test]
    fn test_display() {
        let req = sample_request();
        let s = format!("{req}");
        assert!(s.contains("req-1"));
        assert!(s.contains("128"));
    }

    #[test]
    fn test_validation_bounds() {
        let mut req = sample_request();

        req.prompt.clear();
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::EmptyPrompt
        );

        req = sample_request();
        req.max_tokens = 0;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::MaxTokensOutOfRange {
                max_tokens: 0,
                max_seq_len: 2048
            }
        );

        req.max_tokens = 4096;
        assert_eq!(
            req.validate(2048).unwrap_err(),
            RequestValidationError::MaxTokensOutOfRange {
                max_tokens: 4096,
                max_seq_len: 2048
            }
        );

        for bad_temp in [-0.1, 2.1, f32::NAN] {
            req = sample_request();
            req.temperature = bad_temp;
            let err = req.validate(2048).unwrap_err();
            assert!(matches!(err, RequestValidationError::InvalidTemperature(_)));
        }

        for bad_top_p in [0.0, -0.5, 1.01, f32::NAN] {
            req = sample_request();
            req.top_p = bad_top_p;
            let err = req.validate(2048).unwrap_err();
            assert!(matches!(err, RequestValidationError::InvalidTopP(_)));
        }
    }

    #[test]
    fn test_serde_roundtrip() {
        let req = sample_request();
        let json = serde_json::to_string(&req).unwrap();
        let parsed: InferenceRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(req, parsed);
    }
}
