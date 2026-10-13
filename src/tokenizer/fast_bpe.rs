//! Fast BPE Tokenizer wrapper using `HuggingFace` Tokenizers C-bindings.

use std::path::Path;
use std::str::FromStr;
use std::sync::Arc;
use tokenizers::Tokenizer;

/// Errors that can occur during tokenizer operations.
#[derive(Debug, thiserror::Error)]
pub enum TokenizerError {
    /// Failed to load tokenizer file from the specified path.
    #[error("failed to load tokenizer configuration from '{path}': {source}")]
    Load {
        /// Path of the tokenizer vocabulary file.
        path: String,
        /// Underlying error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },
    /// Failed to parse tokenizer from string/bytes.
    #[error("failed to parse tokenizer configuration: {0}")]
    Parse(String),
    /// Failed to encode input text.
    #[error("failed to encode text: {0}")]
    Encode(String),
    /// Failed to decode token IDs.
    #[error("failed to decode tokens: {0}")]
    Decode(String),
}

impl From<tokenizers::Error> for TokenizerError {
    fn from(err: tokenizers::Error) -> Self {
        Self::Parse(err.to_string())
    }
}

/// High-performance thread-safe wrapper around `HuggingFace` [`Tokenizer`].
#[derive(Debug, Clone)]
pub struct FastTokenizer {
    inner: Arc<Tokenizer>,
}

impl FastTokenizer {
    /// Creates a [`FastTokenizer`] by loading a vocabulary file from disk.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Load`] if the file cannot be read or contains invalid tokenizer data.
    pub fn from_file<P: AsRef<Path>>(vocab_path: P) -> Result<Self, TokenizerError> {
        let path_ref = vocab_path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();

        let tok = Tokenizer::from_file(path_ref).map_err(|source| TokenizerError::Load {
            path: path_str,
            source,
        })?;

        Ok(Self {
            inner: Arc::new(tok),
        })
    }

    /// Creates a [`FastTokenizer`] from raw byte configuration.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Parse`] if the byte slice is not a valid tokenizer definition.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TokenizerError> {
        let tok =
            Tokenizer::from_bytes(bytes).map_err(|err| TokenizerError::Parse(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(tok),
        })
    }

    /// Creates a [`FastTokenizer`] from a JSON-formatted string.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Parse`] if the string is not a valid tokenizer definition.
    pub fn from_json_str(json: &str) -> Result<Self, TokenizerError> {
        json.parse()
    }

    /// Wraps an existing [`Tokenizer`] instance.
    #[must_use]
    pub fn from_tokenizer(tokenizer: Tokenizer) -> Self {
        Self {
            inner: Arc::new(tokenizer),
        }
    }

    /// Encodes a text prompt into a vector of token IDs.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Encode`] if tokenization fails.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>, TokenizerError> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|err| TokenizerError::Encode(err.to_string()))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encodes a batch of text prompts into vectors of token IDs.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Encode`] if any tokenization fails.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>, TokenizerError> {
        let encodings = self
            .inner
            .encode_batch(texts.to_vec(), false)
            .map_err(|err| TokenizerError::Encode(err.to_string()))?;

        Ok(encodings
            .into_iter()
            .map(|e| e.get_ids().to_vec())
            .collect())
    }

    /// Decodes a slice of token IDs back into a reconstructed text string.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Decode`] if decoding fails.
    pub fn decode(&self, ids: &[u32]) -> Result<String, TokenizerError> {
        let text = self
            .inner
            .decode(ids, true)
            .map_err(|err| TokenizerError::Decode(err.to_string()))?;

        Ok(text)
    }

    /// Decodes a batch of token ID sequences into reconstructed text strings.
    ///
    /// # Errors
    /// Returns [`TokenizerError::Decode`] if decoding fails.
    pub fn decode_batch(&self, sequences: &[&[u32]]) -> Result<Vec<String>, TokenizerError> {
        sequences.iter().map(|seq| self.decode(seq)).collect()
    }

    /// Returns the total vocabulary size of the tokenizer.
    #[must_use]
    #[inline]
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(true)
    }

    /// Returns `true` if the vocabulary is empty.
    #[must_use]
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vocab_size() == 0
    }

    /// Returns a shared reference to the underlying [`Tokenizer`].
    #[must_use]
    #[inline]
    pub fn inner(&self) -> &Tokenizer {
        &self.inner
    }

    /// Returns an `Arc` clone of the underlying [`Tokenizer`].
    #[must_use]
    #[inline]
    pub fn inner_arc(&self) -> Arc<Tokenizer> {
        Arc::clone(&self.inner)
    }
}

impl FromStr for FastTokenizer {
    type Err = TokenizerError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let tok = Tokenizer::from_str(s).map_err(|err| TokenizerError::Parse(err.to_string()))?;
        Ok(Self {
            inner: Arc::new(tok),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::thread;
    use tokenizers::models::bpe::BPE;
    use tokenizers::models::wordlevel::WordLevel;

    fn build_test_tokenizer() -> FastTokenizer {
        let vocab = [
            ("<unk>".to_string(), 0),
            ("<s>".to_string(), 1),
            ("</s>".to_string(), 2),
            ("Hello".to_string(), 3),
            ("world".to_string(), 4),
            ("!".to_string(), 5),
        ]
        .into_iter()
        .collect();

        let model = WordLevel::builder()
            .vocab(vocab)
            .unk_token("<unk>".to_string())
            .build()
            .expect("valid model");

        let mut tokenizer = Tokenizer::new(model);
        tokenizer.with_pre_tokenizer(Some(tokenizers::pre_tokenizers::whitespace::Whitespace));
        FastTokenizer::from_tokenizer(tokenizer)
    }

    #[test]
    fn test_encode_and_decode_roundtrip() {
        let tokenizer = build_test_tokenizer();

        let token_ids = tokenizer
            .encode("Hello world !")
            .expect("encoding succeeds");
        assert_eq!(token_ids, vec![3, 4, 5]);

        let decoded = tokenizer.decode(&token_ids).expect("decoding succeeds");
        assert_eq!(decoded, "Hello world !");
    }

    #[test]
    fn test_encode_and_decode_batch() {
        let tokenizer = build_test_tokenizer();

        let batch_ids = tokenizer
            .encode_batch(&["Hello world", "!"])
            .expect("batch encoding succeeds");
        assert_eq!(batch_ids, vec![vec![3, 4], vec![5]]);

        let decoded = tokenizer
            .decode_batch(&[&batch_ids[0], &batch_ids[1]])
            .expect("batch decoding succeeds");
        assert_eq!(decoded, vec!["Hello world", "!"]);
    }

    #[test]
    fn test_vocab_size() {
        let tokenizer = build_test_tokenizer();
        assert_eq!(tokenizer.vocab_size(), 6);
        assert!(!tokenizer.is_empty());
    }

    #[test]
    fn test_from_file_nonexistent() {
        let load_result = FastTokenizer::from_file("nonexistent_vocab_path_9999.json");
        assert!(load_result.is_err());
        let err = load_result.unwrap_err();
        assert!(matches!(err, TokenizerError::Load { .. }));
        assert!(err.to_string().contains("failed to load tokenizer"));
    }

    #[test]
    fn test_from_file_valid_temp() {
        let temp_dir = std::env::temp_dir();
        let file_path = temp_dir.join("test_bpe_tokenizer.json");

        let bpe = BPE::builder().build().expect("valid bpe");
        let tokenizer = Tokenizer::new(bpe);
        tokenizer
            .save(&file_path, true)
            .expect("save should succeed");

        let fast_tok =
            FastTokenizer::from_file(&file_path).expect("loading from temp file succeeds");
        assert_eq!(fast_tok.vocab_size(), 0);
        assert!(fast_tok.is_empty());

        let _ = fs::remove_file(file_path);
    }

    #[test]
    fn test_concurrent_tokenization() {
        let tokenizer = build_test_tokenizer();
        let mut handles = Vec::new();

        for _ in 0..8 {
            let tok_clone = tokenizer.clone();
            handles.push(thread::spawn(move || {
                let ids = tok_clone.encode("Hello world").expect("encode");
                assert_eq!(ids, vec![3, 4]);
                let decoded = tok_clone.decode(&ids).expect("decode");
                assert_eq!(decoded, "Hello world");
            }));
        }

        for handle in handles {
            handle.join().expect("thread should join");
        }
    }
}
