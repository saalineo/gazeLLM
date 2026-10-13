//! High-performance tokenizer wrapper module.

pub mod fast_bpe;

pub use fast_bpe::{FastTokenizer, TokenizerError};
pub use tokenizers;
