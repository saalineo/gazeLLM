//! Shared C ABI FFI type definitions for Rust and Mojo compute execution.

use std::ffi::{CStr, c_char};
use std::fmt;
use std::str::FromStr;

/// Supported tensor data types for Mojo compute kernels.
pub mod dtype {
    /// 32-bit floating point.
    pub const FP32: i32 = 0;
    /// 16-bit floating point.
    pub const FP16: i32 = 1;
    /// Brain floating point (16-bit).
    pub const BF16: i32 = 2;
    /// 8-bit signed integer.
    pub const INT8: i32 = 3;
}

/// Strongly-typed enumeration of tensor data types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(i32)]
pub enum DType {
    /// 32-bit floating point.
    #[default]
    Fp32 = dtype::FP32,
    /// 16-bit floating point.
    Fp16 = dtype::FP16,
    /// Brain floating point (16-bit).
    Bf16 = dtype::BF16,
    /// 8-bit signed integer.
    Int8 = dtype::INT8,
}

impl DType {
    /// Returns the element size in bytes for this data type.
    #[must_use]
    #[inline]
    pub const fn size_bytes(self) -> usize {
        match self {
            Self::Fp32 => 4,
            Self::Fp16 | Self::Bf16 => 2,
            Self::Int8 => 1,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Fp32 => write!(f, "fp32"),
            Self::Fp16 => write!(f, "fp16"),
            Self::Bf16 => write!(f, "bf16"),
            Self::Int8 => write!(f, "int8"),
        }
    }
}

impl From<DType> for i32 {
    #[inline]
    fn from(dt: DType) -> Self {
        dt as Self
    }
}

/// Error type when converting an invalid integer into a [`DType`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[repr(transparent)]
#[error("unknown dtype code: {0}")]
pub struct InvalidDTypeError(pub i32);

impl TryFrom<i32> for DType {
    type Error = InvalidDTypeError;

    #[inline]
    fn try_from(value: i32) -> Result<Self, Self::Error> {
        match value {
            dtype::FP32 => Ok(Self::Fp32),
            dtype::FP16 => Ok(Self::Fp16),
            dtype::BF16 => Ok(Self::Bf16),
            dtype::INT8 => Ok(Self::Int8),
            other => Err(InvalidDTypeError(other)),
        }
    }
}

/// Error type when parsing a [`DType`] from a string.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[repr(transparent)]
#[error("unknown dtype string '{0}', expected 'fp32', 'fp16', 'bf16', or 'int8'")]
pub struct ParseDTypeError(pub String);

impl FromStr for DType {
    type Err = ParseDTypeError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.to_ascii_lowercase().as_str() {
            "fp32" | "f32" | "float32" => Ok(Self::Fp32),
            "fp16" | "f16" | "float16" => Ok(Self::Fp16),
            "bf16" | "bfloat16" => Ok(Self::Bf16),
            "int8" | "i8" => Ok(Self::Int8),
            _ => Err(ParseDTypeError(s.to_string())),
        }
    }
}

/// C-compatible representation of a Mojo tensor memory buffer.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MojoTensorBuffer {
    /// Pointer to the contiguous raw memory buffer.
    pub data_ptr: *mut std::ffi::c_void,
    /// Number of elements in the tensor.
    pub num_elements: usize,
    /// Byte size of each individual element.
    pub element_size_bytes: usize,
    /// Data type identifier (matching [`DType`] or [`dtype`]).
    pub dtype: i32,
}

impl Default for MojoTensorBuffer {
    fn default() -> Self {
        Self {
            data_ptr: std::ptr::null_mut(),
            num_elements: 0,
            element_size_bytes: 0,
            dtype: 0,
        }
    }
}

impl MojoTensorBuffer {
    /// Constructs a new [`MojoTensorBuffer`].
    #[must_use]
    #[inline]
    pub const fn new(
        data_ptr: *mut std::ffi::c_void,
        num_elements: usize,
        element_size_bytes: usize,
        dtype: i32,
    ) -> Self {
        Self {
            data_ptr,
            num_elements,
            element_size_bytes,
            dtype,
        }
    }

    /// Returns `true` if the data pointer is null.
    #[must_use]
    #[inline]
    pub const fn is_null(&self) -> bool {
        self.data_ptr.is_null()
    }

    /// Computes the total byte footprint of the buffer (`num_elements * element_size_bytes`).
    #[must_use]
    #[inline]
    pub const fn total_bytes(&self) -> usize {
        self.num_elements.saturating_mul(self.element_size_bytes)
    }

    /// Returns the strongly-typed [`DType`] enum if valid.
    ///
    /// # Errors
    /// Returns [`InvalidDTypeError`] if the integer code does not match a known data type.
    #[inline]
    pub fn dtype_enum(&self) -> Result<DType, InvalidDTypeError> {
        DType::try_from(self.dtype)
    }
}

/// C-compatible representation of a block table for `PagedAttention` memory management.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct MojoBlockTable {
    /// Pointer to the block indices array.
    pub block_table_ptr: *mut i32,
    /// Total number of mapped blocks.
    pub num_blocks: usize,
    /// Token capacity per block.
    pub block_size: usize,
}

impl Default for MojoBlockTable {
    fn default() -> Self {
        Self {
            block_table_ptr: std::ptr::null_mut(),
            num_blocks: 0,
            block_size: 0,
        }
    }
}

impl MojoBlockTable {
    /// Constructs a new [`MojoBlockTable`].
    #[must_use]
    #[inline]
    pub const fn new(block_table_ptr: *mut i32, num_blocks: usize, block_size: usize) -> Self {
        Self {
            block_table_ptr,
            num_blocks,
            block_size,
        }
    }

    /// Returns `true` if the table pointer is null.
    #[must_use]
    #[inline]
    pub const fn is_null(&self) -> bool {
        self.block_table_ptr.is_null()
    }

    /// Computes the total token capacity across all allocated blocks.
    #[must_use]
    #[inline]
    pub const fn total_capacity(&self) -> usize {
        self.num_blocks.saturating_mul(self.block_size)
    }
}

/// C-compatible status result returned from Mojo FFI kernel invocations.
#[repr(C)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub struct FfiResult {
    /// Exit status code (0 for success, non-zero for error).
    pub status_code: i32,
    /// Pointer to null-terminated C string if an error occurred, or null on success.
    pub error_message: *const c_char,
}

impl Default for FfiResult {
    fn default() -> Self {
        Self::ok()
    }
}

impl FfiResult {
    /// Constructs a success result (`status_code == 0`) with a null error pointer.
    #[must_use]
    pub const fn ok() -> Self {
        Self {
            status_code: 0,
            error_message: std::ptr::null(),
        }
    }

    /// Constructs an error result with the specified non-zero status code and message pointer.
    #[must_use]
    pub const fn err(status_code: i32, message: *const c_char) -> Self {
        Self {
            status_code,
            error_message: message,
        }
    }

    /// Returns `true` if the operation succeeded (`status_code == 0`).
    #[must_use]
    #[inline]
    pub const fn is_ok(&self) -> bool {
        self.status_code == 0
    }

    /// Returns `true` if the operation failed (`status_code != 0`).
    #[must_use]
    #[inline]
    pub const fn is_err(&self) -> bool {
        self.status_code != 0
    }

    /// Returns the error message as a [`CStr`] slice if non-null.
    ///
    /// # Safety
    /// If non-null, `error_message` must point to a valid, null-terminated C string.
    #[must_use]
    pub const unsafe fn message_cstr(&self) -> Option<&CStr> {
        if self.error_message.is_null() {
            None
        } else {
            // SAFETY: Caller guarantees `error_message` points to a valid null-terminated string.
            Some(unsafe { CStr::from_ptr(self.error_message) })
        }
    }
}

// Compile-time static assertions ensuring memory alignment and layout match C ABI.
const _: () = {
    assert!(std::mem::size_of::<MojoTensorBuffer>() == 32);
    assert!(std::mem::align_of::<MojoTensorBuffer>() == 8);
    assert!(std::mem::offset_of!(MojoTensorBuffer, data_ptr) == 0);
    assert!(std::mem::offset_of!(MojoTensorBuffer, num_elements) == 8);
    assert!(std::mem::offset_of!(MojoTensorBuffer, element_size_bytes) == 16);
    assert!(std::mem::offset_of!(MojoTensorBuffer, dtype) == 24);

    assert!(std::mem::size_of::<MojoBlockTable>() == 24);
    assert!(std::mem::align_of::<MojoBlockTable>() == 8);
    assert!(std::mem::offset_of!(MojoBlockTable, block_table_ptr) == 0);
    assert!(std::mem::offset_of!(MojoBlockTable, num_blocks) == 8);
    assert!(std::mem::offset_of!(MojoBlockTable, block_size) == 16);

    assert!(std::mem::size_of::<FfiResult>() == 16);
    assert!(std::mem::align_of::<FfiResult>() == 8);
    assert!(std::mem::offset_of!(FfiResult, status_code) == 0);
    assert!(std::mem::offset_of!(FfiResult, error_message) == 8);
};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tensor_buffer_layout() {
        assert_eq!(std::mem::size_of::<MojoTensorBuffer>(), 32);
        assert_eq!(std::mem::align_of::<MojoTensorBuffer>(), 8);
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, data_ptr), 0);
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, num_elements), 8);
        assert_eq!(
            std::mem::offset_of!(MojoTensorBuffer, element_size_bytes),
            16
        );
        assert_eq!(std::mem::offset_of!(MojoTensorBuffer, dtype), 24);

        let buf = MojoTensorBuffer::new(std::ptr::null_mut(), 100, 4, dtype::FP32);
        assert!(buf.is_null());
        assert_eq!(buf.total_bytes(), 400);
        assert_eq!(buf.dtype_enum(), Ok(DType::Fp32));
    }

    #[test]
    fn test_block_table_layout() {
        assert_eq!(std::mem::size_of::<MojoBlockTable>(), 24);
        assert_eq!(std::mem::align_of::<MojoBlockTable>(), 8);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, block_table_ptr), 0);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, num_blocks), 8);
        assert_eq!(std::mem::offset_of!(MojoBlockTable, block_size), 16);

        let table = MojoBlockTable::new(std::ptr::null_mut(), 16, 32);
        assert!(table.is_null());
        assert_eq!(table.total_capacity(), 512);
    }

    #[test]
    fn test_ffi_result_layout() {
        assert_eq!(std::mem::size_of::<FfiResult>(), 16);
        assert_eq!(std::mem::align_of::<FfiResult>(), 8);
        assert_eq!(std::mem::offset_of!(FfiResult, status_code), 0);
        assert_eq!(std::mem::offset_of!(FfiResult, error_message), 8);
    }

    #[test]
    fn test_ffi_result_helpers() {
        let ok_result = FfiResult::ok();
        assert!(ok_result.is_ok());
        assert!(!ok_result.is_err());
        assert_eq!(ok_result.status_code, 0);
        assert!(ok_result.error_message.is_null());
        // SAFETY: null pointer test for message_cstr
        assert!(unsafe { ok_result.message_cstr() }.is_none());

        let err_result = FfiResult::err(1, std::ptr::null());
        assert!(!err_result.is_ok());
        assert!(err_result.is_err());
        assert_eq!(err_result.status_code, 1);
    }

    #[test]
    fn test_dtype_conversions() {
        assert_eq!(DType::try_from(0), Ok(DType::Fp32));
        assert_eq!(DType::try_from(1), Ok(DType::Fp16));
        assert_eq!(DType::try_from(2), Ok(DType::Bf16));
        assert_eq!(DType::try_from(3), Ok(DType::Int8));
        assert_eq!(DType::try_from(99), Err(InvalidDTypeError(99)));

        assert_eq!(i32::from(DType::Fp32), 0);
        assert_eq!(i32::from(DType::Int8), 3);

        assert_eq!(DType::Fp32.size_bytes(), 4);
        assert_eq!(DType::Fp16.size_bytes(), 2);
        assert_eq!(DType::Bf16.size_bytes(), 2);
        assert_eq!(DType::Int8.size_bytes(), 1);

        assert_eq!(format!("{}", DType::Fp32), "fp32");
        assert_eq!("bf16".parse::<DType>(), Ok(DType::Bf16));
        assert!("invalid".parse::<DType>().is_err());
    }

    #[test]
    fn test_type_defaults_and_equality() {
        assert_eq!(DType::default(), DType::Fp32);

        let default_tensor = MojoTensorBuffer::default();
        assert!(default_tensor.is_null());
        assert_eq!(default_tensor.num_elements, 0);
        assert_eq!(default_tensor, MojoTensorBuffer::default());

        let default_table = MojoBlockTable::default();
        assert!(default_table.is_null());
        assert_eq!(default_table.num_blocks, 0);
        assert_eq!(default_table, MojoBlockTable::default());

        let default_res = FfiResult::default();
        assert!(default_res.is_ok());
        assert_eq!(default_res, FfiResult::ok());
    }
}
