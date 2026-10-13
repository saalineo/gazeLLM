//! Safe Rust wrapper and C-ABI bridge for Mojo kernel execution.

use crate::ffi::types::{FfiResult, MojoTensorBuffer};

/// Errors arising from FFI bridge execution.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum FfiBridgeError {
    /// Kernel returned a non-zero exit status with an error message.
    #[error("mojo kernel execution failed with status {status_code}: {message}")]
    KernelFailed {
        /// Non-zero status code returned by kernel.
        status_code: i32,
        /// Description of the error.
        message: String,
    },
    /// Kernel returned a non-zero exit status without an error message.
    #[error("mojo kernel execution failed with unknown error (status {status_code})")]
    Unknown {
        /// Non-zero status code returned by kernel.
        status_code: i32,
    },
}

/// Decodes an [`FfiResult`] into a Rust [`Result<(), FfiBridgeError>`].
///
/// # Errors
/// Returns [`FfiBridgeError::KernelFailed`] if the status is non-zero with an error message,
/// or [`FfiBridgeError::Unknown`] if no message is provided.
#[inline]
pub fn parse_ffi_result(result: FfiResult) -> Result<(), FfiBridgeError> {
    if result.is_ok() {
        return Ok(());
    }

    // SAFETY: If non-null, `result.error_message` is checked to point to a valid null-terminated C string.
    let message = unsafe { result.message_cstr() }.map(|cstr| cstr.to_string_lossy().into_owned());

    Err(message.map_or(
        FfiBridgeError::Unknown {
            status_code: result.status_code,
        },
        |message| FfiBridgeError::KernelFailed {
            status_code: result.status_code,
            message,
        },
    ))
}

unsafe extern "C" {
    /// Raw C-ABI symbol for executing a Mojo compute kernel.
    ///
    /// # Safety
    /// Pointers `input` and `output` must be valid, non-null, aligned, and point to
    /// properly initialized [`MojoTensorBuffer`] structures.
    pub fn mojo_execute_kernel(
        input: *const MojoTensorBuffer,
        output: *mut MojoTensorBuffer,
    ) -> FfiResult;
}

/// Executes a Mojo compute kernel across the C-ABI FFI boundary.
///
/// # Errors
/// Returns [`FfiBridgeError::KernelFailed`] if the Mojo kernel returns a non-zero status code
/// with an error message, or [`FfiBridgeError::Unknown`] if no message is provided.
#[inline]
pub fn safe_mojo_execute(
    input: &MojoTensorBuffer,
    output: &mut MojoTensorBuffer,
) -> Result<(), FfiBridgeError> {
    // SAFETY: We pass valid references converted via raw reference operators that satisfy Mojo C-ABI layout.
    let kernel_result = unsafe { mojo_execute_kernel(&raw const *input, &raw mut *output) };
    parse_ffi_result(kernel_result)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::{Cell, RefCell};
    use std::ffi::CString;

    thread_local! {
        static MOCK_STATUS: Cell<i32> = const { Cell::new(0) };
        static MOCK_ERROR_MSG: RefCell<Option<CString>> = const { RefCell::new(None) };
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn mojo_execute_kernel(
        input: *const MojoTensorBuffer,
        output: *mut MojoTensorBuffer,
    ) -> FfiResult {
        let result = std::panic::catch_unwind(|| {
            if input.is_null() || output.is_null() {
                return FfiResult::err(1, c"Null buffer pointer".as_ptr());
            }

            let status = MOCK_STATUS.with(Cell::get);
            if status == 0 {
                FfiResult::ok()
            } else {
                let msg_ptr = MOCK_ERROR_MSG.with(|m| {
                    m.borrow()
                        .as_ref()
                        .map_or(std::ptr::null(), |cstr| cstr.as_ptr())
                });
                FfiResult::err(status, msg_ptr)
            }
        });
        result.unwrap_or_else(|_| FfiResult::err(-1, c"Panic caught at FFI boundary".as_ptr()))
    }

    fn set_mock_behavior(status: i32, msg: Option<&str>) {
        MOCK_STATUS.with(|s| s.set(status));
        MOCK_ERROR_MSG.with(|m| {
            *m.borrow_mut() = msg.map(|s| CString::new(s).unwrap());
        });
    }

    fn dummy_buffer() -> MojoTensorBuffer {
        MojoTensorBuffer {
            data_ptr: std::ptr::null_mut(),
            num_elements: 10,
            element_size_bytes: 4,
            dtype: 0,
        }
    }

    #[test]
    fn test_safe_mojo_execute_success() {
        set_mock_behavior(0, None);
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        assert!(safe_mojo_execute(&input, &mut output).is_ok());
    }

    #[test]
    fn test_safe_mojo_execute_error_with_message() {
        set_mock_behavior(42, Some("CUDA out of memory in Mojo kernel"));
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        let err = safe_mojo_execute(&input, &mut output).unwrap_err();
        assert_eq!(
            err,
            FfiBridgeError::KernelFailed {
                status_code: 42,
                message: "CUDA out of memory in Mojo kernel".to_string(),
            }
        );
    }

    #[test]
    fn test_safe_mojo_execute_error_null_message() {
        set_mock_behavior(-1, None);
        let input = dummy_buffer();
        let mut output = dummy_buffer();
        let err = safe_mojo_execute(&input, &mut output).unwrap_err();
        assert_eq!(err, FfiBridgeError::Unknown { status_code: -1 });
    }
}
