//! Dynamic library loader for runtime hot-swapping of Mojo compute kernels.

use crate::ffi::bridge::{FfiBridgeError, parse_ffi_result};
use crate::ffi::types::{FfiResult, MojoTensorBuffer};
use libloading::{Library, Symbol};
use std::path::Path;
use std::sync::Arc;

/// Type alias for the C-ABI Mojo kernel entry point function pointer.
pub type MojoKernelFn =
    unsafe extern "C" fn(*const MojoTensorBuffer, *mut MojoTensorBuffer) -> FfiResult;

/// Errors arising from dynamic loading of Mojo shared libraries.
#[derive(Debug, thiserror::Error)]
pub enum MojoLibraryError {
    /// Failure loading shared library from disk.
    #[error("failed to load dynamic library at '{path}': {source}")]
    Load {
        /// File path of the shared library.
        path: String,
        /// Underlying libloading error.
        #[source]
        source: libloading::Error,
    },
    /// Failure resolving symbol in shared library.
    #[error("failed to resolve kernel symbol 'mojo_execute_kernel': {source}")]
    SymbolNotFound {
        /// Underlying libloading error.
        #[source]
        source: libloading::Error,
    },
}

/// RAII wrapper managing the lifecycle and dynamic symbol table of a compiled Mojo shared object.
pub struct MojoLibrary {
    _lib: Library,
    /// Function symbol pointer for `mojo_execute_kernel`.
    pub execute_kernel: Symbol<'static, MojoKernelFn>,
}

impl std::fmt::Debug for MojoLibrary {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MojoLibrary").finish_non_exhaustive()
    }
}

impl MojoLibrary {
    /// Dynamically loads a Mojo shared object (`.so` / `.dylib`) from the given path
    /// and resolves the entry kernel symbol `mojo_execute_kernel`.
    ///
    /// # Errors
    /// Returns [`MojoLibraryError::Load`] if the library cannot be opened, or
    /// [`MojoLibraryError::SymbolNotFound`] if `mojo_execute_kernel` is missing.
    pub fn load<P: AsRef<Path>>(path: P) -> Result<Arc<Self>, MojoLibraryError> {
        let path_ref = path.as_ref();
        let path_str = path_ref.to_string_lossy().to_string();

        // SAFETY: Loading external shared objects assumes standard C-ABI calling conventions
        // and matching `MojoTensorBuffer` / `FfiResult` layouts.
        let lib = unsafe {
            Library::new(path_ref).map_err(|source| MojoLibraryError::Load {
                path: path_str,
                source,
            })?
        };

        // SAFETY: Symbol name is null-terminated and matches expected C ABI signature.
        let kernel_symbol: Symbol<MojoKernelFn> = unsafe {
            lib.get(b"mojo_execute_kernel\0")
                .map_err(|source| MojoLibraryError::SymbolNotFound { source })?
        };

        // SAFETY: The symbol pointer remains valid for the lifetime of `_lib`,
        // which is co-owned inside the returned `MojoLibrary` struct.
        let execute_kernel_static: Symbol<'static, MojoKernelFn> =
            unsafe { std::mem::transmute(kernel_symbol) };

        Ok(Arc::new(Self {
            _lib: lib,
            execute_kernel: execute_kernel_static,
        }))
    }

    /// Safely invokes the dynamically loaded `mojo_execute_kernel` symbol.
    ///
    /// # Errors
    /// Returns [`FfiBridgeError::KernelFailed`] if the Mojo kernel returns a non-zero status code
    /// with an error message, or [`FfiBridgeError::Unknown`] if no message is provided.
    #[inline]
    pub fn execute(
        &self,
        input: &MojoTensorBuffer,
        output: &mut MojoTensorBuffer,
    ) -> Result<(), FfiBridgeError> {
        // SAFETY: We pass valid references converted via raw reference operators that satisfy Mojo C-ABI layout.
        let kernel_result = unsafe { (self.execute_kernel)(&raw const *input, &raw mut *output) };
        parse_ffi_result(kernel_result)
    }
}

// SAFETY: `MojoLibrary` owns the loaded library mapping and immutable symbol pointer.
// Function symbols in compiled Mojo shared objects are re-entrant and thread-safe.
unsafe impl Send for MojoLibrary {}
unsafe impl Sync for MojoLibrary {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::memory::raw_buffer::RawMemoryBuffer;
    use std::path::PathBuf;
    use std::thread;

    fn get_mojo_lib_path() -> PathBuf {
        let manifest_dir = env!("CARGO_MANIFEST_DIR");
        PathBuf::from(manifest_dir).join("target/libmojo_kernel.so")
    }

    #[test]
    fn test_dlopen_valid_library() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let input_buf = RawMemoryBuffer::allocate(1024, 64).expect("Alloc failed");
        let output_buf = RawMemoryBuffer::allocate(1024, 64).expect("Alloc failed");

        let in_tensor = MojoTensorBuffer {
            data_ptr: input_buf.as_mut_ptr().cast(),
            num_elements: 256,
            element_size_bytes: 4,
            dtype: 0,
        };
        let mut out_tensor = MojoTensorBuffer {
            data_ptr: output_buf.as_mut_ptr().cast(),
            num_elements: 256,
            element_size_bytes: 4,
            dtype: 0,
        };

        // SAFETY: Valid buffer layout and loaded library.
        let kernel_status =
            unsafe { (mojo_lib.execute_kernel)(&raw const in_tensor, &raw mut out_tensor) };
        assert_eq!(kernel_status.status_code, 0);
    }

    #[test]
    fn test_dlopen_execute_method() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let input_buf = RawMemoryBuffer::allocate(512, 64).expect("Alloc failed");
        let output_buf = RawMemoryBuffer::allocate(512, 64).expect("Alloc failed");

        let in_tensor = MojoTensorBuffer {
            data_ptr: input_buf.as_mut_ptr().cast(),
            num_elements: 128,
            element_size_bytes: 4,
            dtype: 0,
        };
        let mut out_tensor = MojoTensorBuffer {
            data_ptr: output_buf.as_mut_ptr().cast(),
            num_elements: 128,
            element_size_bytes: 4,
            dtype: 0,
        };

        let execution_result = mojo_lib.execute(&in_tensor, &mut out_tensor);
        assert!(execution_result.is_ok());
    }

    #[test]
    fn test_dlopen_nonexistent_path_fails() {
        let load_result = MojoLibrary::load("target/nonexistent_lib_invalid.so");
        assert!(load_result.is_err());
        assert!(matches!(
            load_result.unwrap_err(),
            MojoLibraryError::Load { .. }
        ));
    }

    #[test]
    fn test_dlopen_missing_symbol_fails() {
        let libc_candidates = [
            "libc.so.6",
            "/usr/lib/libc.so.6",
            "/lib/x86_64-linux-gnu/libc.so.6",
        ];
        for candidate in libc_candidates {
            if Path::new(candidate).exists() || candidate == "libc.so.6" {
                // SAFETY: Testing loading system libc.
                if let Ok(lib) = unsafe { Library::new(candidate) } {
                    // SAFETY: Testing missing symbol lookup.
                    let missing_sym: Result<Symbol<MojoKernelFn>, _> =
                        unsafe { lib.get(b"mojo_execute_kernel\0") };
                    assert!(missing_sym.is_err());
                    break;
                }
            }
        }
    }

    #[test]
    fn test_dlopen_concurrent_execution() {
        let lib_path = get_mojo_lib_path();
        if !lib_path.exists() {
            return;
        }

        let mojo_lib = MojoLibrary::load(&lib_path).expect("Failed to load Mojo library");
        let mut thread_handles = Vec::new();

        for _ in 0..4 {
            let shared_lib = Arc::clone(&mojo_lib);
            thread_handles.push(thread::spawn(move || {
                let input_buf = RawMemoryBuffer::allocate(256, 64).expect("Alloc failed");
                let output_buf = RawMemoryBuffer::allocate(256, 64).expect("Alloc failed");

                let in_tensor = MojoTensorBuffer {
                    data_ptr: input_buf.as_mut_ptr().cast(),
                    num_elements: 64,
                    element_size_bytes: 4,
                    dtype: 0,
                };
                let mut out_tensor = MojoTensorBuffer {
                    data_ptr: output_buf.as_mut_ptr().cast(),
                    num_elements: 64,
                    element_size_bytes: 4,
                    dtype: 0,
                };

                let execution_result = shared_lib.execute(&in_tensor, &mut out_tensor);
                assert!(execution_result.is_ok());
            }));
        }

        for handle in thread_handles {
            handle.join().expect("Thread join failed");
        }
    }
}
