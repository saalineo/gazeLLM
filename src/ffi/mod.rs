//! Foreign Function Interface (FFI) bindings for external compute runtime (Mojo C-ABI).

pub mod bridge;
pub mod dlopen;
pub mod types;

pub use bridge::{FfiBridgeError, safe_mojo_execute};
pub use dlopen::{MojoKernelFn, MojoLibrary, MojoLibraryError};
pub use libloading;
pub use types::{DType, FfiResult, InvalidDTypeError, MojoBlockTable, MojoTensorBuffer, dtype};

/// Scaffolding function for initializing FFI compute bindings.
pub fn init_ffi() {
    tracing::info!("Initializing FFI compute bindings");
}
