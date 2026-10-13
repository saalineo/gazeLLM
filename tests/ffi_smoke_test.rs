use gazellm::ffi::types::MojoTensorBuffer;
use gazellm::memory::raw_buffer::RawMemoryBuffer;

#[test]
fn test_rust_to_mojo_ffi_smoke() {
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

    // SAFETY: Buffers are allocated with correct size and alignment.
    let kernel_result = unsafe {
        gazellm::ffi::bridge::mojo_execute_kernel(&raw const in_tensor, &raw mut out_tensor)
    };
    assert_eq!(kernel_result.status_code, 0);
}

#[test]
fn test_rust_to_mojo_safe_bridge_smoke() {
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

    let bridge_result = gazellm::ffi::bridge::safe_mojo_execute(&in_tensor, &mut out_tensor);
    assert!(bridge_result.is_ok());
}
