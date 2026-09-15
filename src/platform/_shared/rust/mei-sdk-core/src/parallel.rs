//! Opt-in browser worker cooperation. Each worker owns a private WASM heap;
//! JavaScript coordinates disjoint matrix output rows through shared buffers.
//! This does not require sharing Rust's allocator or standard-library state.
use crate::error::SdkError;
#[cfg(target_arch = "wasm32")]
#[link(wasm_import_module = "mei_parallel")]
extern "C" {
    fn enabled() -> i32;
    fn matmul(
        name: *const u8,
        name_len: usize,
        input: *const f32,
        tokens: usize,
        rows: usize,
        cols: usize,
        output: *mut f32,
    ) -> i32;
}
pub fn active() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        unsafe { enabled() != 0 }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        false
    }
}
#[cfg(target_arch = "wasm32")]
pub fn linear(
    name: &str,
    x: &[f32],
    rows: usize,
    cols: usize,
) -> Option<Result<Vec<f32>, SdkError>> {
    if !active()
        || cols != 512
        || rows < 256
        || rows > 24000
        || rows % 4 != 0
        || x.len() % cols != 0
    {
        return None;
    }
    let tokens = x.len() / cols;
    if tokens == 0 || tokens > 512 || tokens * rows > 512 * 512 {
        return None;
    }
    let mut out = vec![0.0; tokens * rows];
    #[cfg(target_arch = "wasm32")]
    if unsafe {
        matmul(
            name.as_ptr(),
            name.len(),
            x.as_ptr(),
            tokens,
            rows,
            cols,
            out.as_mut_ptr(),
        )
    } != 0
    {
        return Some(Err(SdkError::new(
            "engine_unavailable",
            "parallel matrix worker failed",
        )));
    }
    Some(Ok(out))
}

#[cfg(not(target_arch = "wasm32"))]
pub fn linear(
    _name: &str,
    _x: &[f32],
    _rows: usize,
    _cols: usize,
) -> Option<Result<Vec<f32>, SdkError>> {
    None
}
