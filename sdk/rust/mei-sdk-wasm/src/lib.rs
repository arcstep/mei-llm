//! WASM tier-1 for MEI Runtime: quantized 51M package only.
//! Float / npz weights are refused. Protocol helpers remain available.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use mei_sdk_core::{complete_request, parse_v2_text, sdk_versions, Engine};
use serde_json::{json, Value};

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
}

fn cstr<'a>(ptr: *const c_char) -> Result<&'a str, i32> {
    if ptr.is_null() {
        return Err(1);
    }
    unsafe { CStr::from_ptr(ptr) }.to_str().map_err(|_| 2)
}

fn out_string(text: String, out: *mut *mut c_char) -> i32 {
    if out.is_null() {
        return 1;
    }
    match CString::new(text) {
        Ok(c) => {
            unsafe {
                *out = c.into_raw();
            }
            0
        }
        Err(_) => 2,
    }
}

/// Linear-memory bump for JS glue. Pair with `mei_sdk_wasm_free`.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_alloc(n: usize) -> *mut u8 {
    if n == 0 {
        return ptr::null_mut();
    }
    let mut buf = vec![0u8; n];
    let p = buf.as_mut_ptr();
    std::mem::forget(buf);
    p
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_free(p: *mut u8, n: usize) {
    if p.is_null() || n == 0 {
        return;
    }
    unsafe {
        drop(Vec::from_raw_parts(p, n, n));
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_loaded() -> i32 {
    ENGINE.with(|slot| i32::from(slot.borrow().is_some()))
}

fn dummy_caps() -> Value {
    json!({
        "package_id": "wasm-tier1",
        "release_class": "experimental",
        "inference": false,
        "protocol": true,
        "quantized_only": true,
        "wasm_tier": 1,
        "versions": sdk_versions()
    })
}

/// Tier-1 = quantized in-browser inference. Never 0 once this crate is built.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_tier() -> i32 {
    1
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_version(out_json: *mut *mut c_char) -> i32 {
    match serde_json::to_string(&sdk_versions()) {
        Ok(text) => out_string(text, out_json),
        Err(_) => 2,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_parse(text: *const c_char, out_json: *mut *mut c_char) -> i32 {
    let raw = match cstr(text) {
        Ok(s) => s,
        Err(code) => return code,
    };
    out_string(parse_v2_text(raw).to_value().to_string(), out_json)
}

/// Load a quantized package from memory. `weights` must be mei-q4-packed-v1.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_load_quantized(
    manifest_json: *const c_char,
    weights_ptr: *const u8,
    weights_len: usize,
    vocab_ptr: *const u8,
    vocab_len: usize,
) -> i32 {
    let raw = match cstr(manifest_json) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let manifest: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    if weights_ptr.is_null() || vocab_ptr.is_null() {
        return 1;
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, weights_len) }.to_vec();
    let vocab = unsafe { std::slice::from_raw_parts(vocab_ptr, vocab_len) }.to_vec();
    match Engine::from_quantized_bytes(manifest, weights, vocab) {
        Ok(engine) => {
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            0
        }
        Err(err) => err.code(),
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_complete(request_json: *const c_char, out_json: *mut *mut c_char) -> i32 {
    let raw = match cstr(request_json) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    let result = ENGINE.with(|slot| {
        if let Some(engine) = slot.borrow().as_ref() {
            match engine.create_session() {
                Ok(session) => session.complete(&request).unwrap_or_else(|err| {
                    json!({"ok": false, "error": err.info().to_value(), "refuse": true})
                }),
                Err(err) => json!({"ok": false, "error": err.info().to_value(), "refuse": true}),
            }
        } else {
            complete_request(&request, &dummy_caps(), false)
        }
    });
    match serde_json::to_string(&result) {
        Ok(text) => out_string(text, out_json),
        Err(_) => 2,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_string_free(ptr: *mut c_char) -> i32 {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
    0
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_unload() -> i32 {
    ENGINE.with(|slot| *slot.borrow_mut() = None);
    0
}

#[no_mangle]
pub unsafe extern "C" fn mei_sdk_wasm_copy(dst: *mut u8, src: *const u8, n: usize) -> i32 {
    if dst.is_null() || src.is_null() {
        return 1;
    }
    ptr::copy_nonoverlapping(src, dst, n);
    0
}
