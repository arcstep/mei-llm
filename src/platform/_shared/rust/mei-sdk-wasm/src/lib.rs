//! WASM tier-1 for MEI Runtime: quantized 51M package only.
//! Float / npz weights are refused. Protocol helpers remain available.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::{CStr, CString};
use std::os::raw::c_char;
use std::ptr;

use mei_sdk_core::{
    model::runtime_contract_evidence, parse_v2_text, sdk_versions, vocab::Vocab, Engine, Session,
};
use serde_json::{json, Value};

thread_local! {
    static ENGINE: RefCell<Option<Engine>> = const { RefCell::new(None) };
    static SESSIONS: RefCell<HashMap<u32, Session>> = RefCell::new(HashMap::new());
    static NEXT_SESSION_ID: Cell<u32> = const { Cell::new(1) };
    static LAST_ERROR: RefCell<Value> = RefCell::new(json!({"code":0,"id":"ok","message":"ok"}));
}

fn remember_error(error: &mei_sdk_core::error::SdkError) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = error.info().to_value());
}

fn clear_error() {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = json!({"code":0,"id":"ok","message":"ok"}));
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
    unsafe { *out = ptr::null_mut() };
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
    let mut buf = vec![0u8; n].into_boxed_slice();
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
        drop(Box::from_raw(std::ptr::slice_from_raw_parts_mut(p, n)));
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_loaded() -> i32 {
    ENGINE.with(|slot| i32::from(slot.borrow().is_some()))
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

/// Diagnostic detail for the last coded runtime failure on this WASM thread.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_last_error(out_json: *mut *mut c_char) -> i32 {
    LAST_ERROR.with(|slot| out_string(slot.borrow().to_string(), out_json))
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_parse(text: *const c_char, out_json: *mut *mut c_char) -> i32 {
    let raw = match cstr(text) {
        Ok(s) => s,
        Err(code) => return code,
    };
    out_string(parse_v2_text(raw).to_value().to_string(), out_json)
}

/// Audit-only browser entry for the canonical portable tokenizer.  It uses
/// the same Rust `Vocab` implementation as inference while avoiding the need
/// to construct a model package merely to compare tokenizer parity.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_tokenizer_audit(
    vocab_ptr: *const u8,
    vocab_len: usize,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    if vocab_ptr.is_null() || vocab_len == 0 {
        return 1;
    }
    let raw = match cstr(request_json) {
        Ok(value) => value,
        Err(code) => return code,
    };
    let request: Value = match serde_json::from_str(raw) {
        Ok(value) => value,
        Err(_) => return 2,
    };
    let Some(cases) = request.get("texts").and_then(Value::as_array) else {
        return 2;
    };
    let bytes = unsafe { std::slice::from_raw_parts(vocab_ptr, vocab_len) };
    let vocab = match Vocab::from_package_payload(bytes) {
        Ok(value) => value,
        Err(error) => {
            remember_error(&error);
            return error.code();
        }
    };
    let mut results = Vec::with_capacity(cases.len());
    for value in cases {
        let Some(text) = value.as_str() else {
            return 2;
        };
        let ids = vocab.encode(text, false);
        results.push(json!({"ids":ids,"decoded":vocab.decode(&ids)}));
    }
    clear_error();
    out_string(json!({"cases":results}).to_string(), out_json)
}

/// Load a CQ2 v2 or read-only legacy-Q4 package from memory.
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
            clear_error();
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            SESSIONS.with(|slot| slot.borrow_mut().clear());
            0
        }
        Err(err) => {
            remember_error(&err);
            err.code()
        }
    }
}

/// Native v2 in-memory loader. The frozen tool index is a required package
/// payload because retrieval must execute R1 against its hash-bound f16
/// embeddings instead of rebuilding or substituting a lexical index.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_load_package_v2(
    manifest_json: *const c_char,
    weights_ptr: *const u8,
    weights_len: usize,
    vocab_ptr: *const u8,
    vocab_len: usize,
    tool_index_ptr: *const u8,
    tool_index_len: usize,
) -> i32 {
    let raw = match cstr(manifest_json) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let manifest: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    if weights_ptr.is_null() || vocab_ptr.is_null() || tool_index_ptr.is_null() {
        return 1;
    }
    let weights = unsafe { std::slice::from_raw_parts(weights_ptr, weights_len) }.to_vec();
    let vocab = unsafe { std::slice::from_raw_parts(vocab_ptr, vocab_len) }.to_vec();
    let tool_index = unsafe { std::slice::from_raw_parts(tool_index_ptr, tool_index_len) }.to_vec();
    match Engine::from_quantized_package_bytes(manifest, weights, vocab, Some(tool_index)) {
        Ok(engine) => {
            clear_error();
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            SESSIONS.with(|slot| slot.borrow_mut().clear());
            0
        }
        Err(err) => {
            remember_error(&err);
            err.code()
        }
    }
}

/// Native v2 loader that takes ownership of buffers allocated by
/// `mei_sdk_wasm_alloc`. This avoids holding two 18 MiB tensor containers at
/// once while loading a 51M package. Ownership is consumed on every call once
/// all three pointers are non-null, including error returns.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_load_package_v2_owned(
    manifest_json: *const c_char,
    weights_ptr: *mut u8,
    weights_len: usize,
    vocab_ptr: *mut u8,
    vocab_len: usize,
    tool_index_ptr: *mut u8,
    tool_index_len: usize,
) -> i32 {
    if weights_ptr.is_null() || vocab_ptr.is_null() || tool_index_ptr.is_null() {
        return 1;
    }
    let weights = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(weights_ptr, weights_len)).into_vec()
    };
    let vocab = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(vocab_ptr, vocab_len)).into_vec()
    };
    let tool_index = unsafe {
        Box::from_raw(std::ptr::slice_from_raw_parts_mut(
            tool_index_ptr,
            tool_index_len,
        ))
        .into_vec()
    };
    let raw = match cstr(manifest_json) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let manifest: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    match Engine::from_quantized_package_bytes(manifest, weights, vocab, Some(tool_index)) {
        Ok(engine) => {
            clear_error();
            ENGINE.with(|slot| *slot.borrow_mut() = Some(engine));
            SESSIONS.with(|slot| slot.borrow_mut().clear());
            0
        }
        Err(err) => {
            remember_error(&err);
            err.code()
        }
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_capabilities(out_json: *mut *mut c_char) -> i32 {
    ENGINE.with(|slot| {
        let engine = slot.borrow();
        let Some(engine) = engine.as_ref() else {
            return 6;
        };
        out_string(engine.capabilities().to_string(), out_json)
    })
}

/// Executed Browser-WASM deployment contract.  This is intentionally exposed
/// by the WASM module itself so product gates do not need a separate native
/// Rust CLI merely to attest the browser implementation dependency.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_runtime_contract(out_json: *mut *mut c_char) -> i32 {
    out_string(runtime_contract_evidence().to_string(), out_json)
}

/// Package-specific numerical diagnostic used by portable parity gates.  It
/// executes the same Rust model and sidecars as browser inference, but is not
/// part of the product Session API.
#[no_mangle]
pub extern "C" fn mei_sdk_wasm_diagnose_heads(
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let raw = match cstr(request_json) {
        Ok(raw) => raw,
        Err(code) => return code,
    };
    let request: Value = match serde_json::from_str(raw) {
        Ok(request) => request,
        Err(_) => return 2,
    };
    let Some(text) = request.get("text").and_then(Value::as_str) else {
        return 1;
    };
    let result = ENGINE.with(|slot| {
        let engine = slot.borrow();
        let Some(engine) = engine.as_ref() else {
            return Err(6);
        };
        engine.diagnose_heads(text).map_err(|error| error.code())
    });
    match result {
        Ok(result) => out_string(result.to_string(), out_json),
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_session_open(
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let options = if options_json.is_null() {
        json!({})
    } else {
        let raw = match cstr(options_json) {
            Ok(raw) => raw,
            Err(code) => return code,
        };
        match serde_json::from_str(raw) {
            Ok(options) => options,
            Err(_) => return 2,
        }
    };
    let session = ENGINE.with(|slot| {
        let engine = slot.borrow();
        let Some(engine) = engine.as_ref() else {
            return Err(6);
        };
        engine
            .create_session_with_options(&options)
            .map_err(|error| error.code())
    });
    let session = match session {
        Ok(session) => session,
        Err(code) => return code,
    };
    let session_id = NEXT_SESSION_ID.with(|next| {
        let mut candidate = next.get().max(1);
        SESSIONS.with(|sessions| {
            let sessions = sessions.borrow();
            while sessions.contains_key(&candidate) {
                candidate = candidate.wrapping_add(1).max(1);
            }
        });
        next.set(candidate.wrapping_add(1).max(1));
        candidate
    });
    SESSIONS.with(|slot| slot.borrow_mut().insert(session_id, session));
    out_string(json!({"session_id":session_id}).to_string(), out_json)
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_complete(
    session_id: u32,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let raw = match cstr(request_json) {
        Ok(s) => s,
        Err(code) => return code,
    };
    let request: Value = match serde_json::from_str(raw) {
        Ok(v) => v,
        Err(_) => return 2,
    };
    let result = SESSIONS.with(|slot| {
        let mut sessions = slot.borrow_mut();
        let Some(session) = sessions.get_mut(&session_id) else {
            return Err(1);
        };
        session.complete(&request).map_err(|error| error.code())
    });
    match result {
        Ok(result) => match serde_json::to_string(&result) {
            Ok(text) => out_string(text, out_json),
            Err(_) => 2,
        },
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_register_tools(
    tools_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let raw = match cstr(tools_json) {
        Ok(raw) => raw,
        Err(code) => return code,
    };
    let tools: Vec<Value> = match serde_json::from_str(raw) {
        Ok(tools) => tools,
        Err(_) => return 2,
    };
    let result = ENGINE.with(|slot| {
        let mut engine = slot.borrow_mut();
        let Some(engine) = engine.as_mut() else {
            return Err(6);
        };
        let report = engine
            .register_tools(&tools)
            .map_err(|error| error.code())?;
        Ok(report)
    });
    match result {
        Ok(report) => out_string(report.to_string(), out_json),
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_submit_tool_result(
    session_id: u32,
    result_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let raw = match cstr(result_json) {
        Ok(raw) => raw,
        Err(code) => return code,
    };
    let result: Value = match serde_json::from_str(raw) {
        Ok(result) => result,
        Err(_) => return 2,
    };
    let ack = SESSIONS.with(|slot| {
        let mut sessions = slot.borrow_mut();
        let Some(session) = sessions.get_mut(&session_id) else {
            return Err(1);
        };
        session
            .submit_tool_result(&result)
            .map_err(|error| error.code())
    });
    match ack {
        Ok(ack) => out_string(ack.to_string(), out_json),
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_narrate(
    session_id: u32,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> i32 {
    let raw = match cstr(options_json) {
        Ok(raw) => raw,
        Err(code) => return code,
    };
    let options: Value = match serde_json::from_str(raw) {
        Ok(options) => options,
        Err(_) => return 2,
    };
    let result = SESSIONS.with(|slot| {
        let sessions = slot.borrow();
        let Some(session) = sessions.get(&session_id) else {
            return Err(1);
        };
        session.narrate(&options).map_err(|error| error.code())
    });
    match result {
        Ok(result) => out_string(result.to_string(), out_json),
        Err(code) => code,
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_cancel(session_id: u32) -> i32 {
    SESSIONS.with(|slot| {
        if let Some(session) = slot.borrow_mut().get_mut(&session_id) {
            session.cancel();
            0
        } else {
            1
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_wasm_session_close(session_id: u32) -> i32 {
    SESSIONS.with(|slot| {
        if slot.borrow_mut().remove(&session_id).is_some() {
            0
        } else {
            1
        }
    })
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
    SESSIONS.with(|slot| slot.borrow_mut().clear());
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
