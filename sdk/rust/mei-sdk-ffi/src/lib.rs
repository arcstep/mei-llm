//! C ABI for MEI Runtime. Handles are opaque. Errors are int32 codes from spec/errors.json.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;

use mei_sdk_core::{sdk_versions, Engine, SdkError, Session};
use serde_json::Value;

pub type MeiSdkEngine = Engine;
pub type MeiSdkSession = SessionHandle;

pub struct SessionHandle {
    session: Session,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = msg.into());
}

fn write_cstr(src: &str, buf: *mut c_char, n: usize) -> c_int {
    if buf.is_null() || n == 0 {
        set_error("null buffer");
        return 1;
    }
    let bytes = src.as_bytes();
    if bytes.len() + 1 > n {
        set_error("buffer too small");
        return 15;
    }
    unsafe {
        ptr::copy_nonoverlapping(bytes.as_ptr(), buf as *mut u8, bytes.len());
        *buf.add(bytes.len()) = 0;
    }
    0
}

fn cstr<'a>(ptr: *const c_char) -> Result<&'a str, SdkError> {
    if ptr.is_null() {
        return Err(SdkError::new("invalid_argument", "null string"));
    }
    unsafe { CStr::from_ptr(ptr) }
        .to_str()
        .map_err(|_| SdkError::new("invalid_json", "string is not utf-8"))
}

fn boxed_ok<T>(value: T, out: *mut *mut T) -> c_int {
    if out.is_null() {
        set_error("null out pointer");
        return 1;
    }
    unsafe {
        *out = Box::into_raw(Box::new(value));
    }
    0
}

fn fail(err: SdkError) -> c_int {
    set_error(&err.info().message);
    err.code()
}

#[no_mangle]
pub extern "C" fn mei_sdk_version(buf: *mut c_char, n: usize) -> c_int {
    match serde_json::to_string(&sdk_versions()) {
        Ok(text) => write_cstr(&text, buf, n),
        Err(err) => {
            set_error(err.to_string());
            2
        }
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_abi_version() -> c_int {
    1
}

#[no_mangle]
pub extern "C" fn mei_sdk_last_error(buf: *mut c_char, n: usize) -> c_int {
    let msg = LAST_ERROR.with(|slot| slot.borrow().clone());
    write_cstr(&msg, buf, n)
}

#[no_mangle]
pub extern "C" fn mei_sdk_engine_open(package_dir: *const c_char, out: *mut *mut MeiSdkEngine) -> c_int {
    let path = match cstr(package_dir) {
        Ok(s) => s,
        Err(err) => return fail(err),
    };
    match Engine::load(Path::new(path), true) {
        Ok(engine) => boxed_ok(engine, out),
        Err(err) => fail(err),
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_engine_close(engine: *mut MeiSdkEngine) -> c_int {
    if engine.is_null() {
        return 0;
    }
    unsafe {
        drop(Box::from_raw(engine));
    }
    0
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_open(
    engine: *mut MeiSdkEngine,
    _options_json: *const c_char,
    out: *mut *mut MeiSdkSession,
) -> c_int {
    if engine.is_null() {
        set_error("null engine");
        return 1;
    }
    let engine = unsafe { &*engine };
    match engine.create_session() {
        Ok(session) => boxed_ok(SessionHandle { session }, out),
        Err(err) => fail(err),
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_complete(
    session: *mut MeiSdkSession,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    if session.is_null() || out_json.is_null() {
        set_error("null handle");
        return 1;
    }
    let request_str = match cstr(request_json) {
        Ok(s) => s,
        Err(err) => return fail(err),
    };
    let request: Value = match serde_json::from_str(request_str) {
        Ok(v) => v,
        Err(err) => return fail(SdkError::new("invalid_json", err.to_string())),
    };
    let handle = unsafe { &*session };
    let result = match handle.session.complete(&request) {
        Ok(v) => v,
        Err(err) => return fail(err),
    };
    let text = match serde_json::to_string(&result) {
        Ok(t) => t,
        Err(err) => return fail(SdkError::new("invalid_json", err.to_string())),
    };
    match CString::new(text) {
        Ok(c) => {
            unsafe {
                *out_json = c.into_raw();
            }
            0
        }
        Err(_) => fail(SdkError::new("invalid_json", "result contains interior NUL")),
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_cancel(session: *mut MeiSdkSession) -> c_int {
    if session.is_null() {
        return 1;
    }
    unsafe {
        (*session).session.cancel();
    }
    0
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_close(session: *mut MeiSdkSession) -> c_int {
    if session.is_null() {
        return 0;
    }
    unsafe {
        drop(Box::from_raw(session));
    }
    0
}

#[no_mangle]
pub extern "C" fn mei_sdk_string_free(ptr: *mut c_char) -> c_int {
    if ptr.is_null() {
        return 0;
    }
    unsafe {
        drop(CString::from_raw(ptr));
    }
    0
}
