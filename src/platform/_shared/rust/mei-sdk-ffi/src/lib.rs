//! C ABI for MEI Runtime. Handles are opaque. Errors are int32 codes from spec/errors.json.

use std::cell::RefCell;
use std::ffi::{CStr, CString};
use std::os::raw::{c_char, c_int};
use std::path::Path;
use std::ptr;
use std::sync::Mutex;

use mei_sdk_core::{sdk_versions, Engine, SdkError, Session};
use serde_json::Value;

pub struct MeiSdkEngine {
    inner: Mutex<Engine>,
}

pub struct MeiSdkSession {
    inner: Mutex<Session>,
}

thread_local! {
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

fn set_error(msg: impl Into<String>) {
    LAST_ERROR.with(|slot| *slot.borrow_mut() = msg.into());
}

fn catch_code(f: impl FnOnce() -> c_int) -> c_int {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(code) => code,
        Err(_) => {
            set_error("panic contained at the C ABI boundary");
            6
        }
    }
}

fn lock_failed() -> c_int {
    fail(SdkError::new(
        "engine_unavailable",
        "handle lock is poisoned",
    ))
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
        *out = ptr::null_mut();
        *out = Box::into_raw(Box::new(value));
    }
    0
}

fn fail(err: SdkError) -> c_int {
    set_error(&err.info().message);
    err.code()
}

fn write_json(value: &Value, out_json: *mut *mut c_char) -> c_int {
    if out_json.is_null() {
        set_error("null output pointer");
        return 1;
    }
    unsafe { *out_json = ptr::null_mut() };
    let text = match serde_json::to_string(value) {
        Ok(text) => text,
        Err(error) => return fail(SdkError::new("invalid_json", error.to_string())),
    };
    match CString::new(text) {
        Ok(text) => {
            unsafe { *out_json = text.into_raw() };
            0
        }
        Err(_) => fail(SdkError::new(
            "invalid_json",
            "result contains interior NUL",
        )),
    }
}

#[no_mangle]
pub extern "C" fn mei_sdk_version(buf: *mut c_char, n: usize) -> c_int {
    catch_code(|| match serde_json::to_string(&sdk_versions()) {
        Ok(text) => write_cstr(&text, buf, n),
        Err(err) => {
            set_error(err.to_string());
            2
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_abi_version() -> c_int {
    2
}

#[no_mangle]
pub extern "C" fn mei_sdk_last_error(buf: *mut c_char, n: usize) -> c_int {
    catch_code(|| {
        let msg = LAST_ERROR.with(|slot| slot.borrow().clone());
        write_cstr(&msg, buf, n)
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_engine_open(
    package_dir: *const c_char,
    out: *mut *mut MeiSdkEngine,
) -> c_int {
    catch_code(|| {
        if !out.is_null() {
            unsafe { *out = ptr::null_mut() };
        }
        let path = match cstr(package_dir) {
            Ok(s) => s,
            Err(err) => return fail(err),
        };
        match Engine::load(Path::new(path), true) {
            Ok(engine) => boxed_ok(
                MeiSdkEngine {
                    inner: Mutex::new(engine),
                },
                out,
            ),
            Err(err) => fail(err),
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_engine_close(engine: *mut MeiSdkEngine) -> c_int {
    catch_code(|| {
        if engine.is_null() {
            return 0;
        }
        unsafe {
            drop(Box::from_raw(engine));
        }
        0
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_engine_register_tools(
    engine: *mut MeiSdkEngine,
    tools_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    catch_code(|| {
        if !out_json.is_null() {
            unsafe { *out_json = ptr::null_mut() };
        }
        if engine.is_null() {
            return fail(SdkError::new("invalid_argument", "null engine"));
        }
        let raw = match cstr(tools_json) {
            Ok(raw) => raw,
            Err(error) => return fail(error),
        };
        let tools: Vec<Value> = match serde_json::from_str(raw) {
            Ok(tools) => tools,
            Err(error) => return fail(SdkError::new("invalid_json", error.to_string())),
        };
        let mut engine = match unsafe { &*engine }.inner.lock() {
            Ok(engine) => engine,
            Err(_) => return lock_failed(),
        };
        match engine.register_tools(&tools) {
            Ok(result) => write_json(&result, out_json),
            Err(error) => fail(error),
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_open(
    engine: *mut MeiSdkEngine,
    options_json: *const c_char,
    out: *mut *mut MeiSdkSession,
) -> c_int {
    catch_code(|| {
        if !out.is_null() {
            unsafe { *out = ptr::null_mut() };
        }
        if engine.is_null() {
            set_error("null engine");
            return 1;
        }
        let options = if options_json.is_null() {
            Value::Object(Default::default())
        } else {
            match cstr(options_json).and_then(|raw| {
                serde_json::from_str(raw)
                    .map_err(|error| SdkError::new("invalid_json", error.to_string()))
            }) {
                Ok(options) => options,
                Err(error) => return fail(error),
            }
        };
        let engine = match unsafe { &*engine }.inner.lock() {
            Ok(engine) => engine,
            Err(_) => return lock_failed(),
        };
        match engine.create_session_with_options(&options) {
            Ok(session) => boxed_ok(
                MeiSdkSession {
                    inner: Mutex::new(session),
                },
                out,
            ),
            Err(err) => fail(err),
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_complete(
    session: *mut MeiSdkSession,
    request_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    catch_code(|| {
        if !out_json.is_null() {
            unsafe { *out_json = ptr::null_mut() };
        }
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
        let mut session = match unsafe { &*session }.inner.lock() {
            Ok(session) => session,
            Err(_) => return lock_failed(),
        };
        let result = match session.complete(&request) {
            Ok(v) => v,
            Err(err) => return fail(err),
        };
        write_json(&result, out_json)
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_submit_tool_result(
    session: *mut MeiSdkSession,
    result_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    catch_code(|| {
        if !out_json.is_null() {
            unsafe { *out_json = ptr::null_mut() };
        }
        if session.is_null() {
            return fail(SdkError::new("invalid_argument", "null session"));
        }
        let raw = match cstr(result_json) {
            Ok(raw) => raw,
            Err(error) => return fail(error),
        };
        let result: Value = match serde_json::from_str(raw) {
            Ok(result) => result,
            Err(error) => return fail(SdkError::new("invalid_json", error.to_string())),
        };
        let mut session = match unsafe { &*session }.inner.lock() {
            Ok(session) => session,
            Err(_) => return lock_failed(),
        };
        match session.submit_tool_result(&result) {
            Ok(ack) => write_json(&ack, out_json),
            Err(error) => fail(error),
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_narrate(
    session: *mut MeiSdkSession,
    options_json: *const c_char,
    out_json: *mut *mut c_char,
) -> c_int {
    catch_code(|| {
        if !out_json.is_null() {
            unsafe { *out_json = ptr::null_mut() };
        }
        if session.is_null() || out_json.is_null() {
            return fail(SdkError::new("invalid_argument", "null session/output"));
        }
        let raw = match cstr(options_json) {
            Ok(raw) => raw,
            Err(error) => return fail(error),
        };
        let options: Value = match serde_json::from_str(raw) {
            Ok(options) => options,
            Err(error) => return fail(SdkError::new("invalid_json", error.to_string())),
        };
        let session = match unsafe { &*session }.inner.lock() {
            Ok(session) => session,
            Err(_) => return lock_failed(),
        };
        match session.narrate(&options) {
            Ok(result) => write_json(&result, out_json),
            Err(error) => fail(error),
        }
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_cancel(session: *mut MeiSdkSession) -> c_int {
    catch_code(|| {
        if session.is_null() {
            return 1;
        }
        let mut session = match unsafe { &*session }.inner.lock() {
            Ok(session) => session,
            Err(_) => return lock_failed(),
        };
        session.cancel();
        0
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_session_close(session: *mut MeiSdkSession) -> c_int {
    catch_code(|| {
        if session.is_null() {
            return 0;
        }
        unsafe {
            drop(Box::from_raw(session));
        }
        0
    })
}

#[no_mangle]
pub extern "C" fn mei_sdk_string_free(ptr: *mut c_char) -> c_int {
    catch_code(|| {
        if ptr.is_null() {
            return 0;
        }
        unsafe {
            drop(CString::from_raw(ptr));
        }
        0
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn abi_and_version_report_v2() {
        assert_eq!(mei_sdk_abi_version(), 2);
        let mut buffer = vec![0i8; 1024];
        assert_eq!(mei_sdk_version(buffer.as_mut_ptr(), buffer.len()), 0);
        let text = unsafe { CStr::from_ptr(buffer.as_ptr()) }.to_str().unwrap();
        assert!(text.contains("mei-runtime-wire-v2"));
        assert!(text.contains("mei-runtime-abi-2"));
    }
}
