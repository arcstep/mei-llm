#include "mei_sdk.h"

/* Compile-only ABI surface check. Runtime semantics are covered by mei-sdk-ffi tests. */
int main(void) {
    MeiSdkEngine *engine = 0;
    MeiSdkSession *session = 0;
    char *json = 0;
    (void)mei_sdk_abi_version;
    (void)mei_sdk_version;
    (void)mei_sdk_engine_open;
    (void)mei_sdk_engine_register_tools;
    (void)mei_sdk_session_open;
    (void)mei_sdk_session_complete;
    (void)mei_sdk_session_submit_tool_result;
    (void)mei_sdk_session_narrate;
    (void)mei_sdk_session_cancel;
    (void)mei_sdk_session_close;
    (void)mei_sdk_engine_close;
    (void)mei_sdk_string_free;
    (void)engine;
    (void)session;
    (void)json;
    return 0;
}
