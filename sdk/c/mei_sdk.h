#ifndef MEI_SDK_H
#define MEI_SDK_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * MEI Runtime C ABI (mei-runtime-abi-1)
 *
 * Product: mei-1.0-58m Runtime / MEI Runtime
 * Experimental. Do not treat this as a product Runtime release.
 * This header does not implement Needle / libneedle / .cact compatibility.
 *
 * Handles are opaque. JSON is UTF-8. Strings returned via char** must be
 * released with mei_sdk_string_free. Error codes are defined in spec/errors.json.
 */

typedef struct MeiSdkEngine MeiSdkEngine;
typedef struct MeiSdkSession MeiSdkSession;

int32_t mei_sdk_abi_version(void);
int32_t mei_sdk_version(char *buf, size_t n);
int32_t mei_sdk_last_error(char *buf, size_t n);

int32_t mei_sdk_engine_open(const char *package_dir, MeiSdkEngine **out);
int32_t mei_sdk_engine_close(MeiSdkEngine *engine);

int32_t mei_sdk_session_open(MeiSdkEngine *engine, const char *options_json, MeiSdkSession **out);
int32_t mei_sdk_session_complete(MeiSdkSession *session, const char *request_json, char **out_json);
int32_t mei_sdk_session_cancel(MeiSdkSession *session);
int32_t mei_sdk_session_close(MeiSdkSession *session);

int32_t mei_sdk_string_free(char *ptr);

#ifdef __cplusplus
}
#endif

#endif /* MEI_SDK_H */
