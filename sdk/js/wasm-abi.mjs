/**
 * Quantized WASM C ABI loader. No Node builtins — safe in a browser Worker.
 */
export const wasmTier = 1;

const WASM_TIER1_VERSIONS = {
  sdk_semver: "0.1.0-experimental",
  wire_version: "mei-runtime-wire-v1",
  model_package_version: "mei-model-package-v1",
  runtime_abi_version: "mei-runtime-abi-1",
  protocol_id: "mei-tool-call-protocol-v2",
  serializer_id: "mei-tool-call-serializer-v2",
  release_class: "experimental",
  product: "mei-1.0-58m Runtime",
};

function fail(code, message) {
  const err = new Error(message);
  err.code = code;
  throw err;
}

function wasmExports(wasm) {
  if (wasm && wasm.exports && wasm.exports.memory) return wasm.exports;
  return wasm;
}

function encodeCString(text) {
  const body = new TextEncoder().encode(String(text));
  const out = new Uint8Array(body.length + 1);
  out.set(body, 0);
  return out;
}

function allocBytes(exp, bytes) {
  if (typeof exp.mei_sdk_wasm_alloc !== "function") {
    fail("engine_unavailable", "WASM module missing mei_sdk_wasm_alloc");
  }
  const ptr = exp.mei_sdk_wasm_alloc(bytes.length);
  if (!ptr) {
    fail("engine_unavailable", "WASM alloc returned null");
  }
  new Uint8Array(exp.memory.buffer, ptr, bytes.length).set(bytes);
  return { ptr, len: bytes.length };
}

function freeBytes(exp, slot) {
  if (slot && slot.ptr && typeof exp.mei_sdk_wasm_free === "function") {
    exp.mei_sdk_wasm_free(slot.ptr, slot.len);
  }
}

function readCString(exp, ptr) {
  if (!ptr) return "";
  const mem = new Uint8Array(exp.memory.buffer);
  let end = ptr;
  while (end < mem.length && mem[end] !== 0) end += 1;
  return new TextDecoder().decode(mem.subarray(ptr, end));
}

function loadQuantizedCAbi(exp, manifestText, weightBytes, vocabBytes) {
  const man = allocBytes(exp, encodeCString(manifestText));
  const weights = allocBytes(exp, weightBytes);
  const vocab = allocBytes(exp, vocabBytes);
  try {
    return exp.mei_sdk_wasm_load_quantized(man.ptr, weights.ptr, weights.len, vocab.ptr, vocab.len);
  } finally {
    freeBytes(exp, man);
    freeBytes(exp, weights);
    freeBytes(exp, vocab);
  }
}

function completeCAbi(exp, request) {
  const req = allocBytes(exp, encodeCString(JSON.stringify(request)));
  const outSlot = allocBytes(exp, new Uint8Array(4));
  try {
    const rc = exp.mei_sdk_wasm_complete(req.ptr, outSlot.ptr);
    if (rc !== 0) {
      fail("engine_unavailable", `wasm complete failed code=${rc}`);
    }
    const strPtr = new DataView(exp.memory.buffer).getUint32(outSlot.ptr, true);
    const text = readCString(exp, strPtr);
    if (typeof exp.mei_sdk_wasm_string_free === "function" && strPtr) {
      exp.mei_sdk_wasm_string_free(strPtr);
    }
    return JSON.parse(text);
  } finally {
    freeBytes(exp, req);
    freeBytes(exp, outSlot);
  }
}

/**
 * Load a quantized model into the WASM engine.
 * `weights` must be mei-q4-packed-v1 bytes. Float npz is refused.
 */
export function loadModel(options = {}) {
  const { manifest, weights, vocab, wasm } = options;
  if (!manifest || !weights) {
    fail(
      "engine_unavailable",
      "browser loadModel requires a quantized package (manifest + weights.q4); float checkpoints are not loaded in WASM",
    );
  }
  const fmt = manifest.weights?.format;
  if (fmt !== "mei-q4-packed-v1") {
    fail("engine_unavailable", "WASM tier-1 loads only mei-q4-packed-v1 quantized weights");
  }
  if (!wasm) {
    fail(
      "engine_unavailable",
      "WASM module not provided. Instantiate mei-sdk-wasm and pass { wasm } to loadModel",
    );
  }
  const exp = wasmExports(wasm);
  const manifestText = typeof manifest === "string" ? manifest : JSON.stringify(manifest);
  const weightBytes = weights instanceof Uint8Array ? weights : new Uint8Array(weights);
  const vocabBytes = vocab ? (vocab instanceof Uint8Array ? vocab : new Uint8Array(vocab)) : new Uint8Array();

  let rc;
  if (typeof exp.mei_sdk_wasm_alloc === "function" && typeof exp.mei_sdk_wasm_load_quantized === "function") {
    rc = loadQuantizedCAbi(exp, manifestText, weightBytes, vocabBytes);
  } else if (typeof exp.mei_sdk_wasm_load_quantized === "function") {
    rc = exp.mei_sdk_wasm_load_quantized(manifestText, weightBytes, vocabBytes);
  } else {
    fail(
      "engine_unavailable",
      "WASM module not provided. Instantiate mei-sdk-wasm and pass { wasm } to loadModel",
    );
  }
  if (rc !== 0) {
    fail("package_invalid", `wasm load failed code=${rc}`);
  }
  return {
    wasmTier: 1,
    quantizedOnly: true,
    createSession() {
      return {
        complete(request) {
          if (typeof exp.mei_sdk_wasm_alloc === "function") {
            return completeCAbi(exp, request);
          }
          return JSON.parse(exp.mei_sdk_wasm_complete(JSON.stringify(request)));
        },
        cancel() {
          if (typeof exp.mei_sdk_wasm_unload === "function") {
            /* session cancel is per-request; engine unload is explicit */
          }
        },
      };
    },
    capabilities() {
      return {
        inference: true,
        protocol: true,
        quantized_only: true,
        wasm_tier: 1,
        versions: { ...WASM_TIER1_VERSIONS },
      };
    },
    unload() {
      if (typeof exp.mei_sdk_wasm_unload === "function") {
        exp.mei_sdk_wasm_unload();
      }
    },
  };
}
