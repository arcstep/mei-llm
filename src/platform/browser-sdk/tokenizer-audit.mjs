/** Execute the tokenizer-only audit ABI in Node or a real browser. */
export function auditTokenizer(instance, vocabBytes, text) {
  const exp = instance.exports;
  const body = new TextEncoder().encode(JSON.stringify({ texts: [text] }));
  const source = new Uint8Array(body.length + 1);
  source.set(body);
  const vocabPtr = exp.mei_sdk_wasm_alloc(vocabBytes.length);
  const textPtr = exp.mei_sdk_wasm_alloc(source.length || 1);
  const outPtr = exp.mei_sdk_wasm_alloc(4);
  try {
    new Uint8Array(exp.memory.buffer, vocabPtr, vocabBytes.length).set(vocabBytes);
    new Uint8Array(exp.memory.buffer, textPtr, source.length).set(source);
    new DataView(exp.memory.buffer).setUint32(outPtr, 0, true);
    const code = exp.mei_sdk_wasm_tokenizer_audit(vocabPtr, vocabBytes.length, textPtr, outPtr);
    if (code !== 0) throw new Error(`tokenizer audit failed with code ${code}`);
    const resultPtr = new DataView(exp.memory.buffer).getUint32(outPtr, true);
    let end = resultPtr;
    const memory = new Uint8Array(exp.memory.buffer);
    while (memory[end] !== 0) end += 1;
    const result = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(memory.subarray(resultPtr, end)));
    exp.mei_sdk_wasm_string_free(resultPtr);
    return result.cases[0];
  } finally {
    exp.mei_sdk_wasm_free(vocabPtr, vocabBytes.length);
    exp.mei_sdk_wasm_free(textPtr, source.length || 1);
    exp.mei_sdk_wasm_free(outPtr, 4);
  }
}
