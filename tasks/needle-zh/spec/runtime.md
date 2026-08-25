# needle-zh / mei-1.0-58m runtime contract (phase 1, Route-ID)

- **Runtime**: Apple Silicon MLX. Checkpoint is a local safetensors/npz bundle, not official `.cact`.
- **Canonical family**: `mei-1.0-58m`. `needle-zh` is a historical alias for the same physical task tree. `MW` means Micro-World governance, not the model name.
- **Window**: `max_seq_len=2048`. Product SFT/decode uses the full sequence; `kv_window=512` is a documented budget, **not** an implemented sliding window that may drop the leading `<tools>` / `<routes>` block. Do not slide tools out of attention.
- **Schema sink**: every request **must** pass an explicit toolset/schema. There is **no default VRM**. The model may only select a `route_id` compiled from that request, or output `[]`.
- **Request shape** (train = student = Qwen): locked task contract + `<tools>{canonical compact schema}</tools>` + optional `<scene>` / `<entities>` + `<routes>{finite compiled hypotheses}</routes>` + natural user query. Serializer `mei-route-serializer-v1`. Do not add extra system text on one side only.
- **Subset**: internal `[]` or `{"route_id":N}`; external JSON array of at most one call; `arguments` is an object; `required` / `enum` / `string|boolean|integer|number`. No arrays, nested objects, multi-call, generic JSON Schema, or LM-free-text values.
- **Grounding**: values come only from query/scene/entity/normalizer/const/default. Enum membership is not evidence. Missing/ambiguous/conflict/overflow fail closed to `[]`.
- **Overflow**: if the full tools+routes block does not fit in `seq_len`, **reject the sample**. Never left-trim the prompt to keep the answer. If compiled routes exceed `max_routes`, refuse the whole request rather than truncate.
- **Special IDs**: PAD=0 EOS=1 BOS=2 UNK=3. `<tools>` / `</tools>` are atomic tokenizer symbols.
- **Phase-1 output**: model emits only `[]` or a legal request-local `route_id`. Runtime renders schema-legal JSON. Missing slots, scene conflict, illegal shop/dish pair, offtopic, unknown route, and provenance failure gold `[]`. Never ask the user to fill a slot.
- **Grammar**: `model/route_protocol.py` prefix trie over `[]` and legal route IDs. `model/grammar.py` remains the parse/validate façade (legacy JSON-array parser kept for v1 diagnostics).
- **Confidence head**: trained in phase-1 SFT; calibrate execute threshold separately.
- **Act bits**: decode first token may be `<act_execute>` / `<act_refuse>` (optional wrapper). Phase-2 bits exist in the vocab but are illegal in phase-1 grammar.
- **Oracle**: cactus-needle JAX `architecture.py` is read-only numerical/structure reference (Apache-2.0). See `model/NOTICE`.
