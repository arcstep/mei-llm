# needle-zh runtime contract (phase 1)

- **Runtime**: Apple Silicon MLX. Checkpoint is a local safetensors/npz bundle, not official `.cact`.
- **Window**: `max_seq_len=2048`, `kv_window=512`. Tool catalogue is a **tool sink**: all 16 VRM tools are in the prompt; no top-k retrieval in phase 1.
- **Special IDs**: PAD=0 EOS=1 BOS=2 UNK=3. Chat/tool markers and act bits are tokenizer v1 user-defined symbols.
- **Phase-1 output**: only `[]` or a schema-legal JSON array of one tool call. Missing slots, scene conflict, illegal shop/dish pair, and offtopic gold `[]`. Never ask the user to fill a slot.
- **Grammar**: `model/grammar.py` (Python FSM). Does not call `libneedle`.
- **Confidence head**: trained in phase-1 SFT; calibrate execute threshold separately. Official Needle 2 LoRA must not be copied here (that LoRA skips the head).
- **Act bits**: decode first token may be `<act_execute>` / `<act_refuse>` (optional wrapper). Phase-2 bits exist in the vocab but are illegal in phase-1 grammar.
- **Oracle**: cactus-needle JAX `architecture.py` is read-only numerical/structure reference (Apache-2.0). See `model/NOTICE`.
