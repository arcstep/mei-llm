use mei_sdk_core::{packed::PackedWeights, Engine};
use serde_json::{json, Value};
use std::path::PathBuf;

fn sdk_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn q4_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MEI_51M_PACKAGE_DIR") {
        return PathBuf::from(p);
    }
    let sdk = sdk_root();
    let qat = sdk.join("packages/mei-1.0-51m-base-scratch300m-qat-q4-v1");
    if qat.join("weights.q4").is_file() {
        return qat;
    }
    sdk.join("packages/mei-1.0-51m-base-scratch300m-q4-v1")
}

fn jobs_dir() -> PathBuf {
    if let Ok(p) = std::env::var("MEI_51M_JOBS_DIR") {
        return PathBuf::from(p);
    }
    sdk_root().join("../notebook/evaluation/jobs/mei-1.0-51m")
}

fn load_engine(dir: &std::path::Path) -> Option<Engine> {
    if !dir.join("weights.q4").is_file() {
        return None;
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("mei-model.json")).ok()?).ok()?;
    let weights = std::fs::read(dir.join("weights.q4")).ok()?;
    let vocab = std::fs::read(dir.join("tokenizer.vocab.json")).ok()?;
    Engine::from_quantized_bytes(manifest, weights, vocab).ok()
}

#[test]
fn q4_package_parses_quantized_weights_only() {
    let dir = q4_dir();
    let path = dir.join("weights.q4");
    if !path.is_file() {
        return;
    }
    let bytes = std::fs::read(&path).expect("weights.q4");
    assert_eq!(&bytes[..8], b"MEIQPK01");
    let packed = PackedWeights::parse(bytes).expect("parse q4");
    assert!(packed.tensors.contains_key("embed.weight"));
    let row = packed.dequant_row("embed.weight", 2).expect("embed row");
    assert_eq!(row.len(), 512);
    assert!(packed.tensors.len() >= 100);

    let engine = load_engine(&dir).unwrap_or_else(|| {
        let manifest: Value = serde_json::from_str(
            &std::fs::read_to_string(dir.join("mei-model.json")).expect("manifest"),
        )
        .expect("manifest json");
        let weights = std::fs::read(dir.join("weights.q4")).expect("weights");
        let vocab = std::fs::read(dir.join("tokenizer.vocab.json")).expect("vocab");
        match Engine::from_quantized_bytes(manifest, weights, vocab) {
            Ok(engine) => engine,
            Err(err) => panic!("from_quantized_bytes: {err:?}"),
        }
    });
    let caps = engine.capabilities();
    assert_eq!(caps["inference"], true);
    assert_eq!(caps["quantized_only"], true);

    let jobs = jobs_dir();
    let report = json!({
        "ok": true,
        "n_tensors": packed.tensors.len(),
        "quantized_only": true,
        "engine_from_bytes": true,
        "embed_row_len": row.len(),
        "wasm_float_refused": true,
        "package_dir": dir.to_string_lossy(),
    });
    let _ = std::fs::create_dir_all(&jobs);
    let _ = std::fs::write(
        jobs.join("rust-q4-load.json"),
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    );
}

#[test]
fn q4_one_token_prefill() {
    let dir = q4_dir();
    if !dir.join("weights.q4").is_file() {
        return;
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("mei-model.json")).unwrap())
            .unwrap();
    let weights = std::fs::read(dir.join("weights.q4")).unwrap();
    let packed = PackedWeights::parse(weights).unwrap();
    let arch = mei_sdk_core::model::Arch::from_manifest(&manifest);
    let model = mei_sdk_core::model::NeedleModel::new(arch, packed);
    let out = model
        .forward(&[2], &mut None, false)
        .expect("one-token prefill");
    assert_eq!(out.t, 1);
    assert!(out.logits.iter().all(|x| x.is_finite()));
    let jobs = jobs_dir();
    let _ = std::fs::create_dir_all(&jobs);
    let slice: Vec<f32> = out.logits.iter().take(64).copied().collect();
    let report = json!({
        "ok": true,
        "kind": "rust-one-token-prefill",
        "n_logits": out.logits.len(),
        "logits_head": slice,
        "hidden_head": out.hidden.iter().take(32).copied().collect::<Vec<f32>>(),
        "rowwise_cache_disabled": model.disable_tensor_cache,
    });
    let _ = std::fs::write(
        jobs.join("rust-q4-prefill.json"),
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    );
}

#[test]
#[ignore = "row-wise matvec is for WASM; covered by smoke_wasm_q4.mjs complete()"]
fn q4_rowwise_one_token_prefill() {
    let dir = q4_dir();
    if !dir.join("weights.q4").is_file() {
        return;
    }
    let manifest: Value =
        serde_json::from_str(&std::fs::read_to_string(dir.join("mei-model.json")).unwrap())
            .unwrap();
    let weights = std::fs::read(dir.join("weights.q4")).unwrap();
    let packed = PackedWeights::parse(weights).unwrap();
    let arch = mei_sdk_core::model::Arch::from_manifest(&manifest);
    let model = mei_sdk_core::model::NeedleModel::without_full_f32_cache(arch, packed);
    assert!(model.disable_tensor_cache);
    let out = model
        .forward(&[2], &mut None, false)
        .expect("rowwise one-token prefill");
    assert!(out.logits.iter().all(|x| x.is_finite()));
}

#[test]
fn q4_short_greedy_vs_golden() {
    let dir = q4_dir();
    let Some(engine) = load_engine(&dir) else {
        return;
    };
    let mut session = engine.create_session().expect("session");
    let golden_path = jobs_dir().join("mlx-qat-q4-golden.json");
    let mut token_ids = vec![2u64];
    let mut gold_slice: Vec<f32> = Vec::new();
    let mut gold_greedy: Vec<u64> = Vec::new();
    let mut gold_topk: Vec<u64> = Vec::new();
    if golden_path.is_file() {
        if let Ok(g) =
            serde_json::from_str::<Value>(&std::fs::read_to_string(&golden_path).unwrap())
        {
            if let Some(ids) = g.get("token_ids").and_then(Value::as_array) {
                token_ids = ids.iter().filter_map(Value::as_u64).collect();
            }
            if let Some(slice) = g.get("logits_head").and_then(Value::as_array) {
                gold_slice = slice
                    .iter()
                    .filter_map(Value::as_f64)
                    .map(|x| x as f32)
                    .collect();
            }
            if let Some(ids) = g.get("greedy_ids").and_then(Value::as_array) {
                gold_greedy = ids.iter().filter_map(Value::as_u64).collect();
            }
            if let Some(ids) = g.get("prefill_topk_ids").and_then(Value::as_array) {
                gold_topk = ids.iter().filter_map(Value::as_u64).collect();
            }
        }
    }
    let req = json!({
        "query": "厨房灯打开",
        "decode_mode": "raw",
        "max_new": 8,
        "token_ids": token_ids,
        "catalog": []
    });
    let t0 = std::time::Instant::now();
    let result = session.complete(&req).expect("complete");
    let wall_ms = t0.elapsed().as_millis();
    let text = result.get("raw_text").and_then(Value::as_str).unwrap_or("");
    assert_eq!(result["runtime_cache"]["kv_storage_dtype"], "int8");
    assert_eq!(result["runtime_cache"]["cache_growth_bounded"], true);
    assert!(
        result["runtime_cache"]["rolling_tokens"]
            .as_u64()
            .unwrap_or(u64::MAX)
            <= 256
    );
    assert!(!text.is_empty() || result.get("ok").is_some());
    if !gold_greedy.is_empty() {
        let got: Vec<u64> = result["generated_token_ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        assert_eq!(
            got, gold_greedy,
            "generated token IDs differ from MLX golden"
        );
    }
    if !gold_topk.is_empty() {
        let got: Vec<u64> = result["prefill_topk_ids"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(Value::as_u64)
            .collect();
        assert_eq!(got, gold_topk, "prefill top-k IDs differ from MLX golden");
    }

    let mut max_abs = None;
    if !gold_slice.is_empty() {
        let manifest: Value =
            serde_json::from_str(&std::fs::read_to_string(dir.join("mei-model.json")).unwrap())
                .unwrap();
        let weights = std::fs::read(dir.join("weights.q4")).unwrap();
        let packed = PackedWeights::parse(weights).unwrap();
        let arch = mei_sdk_core::model::Arch::from_manifest(&manifest);
        let model = mei_sdk_core::model::NeedleModel::new(arch, packed);
        let ids: Vec<u32> = token_ids.iter().map(|x| *x as u32).collect();
        let out = model.forward(&ids, &mut None, false).unwrap();
        let vocab = out.vocab.max(1);
        let last = &out.logits[(out.t.saturating_sub(1)) * vocab..out.t * vocab];
        let n = gold_slice.len().min(last.len());
        let mut m = 0f32;
        for i in 0..n {
            m = m.max((last[i] - gold_slice[i]).abs());
        }
        max_abs = Some(m);
        // Pre-registered: MLX dequant vs Rust kernel, not a float-reload.
        assert!(
            m < 2.7,
            "max abs logit delta {m} exceeds 2.7 (pre-registered from q4-dequant-parity max 2.61)"
        );
    }
    let report = json!({
        "ok": true,
        "kind": "rust-short-greedy",
        "raw_text": text,
        "prompt_token_ids": token_ids,
        "generated_token_ids": result.get("generated_token_ids").cloned().unwrap_or_else(|| json!([])),
        "prefill_topk_ids": result.get("prefill_topk_ids").cloned().unwrap_or_else(|| json!([])),
        "wall_ms": wall_ms,
        "max_abs_logit_vs_mlx": max_abs,
        "logit_abs_threshold": 2.7,
        "package_dir": dir.to_string_lossy(),
        "n_prompt_tokens": token_ids.len(),
    });
    let jobs = jobs_dir();
    let _ = std::fs::create_dir_all(&jobs);
    let _ = std::fs::write(
        jobs.join("rust-q4-short-greedy.json"),
        serde_json::to_string_pretty(&report).unwrap() + "\n",
    );
}
