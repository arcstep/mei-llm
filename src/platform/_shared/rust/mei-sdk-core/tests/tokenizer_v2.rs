use mei_sdk_core::vocab::Vocab;
use serde_json::Value;
use std::path::PathBuf;

fn sdk_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn package_dir() -> PathBuf {
    std::env::var("MEI_51M_PACKAGE_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| {
            sdk_root().join(
                "../../.local/artifacts/mei-1.0-51m/exp-000300m/packages/mei-1.0-51m-base-scratch300m-qat-q4-v1",
            )
        })
}

#[test]
fn zh_24k_unigram_matches_sentencepiece_golden() {
    let dir = package_dir();
    if !dir.join("mei-model.json").is_file() {
        return;
    }
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(dir.join("mei-model.json")).expect("package manifest"),
    )
    .expect("package manifest json");
    let vocab_name = manifest
        .pointer("/tokenizer/file")
        .and_then(Value::as_str)
        .unwrap_or("tokenizer.model");
    let vocab = Vocab::from_package_payload(
        &std::fs::read(dir.join(vocab_name)).expect("portable tokenizer payload"),
    )
    .expect("valid portable tokenizer payload");
    let golden: Value = serde_json::from_slice(
        &std::fs::read(sdk_root().join("spec/golden/tokenizer_v2.json")).expect("tokenizer golden"),
    )
    .expect("tokenizer golden json");
    assert_eq!(
        manifest.pointer("/tokenizer/id").and_then(Value::as_str),
        golden.get("tokenizer_id").and_then(Value::as_str)
    );
    assert_eq!(
        manifest
            .pointer("/tokenizer/sha256")
            .and_then(Value::as_str),
        golden.get("model_sha256").and_then(Value::as_str)
    );
    for case in golden["cases"].as_array().expect("golden cases") {
        let text = case["text"].as_str().expect("text");
        let expected_ids: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("token id") as u32)
            .collect();
        let got = vocab.encode(text, false);
        assert_eq!(got, expected_ids, "token IDs differ for {text:?}");
        assert_eq!(
            vocab.decode(&got),
            case["decoded"].as_str().expect("decoded"),
            "decode differs for {text:?}"
        );
    }
}
