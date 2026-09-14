use mei_sdk_core::vocab::Vocab;
use serde_json::Value;

#[test]
fn candidate_matches_python_and_round_trips() {
    let Ok(model_path) = std::env::var("MEI_LOSSLESS_TOKENIZER_MODEL") else {
        return;
    };
    let golden_path = std::env::var("MEI_LOSSLESS_TOKENIZER_GOLDEN").expect("golden path");
    let vocab = Vocab::from_package_payload(&std::fs::read(model_path).expect("model"))
        .expect("portable candidate");
    let golden: Value =
        serde_json::from_slice(&std::fs::read(golden_path).expect("golden")).expect("golden json");
    for case in golden["cases"].as_array().expect("cases") {
        let text = case["text"].as_str().expect("text");
        let expected: Vec<u32> = case["ids"]
            .as_array()
            .expect("ids")
            .iter()
            .map(|value| value.as_u64().expect("id") as u32)
            .collect();
        let actual = vocab.encode(text, false);
        assert_eq!(actual, expected, "token IDs differ for {text:?}");
        assert_eq!(
            vocab.decode(&actual),
            text,
            "round trip differs for {text:?}"
        );
    }
}
