use mei_sdk_core::cq2::{quantize, scales_to_le_bytes, GROUP_SIZE};
use mei_sdk_core::package::{load_package, package_from_manifest};
use mei_sdk_core::packed::{PackedWeights, CQ2_MAGIC, CQ2_PACK_VERSION};
use mei_sdk_core::{complete_request, Engine};
use serde_json::{json, Value};

fn sha(ch: char) -> String {
    std::iter::repeat(ch).take(64).collect()
}

fn manifest(directory: Value) -> Value {
    json!({
        "package_format": "mei-model-package-v2",
        "product": "mei-1.0-51m",
        "package_id": "tiny-v2",
        "runtime_min": "mei-runtime-abi-2",
        "contracts": {
            "weight_contract_sha256": "c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b",
            "runtime_profile_sha256": "74839b08155e624f14318ca8646166ddc68ee6496720dedac26aa91fdc8bdf43",
            "training_aux_sha256": "83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac"
        },
        "architecture": {
            "id":"mei-1.0-51m-arch-v1", "d_model":512, "n_layers":27,
            "n_heads":8, "n_kv_heads":4, "head_dim":64,
            "vocab_size":24000, "max_seq_len":2048, "parameter_count":51463797,
            "rope_theta":100000.0, "engram_layers":[2,15], "engram_orders":[2,3],
            "engram_slots":8192, "engram_conv_taps":4, "mhc_lanes":4,
            "sinkhorn_iters":20, "tie_embeddings":true, "rms_eps":0.000001,
            "conf_probes":8, "mlp":"FixedWalshHadamardMLP", "confidence_head":true
        },
        "runtime_profile": {"max_context_tokens":2048,"stable_prefix_tokens":1024,"rolling_window_tokens":256,"default_output_tokens":128,"kv_dtype":"i8","activation_dtype":"i8"},
        "tokenizer": {"id":"zh-24k-v1","file":"tokenizer.model","sha256":sha('e'),"pad_id":0,"eos_id":1,"bos_id":2,"unk_id":3},
        "tensor_container": {
            "file": "tensors.bin", "format": "mei-cq-tensor-v2", "sha256": sha('d'),
            "payload_bytes": 256, "quant_math_id": "mei-cq-v2-g128-wht-codebook",
            "directory": directory
        },
        "files": [
            {"path":"tokenizer.model","sha256":sha('e'),"nbytes":1,"role":"tokenizer"},
            {"path":"tensors.bin","sha256":sha('d'),"nbytes":256,"role":"tensor_container"}
        ],
        "heads": {
            "lm": {"present":true,"trained":false,"status":"untrained","tensor_prefixes":["lm"]},
            "contrastive": {"present":true,"trained":false,"status":"untrained","tensor_prefixes":["heads.contrastive"]},
            "mw_disposition": {"present":true,"trained":false,"status":"untrained","tensor_prefixes":["heads.mw"]},
            "confidence": {"present":true,"trained":false,"status":"untrained","tensor_prefixes":["heads.confidence"]}
        },
        "capabilities": {"retrieval":false,"full_call":false,"mw_disposition":false,"confidence":false,"multi_step":false},
        "training_receipts": [],
        "resources": {"package_bytes":256,"rust_session_peak_bytes":1,"wasm_heap_peak_bytes":1},
        "release_class": "experimental"
    })
}

fn runtime_quantization() -> Value {
    json!({
        "weight_math_id": "mei-cq-v2-g128-wht-codebook",
        "activation_semantics_id": "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2",
        "activation_sites": ["engram_input", "attention_input", "attention_output", "lm_head_input"],
        "activation_q_quantized": false,
        "activation_reconstruction_dtype": "f32",
        "kv_semantics_id": "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1",
        "kv_sites": ["attention_key_after_rope", "attention_value"],
        "kv_storage_dtype": "int8",
        "kv_scale_granularity": "per-head-vector",
        "int8_code_min": -128,
        "int8_code_max": 127,
        "scale_denominator": 127
    })
}

fn four_entries() -> Value {
    json!([
        {"name":"lm.w","role":"lm","shape":[],"n_params":1,"dtype":"f16","data":{"offset":0,"nbytes":2},"transform":"none","codebook":"none"},
        {"name":"heads.contrastive.w","role":"contrastive","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":2,"nbytes":2},"transform":"none","codebook":"none"},
        {"name":"heads.mw.w","role":"mw_disposition","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":4,"nbytes":2},"transform":"none","codebook":"none"},
        {"name":"heads.confidence.w","role":"confidence","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":6,"nbytes":2},"transform":"none","codebook":"none"}
    ])
}

fn safe_container_fixture() -> (Vec<u8>, Value) {
    let mut header = json!({
        "quant_math_id":"mei-cq-v2-g128-wht-codebook",
        "tensors":[
            {"name":"lm.w","role":"lm","shape":[],"n_params":1,"dtype":"f16","data":{"offset":0,"nbytes":2},"transform":"none","codebook":"none"},
            {"name":"heads.contrastive.w","role":"contrastive","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":0,"nbytes":2},"transform":"none","codebook":"none"},
            {"name":"heads.mw.w","role":"mw_disposition","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":0,"nbytes":2},"transform":"none","codebook":"none"},
            {"name":"heads.confidence.w","role":"confidence","shape":[1],"n_params":1,"dtype":"f16","data":{"offset":0,"nbytes":2},"transform":"none","codebook":"none"}
        ]
    });
    for _ in 0..8 {
        let base = 16 + serde_json::to_vec(&header).unwrap().len();
        for (index, entry) in header["tensors"]
            .as_array_mut()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            entry["data"]["offset"] = json!(base + index * 2);
        }
    }
    let header_bytes = serde_json::to_vec(&header).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(CQ2_MAGIC);
    bytes.extend_from_slice(&CQ2_PACK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&[0; 8]);
    (bytes, header["tensors"].clone())
}

fn write_manifest_with_measured_size(root: &std::path::Path, manifest: &mut Value) {
    for _ in 0..16 {
        let bytes = serde_json::to_vec(manifest).unwrap();
        let total = bytes.len() as u64
            + manifest["files"]
                .as_array()
                .unwrap()
                .iter()
                .map(|entry| entry["nbytes"].as_u64().unwrap())
                .sum::<u64>();
        if manifest["resources"]["package_bytes"].as_u64() == Some(total) {
            std::fs::write(root.join("mei-model.json"), bytes).unwrap();
            return;
        }
        manifest["resources"]["package_bytes"] = json!(total);
    }
    panic!("manifest size did not stabilize");
}

#[test]
fn package_v2_rejects_duplicates_overlap_and_bad_contracts() {
    let package = package_from_manifest(manifest(four_entries()), true).unwrap();
    assert!(!package.runtime_quantization_complete());
    assert!(!package.verified_hashes);
    assert!(!package.capabilities()["resource_eligible"]
        .as_bool()
        .unwrap());
    let mut duplicate = four_entries();
    duplicate[1]["name"] = json!("lm.w");
    assert_eq!(
        package_from_manifest(manifest(duplicate), true)
            .unwrap_err()
            .info()
            .id,
        "duplicate_tensor"
    );
    let mut overlap = four_entries();
    overlap[1]["data"]["offset"] = json!(1);
    assert_eq!(
        package_from_manifest(manifest(overlap), true)
            .unwrap_err()
            .info()
            .id,
        "package_range_invalid"
    );
    let mut bad = manifest(four_entries());
    bad["contracts"]["weight_contract_sha256"] = json!("not-a-hash");
    assert_eq!(
        package_from_manifest(bad, true).unwrap_err().info().id,
        "package_invalid"
    );

    let mut false_ready = manifest(four_entries());
    false_ready["heads"]["contrastive"]["trained"] = json!(true);
    false_ready["heads"]["contrastive"]["status"] = json!("ready");
    false_ready["heads"]["contrastive"]["training_receipt_sha256"] = json!(sha('f'));
    false_ready["training_receipts"] = json!([sha('f')]);
    false_ready["files"].as_array_mut().unwrap().push(json!({
        "path":"contrastive-receipt.json", "sha256":sha('f'), "nbytes":1,
        "role":"training_receipt"
    }));
    assert_eq!(
        package_from_manifest(false_ready, true)
            .unwrap_err()
            .info()
            .id,
        "package_invalid"
    );

    let mut current = manifest(four_entries());
    current["runtime_quantization"] = runtime_quantization();
    assert!(package_from_manifest(current.clone(), true)
        .unwrap()
        .runtime_quantization_complete());
    current["runtime_quantization"]["activation_q_quantized"] = json!(true);
    assert_eq!(
        package_from_manifest(current, true).unwrap_err().info().id,
        "package_invalid"
    );
}

#[test]
fn filesystem_v2_requires_complete_hashed_inventory() {
    let unique = format!(
        "mei-sdk-files-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    std::fs::create_dir(&root).unwrap();
    let tokenizer = b"tokenizer";
    let (container, directory) = safe_container_fixture();
    std::fs::write(root.join("tokenizer.model"), tokenizer).unwrap();
    std::fs::write(root.join("tensors.bin"), &container).unwrap();
    let tokenizer_sha = mei_sdk_core::canonical::sha256_bytes(tokenizer);
    let container_sha = mei_sdk_core::canonical::sha256_bytes(&container);
    let mut package = manifest(directory);
    package["tokenizer"]["sha256"] = json!(tokenizer_sha);
    package["tensor_container"]["sha256"] = json!(container_sha);
    package["tensor_container"]["payload_bytes"] = json!(container.len());
    package["files"] = json!([
        {"path":"tokenizer.model","sha256":tokenizer_sha,"nbytes":tokenizer.len(),"role":"tokenizer"},
        {"path":"tensors.bin","sha256":container_sha,"nbytes":container.len(),"role":"tensor_container"}
    ]);
    let lm_directory = package["tensor_container"]["directory"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|entry| entry["role"] == "lm")
        .cloned()
        .collect::<Vec<_>>();
    let directory_sha = mei_sdk_core::canonical::sha256_bytes(
        mei_sdk_core::canonical::dumps_canonical(&json!(lm_directory)).as_bytes(),
    );
    let mut receipt = json!({
        "schema":"mei-training-receipt-v2", "product":"mei-1.0-51m",
        "package_id":"tiny-v2", "component":"lm", "stage_id":"fixture-lm",
        "stage_fingerprint_sha256":sha('a'), "status":"failed",
        "contracts":package["contracts"].clone(),
        "tensor_container_sha256":container_sha,
        "tensor_directory_sha256":directory_sha
    });
    let mut receipt_bytes = serde_json::to_vec(&receipt).unwrap();
    std::fs::write(root.join("lm-receipt.json"), &receipt_bytes).unwrap();
    let mut receipt_sha = mei_sdk_core::canonical::sha256_bytes(&receipt_bytes);
    package["training_receipts"] = json!([receipt_sha]);
    package["files"].as_array_mut().unwrap().push(json!({
        "path":"lm-receipt.json", "sha256":receipt_sha, "nbytes":receipt_bytes.len(),
        "role":"training_receipt"
    }));
    write_manifest_with_measured_size(&root, &mut package);
    assert_eq!(
        load_package(&root, true).unwrap_err().info().id,
        "package_invalid"
    );
    receipt["status"] = json!("passed");
    receipt_bytes = serde_json::to_vec(&receipt).unwrap();
    std::fs::write(root.join("lm-receipt.json"), &receipt_bytes).unwrap();
    receipt_sha = mei_sdk_core::canonical::sha256_bytes(&receipt_bytes);
    package["training_receipts"] = json!([receipt_sha]);
    let receipt_file = package["files"].as_array_mut().unwrap().last_mut().unwrap();
    receipt_file["sha256"] = json!(receipt_sha);
    receipt_file["nbytes"] = json!(receipt_bytes.len());
    let packed = PackedWeights::parse(container.clone()).unwrap();
    let contrastive = packed.entry("heads.contrastive.w").unwrap();
    let mut head_identity = contrastive.name.as_bytes().to_vec();
    head_identity.extend_from_slice(
        &packed.bytes
            [contrastive.packed_offset..contrastive.packed_offset + contrastive.packed_nbytes],
    );
    let head_sha = mei_sdk_core::canonical::sha256_bytes(&head_identity);
    let mut tool_index: Value = serde_json::from_slice(
        &std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../spec/golden/tool_index_v2.json"),
        )
        .unwrap(),
    )
    .unwrap();
    tool_index["model_sha256"] = package["tensor_container"]["sha256"].clone();
    tool_index["head_sha256"] = json!(head_sha);
    tool_index["tokenizer_sha256"] = package["tokenizer"]["sha256"].clone();
    let schema_map = Value::Object(
        tool_index["records"]
            .as_array()
            .unwrap()
            .iter()
            .map(|row| {
                (
                    row["tool_id"].as_str().unwrap().to_string(),
                    row["schema_sha256"].clone(),
                )
            })
            .collect(),
    );
    let combined_schema = mei_sdk_core::canonical::sha256_bytes(
        mei_sdk_core::canonical::dumps_canonical(&schema_map).as_bytes(),
    );
    let index_identity = json!({
        "catalog_sha256":tool_index["catalog_sha256"],
        "head_sha256":tool_index["head_sha256"],
        "model_sha256":tool_index["model_sha256"],
        "schema_sha256":combined_schema,
        "serializer_id":tool_index["serializer_id"],
        "tokenizer_sha256":tool_index["tokenizer_sha256"]
    });
    tool_index["fingerprint"] = json!(mei_sdk_core::canonical::sha256_bytes(
        mei_sdk_core::canonical::dumps_canonical(&index_identity).as_bytes(),
    ));
    let index_bytes = serde_json::to_vec(&tool_index).unwrap();
    std::fs::write(root.join("tool-index.json"), &index_bytes).unwrap();
    let index_sha = mei_sdk_core::canonical::sha256_bytes(&index_bytes);
    package["files"].as_array_mut().unwrap().push(json!({
        "path":"tool-index.json", "sha256":index_sha, "nbytes":index_bytes.len(), "role":"tool_index"
    }));
    write_manifest_with_measured_size(&root, &mut package);
    let loaded = load_package(&root, true).unwrap();
    assert!(loaded.verified_hashes);
    assert!(loaded.training_receipts_verified);
    assert!(loaded.tool_index.is_some());
    assert!(!loaded.tensor_identity_complete());

    std::fs::write(root.join("undeclared.txt"), b"hidden").unwrap();
    assert_eq!(
        load_package(&root, true).unwrap_err().info().id,
        "package_invalid"
    );
    std::fs::remove_file(root.join("undeclared.txt")).unwrap();

    package["resources"]["package_bytes"] = json!(1);
    std::fs::write(
        root.join("mei-model.json"),
        serde_json::to_vec(&package).unwrap(),
    )
    .unwrap();
    assert_eq!(
        load_package(&root, true).unwrap_err().info().id,
        "package_range_invalid"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn session_enforces_pending_call_and_verified_tool_result() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    engine
        .register_tools(&[json!({
            "name":"light.set", "description":"set light",
            "parameters":{"type":"object","properties":{}}
        })])
        .unwrap();
    let mut session = engine.create_session().unwrap();
    let request = json!({
        "wire_version":"mei-runtime-wire-v2", "query":"开灯",
        "candidate_text":"[{\"name\":\"light.set\",\"arguments\":{}}]"
    });
    let call = session.complete(&request).unwrap();
    assert_eq!(call["kind"], "call");
    let call_id = call["call"]["call_id"].as_str().unwrap().to_string();
    assert_eq!(
        session.complete(&request).unwrap()["error"]["id"],
        "tool_result_required"
    );
    let wrong = json!({
        "wire_version":"mei-runtime-wire-v2", "call_id":"stale", "status":"ok", "payload":{},
        "provenance":{"source":"test","verified":true}
    });
    assert_eq!(
        session.submit_tool_result(&wrong).unwrap_err().info().id,
        "stale_call_id"
    );
    let result = json!({
        "wire_version":"mei-runtime-wire-v2", "call_id":call_id, "status":"ok", "payload":{"on":true},
        "provenance":{"source":"test","verified":true}
    });
    assert_eq!(
        session.submit_tool_result(&result).unwrap()["accepted"],
        true
    );
    let next_tool = json!({
        "name":"light.set",
        "parameters":{
            "type":"object", "additionalProperties":false,
            "properties":{"on":{"type":"boolean"}}, "required":["on"]
        }
    });
    let chained = session
        .complete(&json!({
            "wire_version":"mei-runtime-wire-v2", "query":"应用刚才结果",
            "oracle_tools":[next_tool.clone()],
            "candidate_text":"[{\"name\":\"light.set\",\"arguments\":{\"on\":true}}]"
        }))
        .unwrap();
    assert_eq!(chained["kind"], "call");
    assert_eq!(
        chained["provenance"]["arguments"]["on"]["source"],
        "verified_tool_result"
    );

    let mut forged = engine.create_session().unwrap();
    let forged_turn = forged
        .complete(&json!({
            "wire_version":"mei-runtime-wire-v2", "query":"伪造结果",
            "oracle_tools":[next_tool],
            "tool_results":[result],
            "candidate_text":"[{\"name\":\"light.set\",\"arguments\":{\"on\":true}}]"
        }))
        .unwrap();
    assert_eq!(forged_turn["kind"], "refuse");
    assert_eq!(forged_turn["refusal"]["reason"], "provenance_missing");
}

#[test]
fn verified_tool_success_followed_by_empty_action_is_respond() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    let tool = json!({"name":"clock.read","parameters":{"type":"object","properties":{}}});
    engine.register_tools(&[tool.clone()]).unwrap();
    let mut session = engine.create_session().unwrap();
    let call = session
        .complete(&json!({
            "wire_version":"mei-runtime-wire-v2", "query":"几点",
            "oracle_tools":[tool.clone()],
            "candidate_text":"[{\"name\":\"clock.read\",\"arguments\":{}}]"
        }))
        .unwrap();
    session
        .submit_tool_result(&json!({
            "wire_version":"mei-runtime-wire-v2", "call_id":call["call"]["call_id"],
            "status":"ok", "payload":{"hour":12},
            "provenance":{"source":"test","verified":true}
        }))
        .unwrap();
    assert_eq!(
        session.narrate(&json!({})).unwrap_err().info().message,
        "narration_requires_respond"
    );
    let terminal = session
        .complete(&json!({
            "wire_version":"mei-runtime-wire-v2", "query":"几点",
            "oracle_tools":[tool], "candidate_text":"[]"
        }))
        .unwrap();
    assert_eq!(terminal["kind"], "respond");
    assert_eq!(terminal["refuse"], false);
    assert!(terminal["refusal"].is_null());
    let narration = session.narrate(&json!({"mode":"adapter"})).unwrap();
    assert_eq!(narration["mode"], "deterministic");
    assert_eq!(narration["fallback_used"], true);
    assert_eq!(narration["grounded"], true);
    assert_eq!(narration["text"], "clock.read已执行完成，hour为12。");
}

#[test]
fn call_ids_are_session_bound_and_reject_cross_session_results() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    let request = json!({
        "wire_version":"mei-runtime-wire-v2", "query":"开灯",
        "oracle_tools":[{"name":"light.set","parameters":{"type":"object","properties":{}}}],
        "candidate_text":"[{\"name\":\"light.set\",\"arguments\":{}}]"
    });
    let mut first = engine.create_session().unwrap();
    let mut second = engine.create_session().unwrap();
    let first_id = first.complete(&request).unwrap()["call"]["call_id"]
        .as_str()
        .unwrap()
        .to_string();
    let second_id = second.complete(&request).unwrap()["call"]["call_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(first_id, second_id);
    assert!(regex::Regex::new(r"^call-s[0-9a-f]{8}-1-[0-9a-f]{12}$")
        .unwrap()
        .is_match(&first_id));
    let crossed = json!({
        "wire_version":"mei-runtime-wire-v2", "call_id":first_id, "status":"ok", "payload":{},
        "provenance":{"source":"test","verified":true}
    });
    assert_eq!(
        second.submit_tool_result(&crossed).unwrap_err().info().id,
        "stale_call_id"
    );
}

#[test]
fn deterministic_gate_order_is_fail_closed_before_confidence() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    engine
        .register_tools(&[json!({
            "name":"light.set",
            "parameters":{
                "type":"object", "additionalProperties":false,
                "properties":{"on":{"type":"boolean"}}, "required":["on"]
            },
            "required_permissions":["device.write"],
            "required_state":{"online":true}
        })])
        .unwrap();
    let request = json!({
        "wire_version":"mei-runtime-wire-v2", "query":"开灯",
        "candidate_text":"[{\"name\":\"light.set\",\"arguments\":{\"on\":true}}]",
        "evidence":[{"tool":"light.set","argument":"on","value":true,"source":"ui","verified":true}],
        "permissions":{"scopes":["device.write"]}, "state":{"online":true},
        "mw":{"decision":"continue","source":"protocol-test"},
        "confidence":{"value":0.9,"source":"protocol-test"}, "enforce_confidence":true
    });
    let mut allowed = engine.create_session().unwrap();
    let call = allowed.complete(&request).unwrap();
    assert_eq!(call["kind"], "call");
    let gates = call["provenance"]["gates"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|gate| gate.get("gate").and_then(Value::as_str))
        .collect::<Vec<_>>();
    assert_eq!(
        gates,
        vec![
            "retrieval",
            "grammar",
            "schema",
            "provenance",
            "permission",
            "state",
            "mw",
            "confidence"
        ]
    );

    let mut denied_request = request.clone();
    denied_request["permissions"]["denied_tools"] = json!(["light.set"]);
    denied_request["confidence"] = json!({"value":1.0,"source":"protocol-test"});
    let mut denied = engine.create_session().unwrap();
    let refusal = denied.complete(&denied_request).unwrap();
    assert_eq!(refusal["kind"], "refuse");
    assert_eq!(refusal["refusal"]["reason"], "permission_denied");
    assert_eq!(
        refusal["provenance"]["gates"]
            .as_array()
            .unwrap()
            .last()
            .unwrap()["gate"],
        "permission"
    );

    let mut unbound_request = request.clone();
    unbound_request["evidence"] = json!([{
        "id":"same-value-only", "kind":"fact", "value":true, "source":"ui", "verified":true
    }]);
    let mut unbound = engine.create_session().unwrap();
    assert_eq!(
        unbound.complete(&unbound_request).unwrap()["refusal"]["reason"],
        "provenance_missing"
    );

    let mut invalid_confidence = request.clone();
    invalid_confidence["confidence"] = json!({"value":0.9,"source":"fixture"});
    let mut invalid = engine.create_session().unwrap();
    assert_eq!(
        invalid.complete(&invalid_confidence).unwrap()["refusal"]["reason"],
        "confidence_invalid"
    );
    let mut invalid_mw = request;
    invalid_mw["mw"] = json!({"decision":"continue","source":"mw-head"});
    let mut invalid = engine.create_session().unwrap();
    assert_eq!(
        invalid.complete(&invalid_mw).unwrap()["refusal"]["reason"],
        "mw_invalid"
    );
    invalid_mw["mw"] = json!({"decision":"constrain","allowed_tools":[],"source":"protocol-test"});
    let mut invalid = engine.create_session().unwrap();
    assert_eq!(
        invalid.complete(&invalid_mw).unwrap()["refusal"]["reason"],
        "mw_invalid"
    );
}

#[test]
fn catalog_identity_is_order_independent_and_pattern_subset_is_portable() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("spec/golden/catalog_fingerprint_v2.json")).unwrap(),
    )
    .unwrap();
    let tools = fixture["catalog"].as_array().unwrap().clone();
    let mut reversed = tools.clone();
    reversed.reverse();
    assert_eq!(
        mei_sdk_core::catalog_fingerprint(&tools),
        fixture["sha256"].as_str().unwrap()
    );
    assert_eq!(
        mei_sdk_core::catalog_fingerprint(&tools),
        mei_sdk_core::catalog_fingerprint(&reversed)
    );
    let unsafe_tool = json!({"name":"unsafe","parameters":{"type":"object","properties":{
        "value":{"type":"string","pattern":r"^(?=x)\w+$"}
    }}});
    assert!(mei_sdk_core::validate_tool_schema(&unsafe_tool)
        .unwrap_err()
        .contains("non-portable"));
    let unknown =
        json!({"name":"unknown","extra":true,"parameters":{"type":"object","properties":{}}});
    assert!(mei_sdk_core::validate_tool_schema(&unknown).is_err());
    let empty_integer = json!({"name":"empty-range","parameters":{"type":"object","properties":{
        "value":{"type":"integer","exclusiveMinimum":5,"maximum":5}
    }}});
    assert!(mei_sdk_core::validate_tool_schema(&empty_integer).is_err());
    let empty_string = json!({"name":"empty-string","parameters":{"type":"object","properties":{
        "value":{"type":"string","minLength":3,"maxLength":2}
    }}});
    assert!(mei_sdk_core::validate_tool_schema(&empty_string).is_err());
}

#[test]
fn portable_format_subset_matches_cross_runtime_golden() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("spec/golden/format_cases_v2.json")).unwrap(),
    )
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let schema = json!({"type":"string", "format":case["format"]});
        let actual = mei_sdk_core::validate_json_value(&case["value"], &schema).is_ok();
        assert_eq!(
            actual,
            case["valid"].as_bool().unwrap(),
            "format case {}",
            case
        );
    }
}

#[test]
fn portable_tool_index_parses_and_uses_stable_id_ties() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let bytes = std::fs::read(root.join("spec/golden/tool_index_v2.json")).unwrap();
    let index = mei_sdk_core::tool_index::ToolIndex::parse(&bytes).unwrap();
    assert_eq!(index.dimension, 2);
    assert_eq!(index.records.len(), 2);
    assert_eq!(
        index
            .topk(&[1.0, 0.0], 2)
            .unwrap()
            .iter()
            .map(|record| record.tool_id.as_str())
            .collect::<Vec<_>>(),
        vec!["a", "b"]
    );
    let mut tampered: Value = serde_json::from_slice(&bytes).unwrap();
    tampered["records"][0]["embedding_f16_base64"] = json!("AAAAAA==");
    assert!(
        mei_sdk_core::tool_index::ToolIndex::parse(&serde_json::to_vec(&tampered).unwrap())
            .is_err()
    );
}

#[test]
fn canonical_number_domain_matches_cross_runtime_golden() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let fixture: Value = serde_json::from_slice(
        &std::fs::read(root.join("spec/golden/canonical_numbers_v2.json")).unwrap(),
    )
    .unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let value = &case["input"];
        mei_sdk_core::canonical::validate_semantic_json(value).unwrap();
        assert_eq!(
            mei_sdk_core::canonical::dumps_canonical(value),
            case["canonical"].as_str().unwrap()
        );
    }
    for literal in fixture["invalid_json_literals"].as_array().unwrap() {
        let value: Value = serde_json::from_str(literal.as_str().unwrap()).unwrap();
        assert!(mei_sdk_core::canonical::validate_semantic_json(&value).is_err());
    }
}

#[test]
fn run_is_fail_closed_at_max_steps() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let mut engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    engine
        .register_tools(&[json!({
            "name":"clock.now", "parameters":{"type":"object","properties":{}}
        })])
        .unwrap();
    let mut session = engine
        .create_session_with_options(&json!({"max_steps":2}))
        .unwrap();
    let loop_result = session
        .run(
            &json!({
                "wire_version":"mei-runtime-wire-v2", "query":"几点",
                "candidate_text":"[{\"name\":\"clock.now\",\"arguments\":{}}]"
            }),
            |_name, _arguments| Ok(json!({"hour":12})),
        )
        .unwrap();
    assert_eq!(loop_result["stopped_reason"], "max_steps");
    assert_eq!(loop_result["ok"], false);
    assert_eq!(loop_result["tool_results"].as_array().unwrap().len(), 2);
}

#[test]
fn session_options_are_strict_and_not_clamped() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let engine = Engine::load(&root.join("fixtures/packages/tiny-protocol-v1"), true).unwrap();
    for options in [
        json!({"max_steps":0}),
        json!({"max_tool_result_bytes":0}),
        json!({"max_tool_result_bytes":1_048_577}),
        json!({"unknown":true}),
    ] {
        assert_eq!(
            engine
                .create_session_with_options(&options)
                .err()
                .unwrap()
                .info()
                .id,
            "invalid_argument"
        );
    }
}

#[test]
fn candidate_and_release_packages_forbid_raw_and_fixture_overrides() {
    let result = complete_request(
        &json!({"wire_version":"mei-runtime-wire-v2","query":"x","decode_mode":"raw","candidate_text":"[]"}),
        &json!({"release_class":"candidate"}),
        false,
    );
    assert_eq!(result["error"]["id"], "decode_mode_forbidden");
    assert_eq!(result["stats"]["decode_mode"], "raw");
    let result = complete_request(
        &json!({"wire_version":"mei-runtime-wire-v2","query":"x","decode_mode":"constrained","candidate_text":"[]"}),
        &json!({"release_class":"release"}),
        false,
    );
    assert_eq!(result["error"]["id"], "decode_mode_forbidden");
}

#[test]
fn cq2_container_parses_and_dequantizes() {
    let values: Vec<f32> = (0..GROUP_SIZE)
        .map(|index| (index as f32 * 0.07).sin())
        .collect();
    let tensor = quantize(&values, &[2]).unwrap();
    let scale_bytes = scales_to_le_bytes(&tensor.scales_f16);
    let mut header = json!({
        "quant_math_id":"mei-cq-v2-g128-wht-codebook",
        "tensors":[{
            "name":"lm.test","role":"lm","shape":[GROUP_SIZE],"n_params":GROUP_SIZE,"dtype":"cq2",
            "group_size":128,"transform":"wht","codebook":"gaussian-lloyd-q2-v1",
            "data":{"offset":0,"nbytes":tensor.data.len()},
            "scales":{"offset":0,"nbytes":scale_bytes.len()},
            "bit_map":{"offset":0,"nbytes":tensor.bit_map.len()}
        }]
    });
    let mut header_bytes;
    for _ in 0..4 {
        header_bytes = serde_json::to_vec(&header).unwrap();
        let base = 16 + header_bytes.len();
        header["tensors"][0]["data"]["offset"] = json!(base);
        header["tensors"][0]["scales"]["offset"] = json!(base + tensor.data.len());
        header["tensors"][0]["bit_map"]["offset"] =
            json!(base + tensor.data.len() + scale_bytes.len());
    }
    header_bytes = serde_json::to_vec(&header).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(CQ2_MAGIC);
    bytes.extend_from_slice(&CQ2_PACK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&tensor.data);
    bytes.extend_from_slice(&scale_bytes);
    bytes.extend_from_slice(&tensor.bit_map);
    let packed = PackedWeights::parse(bytes.clone()).unwrap();
    let decoded = packed.dequant("lm.test").unwrap();
    let mse = values
        .iter()
        .zip(decoded)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        / GROUP_SIZE as f32;
    assert!(mse < 0.4, "CQ2 Q2-only MSE {mse}");

    let bit_map_offset = header["tensors"][0]["bit_map"]["offset"].as_u64().unwrap() as usize;
    let mut all_q4_disguised = bytes.clone();
    all_q4_disguised[bit_map_offset] = 1;
    assert_eq!(
        PackedWeights::parse(all_q4_disguised)
            .unwrap_err()
            .info()
            .id,
        "package_invalid"
    );
    let mut unused_bit = bytes;
    unused_bit[bit_map_offset] = 0b1000_0000;
    assert_eq!(
        PackedWeights::parse(unused_bit).unwrap_err().info().id,
        "package_invalid"
    );
}

#[test]
fn cq2_row_and_matvec_decode_each_group_once_with_unaligned_rows() {
    let rows = 4usize;
    let cols = 96usize;
    let values: Vec<f32> = (0..rows * cols)
        .map(|index| ((index as f32 * 0.031).sin() * 0.7) + ((index % 13) as f32 - 6.0) * 0.02)
        .collect();
    let tensor = quantize(&values, &[2, 4, 2]).unwrap();
    let scale_bytes = scales_to_le_bytes(&tensor.scales_f16);
    let mut header = json!({
        "quant_math_id":"mei-cq-v2-g128-wht-codebook",
        "tensors":[{
            "name":"lm.matrix","role":"lm","shape":[rows, cols],"n_params":rows * cols,
            "dtype":"cq2","group_size":128,"transform":"wht",
            "codebook":"gaussian-lloyd-q2-v1",
            "data":{"offset":0,"nbytes":tensor.data.len()},
            "scales":{"offset":0,"nbytes":scale_bytes.len()},
            "bit_map":{"offset":0,"nbytes":tensor.bit_map.len()}
        }]
    });
    let mut header_bytes;
    for _ in 0..4 {
        header_bytes = serde_json::to_vec(&header).unwrap();
        let base = 16 + header_bytes.len();
        header["tensors"][0]["data"]["offset"] = json!(base);
        header["tensors"][0]["scales"]["offset"] = json!(base + tensor.data.len());
        header["tensors"][0]["bit_map"]["offset"] =
            json!(base + tensor.data.len() + scale_bytes.len());
    }
    header_bytes = serde_json::to_vec(&header).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(CQ2_MAGIC);
    bytes.extend_from_slice(&CQ2_PACK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&tensor.data);
    bytes.extend_from_slice(&scale_bytes);
    bytes.extend_from_slice(&tensor.bit_map);

    let packed = PackedWeights::parse(bytes).unwrap();
    let full = packed.dequant("lm.matrix").unwrap();
    for row in 0..rows {
        assert_eq!(
            packed.dequant_row("lm.matrix", row).unwrap(),
            full[row * cols..(row + 1) * cols]
        );
    }
    let input: Vec<f32> = (0..cols)
        .map(|index| ((index as f32 * 0.11).cos() * 0.5) - 0.1)
        .collect();
    let expected: Vec<f32> = (0..rows)
        .map(|row| {
            full[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(left, right)| left * right)
                .sum()
        })
        .collect();
    let actual = packed.matvec("lm.matrix", &input).unwrap();
    for (left, right) in actual.iter().zip(expected) {
        assert!((left - right).abs() <= 1.0e-6, "{left} != {right}");
    }
}

#[test]
fn cq2_aligned_mixed_group_matvec_uses_transformed_domain() {
    let rows = 3usize;
    let cols = GROUP_SIZE * 2;
    let values: Vec<f32> = (0..rows * cols)
        .map(|index| ((index as f32 * 0.019).sin() * 0.65) + ((index % 17) as f32 - 8.0) * 0.015)
        .collect();
    let tensor = quantize(&values, &[2, 4, 4, 2, 2, 4]).unwrap();
    let scale_bytes = scales_to_le_bytes(&tensor.scales_f16);
    let mut header = json!({
        "quant_math_id":"mei-cq-v2-g128-wht-codebook",
        "tensors":[{
            "name":"lm.aligned","role":"lm","shape":[rows, cols],"n_params":rows * cols,
            "dtype":"cq2","group_size":128,"transform":"wht",
            "codebook":"gaussian-lloyd-q2-v1",
            "data":{"offset":0,"nbytes":tensor.data.len()},
            "scales":{"offset":0,"nbytes":scale_bytes.len()},
            "bit_map":{"offset":0,"nbytes":tensor.bit_map.len()}
        }]
    });
    let mut header_bytes;
    for _ in 0..4 {
        header_bytes = serde_json::to_vec(&header).unwrap();
        let base = 16 + header_bytes.len();
        header["tensors"][0]["data"]["offset"] = json!(base);
        header["tensors"][0]["scales"]["offset"] = json!(base + tensor.data.len());
        header["tensors"][0]["bit_map"]["offset"] =
            json!(base + tensor.data.len() + scale_bytes.len());
    }
    header_bytes = serde_json::to_vec(&header).unwrap();
    let mut bytes = Vec::new();
    bytes.extend_from_slice(CQ2_MAGIC);
    bytes.extend_from_slice(&CQ2_PACK_VERSION.to_le_bytes());
    bytes.extend_from_slice(&(header_bytes.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header_bytes);
    bytes.extend_from_slice(&tensor.data);
    bytes.extend_from_slice(&scale_bytes);
    bytes.extend_from_slice(&tensor.bit_map);

    let packed = PackedWeights::parse(bytes).unwrap();
    let full = packed.dequant("lm.aligned").unwrap();
    let input: Vec<f32> = (0..cols)
        .map(|index| ((index as f32 * 0.083).cos() * 0.55) - 0.04)
        .collect();
    let expected = (0..rows)
        .map(|row| {
            full[row * cols..(row + 1) * cols]
                .iter()
                .zip(&input)
                .map(|(weight, value)| weight * value)
                .sum::<f32>()
        })
        .collect::<Vec<_>>();
    let actual = packed.matvec("lm.aligned", &input).unwrap();
    for (left, right) in actual.iter().zip(expected) {
        assert!((left - right).abs() <= 5.0e-5, "{left} != {right}");
    }
}

#[test]
fn cq2_matches_cross_language_golden() {
    let root = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let golden: Value = serde_json::from_str(
        &std::fs::read_to_string(root.join("spec/golden/cq2_v2.json")).unwrap(),
    )
    .unwrap();
    let values: Vec<f32> = (0..256)
        .map(|index| ((index % 17) as f32 - 8.0) / 8.0)
        .collect();
    let tensor = quantize(&values, &[2, 4]).unwrap();
    assert_eq!(
        tensor.data.len(),
        golden["data_nbytes"].as_u64().unwrap() as usize
    );
    assert_eq!(
        mei_sdk_core::canonical::sha256_bytes(&tensor.data),
        golden["data_sha256"].as_str().unwrap()
    );
    assert_eq!(
        hex::encode(scales_to_le_bytes(&tensor.scales_f16)),
        golden["scales_f16_le_hex"]
    );
    assert_eq!(hex::encode(&tensor.bit_map), golden["bit_map_hex"]);
}

#[test]
fn filesystem_loader_rejects_hash_mismatch_and_traversal() {
    let unique = format!(
        "mei-sdk-v2-{}-{}",
        std::process::id(),
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    );
    let root = std::env::temp_dir().join(unique);
    std::fs::create_dir(&root).unwrap();
    std::fs::write(root.join("tokenizer.model"), b"tokenizer").unwrap();
    std::fs::write(root.join("weights.bin"), b"weights").unwrap();
    let tokenizer_sha = mei_sdk_core::canonical::sha256_bytes(b"tokenizer");
    let mut manifest = json!({
        "package_format":"mei-model-package-v1", "product":"mei-1.0-51m", "package_id":"bad-v1",
        "runtime_min":"mei-runtime-abi-1",
        "tokenizer":{"id":"zh-24k-v1","file":"tokenizer.model","sha256":tokenizer_sha},
        "weights":{"file":"weights.bin","format":"raw","sha256":sha('f')},
        "heads":{
            "lm":{"present":true,"trained":false,"status":"untrained"},
            "contrastive":{"present":false,"trained":false,"status":"missing"},
            "mw_disposition":{"present":false,"trained":false,"status":"missing"},
            "confidence":{"present":false,"trained":false,"status":"missing"}
        },
        "release_class":"experimental"
    });
    std::fs::write(
        root.join("mei-model.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert_eq!(
        load_package(&root, true).unwrap_err().info().id,
        "package_hash_mismatch"
    );
    manifest["weights"]["file"] = json!("../escape.bin");
    std::fs::write(
        root.join("mei-model.json"),
        serde_json::to_vec(&manifest).unwrap(),
    )
    .unwrap();
    assert_eq!(
        load_package(&root, true).unwrap_err().info().id,
        "package_path_unsafe"
    );
    std::fs::remove_dir_all(root).unwrap();
}

#[test]
fn real_v2_package_loads_from_the_same_in_memory_payloads_as_wasm() {
    let Ok(raw) = std::env::var("MEI_51M_PACKAGE_DIR") else {
        return;
    };
    let root = std::path::PathBuf::from(raw);
    let manifest: Value = serde_json::from_slice(
        &std::fs::read(root.join("mei-model.json")).expect("real v2 manifest"),
    )
    .expect("real v2 manifest JSON");
    if manifest.get("package_format").and_then(Value::as_str) != Some("mei-model-package-v2") {
        return;
    }
    let weights_rel = manifest
        .pointer("/tensor_container/file")
        .and_then(Value::as_str)
        .expect("tensor container path")
        .to_string();
    let tokenizer_rel = manifest
        .pointer("/tokenizer/vocab_file")
        .and_then(Value::as_str)
        .or_else(|| manifest.pointer("/tokenizer/file").and_then(Value::as_str))
        .expect("tokenizer payload path")
        .to_string();
    let index_rel = manifest["files"]
        .as_array()
        .expect("files")
        .iter()
        .find(|row| row.get("role").and_then(Value::as_str) == Some("tool_index"))
        .and_then(|row| row.get("path"))
        .and_then(Value::as_str)
        .expect("tool index path")
        .to_string();
    let engine = Engine::from_quantized_package_bytes(
        manifest,
        std::fs::read(root.join(weights_rel)).expect("tensor container"),
        std::fs::read(root.join(tokenizer_rel)).expect("tokenizer payload"),
        Some(std::fs::read(root.join(index_rel)).expect("tool index")),
    )
    .expect("in-memory v2 load");
    assert_eq!(engine.capabilities()["backend_loaded"], true);
}

#[test]
fn javascript_integer_float_reserialization_preserves_architecture_identity() {
    let mut value = manifest(four_entries());
    value["architecture"]["rope_theta"] = json!(100_000);
    package_from_manifest(value, false)
        .expect("integer-valued JSON float is numerically identical");
}
