import { createHash } from "node:crypto";
import { lstatSync, readFileSync, readdirSync, realpathSync, statSync } from "node:fs";
import { isAbsolute, join, relative as relativePath, resolve, sep } from "node:path";
import { parseCq2Container } from "./cq2.mjs";
import { dumpsCanonical } from "./protocol.mjs";
import { parseToolIndex } from "./tool-index.mjs";
import { ERRORS } from "./version.mjs";

const REQUIRED = ["lm", "contrastive", "mw_disposition", "confidence"];
const OPTIONAL = ["narration_adapter"];
const ALL_HEADS = [...REQUIRED, ...OPTIONAL];
const SHA256 = /^[0-9a-f]{64}$/;
const V1 = "mei-model-package-v1";
const V2 = "mei-model-package-v2";
// runtime_profile_sha256 随架构 spec 演进（见 spec/runtime-profile-compatibility.json：
// 74839b08 → 7d2d97 → f3a4ab11，caps 恒为 compact 1024 / standard 1536）
const CONTRACTS = Object.freeze({
  weight_contract_sha256: "c468b96453f0a377b1ffbcfef00ed9e108c82b2c44847ee5345509a181581d9b",
  runtime_profile_sha256: "f3a4ab1151e82299fee0214c5f78a18c20d83367d24a9dc073bbcd56312fa512",
  training_aux_sha256: "83849db3926693e49c0896a58c172ae15e4b203550cee0ef12a4c37a8c1d48ac",
});
const RUNTIME_QUANTIZATION = Object.freeze({
  weight_math_id: "mei-cq-v2-g128-wht-codebook",
  activation_semantics_id: "int8-symmetric-per-last-axis-vector-qdq-forward_identity-backward-v2",
  activation_sites: ["engram_input", "attention_input", "attention_output", "lm_head_input"],
  activation_q_quantized: false,
  activation_reconstruction_dtype: "f32",
  kv_semantics_id: "mei-int8-kv-per-head-vector-qdq-forward_identity-backward-v1",
  kv_sites: ["attention_key_after_rope", "attention_value"],
  kv_storage_dtype: "int8",
  kv_scale_granularity: "per-head-vector",
  int8_code_min: -128,
  int8_code_max: 127,
  scale_denominator: 127,
});
const MW_LABEL_CODEBOOK_SHA256 = "913d2c4bca8a796c9baddb9a79539cc703aa6420af4260be2c5ca0e0d5a68d40";
const READY_HEAD_CONTRACTS = {
  contrastive: {
    prefix: "heads.contrastive",
    tensors: [
      ["heads.contrastive.tok_probes", [4, 512]],
      ["heads.contrastive.lay_probes", [4, 512]],
      ["heads.contrastive.proj.weight", [128, 2048]],
    ],
  },
  mw_disposition: {
    prefix: "heads.mw_disposition",
    tensors: [
      ["heads.mw_disposition.proj.weight", [20, 512]],
      ["heads.mw_disposition.proj.bias", [20]],
    ],
  },
  confidence: {
    prefix: "heads.confidence",
    tensors: [
      ["heads.confidence.cell_probes", [8, 512]],
      ["heads.confidence.proj.weight", [1, 4096]],
      ["heads.confidence.proj.bias", [1]],
    ],
  },
  narration_adapter: {
    prefix: "heads.narration_adapter",
    tensors: [
      ["heads.narration_adapter.down.weight", [16, 512]],
      ["heads.narration_adapter.up.weight", [24000, 16]],
    ],
  },
};

function canonicalLmTensorGeometry() {
  const tensors = [];
  const add = (name, shape) => tensors.push([name, shape]);
  add("embed.weight", [24000, 512]);
  for (let layer = 0; layer < 27; layer += 1) {
    const prefix = `blocks.${layer}`;
    add(`${prefix}.attn_norm.scale`, [512]);
    add(`${prefix}.attn.q_proj.weight`, [512, 512]);
    add(`${prefix}.attn.k_proj.weight`, [256, 512]);
    add(`${prefix}.attn.v_proj.weight`, [256, 512]);
    add(`${prefix}.attn.gate_proj.weight`, [512, 512]);
    add(`${prefix}.attn.o_proj.weight`, [512, 512]);
    add(`${prefix}.attn.q_norm.scale`, [64]);
    add(`${prefix}.attn.k_norm.scale`, [64]);
    add(`${prefix}.post_attn_norm.scale`, [512]);
    add(`${prefix}.attn_gate`, []);
    add(`${prefix}.mlp_norm.scale`, [512]);
    add(`${prefix}.mlp.d1`, [512]);
    add(`${prefix}.mlp.d2`, [512]);
    add(`${prefix}.mlp.d3`, [512]);
  }
  add("final_norm.scale", [512]);
  for (let site = 0; site < 2; site += 1) {
    const prefix = `engrams.${site}`;
    add(`${prefix}.tables`, [4, 8192, 128]);
    add(`${prefix}.key_proj.weight`, [512, 512]);
    add(`${prefix}.value_proj.weight`, [512, 512]);
    add(`${prefix}.taps`, [4, 512]);
  }
  add("mhc_phi_pre", [27, 2048, 4]);
  add("mhc_phi_post", [27, 2048, 4]);
  add("mhc_phi_res", [27, 2048, 16]);
  add("mhc_b_pre", [27, 4]);
  add("mhc_b_post", [27, 4]);
  add("mhc_b_res", [27, 4, 4]);
  add("mhc_a_pre", [27]);
  add("mhc_a_post", [27]);
  add("mhc_a_res", [27]);
  add("conf_probes", [8, 512]);
  add("conf_proj.weight", [1, 4096]);
  add("conf_proj.bias", [1]);
  if (tensors.length !== 400) fail("package_invalid", "internal 51M tensor contract is corrupt");
  return tensors;
}

function tensorIdentityComplete(manifest) {
  if (manifest?.contracts?.weight_contract_sha256 !== CONTRACTS.weight_contract_sha256) return false;
  const directory = manifest?.tensor_container?.directory;
  if (!Array.isArray(directory)) return false;
  if (directory.some((entry) => String(entry.name || "").split(".")
    .some((component) => component === "mtp" || component.startsWith("mtp_")))) return false;
  const actual = directory.filter((entry) => entry.role === "lm").map((entry) => [entry.name, entry.shape]);
  return dumpsCanonical(actual) === dumpsCanonical(canonicalLmTensorGeometry());
}

function portableLmDtype(name) {
  if (name === "embed.weight" || name.startsWith("mhc_")) return "cq4";
  if ((name.startsWith("engrams.") && !name.endsWith(".taps"))
      || (name.includes(".attn.") && name.endsWith("_proj.weight"))) return "cq2";
  return "f16";
}

function portableQuantizationPolicyComplete(manifest) {
  return tensorIdentityComplete(manifest)
    && manifest.tensor_container.directory.filter((entry) => entry.role === "lm")
      .every((entry) => entry.dtype === portableLmDtype(entry.name));
}

function runtimeQuantizationComplete(manifest) {
  return dumpsCanonical(manifest?.runtime_quantization ?? null)
    === dumpsCanonical(RUNTIME_QUANTIZATION);
}

function fail(id, message) {
  const err = new Error(message || ERRORS[id]?.message || id);
  err.code = id;
  err.abiCode = ERRORS[id]?.code;
  throw err;
}

function safeRelativeString(relative) {
  if (typeof relative !== "string" || !relative || relative.includes("\\") || isAbsolute(relative)) {
    fail("package_path_unsafe", `unsafe package path: ${String(relative)}`);
  }
  const segments = relative.split("/");
  if (segments.some((segment) => !segment || segment === "." || segment === "..")) {
    fail("package_path_unsafe", `unsafe package path: ${relative}`);
  }
  return relative;
}

function onlyKeys(value, allowed, path) {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail("package_invalid", `${path} must be an object`);
  const unknown = Object.keys(value).find((key) => !allowed.includes(key));
  if (unknown) fail("package_invalid", `unknown ${path}.${unknown}`);
}

function isComponentPath(value) {
  return typeof value === "string" && /^[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*$/.test(value);
}

function prefixMatches(name, prefix) {
  return name === prefix || name.startsWith(`${prefix}.`);
}

function safePath(root, relative) {
  safeRelativeString(relative);
  const path = resolve(root, relative);
  if (path !== root && !path.startsWith(`${root}${sep}`)) {
    fail("package_path_unsafe", `unsafe package path: ${relative}`);
  }
  let real;
  try {
    real = realpathSync(path);
  } catch {
    fail("file_not_found", `missing package file: ${relative}`);
  }
  if (real !== root && !real.startsWith(`${root}${sep}`)) {
    fail("package_path_unsafe", `symlink escapes package: ${relative}`);
  }
  return real;
}

function sha256File(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

function validateHeads(manifest, generation, tensors = []) {
  const heads = {};
  for (const name of ALL_HEADS) {
    const item = manifest.heads?.[name];
    if (!item && OPTIONAL.includes(name)) {
      heads[name] = {
        present: false,
        trained: false,
        status: "missing",
        tensor_prefixes: [],
        training_receipt_sha256: null,
        label_codebook_sha256: null,
      };
      continue;
    }
    if (!item) fail("package_invalid", `heads.${name} must be listed explicitly`);
    const status = String(item.status || "missing");
    if (!["ready", "untrained", "missing", "disabled"].includes(status)) {
      fail("package_invalid", `heads.${name}.status is invalid`);
    }
    const tensorPrefixes = Array.isArray(item.tensor_prefixes) ? item.tensor_prefixes.map(String) : [];
    const present = Boolean(item.present);
    if (typeof item.present !== "boolean" || typeof item.trained !== "boolean") {
      fail("package_invalid", `heads.${name} present/trained must be boolean`);
    }
    if (status === "ready" && (!item.present || !item.trained)) {
      fail("package_invalid", `heads.${name} ready status requires present and trained`);
    }
    if (generation === V2 && present && !tensorPrefixes.length) {
      fail("package_invalid", `heads.${name}.tensor_prefixes must identify tensors`);
    }
    for (const prefix of tensorPrefixes) {
      if (!isComponentPath(prefix) || !tensors.some((tensor) => tensor.role === name && prefixMatches(tensor.name, prefix))) {
        fail("package_invalid", `heads.${name} prefix ${prefix} matches no same-role tensor`);
      }
    }
    const contract = READY_HEAD_CONTRACTS[name];
    if (generation === V2 && status === "ready") {
      const receipt = item.training_receipt_sha256;
      if (!SHA256.test(String(receipt || ""))) {
        fail("package_invalid", `heads.${name} ready status requires training_receipt_sha256`);
      }
      if (!manifest.training_receipts?.includes(receipt)) {
        fail("package_invalid", `heads.${name} training receipt is absent from training_receipts`);
      }
      if (!manifest.files?.some((row) => row.role === "training_receipt" && row.sha256 === receipt)) {
        fail("package_invalid", `heads.${name} training receipt has no inventoried payload`);
      }
      if (name === "mw_disposition") {
        if (item.label_codebook_sha256 !== MW_LABEL_CODEBOOK_SHA256) {
          fail("package_invalid", "heads.mw_disposition ready status requires the frozen 20-class label codebook");
        }
        if (manifest.files?.filter(
          (row) => row.role === "head_codebook" && row.sha256 === MW_LABEL_CODEBOOK_SHA256,
        ).length !== 1) {
          fail("package_invalid", "heads.mw_disposition label codebook has no inventoried payload");
        }
      }
      if (contract && (tensorPrefixes.length !== 1 || tensorPrefixes[0] !== contract.prefix)) {
        fail("package_invalid", `heads.${name} must use canonical prefix ${contract.prefix}`);
      }
      const roleTensors = contract ? tensors.filter((tensor) => tensor.role === name) : [];
      if (contract && roleTensors.length !== contract.tensors.length) {
        fail("package_invalid", `heads.${name} ready tensor set is not canonical`);
      }
      for (const [tensorName, shape] of contract?.tensors || []) {
        const tensor = roleTensors.find((candidate) => candidate.name === tensorName);
        const narrationQuantized = name === "narration_adapter"
          && ["cq2", "cq4"].includes(tensor?.dtype)
          && tensor?.transform === "wht"
          && tensor?.codebook === (tensor.dtype === "cq2"
            ? "gaussian-lloyd-q2-v1" : "gaussian-lloyd-q4-v1");
        const ordinaryF16 = name !== "narration_adapter"
          && tensor?.dtype === "f16" && tensor?.transform === "none" && tensor?.codebook === "none";
        if (!tensor || JSON.stringify(tensor.shape) !== JSON.stringify(shape)
            || !(narrationQuantized || ordinaryF16)) {
          fail("package_invalid", `heads.${name} tensor ${tensorName} violates the canonical contract`);
        }
      }
    }
    heads[name] = {
      present,
      trained: Boolean(item.trained),
      status,
      tensor_prefixes: tensorPrefixes,
      training_receipt_sha256: item.training_receipt_sha256 ?? null,
      label_codebook_sha256: item.label_codebook_sha256 ?? null,
    };
  }
  return heads;
}

function validateDirectory(manifest, actualBytes) {
  const container = manifest.tensor_container;
  if (container?.format !== "mei-cq-tensor-v2" || container.quant_math_id !== "mei-cq-v2-g128-wht-codebook") {
    fail("package_invalid", "v2 requires mei-cq-tensor-v2 / mei-cq-v2-g128-wht-codebook");
  }
  if (!Number.isSafeInteger(container.payload_bytes) || container.payload_bytes <= 0) {
    fail("package_invalid", "tensor_container.payload_bytes is invalid");
  }
  if (actualBytes != null && actualBytes !== container.payload_bytes) {
    fail("package_range_invalid", `payload_bytes=${container.payload_bytes}, file=${actualBytes}`);
  }
  if (!Array.isArray(container.directory) || !container.directory.length) {
    fail("package_invalid", "tensor_container.directory must not be empty");
  }
  const names = new Set();
  const ranges = [];
  const tensors = [];
  for (const entry of container.directory) {
    if (!isComponentPath(entry.name)) fail("package_invalid", `invalid tensor name: ${String(entry.name || "")}`);
    onlyKeys(entry, ["name", "role", "shape", "n_params", "dtype", "data", "scales", "bit_map", "group_size", "transform", "codebook"], `tensor.${entry.name}`);
    if (names.has(entry.name)) fail("duplicate_tensor", entry.name);
    names.add(entry.name);
    if (!Array.isArray(entry.shape) || entry.shape.some((dim) => !Number.isSafeInteger(dim) || dim <= 0)) {
      fail("package_invalid", `${entry.name}.shape is invalid`);
    }
    const product = entry.shape.reduce((value, dim) => value * dim, 1);
    if (!Number.isSafeInteger(product) || product !== entry.n_params) {
      fail("package_invalid", `${entry.name}.n_params != product(shape)`);
    }
    if (!["lm", "contrastive", "mw_disposition", "confidence", "narration_adapter"].includes(entry.role)) {
      fail("package_invalid", `${entry.name}.role is invalid`);
    }
    if (!["cq2", "cq4", "f16", "f32", "i8"].includes(entry.dtype)) {
      fail("package_invalid", `${entry.name}.dtype is invalid`);
    }
    if (["cq2", "cq4"].includes(entry.dtype)) {
      const groups = Math.ceil(entry.n_params / 128);
      const expectedCodebook = entry.dtype === "cq2" ? "gaussian-lloyd-q2-v1" : "gaussian-lloyd-q4-v1";
      const dataBytes = entry.data?.nbytes;
      if (entry.group_size !== 128 || entry.transform !== "wht" || entry.codebook !== expectedCodebook
          || entry.scales?.nbytes !== groups * 2 || entry.bit_map?.nbytes !== Math.ceil(groups / 8)
          || !Number.isSafeInteger(dataBytes) || dataBytes < groups * 32 || dataBytes > groups * 64
          || dataBytes % 32 !== 0 || (entry.dtype === "cq2" && dataBytes === groups * 64)
          || (entry.dtype === "cq4" && dataBytes !== groups * 64)) {
        fail("package_invalid", `${entry.name} CQ2 metadata is invalid`);
      }
    } else {
      const width = { f32: 4, f16: 2, i8: 1 }[entry.dtype];
      if (entry.data?.nbytes !== entry.n_params * width || entry.transform !== "none" || entry.codebook !== "none"
          || entry.group_size != null || entry.scales != null || entry.bit_map != null) {
        fail("package_invalid", `${entry.name} safe tensor metadata is invalid`);
      }
    }
    for (const field of ["data", "scales", "bit_map"]) {
      const range = entry[field];
      if (field === "data" && !range) fail("package_invalid", `${entry.name}.data is required`);
      if (!range) continue;
      onlyKeys(range, ["offset", "nbytes"], `tensor.${entry.name}.${field}`);
      const { offset, nbytes } = range;
      if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(nbytes) || offset < 0 || nbytes < 0 || offset + nbytes > container.payload_bytes) {
        fail("package_range_invalid", `${entry.name}.${field} exceeds payload`);
      }
      if (nbytes > 0) ranges.push({ start: offset, end: offset + nbytes, label: `${entry.name}.${field}` });
    }
    tensors.push({
      name: entry.name,
      role: entry.role,
      shape: entry.shape,
      dtype: entry.dtype,
      transform: entry.transform,
      codebook: entry.codebook,
    });
  }
  ranges.sort((a, b) => a.start - b.start);
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index - 1].end > ranges[index].start) {
      fail("package_range_invalid", `${ranges[index - 1].label} overlaps ${ranges[index].label}`);
    }
  }
  return tensors;
}

function verifyFile(root, spec) {
  if (!SHA256.test(String(spec?.sha256 || ""))) fail("package_invalid", "invalid package file sha256");
  const path = safePath(root, spec.file);
  const digest = sha256File(path);
  if (digest !== spec.sha256) fail("package_hash_mismatch", `${spec.file} sha256 mismatch: expected ${spec.sha256}, got ${digest}`);
  return path;
}

function validateFiles(manifest) {
  if (!Array.isArray(manifest.files) || !manifest.files.length) {
    fail("package_invalid", "files must inventory every payload");
  }
  const files = new Map();
  for (const row of manifest.files) {
    onlyKeys(row, ["path", "sha256", "nbytes", "role"], "files[]");
    safeRelativeString(row.path);
    if (row.path === "mei-model.json") fail("package_invalid", "mei-model.json is the self-hash exception");
    if (!SHA256.test(String(row.sha256 || "")) || !Number.isSafeInteger(row.nbytes) || row.nbytes < 0
        || !["tokenizer", "tensor_container", "tool_index", "training_receipt", "head_codebook", "resource_receipt", "auxiliary"].includes(row.role)) {
      fail("package_invalid", `invalid files entry for ${String(row.path)}`);
    }
    if (files.has(row.path)) fail("package_invalid", `duplicate files path ${row.path}`);
    files.set(row.path, { sha256: row.sha256, nbytes: row.nbytes, role: row.role });
  }
  if (manifest.files.filter((row) => row.role === "tool_index").length > 1) {
    fail("package_invalid", "package supports at most one canonical tool index");
  }
  const required = [
    [manifest.tokenizer?.file, manifest.tokenizer?.sha256, null, "tokenizer"],
    [manifest.tokenizer?.vocab_file, manifest.tokenizer?.vocab_sha256, null, "tokenizer"],
    [manifest.tensor_container?.file, manifest.tensor_container?.sha256, manifest.tensor_container?.payload_bytes, "tensor_container"],
  ];
  for (const [path, hash, nbytes, role] of required) {
    if (path == null && hash == null) continue;
    if (typeof path !== "string" || typeof hash !== "string") fail("package_invalid", "file/hash reference pair");
    const listed = files.get(path);
    if (!listed || listed.sha256 !== hash || listed.role !== role || (nbytes != null && listed.nbytes !== nbytes)) {
      fail("package_invalid", `files entry disagrees with ${path} reference`);
    }
  }
  return files;
}

function collectPayloadFiles(root, directory = root, out = new Set()) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const full = join(directory, entry.name);
    const metadata = lstatSync(full);
    if (metadata.isSymbolicLink()) fail("package_path_unsafe", `package symlink is forbidden: ${full}`);
    if (metadata.isDirectory()) collectPayloadFiles(root, full, out);
    else if (metadata.isFile()) {
      const relative = relativePath(root, full).split(sep).join("/");
      safeRelativeString(relative);
      out.add(relative);
    } else fail("package_path_unsafe", `special package file is forbidden: ${full}`);
  }
  return out;
}

function verifyFiles(root, manifest) {
  const declared = validateFiles(manifest);
  const actual = collectPayloadFiles(root);
  if (!actual.delete("mei-model.json")) fail("file_not_found", "mei-model.json");
  const declaredPaths = new Set(declared.keys());
  const undeclared = [...actual].filter((path) => !declaredPaths.has(path));
  const missing = [...declaredPaths].filter((path) => !actual.has(path));
  if (undeclared.length || missing.length) {
    fail("package_invalid", `payload inventory mismatch undeclared=${JSON.stringify(undeclared)} missing=${JSON.stringify(missing)}`);
  }
  let payloadBytes = 0;
  for (const [path, row] of declared) {
    const full = safePath(root, path);
    const size = statSync(full).size;
    if (size !== row.nbytes) fail("package_range_invalid", `${path} declares ${row.nbytes} bytes, file has ${size}`);
    if (sha256File(full) !== row.sha256) fail("package_hash_mismatch", `${path} sha256 mismatch`);
    payloadBytes += size;
  }
  const packageBytes = payloadBytes + statSync(join(root, "mei-model.json")).size;
  if (manifest.resources?.package_bytes !== packageBytes) {
    fail("package_range_invalid", `resources.package_bytes declares ${manifest.resources?.package_bytes}, measured ${packageBytes}`);
  }
}

function verifyResourceMeasurementReceipt(root, manifest) {
  const spec = manifest.resources?.measurement_receipt;
  if (spec == null) return false;
  const path = safePath(root, spec.file);
  if (sha256File(path) !== spec.sha256) fail("package_hash_mismatch", "resource receipt sha256 mismatch");
  let receipt;
  try { receipt = JSON.parse(readFileSync(path, "utf8")); } catch { fail("invalid_json", "resource receipt is not valid JSON"); }
  onlyKeys(receipt, [
    "schema", "package_id", "tensor_container_sha256", "runtime_abi", "measurements", "rust", "wasm",
  ], "resource_receipt");
  if (receipt.schema !== "mei-resource-measurement-receipt-v1"
      || receipt.package_id !== manifest.package_id
      || receipt.tensor_container_sha256 !== manifest.tensor_container.sha256
      || receipt.runtime_abi !== "mei-runtime-abi-2") {
    fail("package_invalid", "resource measurement receipt identity mismatch");
  }
  const measurementKeys = ["package_bytes", "rust_session_peak_bytes", "wasm_heap_peak_bytes"];
  onlyKeys(receipt.measurements, measurementKeys, "resource_receipt.measurements");
  for (const key of measurementKeys) {
    if (!Number.isSafeInteger(receipt.measurements?.[key]) || receipt.measurements[key] !== manifest.resources[key]) {
      fail("package_invalid", `resource measurement receipt disagrees on ${key}`);
    }
  }
  for (const runtime of ["rust", "wasm"]) {
    const runner = receipt[runtime];
    onlyKeys(runner, ["target", "build_profile", "runner_id", "runner_sha256", "command_sha256"], `resource_receipt.${runtime}`);
    if (typeof runner.target !== "string" || !runner.target || runner.build_profile !== "release"
        || typeof runner.runner_id !== "string" || !runner.runner_id
        || !SHA256.test(String(runner.runner_sha256 || "")) || !SHA256.test(String(runner.command_sha256 || ""))) {
      fail("package_invalid", `resource receipt ${runtime} runner evidence`);
    }
  }
  return true;
}

function contrastiveHeadSha256(containerBytes, parsed) {
  const digest = createHash("sha256");
  const rows = parsed.tensors.filter((tensor) => tensor.role === "contrastive")
    .sort((left, right) => Buffer.compare(Buffer.from(left.name), Buffer.from(right.name)));
  if (!rows.length) fail("package_invalid", "tool index requires a contrastive head payload");
  for (const tensor of rows) {
    digest.update(tensor.name, "utf8");
    for (const range of [tensor.data, tensor.scales, tensor.bit_map]) {
      if (range?.nbytes) digest.update(containerBytes.subarray(range.offset, range.offset + range.nbytes));
    }
  }
  return digest.digest("hex");
}

function verifyToolIndex(root, manifest, containerBytes, parsed) {
  const rows = manifest.files.filter((row) => row.role === "tool_index");
  if (!rows.length) return null;
  if (rows.length !== 1) fail("package_invalid", "package supports exactly one canonical tool index");
  const index = parseToolIndex(readFileSync(safePath(root, rows[0].path)));
  if (index.model_sha256 !== manifest.tensor_container.sha256
      || index.head_sha256 !== contrastiveHeadSha256(containerBytes, parsed)
      || index.tokenizer_sha256 !== manifest.tokenizer.sha256) {
    fail("package_hash_mismatch", "tool index model/head/tokenizer fingerprint mismatch");
  }
  return index;
}

function roleDirectorySha256(manifest, component) {
  const entries = manifest.tensor_container?.directory?.filter((entry) => entry.role === component) || [];
  if (!entries.length) fail("package_invalid", `training receipt component ${component} has no tensors`);
  return createHash("sha256").update(dumpsCanonical(entries), "utf8").digest("hex");
}

function verifyTrainingReceipts(root, manifest) {
  const components = new Map();
  for (const digest of manifest.training_receipts) {
    const row = manifest.files.find((candidate) => candidate.role === "training_receipt" && candidate.sha256 === digest);
    if (!row) fail("package_invalid", "training receipt payload is absent");
    let receipt;
    try { receipt = JSON.parse(readFileSync(safePath(root, row.path), "utf8")); }
    catch { fail("invalid_json", `training receipt ${row.path} is not valid JSON`); }
    onlyKeys(receipt, [
      "schema", "product", "package_id", "component", "stage_id", "stage_fingerprint_sha256",
      "status", "contracts", "tensor_container_sha256", "tensor_directory_sha256",
    ], "training_receipt");
    const component = receipt.component;
    if (receipt.schema !== "mei-training-receipt-v2"
        || receipt.product !== manifest.product
        || receipt.package_id !== manifest.package_id
        || !ALL_HEADS.includes(component)
        || typeof receipt.stage_id !== "string" || !receipt.stage_id
        || !SHA256.test(String(receipt.stage_fingerprint_sha256 || ""))
        || receipt.status !== "passed"
        || dumpsCanonical(receipt.contracts) !== dumpsCanonical(manifest.contracts)
        || receipt.tensor_container_sha256 !== manifest.tensor_container.sha256
        || receipt.tensor_directory_sha256 !== roleDirectorySha256(manifest, component)) {
      fail("package_invalid", `training receipt ${digest} identity or terminal status mismatch`);
    }
    components.set(digest, component);
  }
  for (const name of ALL_HEADS) {
    const head = manifest.heads[name];
    if (!head) continue;
    if (head.status === "ready" && components.get(head.training_receipt_sha256) !== name) {
      fail("package_invalid", `heads.${name} receipt is not bound to that component`);
    }
  }
  return true;
}

export function validatePackageManifest(manifest, { root = null, verifyHashes = false } = {}) {
  const generation = manifest?.package_format;
  if (![V1, V2].includes(generation)) fail("package_invalid", "unsupported package_format");
  if (!["mei-1.0-51m", "mei-1.1-51m", "mei-1.2-51m"].includes(manifest.product)) fail("package_invalid", "product must be mei-1.0-51m or mei-1.2-51m");
  let tensors = [];
  if (generation === V2) {
    onlyKeys(manifest, [
      "package_format", "product", "package_id", "runtime_min", "parent_package_id", "contracts",
      "architecture", "runtime_profile", "runtime_quantization", "tokenizer", "tensor_container", "files", "heads", "capabilities",
      "training_receipts", "resources", "release_class", "retrieval_calibration",
    ], "manifest");
    if (typeof manifest.package_id !== "string" || !manifest.package_id) fail("package_invalid", "package_id is required");
    if (!["experimental", "candidate", "release"].includes(manifest.release_class)) fail("package_invalid", "release_class is required");
    if (manifest.parent_package_id != null && (typeof manifest.parent_package_id !== "string" || !manifest.parent_package_id)) {
      fail("package_invalid", "parent_package_id is invalid");
    }
    if (manifest.runtime_min !== "mei-runtime-abi-2") fail("abi_version_mismatch", "v2 package requires ABI 2");
    for (const [key, expected] of Object.entries(CONTRACTS)) {
      if (manifest.contracts?.[key] !== expected) {
        fail("package_invalid", `contracts.${key} does not match the frozen 51M contract`);
      }
    }
    onlyKeys(manifest.contracts, ["weight_contract_sha256", "runtime_profile_sha256", "training_aux_sha256"], "contracts");
    onlyKeys(manifest.architecture, [
      "id", "d_model", "n_layers", "n_heads", "n_kv_heads", "head_dim", "vocab_size", "max_seq_len",
      "parameter_count", "rope_theta", "engram_layers", "engram_orders", "engram_slots", "engram_conv_taps",
      "mhc_lanes", "sinkhorn_iters", "tie_embeddings", "rms_eps", "conf_probes", "mlp", "confidence_head",
    ], "architecture");
    const expectedArchitecture = {
      d_model: 512, n_layers: 27, n_heads: 8, n_kv_heads: 4,
      head_dim: 64, vocab_size: 24000, max_seq_len: 2048, parameter_count: 51463797,
      id: "mei-1.0-51m-arch-v1", rope_theta: 100000,
      engram_layers: [2, 15], engram_orders: [2, 3], engram_slots: 8192, engram_conv_taps: 4,
      mhc_lanes: 4, sinkhorn_iters: 20, tie_embeddings: true, rms_eps: 1e-6,
      conf_probes: 8, mlp: "FixedWalshHadamardMLP", confidence_head: true,
    };
    if (Object.entries(expectedArchitecture)
      .some(([key, value]) => dumpsCanonical(manifest.architecture?.[key]) !== dumpsCanonical(value))) {
      fail("package_invalid", "51M architecture identity mismatch");
    }
    // 镜像 python schema 的 oneOf：分支1 扁平 portable / 分支2 adaptive profile
    let profileKeys;
    if (manifest.runtime_profile?.stable_prefix_profiles != null) {
      const expectedAdaptive = {
        max_context_tokens: 2048, default_profile: "standard", ordinary_window_policy: "dynamic_remainder",
        default_output_tokens: 128, candidate_batch_size: 5, kv_dtype: "i8", activation_dtype: "i8",
      };
      if (Object.entries(expectedAdaptive).some(([key, value]) => manifest.runtime_profile?.[key] !== value)) {
        fail("package_invalid", "runtime_profile does not match portable v2 (adaptive branch)");
      }
      const expectedPrefixes = JSON.stringify({ compact: 1024, standard: 1536 });
      if (JSON.stringify(manifest.runtime_profile?.stable_prefix_profiles) !== expectedPrefixes) {
        fail("package_invalid", "runtime_profile does not match portable v2 (adaptive prefixes)");
      }
      profileKeys = Object.keys(expectedAdaptive).concat(["stable_prefix_profiles", "context_packer_id", "retrieval_batch_policy_id", "prompt_framing_id", "assistant_suffix"]);
    } else {
      const expectedProfile = {
        max_context_tokens: 2048, stable_prefix_tokens: 1024, rolling_window_tokens: 256,
        default_output_tokens: 128, kv_dtype: "i8", activation_dtype: "i8",
      };
      if (Object.entries(expectedProfile).some(([key, value]) => manifest.runtime_profile?.[key] !== value)) {
        fail("package_invalid", "runtime_profile does not match portable v2");
      }
      profileKeys = Object.keys(expectedProfile);
    }
    onlyKeys(manifest.runtime_profile, profileKeys, "runtime_profile");
    if (manifest.runtime_quantization != null) {
      onlyKeys(manifest.runtime_quantization, Object.keys(RUNTIME_QUANTIZATION), "runtime_quantization");
      if (!runtimeQuantizationComplete(manifest)) {
        fail("package_invalid", "runtime_quantization does not match native v2 semantics");
      }
    }
    if (!["zh-24k-v1", "zh-24k-v3"].includes(manifest.tokenizer?.id)) fail("package_invalid", "tokenizer.id must be zh-24k-v1 or zh-24k-v3");
    onlyKeys(manifest.tokenizer, [
      "id", "file", "sha256", "vocab_file", "vocab_sha256", "pad_id", "eos_id", "bos_id", "unk_id",
    ], "tokenizer");
    for (const [key, expected] of Object.entries({ pad_id: 0, eos_id: 1, bos_id: 2, unk_id: 3 })) {
      if (manifest.tokenizer?.[key] !== expected) fail("package_invalid", `tokenizer.${key}`);
    }
    if (!SHA256.test(String(manifest.tokenizer?.sha256 || ""))) fail("package_invalid", "tokenizer.sha256");
    if (Boolean(manifest.tokenizer?.vocab_file) !== Boolean(manifest.tokenizer?.vocab_sha256)
        || (manifest.tokenizer?.vocab_sha256 && !SHA256.test(manifest.tokenizer.vocab_sha256))) {
      fail("package_invalid", "tokenizer vocab file/hash pair");
    }
    safeRelativeString(manifest.tokenizer?.file);
    if (manifest.tokenizer?.vocab_file) safeRelativeString(manifest.tokenizer.vocab_file);
    safeRelativeString(manifest.tensor_container?.file);
    for (const key of ["retrieval", "full_call", "mw_disposition", "confidence", "multi_step"]) {
      if (typeof manifest.capabilities?.[key] !== "boolean") fail("package_invalid", `capabilities.${key}`);
    }
    onlyKeys(manifest.capabilities, ["retrieval", "full_call", "mw_disposition", "confidence", "multi_step", "narration"], "capabilities");
    if (manifest.capabilities.narration != null && typeof manifest.capabilities.narration !== "boolean") {
      fail("package_invalid", "capabilities.narration");
    }
    if (!Array.isArray(manifest.training_receipts) || manifest.training_receipts.some((receipt) => !SHA256.test(receipt))) {
      fail("package_invalid", "invalid training_receipts");
    }
    if (new Set(manifest.training_receipts).size !== manifest.training_receipts.length
        || manifest.training_receipts.some((receipt) => manifest.files?.filter(
          (row) => row.role === "training_receipt" && row.sha256 === receipt,
        ).length !== 1)) {
      fail("package_invalid", "every training_receipts entry must have one inventoried payload");
    }
    const inventoriedReceipts = manifest.files?.filter((row) => row.role === "training_receipt") || [];
    if (inventoriedReceipts.length !== manifest.training_receipts.length
        || inventoriedReceipts.some((row) => !manifest.training_receipts.includes(row.sha256))) {
      fail("package_invalid", "training_receipts and inventoried payloads must be one-to-one");
    }
    for (const key of ["package_bytes", "rust_session_peak_bytes", "wasm_heap_peak_bytes"]) {
      if (!Number.isSafeInteger(manifest.resources?.[key]) || manifest.resources[key] < 0) fail("package_invalid", `resources.${key}`);
    }
    onlyKeys(manifest.resources, [
      "package_bytes", "rust_session_peak_bytes", "wasm_heap_peak_bytes", "measurement_receipt",
    ], "resources");
    if (manifest.resources.measurement_receipt != null) {
      const receipt = manifest.resources.measurement_receipt;
      onlyKeys(receipt, ["file", "sha256"], "resources.measurement_receipt");
      safeRelativeString(receipt.file);
      if (!SHA256.test(String(receipt.sha256 || "")) || manifest.files?.filter(
        (row) => row.path === receipt.file && row.sha256 === receipt.sha256 && row.role === "resource_receipt",
      ).length !== 1) fail("package_invalid", "resource measurement receipt is not uniquely inventoried");
    }
    onlyKeys(manifest.tensor_container, ["file", "format", "sha256", "payload_bytes", "quant_math_id", "directory"], "tensor_container");
    onlyKeys(manifest.heads, ALL_HEADS, "heads");
    for (const [name, head] of Object.entries(manifest.heads)) {
      onlyKeys(head, [
        "present", "trained", "status", "tensor_prefixes", "training_receipt_sha256", "label_codebook_sha256",
      ], `heads.${name}`);
      if (head.training_receipt_sha256 && !SHA256.test(head.training_receipt_sha256)) fail("package_invalid", `heads.${name}.training_receipt_sha256`);
      if (head.label_codebook_sha256 && !SHA256.test(head.label_codebook_sha256)) fail("package_invalid", `heads.${name}.label_codebook_sha256`);
    }
    if (!SHA256.test(String(manifest.tensor_container?.sha256 || ""))) fail("package_invalid", "tensor_container.sha256");
    validateFiles(manifest);
    tensors = validateDirectory(manifest, null);
  }
  const heads = validateHeads(manifest, generation, tensors);
  let trainingReceiptsVerified = false;
  let resourceMeasurementVerified = false;
  let toolIndex = null;
  if (verifyHashes) {
    if (!root) fail("invalid_argument", "root is required for hash verification");
    // macOS may surface the same directory through /var and /private/var.
    // Canonicalize before doing prefix checks so that a legitimate package is
    // not mistaken for a symlink escape while still confining every payload to
    // the real package root.
    const verifiedRoot = realpathSync(resolve(root));
    if (generation === V2) {
      verifyFiles(verifiedRoot, manifest);
      trainingReceiptsVerified = verifyTrainingReceipts(verifiedRoot, manifest);
      resourceMeasurementVerified = verifyResourceMeasurementReceipt(verifiedRoot, manifest);
    }
    verifyFile(verifiedRoot, manifest.tokenizer);
    if (manifest.tokenizer?.vocab_file && manifest.tokenizer?.vocab_sha256) {
      verifyFile(verifiedRoot, { file: manifest.tokenizer.vocab_file, sha256: manifest.tokenizer.vocab_sha256 });
    }
    const tensorPath = verifyFile(verifiedRoot, generation === V2 ? manifest.tensor_container : manifest.weights);
    if (manifest.heads?.artifact) verifyFile(verifiedRoot, manifest.heads.artifact);
    if (generation === V2) {
      validateDirectory(manifest, statSync(tensorPath).size);
      const containerBytes = readFileSync(tensorPath);
      const parsed = parseCq2Container(containerBytes);
      const declared = new Map(manifest.tensor_container.directory.map((entry) => [entry.name, entry]));
      if (parsed.tensors.length !== declared.size) fail("package_invalid", "container and manifest tensor counts differ");
      for (const actual of parsed.tensors) {
        const expected = declared.get(actual.name);
        if (!expected) fail("package_invalid", `container tensor ${actual.name} is undeclared`);
        const same = (left, right) => dumpsCanonical(left ?? null) === dumpsCanonical(right ?? null);
        if (!same(actual.shape, expected.shape) || actual.n_params !== expected.n_params || actual.dtype !== expected.dtype
            || actual.role !== expected.role || actual.transform !== expected.transform
            || actual.codebook !== expected.codebook
            || !same(actual.data, expected.data) || !same(actual.scales, expected.scales)
            || !same(actual.bit_map, expected.bit_map)
            || (["cq2", "cq4"].includes(actual.dtype) && actual.group_size !== expected.group_size)) {
          fail("package_invalid", `container manifest mismatch: ${actual.name}`);
        }
      }
      toolIndex = verifyToolIndex(verifiedRoot, manifest, containerBytes, parsed);
    }
  }
  const missing = REQUIRED.filter((name) => {
    const head = heads[name];
    return head.status !== "ready" || !head.present || !head.trained;
  });
  const identityComplete = generation === V2 && tensorIdentityComplete(manifest);
  const quantizationPolicyComplete = generation === V2 && portableQuantizationPolicyComplete(manifest);
  const runtimeQuantizationReady = generation === V2 && runtimeQuantizationComplete(manifest);
  const declaredCapabilities = generation === V2
    && ["retrieval", "full_call", "mw_disposition", "confidence", "multi_step"].every((key) => manifest.capabilities[key]);
  const packedStructure = generation === V2
    ? manifest.tensor_container?.format === "mei-cq-tensor-v2" && identityComplete && quantizationPolicyComplete
      && missing.length === 0 && declaredCapabilities
    : manifest.weights?.format === "mei-q4-packed-v1";
  const packed = Boolean(verifyHashes) && trainingReceiptsVerified && toolIndex != null && packedStructure;
  // This generic loader verifies package structure only. The Node runtime
  // wrapper binds the same Rust/WASM numerical core and upgrades capabilities
  // only after it has loaded every declared head and the frozen tool index.
  const productReady = false;
  const resourceLimitsReported = generation === V2 && Boolean(verifyHashes) && resourceMeasurementVerified
    && manifest.resources.package_bytes <= 18 * 1024 * 1024
    && manifest.resources.rust_session_peak_bytes <= 64 * 1024 * 1024
    && manifest.resources.wasm_heap_peak_bytes <= 96 * 1024 * 1024;
  // The receipt binds the package, runner/command identities and numbers, but
  // a trusted runner allow-list and reproducible peak-memory harness are not
  // frozen yet. Never turn producer-authored evidence into release eligibility.
  const resourceEligible = false;
  return {
    generation, heads, missing, packed, tensorIdentityComplete: identityComplete,
    quantizationPolicyComplete, runtimeQuantizationReady, trainingReceiptsVerified, toolIndex,
    toolIndexVerified: toolIndex != null, resourceMeasurementVerified, resourceLimitsReported,
    productReady, resourceEligible, releaseEligible: false,
  };
}

export function loadPackage(packageDir, { verifyHashes = true } = {}) {
  const root = realpathSync(resolve(packageDir));
  const manifestCandidate = join(root, "mei-model.json");
  const manifestMetadata = lstatSync(manifestCandidate);
  if (manifestMetadata.isSymbolicLink() || !manifestMetadata.isFile()) {
    fail("package_path_unsafe", "mei-model.json must be a regular non-symlink file");
  }
  const manifest = JSON.parse(readFileSync(manifestCandidate, "utf8"));
  const report = validatePackageManifest(manifest, { root, verifyHashes });
  return {
    path: root,
    manifest,
    heads: report.heads,
    tool_index: report.toolIndex,
    verified_hashes: Boolean(verifyHashes),
    capabilities(versions) {
      const legacy = report.generation === V1;
      return {
        package_id: manifest.package_id,
        package_format: report.generation,
        release_class: manifest.release_class,
        inference: false,
        diagnostic_payload_ready: report.packed,
        protocol: true,
        heads: report.heads,
        missing_or_untrained_heads: report.missing,
        hash_verified: Boolean(verifyHashes),
        training_receipts_verified: report.trainingReceiptsVerified,
        tool_index_verified: report.toolIndexVerified,
        resource_measurement_verified: report.resourceMeasurementVerified,
        resource_limits_reported: report.resourceLimitsReported,
        tensor_identity_complete: report.tensorIdentityComplete,
        portable_quantization_policy_complete: report.quantizationPolicyComplete,
        runtime_quantization_complete: report.runtimeQuantizationReady,
        compatibility_mode: legacy ? "v1-read-only-degraded" : "v2-native",
        read_only: legacy,
        degraded: legacy || !report.productReady,
        implementation_complete: false,
        open_capabilities: [
          "native-inference", "portable-tool-index-retrieval", "independent-head-execution",
          "utf8-constrained-decoder", "int8-bounded-kv", "trusted-resource-measurement",
        ],
        product_ready: report.productReady,
        resource_eligible: report.resourceEligible,
        release_eligible: report.releaseEligible,
        quantized_only: report.packed,
        narration: Boolean(manifest.capabilities?.narration)
          && report.heads.narration_adapter?.status === "ready",
        versions,
      };
    },
  };
}
