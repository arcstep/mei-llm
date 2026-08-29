import { createHash } from "node:crypto";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";
import { ERRORS } from "./version.mjs";

const REQUIRED = ["lm", "contrastive", "mw_disposition", "confidence"];

function fail(id, message) {
  const err = new Error(message || ERRORS[id].message);
  err.code = id;
  err.abiCode = ERRORS[id].code;
  throw err;
}

function sha256File(path) {
  return createHash("sha256").update(readFileSync(path)).digest("hex");
}

export function loadPackage(packageDir, { verifyHashes = true } = {}) {
  const root = resolve(packageDir);
  const manifest = JSON.parse(readFileSync(resolve(root, "mei-model.json"), "utf8"));
  if (manifest.package_format !== "mei-model-package-v1") fail("package_invalid", "package_format must be mei-model-package-v1");
  if (manifest.product !== "mei-1.0-58m" && manifest.product !== "mei-1.0-51m") {
    fail("package_invalid", "product must be mei-1.0-58m or mei-1.0-51m");
  }
  const heads = {};
  for (const name of REQUIRED) {
    if (!manifest.heads?.[name]) fail("package_invalid", `heads.${name} must be listed explicitly`);
    heads[name] = {
      present: Boolean(manifest.heads[name].present),
      trained: Boolean(manifest.heads[name].trained),
      status: String(manifest.heads[name].status || "missing"),
    };
  }
  const missing = REQUIRED.filter((name) => {
    const head = heads[name];
    return head.status === "missing" || head.status === "untrained" || !head.present || !head.trained;
  });
  if (verifyHashes) {
    for (const key of ["tokenizer", "weights"]) {
      const spec = manifest[key];
      const path = resolve(root, spec.file);
      const digest = sha256File(path);
      if (digest !== String(spec.sha256).toLowerCase()) {
        fail("package_hash_mismatch", `${key} sha256 mismatch: expected ${spec.sha256}, got ${digest}`);
      }
    }
  }
  return {
    path: root,
    manifest,
    heads,
    verified_hashes: Boolean(verifyHashes),
    capabilities(versions) {
      const packed = manifest.weights?.format === "mei-q4-packed-v1";
      const caps = {
        package_id: manifest.package_id,
        release_class: manifest.release_class,
        inference: packed,
        protocol: true,
        heads,
        missing_or_untrained_heads: missing,
        hash_verified: Boolean(verifyHashes),
        versions,
      };
      if (packed) {
        caps.quantized_only = true;
      }
      return caps;
    },
  };
}
