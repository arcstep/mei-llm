import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const here = dirname(fileURLToPath(import.meta.url));
export const PLATFORM_ROOT = join(here, "..");
export const SDK_ROOT = join(PLATFORM_ROOT, "_shared");
export const SPEC_DIR = join(SDK_ROOT, "spec");

export function loadSpec(name) {
  return JSON.parse(readFileSync(join(SPEC_DIR, name), "utf8"));
}

const versions = loadSpec("versions.json");
const protocol = loadSpec("protocol.json");
const errors = loadSpec("errors.json");

export function sdkVersions() {
  return {
    sdk_semver: versions.sdk_semver,
    wire_version: versions.wire_version,
    model_package_version: versions.model_package_version,
    runtime_abi_version: versions.runtime_abi_version,
    protocol_id: versions.protocol_id,
    serializer_id: versions.serializer_id,
    release_class: versions.release_class,
    product: versions.product,
  };
}

export const PROTOCOL = protocol;
export const ERRORS = Object.fromEntries(errors.codes.map((row) => [row.id, row]));
