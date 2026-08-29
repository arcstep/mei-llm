import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import test from "node:test";
import { Engine } from "./engine.mjs";
import { parseV2Text, renderRequest, schemaFingerprint } from "./protocol.mjs";
import { SDK_ROOT, SPEC_DIR, sdkVersions } from "./version.mjs";
import { loadModel } from "./browser.mjs";

const golden = (name) => JSON.parse(readFileSync(join(SPEC_DIR, "golden", name), "utf8"));
const TINY = join(SDK_ROOT, "fixtures/packages/tiny-protocol-v1");

test("versions omit needle2", () => {
  const blob = JSON.stringify(sdkVersions());
  assert.equal(blob.toLowerCase().includes("needle"), false);
  assert.match(sdkVersions().sdk_semver, /experimental/);
});

test("fingerprint matches golden", () => {
  const gold = golden("schema_fingerprint.json");
  assert.equal(schemaFingerprint(gold.tools), gold.sha256);
});

test("parse cases", () => {
  for (const c of golden("parse_cases.json").cases) {
    assert.deepEqual(parseV2Text(c.text), c.parsed);
  }
});

test("render matches golden", () => {
  const gold = golden("render_request.json");
  const got = renderRequest(gold.request, gold.request.oracle_tools);
  assert.equal(got.prompt, gold.prompt);
  assert.equal(got.schema_fingerprint, gold.schema_fingerprint);
});

test("engine turns match golden", () => {
  const gold = golden("turn_results.json");
  const session = Engine.load(TINY).createSession();
  const light = { name: "light.set", parameters: { type: "object", properties: {} } };
  const cases = {
    refuse: { query: "开灯", oracle_tools: [light], candidate_text: "[]" },
    call: { query: "开灯", oracle_tools: [light], candidate_text: '[{"name":"light.set","arguments":{}}]' },
    leak: { query: "gold_route_id=1", oracle_tools: [light], candidate_text: "[]" },
    unavailable: { query: "开灯", oracle_tools: [light] },
  };
  for (const [name, request] of Object.entries(cases)) {
    const got = session.complete(request);
    got.stats.wall_ms = 0;
    assert.deepEqual(got, gold.turns[name], name);
  }
});

test("browser loadModel refuses missing quantized package", () => {
  assert.throws(() => {
    loadModel();
  }, { code: "engine_unavailable" });
});

test("browser loadModel refuses float npz format", () => {
  assert.throws(() => {
    loadModel({
      manifest: { weights: { format: "mlx-npz" } },
      weights: new Uint8Array([1, 2, 3]),
    });
  }, { code: "engine_unavailable" });
});
