import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { VRMLoaderPlugin, VRMUtils } from "@pixiv/three-vrm";

const BANK_URL = "../../banks/needle-vrm-agent-v0/eval-bank-v0.jsonl";
const CLIP_URL = "./clip-map.json";
const DEFAULT_VRM =
  "https://pixiv.github.io/three-vrm/packages/three-vrm/examples/models/VRM1_Constraint_Twist_Sample.vrm";

const places = {
  kitchen: new THREE.Vector3(-1.6, 0, -0.4),
  living: new THREE.Vector3(1.4, 0, -0.2),
  entry: new THREE.Vector3(0, 0, 1.6),
  trash: new THREE.Vector3(1.6, 0, 1.2),
  user: new THREE.Vector3(0, 0, 2.2),
};

const state = {
  items: [],
  clips: {},
  clock: new THREE.Clock(),
  anim: null,
  vrm: null,
  rig: null,
  world: {},
};

const metaEl = document.getElementById("meta");
const listEl = document.getElementById("list");
const hudGold = document.getElementById("hud-gold");
const hudItem = document.getElementById("hud-item");
const stage = document.getElementById("stage");

function parseJsonl(text) {
  return text
    .split(/\n/)
    .map((l) => l.trim())
    .filter(Boolean)
    .map((l) => JSON.parse(l));
}

async function loadText(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} ${res.status}`);
  return res.text();
}

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
stage.appendChild(renderer.domElement);

const scene = new THREE.Scene();
scene.background = new THREE.Color(0x0b0d12);
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 40);
camera.position.set(0, 1.6, 4.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(0, 0.9, 0);

scene.add(new THREE.HemisphereLight(0xc8d6ff, 0x1a140c, 1.1));
const dir = new THREE.DirectionalLight(0xffffff, 1.2);
dir.position.set(2, 4, 3);
scene.add(dir);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(4.2, 48),
  new THREE.MeshStandardMaterial({ color: 0x1a1f28, roughness: 0.9 }),
);
floor.rotation.x = -Math.PI / 2;
scene.add(floor);

const markers = {};
function addBox(id, color, pos, size) {
  const mesh = new THREE.Mesh(
    new THREE.BoxGeometry(...size),
    new THREE.MeshStandardMaterial({ color, roughness: 0.55 }),
  );
  mesh.position.copy(pos);
  mesh.position.y = size[1] / 2;
  scene.add(mesh);
  markers[id] = mesh;
  return mesh;
}
addBox("front", 0x8aa0c8, new THREE.Vector3(-0.7, 0, 1.7), [0.7, 1.5, 0.08]);
addBox("back", 0x8aa0c8, new THREE.Vector3(0.7, 0, -1.8), [0.7, 1.5, 0.08]);
addBox("trash", 0x3d6b4f, new THREE.Vector3(1.6, 0, 1.2), [0.35, 0.55, 0.35]);

function makeLamp(x, z) {
  const mesh = new THREE.Mesh(
    new THREE.SphereGeometry(0.12, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xfff2b0, emissive: 0xffee88, emissiveIntensity: 1.4 }),
  );
  mesh.position.set(x, 1.5, z);
  scene.add(mesh);
  return mesh;
}
markers.kitchen_light = makeLamp(-1.6, -0.4);
markers.living_light = makeLamp(1.4, -0.2);

function makeCapsuleRig() {
  const g = new THREE.Group();
  const body = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.22, 0.7, 6, 12),
    new THREE.MeshStandardMaterial({ color: 0xd7c4a8 }),
  );
  body.position.y = 0.8;
  const head = new THREE.Mesh(
    new THREE.SphereGeometry(0.18, 16, 16),
    new THREE.MeshStandardMaterial({ color: 0xe8d7c3 }),
  );
  head.position.y = 1.42;
  const arm = new THREE.Mesh(
    new THREE.CapsuleGeometry(0.05, 0.35, 4, 8),
    new THREE.MeshStandardMaterial({ color: 0xd7c4a8 }),
  );
  arm.position.set(0.32, 1.05, 0);
  g.add(body, head, arm);
  g.userData = { head, arm, body };
  scene.add(g);
  return g;
}

function resize() {
  const w = stage.clientWidth;
  const h = Math.max(stage.clientHeight, 1);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
  renderer.setSize(w, h, false);
}
window.addEventListener("resize", resize);
resize();

function setLamp(id, on) {
  const m = markers[id];
  if (!m?.material) return;
  m.material.emissiveIntensity = on ? 1.4 : 0.05;
  m.material.color.set(on ? 0xfff2b0 : 0x334155);
  state.world[id] = on;
}

function setDoor(id, open) {
  const m = markers[id];
  if (!m) return;
  m.rotation.y = open ? -Math.PI / 2.2 : 0;
  state.world[id] = open;
}

function resetWorld() {
  setLamp("kitchen_light", true);
  setLamp("living_light", true);
  setDoor("front", false);
  setDoor("back", false);
  if (!state.rig) return;
  state.rig.position.set(0, 0, 0);
  state.rig.rotation.set(0, 0, 0);
  const { head, arm, body } = state.rig.userData || {};
  if (head?.rotation) head.rotation.set(0, 0, 0);
  if (arm?.rotation) arm.rotation.set(0, 0, 0);
  if (body) {
    body.rotation.set(0, 0, 0);
    body.scale.set(1, 1, 1);
  }
}

function playCall(call) {
  if (!call) {
    state.anim = { kind: "idle", args: {}, t0: state.clock.elapsedTime, clip: {} };
    return;
  }
  const name = call.name || "";
  const args = call.arguments || {};
  const clip = state.clips[name] || { kind: "idle" };
  const kind = clip.kind;
  state.anim = { kind, args, t0: state.clock.elapsedTime, clip };
  if (kind === "switch") setLamp(args.id, !!args.on);
  if (kind === "door") setDoor(args.door, !!clip.open);
}

function tickAnim(dt, t) {
  const a = state.anim;
  const rig = state.rig;
  if (!a || !rig) return;
  const { head, arm, body } = rig.userData || {};
  const u = Math.min(1, (t - a.t0) / 0.7);
  if (a.kind === "head_pitch" && head?.rotation) head.rotation.x = Math.sin(t * 8) * 0.35;
  if (a.kind === "head_yaw" && head?.rotation) head.rotation.y = Math.sin(t * 7) * 0.5;
  if (a.kind === "wave" && arm?.rotation) arm.rotation.z = -0.4 + Math.sin(t * 10) * 0.7;
  if (a.kind === "point" && arm?.rotation) {
    const dir = a.args.target === "left" ? 0.9 : a.args.target === "right" ? -0.9 : 0.1;
    arm.rotation.z = dir;
    arm.rotation.x = a.args.target === "user" ? -0.6 : -0.2;
  }
  if (a.kind === "bow" && body?.rotation) body.rotation.x = Math.sin(Math.min(Math.PI, u * Math.PI)) * 0.45;
  if (a.kind === "sit" && body) {
    body.scale.y = 0.72;
    rig.position.y = -0.12;
  }
  if (a.kind === "stand" && body) {
    body.scale.y = 1;
    rig.position.y = 0;
  }
  if (a.kind === "walk") {
    const key = a.args.place || a.clip.place || "entry";
    const dest = places[key] || places.entry;
    rig.position.lerp(dest, 1 - Math.pow(0.02, dt));
  }
  if (a.kind === "idle") {
    if (arm?.rotation) arm.rotation.set(0, 0, 0);
    if (head?.rotation) head.rotation.set(0, 0, 0);
  }
}

function selectItem(item, btn) {
  for (const el of listEl.querySelectorAll("button.item")) el.classList.remove("active");
  btn?.classList.add("active");
  const calls = item.gold?.function_calls || [];
  hudGold.textContent = JSON.stringify(calls, null, 2);
  hudItem.textContent = JSON.stringify(
    { item_id: item.item_id, family: item.family, scene: item.scene || null, query: item.query },
    null,
    2,
  );
  metaEl.innerHTML = `<span class="badge">${item.item_id}</span>${item.family}${
    item.scene ? ` · ${item.scene}` : ""
  }<br>${item.query}`;
  resetWorld();
  if (item.scene && item.scene.includes("前门开着")) setDoor("front", true);
  if (item.scene && item.scene.includes("后门开着")) setDoor("back", true);
  playCall(calls[0] || null);
}

function renderList(items) {
  listEl.innerHTML = "";
  for (const item of items) {
    const empty = !(item.gold?.function_calls || []).length;
    const btn = document.createElement("button");
    btn.className = "item" + (empty ? " empty" : "");
    btn.innerHTML = `<span class="id">${item.item_id}</span> ${item.query}`;
    btn.addEventListener("click", () => selectItem(item, btn));
    listEl.appendChild(btn);
  }
}

async function tryLoadVrm(url) {
  try {
    const loader = new GLTFLoader();
    loader.register((parser) => new VRMLoaderPlugin(parser));
    const gltf = await loader.loadAsync(url);
    const vrm = gltf.userData.vrm;
    if (!vrm) throw new Error("no vrm on gltf");
    VRMUtils.removeUnnecessaryVertices(gltf.scene);
    vrm.scene.rotation.y = Math.PI;
    vrm.scene.userData = { head: vrm.scene, arm: vrm.scene, body: vrm.scene };
    scene.add(vrm.scene);
    state.vrm = vrm;
    state.rig = vrm.scene;
    return true;
  } catch (err) {
    console.warn("VRM load skipped", err);
    return false;
  }
}

function loop() {
  const dt = state.clock.getDelta();
  const t = state.clock.elapsedTime;
  tickAnim(dt, t);
  if (state.vrm?.update) state.vrm.update(dt);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(loop);
}

async function main() {
  const params = new URLSearchParams(location.search);
  const vrmUrl = params.get("vrm") || DEFAULT_VRM;
  const [bankText, clip] = await Promise.all([loadText(BANK_URL), loadText(CLIP_URL).then(JSON.parse)]);
  state.items = parseJsonl(bankText);
  state.clips = clip;
  const ok = await tryLoadVrm(vrmUrl);
  if (!ok) {
    state.rig = makeCapsuleRig();
    metaEl.textContent = "VRM unavailable — capsule fallback";
  }
  renderList(state.items);
  const first = listEl.querySelector("button.item");
  if (state.items[0]) selectItem(state.items[0], first);
  loop();
}

main().catch((err) => {
  metaEl.textContent = String(err);
  console.error(err);
});
