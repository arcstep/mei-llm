// Offline, CPU-only index preparation. Never modifies canonical package bytes.
import {readFile,writeFile,mkdir,copyFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';
import {spawnSync} from 'node:child_process';
import {fileURLToPath} from 'node:url';
import {resolve,join} from 'node:path';
import {loadModel} from '../../platform/browser-sdk/browser.mjs';
import {TOOLS} from './catalog.mjs';
const root=fileURLToPath(new URL('../../../',import.meta.url));
const pkg=join(root,'models/mei-1.2-51m/exp-00900m/products/mei-1.2-51m-cpt900m-tool-sft-cq2-v1');
const wasmPath=join(root,'models/mei-1.2-51m/runtime/wasm/mei_sdk_wasm-v1.wasm');
const sha=b=>createHash('sha256').update(b).digest('hex');
const [manifestBytes,wasm]=await Promise.all([readFile(join(pkg,'mei-model.json')),readFile(wasmPath)]);
const manifest=JSON.parse(manifestBytes);
for(const row of manifest.files) {const b=await readFile(join(pkg,row.path));if(sha(b)!==row.sha256)throw Error('Source hash mismatch: '+row.path);}
const wm=JSON.parse(await readFile(join(root,'models/mei-1.2-51m/runtime/wasm/manifest.json')));
if(sha(wasm)!==wm.sha256)throw Error('WASM hash mismatch');
const id='data-check-'+sha(JSON.stringify(TOOLS)+sha(manifestBytes)+sha(wasm)).slice(0,12);
const out=join(root,'.local/cache/data-check',id);
let cached=false;try{await readFile(join(out,'receipt.json'));cached=true;}catch{}
if(cached){
 const saved=JSON.parse(await readFile(join(out,'mei-model.json')));
 if(saved.tensor_container.sha256!==manifest.tensor_container.sha256||sha(await readFile(join(out,'runtime.wasm')))!==sha(wasm))throw Error('Cached adoption mismatch');
 for(const row of saved.files){const b=await readFile(join(out,row.path));if(b.length!==row.nbytes||sha(b)!==row.sha256)throw Error('Cached adoption corrupt: '+row.path);}
 if(JSON.stringify(JSON.parse(await readFile(join(out,'catalog.json'))))!==JSON.stringify(TOOLS))throw Error('Cached catalog mismatch');
 console.log(out);process.exit(0);
}
await mkdir(out,{recursive:true});
const {instance}=await WebAssembly.instantiate(wasm,{});
const model=loadModel({manifest,weights:await readFile(join(pkg,'tensors.bin')),vocab:await readFile(join(pkg,'tokenizer.model')),toolIndex:await readFile(join(pkg,'tool-index.json')),wasm:instance});
function diagnose(text) {
 const e=instance.exports,b=new TextEncoder().encode(JSON.stringify({text})+'\0');
 const p=e.mei_sdk_wasm_alloc(b.length), o=e.mei_sdk_wasm_alloc(4);
 new Uint8Array(e.memory.buffer,p,b.length).set(b);
 try {
  const rc=e.mei_sdk_wasm_diagnose_heads(p,o); if(rc)throw Error('diagnose_heads '+rc);
  const ptr=new DataView(e.memory.buffer).getUint32(o,true);let end=ptr;
  const mem=new Uint8Array(e.memory.buffer);while(mem[end])end++;
  const result=JSON.parse(new TextDecoder().decode(mem.subarray(ptr,end)));
  e.mei_sdk_wasm_string_free(ptr);return result.contrastive_embedding;
 } finally{e.mei_sdk_wasm_free(p,b.length);e.mei_sdk_wasm_free(o,4);}
}
const vectors={};
for(const t of TOOLS) {
 const text=JSON.stringify({description:t.description,name:t.name,parameters:t.parameters},function(k,v){if(v&&typeof v==='object'&&!Array.isArray(v))return Object.fromEntries(Object.keys(v).sort().map(k=>[k,v[k]]));return v;});
 console.log('Embedding',t.name);vectors[t.name]=diagnose(text);
}
await writeFile(join(out,'vectors.json'),JSON.stringify(vectors));
await writeFile(join(out,'catalog.json'),JSON.stringify(TOOLS));
const code=`import sys,json,pathlib\nsys.path.insert(0,sys.argv[1]+'/src/platform/_shared/runtime')\nfrom tool_index import ToolIndex\np=pathlib.Path(sys.argv[2]); old=json.loads(pathlib.Path(sys.argv[3]).read_text()); v=json.loads((p/'vectors.json').read_text()); tools=json.loads((p/'catalog.json').read_text())\ni=ToolIndex(model_hash=old['model_sha256'],head_hash=old['head_sha256'],tokenizer_hash=old['tokenizer_sha256'])\ni.build(tools,lambda text:v[json.loads(text)['name']])\n(p/'tool-index.json').write_text(json.dumps(i.as_dict(),ensure_ascii=False,separators=(',',':')))\n`;
const result=spawnSync(join(root,'.venv/bin/python'),['-c',code,root,out,join(pkg,'tool-index.json')],{encoding:'utf8'});if(result.status)throw Error(result.stderr);
const ib=await readFile(join(out,'tool-index.json'));
manifest.parent_package_id=manifest.package_id;manifest.package_id=id;
// Keep candidate gates and every head; only change the catalog binding.
const indexRow=manifest.files.find(r=>r.role==='tool_index');indexRow.nbytes=ib.length;indexRow.sha256=sha(ib);
for(const row of manifest.files)if(row.role!=='tool_index')await copyFile(join(pkg,row.path),join(out,row.path));
await copyFile(wasmPath,join(out,'runtime.wasm'));
await writeFile(join(out,'mei-model.json'),JSON.stringify(manifest));
await writeFile(join(out,'receipt.json'),JSON.stringify({id,source_package:pkg,source_manifest_sha256:sha(manifestBytes),tensor_sha256:manifest.tensor_container.sha256,wasm_sha256:sha(wasm),index_sha256:sha(ib),catalog:TOOLS,embedding_source:'same-package Rust/WASM contrastive head; offline preparation',release_eligible:false,weights_changed:false,created_at:new Date().toISOString()},null,2));
console.log(out);
