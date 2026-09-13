import {loadModel} from '/sdk/wasm-abi.mjs';
import {TOOLS} from './catalog.mjs';
import {runNode} from './node-task.mjs';
let model;
const json=async url=>{const r=await fetch(url);if(!r.ok)throw Error('加载失败 '+url);return r.json();};
const bytes=async url=>{const r=await fetch(url);if(!r.ok)throw Error('加载失败 '+url);return new Uint8Array(await r.arrayBuffer());};
async function init(){
 if(model)return;
 const manifest=await json('/assets/mei-model.json');
 const [wasm,weights,vocab,toolIndex]=await Promise.all(['runtime.wasm','tensors.bin','tokenizer.model','tool-index.json'].map(p=>bytes('/assets/'+p)));
 const {instance}=await WebAssembly.instantiate(wasm,{});
 model=loadModel({manifest,weights,vocab,toolIndex,wasm:instance});model.registerTools(TOOLS);
}
self.onmessage=async({data})=>{
 const {id,rule,sheet}=data;
 try {
  if(data.type==='init'){await init();self.postMessage({id,ready:true});return;}
  await init();const report=runNode(model,rule,sheet);
  self.postMessage({id,report});
 }catch(e){self.postMessage({id,error:e.message});}
};
