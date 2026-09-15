import {WebGPU51m} from '/gpu/webgpu-51m.mjs';
import {loadModel} from '/sdk/wasm-abi.mjs';
const argmax=a=>{let j=0;for(let i=1;i<a.length;i++)if(a[i]>a[j])j=i;return j;};
const median=a=>{a=[...a].sort((x,y)=>x-y);return a[Math.floor(a.length/2)];};
const sha=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
self.onmessage=async({data})=>{let gpu,cpu;try{
 const sourcePaths=['/gpu/webgpu-51m.mjs','/bench-worker.mjs','/sdk/wasm-abi.mjs'];
 const sourceHashes={};for(const p of sourcePaths)sourceHashes[p]=await sha(await fetch(p).then(r=>r.arrayBuffer()));
 const base=performance.now();const manifest=await fetch('/assets/mei-model.json').then(r=>r.json());
 const weights=new Uint8Array(await fetch('/assets/tensors.bin').then(r=>r.arrayBuffer()));
 gpu=await WebGPU51m.create(manifest,weights);await gpu.prepareRouting(weights);
 postMessage({progress:'GPU 权重已加载，首次编译与数值检查'});
 const load_ms=performance.now()-base;const st=performance.now();let first=await gpu.forward([2]);const first_ms=performance.now()-st;
 const report={schema:'mei-browser-backend-experiment-v2',date:new Date().toISOString(),userAgent:navigator.userAgent,adapter:gpu.adapter,weights_sha256:gpu.hash,source_hashes:sourceHashes,gpu_allocated_bytes:gpu.bytes,gpu_storage:'dense-f32-from-CQ2; f32 storage of int8-QDQ KV',scope:'Both backends: 27-layer LM + mHC + Engram + tied output, raw numerical generation only; no heads/grammar/tools or cross-request prefix reuse. CPU per-step ABI timing; GPU includes full logits readback. No release claim.',load_ms,first_use_forward_ms:first_ms,first_use_note:'May reuse browser/driver shader cache; not guaranteed cold shader compilation.',first_token:argmax(first),first_logits:Array.from(first)};
 if(data.mode==='smoke'){
  const s=performance.now();const second=await gpu.forward([37]);report.second_ms=performance.now()-s;report.second_logits=Array.from(second);gpu.reset();report.two_token_logits=Array.from(await gpu.forward([2,37]));
  const ids=Array.from({length:128},(_,i)=>i===0?2:(i*137+37)%24000);gpu.reset();const ref=await gpu.forward(ids);report.varied128_logits=Array.from(ref);report.next128_logits=Array.from(await gpu.forward([37]));report.repeat_max_abs=[];
  for(let i=0;i<3;i++){gpu.reset();await gpu.forward([2,37]);gpu.reset();const v=await gpu.forward(ids);const diff=Math.max(...v.map((x,j)=>Math.abs(x-ref[j])));report.repeat_max_abs.push(diff);if(diff>0.00001)throw Error('GPU repeated input unstable: '+diff);}
  report.gpu_allocated_bytes=gpu.bytes;postMessage({report});return;
 }
 const wasmBytes=await fetch('/numeric.wasm').then(r=>r.arrayBuffer());const wasmHash=await sha(wasmBytes);
 const {instance}=await WebAssembly.instantiate(wasmBytes,{});const [vocab,toolIndex]=await Promise.all(['tokenizer.model','tool-index.json'].map(async f=>new Uint8Array(await fetch('/assets/'+f).then(r=>r.arrayBuffer()))));
 cpu=loadModel({manifest,weights,vocab,toolIndex,wasm:instance});
 const x=instance.exports,ip=x.mei_sdk_wasm_alloc(512*4),op=x.mei_sdk_wasm_alloc(24000*4);
 const cpuForward=(ids,reset)=>{new Uint32Array(x.memory.buffer,ip,ids.length).set(ids);const code=x.mei_sdk_wasm_numeric_forward(ip,ids.length,reset?1:0,op,24000);if(code)throw Error('CPU numeric ABI rejected input '+code);return new Float32Array(x.memory.buffer,op,24000).slice();};
 // Guards are tested before timed work; previous model state must not survive reset.
 if(x.mei_sdk_wasm_numeric_forward(ip,0,1,op,24000)===0)throw Error('Empty input accepted');
 if(x.mei_sdk_wasm_numeric_forward(ip,513,1,op,24000)===0)throw Error('Oversized batch accepted');
 if(x.mei_sdk_wasm_numeric_forward(ip,1,1,op,1)===0)throw Error('Small output accepted');
 report.numeric_abi_guards_passed=true;report.cpu_wasm_sha256=wasmHash;report.rounds=[];
 const inputs=len=>Array.from({length:len},(_,i)=>i===0?2:(i*137+37)%24000);
 const run=async(backend,len)=>{
  if(backend==='gpu')gpu.reset();const forward=async(ids,reset)=>backend==='cpu'?cpuForward(ids,reset):gpu.forward(ids);
  const start=performance.now();let logits=await forward(inputs(len),true);const prefill=performance.now()-start;const ids=[argmax(logits)],steps=[];
  for(let i=1;i<32;i++){const s=performance.now();logits=await forward([ids.at(-1)],false);steps.push(performance.now()-s);ids.push(argmax(logits));}
  return {backend,len,prefill_plus_first_ms:prefill,decode_tok_s:31000/steps.reduce((a,b)=>a+b,0),total32_ms:performance.now()-start,step_ms:steps,generated:ids,...(backend==='cpu'?{heap_bytes:instance.exports.memory.buffer.byteLength}:{})};
 };
 for(const len of [1,128,512]){
  postMessage({progress:`预热 ${len}-token 输入与连续解码`});await run('gpu',len);await run('cpu',len);
  for(let round=0;round<3;round++){
   for(const backend of round%2?['gpu','cpu']:['cpu','gpu'])report.rounds.push({...await run(backend,len),round});
   postMessage({progress:`已完成 ${len} tokens 第 ${round+1}/3 轮 CPU/GPU 对测`});
  }
 }
 report.inputs={kind:'deterministic varied token IDs; not a natural-language quality test',formula:'i == 0 ? 2 : (i * 137 + 37) % 24000'};
 report.summary=[];for(const backend of ['cpu','gpu'])for(const len of [1,128,512]){const r=report.rounds.filter(x=>x.backend===backend&&x.len===len);report.summary.push({backend,len,prefill_plus_first_ms:median(r.map(x=>x.prefill_plus_first_ms)),prefill_effective_tok_s:len*1000/median(r.map(x=>x.prefill_plus_first_ms)),decode_tok_s:median(r.map(x=>x.decode_tok_s)),total32_ms:median(r.map(x=>x.total32_ms))});}
 for(const p of sourcePaths)if(sourceHashes[p]!==await sha(await fetch(p).then(r=>r.arrayBuffer())))throw Error('Source changed during benchmark: '+p);
 report.gpu_allocated_bytes=gpu.bytes;delete report.first_logits;x.mei_sdk_wasm_free(ip,512*4);x.mei_sdk_wasm_free(op,24000*4);postMessage({report});
 }catch(e){postMessage({error:String(e.stack||e)});}finally{gpu?.close();cpu?.unload();}};
