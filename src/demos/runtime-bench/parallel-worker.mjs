import {MatrixPool,assets,createCPU} from './parallel-pool.mjs';
import {ParallelGPU51m} from '/gpu/webgpu-parallel.mjs';
const argmax=a=>{let j=0;for(let i=1;i<a.length;i++)if(a[i]>a[j])j=i;return j;};
const sha=async bytes=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
const median=a=>[...a].sort((x,y)=>x-y)[Math.floor(a.length/2)];
const inputs=len=>Array.from({length:len},(_,i)=>i===0?2:(i*137+37)%24000);
function difference(a,b){let max=0,dot=0,aa=0,bb=0;for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('Nonfinite logits');max=Math.max(max,Math.abs(a[i]-b[i]));dot+=a[i]*b[i];aa+=a[i]*a[i];bb+=b[i]*b[i];}return {max_abs:max,cosine:dot/Math.sqrt(aa*bb),same_argmax:argmax(a)===argmax(b)};}
self.onmessage=async({data})=>{let pool,cpu,gpu;try{
 if(!crossOriginIsolated||typeof SharedArrayBuffer==='undefined')throw Error('Requires cross-origin isolation');
 const paths=['/parallel-worker.mjs','/parallel-pool.mjs','/parallel-helper.mjs','/gpu/webgpu-parallel.mjs','/gpu/webgpu-51m.mjs','/sdk/wasm-abi.mjs','/parallel.wasm'];
 const hashes={};for(const p of paths)hashes[p]=await sha(await fetch(p).then(r=>r.arrayBuffer()));
 const report={schema:'mei-browser-parallel-experiment-v1',date:new Date().toISOString(),mode:data.mode,userAgent:navigator.userAgent,hardwareConcurrency:navigator.hardwareConcurrency,crossOriginIsolated,source_hashes:hashes,focused:!!data.focused,scope:'Raw full LM only. Reset KV each trial. Warm code; no prefix/result reuse. CPU separate private heaps + shared matrix input/output, not independent requests. GPU hardware workgroup scheduling, not CPU workers. No accuracy or release claim.',rounds:[],checks:[]};
 const asset=await assets();report.weights_sha256=await sha(asset.weights);const rounds=data.rounds||3,lens=data.lens||[1,128,512],newTokens=data.newTokens||32;
 const run=async(name,len,forward,reset)=>{
  reset?.();pool?.clearStats();let start=performance.now(),a=await forward(inputs(len),true),prefill=performance.now()-start;
  const generated=[argmax(a)],steps=[];
  for(let i=1;i<newTokens;i++){const s=performance.now();a=await forward([generated.at(-1)],false);steps.push(performance.now()-s);generated.push(argmax(a));}
  return {name,len,prefill_plus_first_ms:prefill,decode_tok_s:(newTokens-1)*1000/steps.reduce((a,b)=>a+b,0),total_ms:performance.now()-start,generated,step_ms:steps,...(pool?{matrix_dispatch:{...pool.stats},coordinator_heap_bytes:cpu.x.memory.buffer.byteLength,helper_heap_bytes:pool.heaps.map((initial,i)=>Atomics.load(pool.control,48+i)||initial),shared_bytes:pool.shared.byteLength}:{gpu_allocated_bytes:gpu.bytes})};
 };
 if(data.mode==='cpu'){
  pool=new MatrixPool();postMessage({progress:'加载协调器与 8 个 CPU 矩阵 Worker'});await pool.init(asset,8);cpu=await createCPU(asset,pool);report.worker_abi_guards_passed=pool.guards;
  const names=data.focused?[0,4,'adaptive']:[0,1,2,4,8];report.configurations=names.map(n=>({name:'cpu-'+n,compute_helpers:n==='adaptive'?{prefill:8,decode:4}:n,coordinator_workers:1,private_heap_copies:9,resident_helpers:8,note:n?'Coordinator synchronously waits for disjoint output rows':'Coordinator computes alone; helper workers sleep'}));
  for(const len of lens){
   pool.count=0;const ref=cpu.forward(inputs(len),true),next=cpu.forward([37],false);
   for(const n of names.slice(1)){pool.count=n==='adaptive'?8:n;const a=cpu.forward(inputs(len),true);pool.count=n==='adaptive'?4:n;const b=cpu.forward([37],false);const check={name:'cpu-'+n,len,prefill:difference(ref,a),next:difference(next,b)};report.checks.push(check);if(check.prefill.max_abs>1e-5||check.next.max_abs>1e-5)throw Error('CPU partition changed logits '+JSON.stringify(check));}
   for(const n of names){pool.count=n==='adaptive'?8:n;await run('cpu-'+n,len,(ids,r)=>{pool.count=n==='adaptive'?(r?8:4):n;return cpu.forward(ids,r);});}
   for(let round=0;round<rounds;round++)for(const n of round%2?[...names].reverse():names){pool.count=n==='adaptive'?8:n;postMessage({progress:`CPU ${n} 协同 Worker · 输入 ${len} · 第 ${round+1}/${rounds} 轮`});report.rounds.push({...await run('cpu-'+n,len,(ids,r)=>{pool.count=n==='adaptive'?(r?8:4):n;return cpu.forward(ids,r);}),round});}
  }
 }else{
  const allConfigs=[{name:'gpu-baseline',mv:64,tile:16},{name:'gpu-wg32',mv:32,tile:16},{name:'gpu-wg128',mv:128,tile:16},{name:'gpu-wg256',mv:256,tile:16},{name:'gpu-tile8',mv:64,tile:8},{name:'gpu-tile32',mv:64,tile:32}];
  const configs=data.focused?[allConfigs[0],{name:'gpu-adaptive',mv:64,tile:'auto'}]:allConfigs;
  report.configurations=configs;gpu=await ParallelGPU51m.create(asset.manifest,asset.weights,configs[0]);await gpu.prepareRouting(asset.weights);report.adapter=gpu.adapter;
  for(const len of lens){
   gpu.config=configs[0];gpu.reset();const ref=await gpu.forward(inputs(len)),next=await gpu.forward([37]);
   for(const config of configs){gpu.config=config;gpu.reset();const a=await gpu.forward(inputs(len)),b=await gpu.forward([37]);gpu.reset();const repeat=await gpu.forward(inputs(len));const check={name:config.name,len,prefill:difference(ref,a),next:difference(next,b),repeat:difference(a,repeat)};report.checks.push(check);if(check.repeat.max_abs>1e-5)throw Error('Unstable GPU '+JSON.stringify(check));await run(config.name,len,ids=>gpu.forward(ids),()=>gpu.reset());}
   for(let round=0;round<rounds;round++)for(const config of round%2?[...configs].reverse():configs){gpu.config=config;postMessage({progress:`${config.name} · 输入 ${len} · 第 ${round+1}/${rounds} 轮`});report.rounds.push({...await run(config.name,len,ids=>gpu.forward(ids),()=>gpu.reset()),round});}
  }
 }
 report.summary=[];for(const name of [...new Set(report.rounds.map(r=>r.name))])for(const len of lens){const r=report.rounds.filter(r=>r.name===name&&r.len===len),ms=median(r.map(x=>x.prefill_plus_first_ms));report.summary.push({name,len,prefill_ms:ms,prefill_tok_s:len*1000/ms,decode_tok_s:median(r.map(x=>x.decode_tok_s)),total_ms:median(r.map(x=>x.total_ms))});}
 for(const p of paths)if(hashes[p]!==await sha(await fetch(p).then(r=>r.arrayBuffer())))throw Error('Source changed '+p);
 postMessage({report});
}catch(e){postMessage({error:String(e.stack||e)});}finally{pool?.close();cpu?.close();gpu?.close();}};
