import {DecodeGPU51m} from '/gpu/webgpu-decode.mjs';
const sha=async b=>[...new Uint8Array(await crypto.subtle.digest('SHA-256',b))].map(x=>x.toString(16).padStart(2,'0')).join('');
const argmax=a=>{let j=0;for(let i=1;i<a.length;i++)if(a[i]>a[j])j=i;return j;};
const median=a=>[...a].sort((a,b)=>a-b)[Math.floor(a.length/2)];
function diff(a,b){let max=0;for(let i=0;i<a.length;i++){if(!Number.isFinite(a[i])||!Number.isFinite(b[i]))throw Error('Nonfinite logits');max=Math.max(max,Math.abs(a[i]-b[i]));}return max;}
const inputs=n=>Array.from({length:n},(_,i)=>i===0?2:(i*137+37)%24000);
self.onmessage=async()=>{let gpu;try{
 const paths=['/gpu/webgpu-51m.mjs','/gpu/webgpu-parallel.mjs','/gpu/webgpu-decode.mjs','/decode-worker.mjs'],hashes={};for(const p of paths)hashes[p]=await sha(await fetch(p).then(r=>r.arrayBuffer()));
 const manifest=await fetch('/assets/mei-model.json').then(r=>r.json()),weights=new Uint8Array(await fetch('/assets/tensors.bin').then(r=>r.arrayBuffer()));
 gpu=await DecodeGPU51m.create(manifest,weights);const report={schema:'mei-gpu-decode-validation-v1',date:new Date().toISOString(),hashes,weights_sha256:gpu.hash,adapter:gpu.adapter,userAgent:navigator.userAgent,scope:'raw full LM; no heads/grammar/tools; real same weights; warm alternating 5 rounds, no prefix cache; compact is greedy selection, not general sampler',runs:[],checks:[],guards:[]};
 const configs=[{name:'baseline',routingMode:null},{name:'fused-logits',routingMode:'register',fusePost:true,fuseProjection:true,fuseRouting:true},{name:'fused-token',routingMode:'register',fusePost:true,fuseProjection:true,fuseRouting:true,compact:true}];report.configurations=configs;
 const configure=c=>{gpu.routingMode=c.routingMode;gpu.fusePost=!!c.fusePost;gpu.fuseProjection=!!c.fuseProjection;gpu.fuseRouting=!!c.fuseRouting;};
 gpu.profileMode='none';
 // GPU argmax: ties select lowest id, all-negative values and last-vocabulary id work;
 // any NaN/Infinity produces the invalid-id sentinel, never a valid action token.
 for(const name of ['ties','negative','last','nan','infinity']){
  const x=new Float32Array(24000).fill(name==='ties'?0:-10);if(name==='negative')x[137]=-1;if(name==='last')x[23999]=1;if(name==='nan')x[251]=NaN;if(name==='infinity')x[23999]=Infinity;
  gpu.device.queue.writeBuffer(gpu.logits,0,x);gpu.labels=[];gpu.beginTime=performance.now();gpu.compactRead=true;const got=(await gpu.read(gpu.device.createCommandEncoder()))[0];gpu.compactRead=false;
  const expected=({ties:0,negative:137,last:23999,nan:0xffffffff,infinity:0xffffffff})[name];if(got!==expected)throw Error('argmax guard '+name+' '+got);report.guards.push({name,passed:true});
 }
 for(const ids of [[],[24000],[NaN],Array(513).fill(2)]){let rejected=false;try{await gpu.forward(ids);}catch(e){rejected=true;}if(!rejected)throw Error('Invalid input accepted');}report.guards.push({name:'empty,batch,vocab,NaN inputs rejected',passed:true});
 async function run(config,profile,len){configure(config);gpu.profileMode=profile;gpu.reset();const begin=performance.now();let out=config.compact?await gpu.forwardToken(inputs(len)):await gpu.forward(inputs(len));const prefill=performance.now()-begin;const generated=[config.compact?out:argmax(out)],steps=[];
  for(let i=0;i<31;i++){const s=performance.now();out=config.compact?await gpu.forwardToken([generated.at(-1)]):await gpu.forward([generated.at(-1)]);steps.push({wall_ms:performance.now()-s,...gpu.lastTiming});generated.push(config.compact?out:argmax(out));}
  return {name:config.name,len,profile,prefill_ms:prefill,decode_tok_s:31000/steps.reduce((a,b)=>a+b.wall_ms,0),total_ms:performance.now()-begin,generated,steps};
 }
 for(const len of [1,128,512]){
  configure(configs[0]);gpu.reset();const ref=await gpu.forward(inputs(len)),refNext=await gpu.forward([37]);
  for(const c of configs){configure(c);gpu.reset();const a=await gpu.forward(inputs(len)),b=await gpu.forward([37]);gpu.reset();const repeat=await gpu.forward(inputs(len));const check={name:c.name,len,prefill_max_abs:diff(ref,a),next_max_abs:diff(refNext,b),repeat_max_abs:diff(a,repeat)};report.checks.push(check);if(check.prefill_max_abs>1e-5||check.next_max_abs>1e-5||check.repeat_max_abs>1e-5)throw Error('Numerical check failed '+JSON.stringify(check));await run(c,'none',len);}
  for(let round=0;round<5;round++)for(const c of round%2?[...configs].reverse():configs){postMessage({progress:`${len} tokens · ${c.name} · ${round+1}/5`});report.runs.push({...await run(c,'none',len),round});}
  const runs=report.runs.filter(r=>r.len===len&&r.profile==='none');if(new Set(runs.map(r=>JSON.stringify(r.generated))).size!==1)throw Error('Generated sequence differs');
  for(const c of [configs[0],configs[2]])report.runs.push(await run(c,'whole',len));
 }
 report.summary=[];for(const c of configs)for(const len of [1,128,512]){const r=report.runs.filter(r=>r.name===c.name&&r.len===len&&r.profile==='none');report.summary.push({name:c.name,len,prefill_ms:median(r.map(x=>x.prefill_ms)),decode_tok_s:median(r.map(x=>x.decode_tok_s)),total_ms:median(r.map(x=>x.total_ms)),dispatches:r[0].steps[0].dispatches});}
 report.gpu_explicit_allocated_bytes=gpu.bytes+32768;report.timestamp_quantization_note='Observed timestamps quantized in 65536 ns increments; sub-kernel durations are coarse. Timestamp-enabled passes measured separately from final throughput.';
 for(const p of paths)if(hashes[p]!==await sha(await fetch(p).then(r=>r.arrayBuffer())))throw Error('Source changed '+p);
 postMessage({report});
}catch(e){postMessage({error:String(e.stack||e)});}finally{gpu?.close();}};
