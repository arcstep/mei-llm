// Alternating A/B/B/A measurements in one browser; no simultaneous inference.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/xuehongwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs/promises'),crypto=require('node:crypto');
(async()=>{
 const paths={baseline:'models/mei-1.2-51m/runtime/wasm/mei_sdk_wasm-v1.wasm',candidate:process.env.MEI_BENCH_WASM};
 const browser=await chromium.launch({headless:true});
 try {
  const context=await browser.newContext({serviceWorkers:'block'});const hashes={};
  for(const [key,path] of Object.entries(paths)){const bytes=await fs.readFile(path);hashes[key]=crypto.createHash('sha256').update(bytes).digest('hex');await context.route(`**/speed-${key}.wasm`,r=>r.fulfill({body:bytes,contentType:'application/wasm'}));}
  const page=await context.newPage();await page.goto('http://127.0.0.1:8765');
  await page.evaluate(async()=>{
   const {loadModel}=await import('/sdk/wasm-abi.mjs');const {TOOLS,SAMPLE}=await import('/catalog.mjs');
   const manifest=await fetch('/assets/mei-model.json').then(r=>r.json());
   const [weights,vocab,toolIndex]=await Promise.all(['tensors.bin','tokenizer.model','tool-index.json'].map(async f=>new Uint8Array(await fetch('/assets/'+f).then(r=>r.arrayBuffer()))));
   globalThis.runVariant=async key=>{
    const wasm=await fetch(`/speed-${key}.wasm`).then(r=>r.arrayBuffer());const {instance}=await WebAssembly.instantiate(wasm,{});
    const load=diagnostic=>{const m=loadModel({manifest:diagnostic?{...manifest,release_class:'experimental'}:manifest,weights,vocab,toolIndex,wasm:instance});if(!diagnostic)m.registerTools(TOOLS);return m;};
    let model=load(true);
    const request=(raw,max_new=128,len=1)=>{
     const s=model.createSession({max_steps:2,runtime_profile:'compact'});const start=performance.now();
     try{const result=s.complete({wire_version:'mei-runtime-wire-v2',query:'检查订单号列是否存在空值',context:raw?{}:{columns:SAMPLE[0]},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:raw?'raw':'constrained',max_new,...(raw?{token_ids:Array(len).fill(2)}:{})});return {ms:performance.now()-start,stats:result.stats,generated:result.generated_token_ids,kind:result.kind,cache:result.runtime_cache};}
     finally{s.close();}
    };
    try{
     request(true,32);const decode=[];
     for(let i=0;i<3;i++){const short=request(true,1),long=request(true,32);if(long.stats.output_tokens-short.stats.output_tokens!==31)throw Error('Invalid throughput probe');decode.push(31000/(long.ms-short.ms));}
     const prefill=request(true,1,512);model.unload();model=load(false);
     const cold=request(false),warm=request(false);
     return {key,decode_tok_s:decode,prefill512_plus_one_ms:prefill.ms,task_cold_ms:cold.ms,task_warm_ms:warm.ms,cold,warm,heap_mib:instance.exports.memory.buffer.byteLength/1048576};
    }finally{model.unload();}
   };
  });
  const rounds=[];
  for(const key of ['baseline','candidate','candidate','baseline']){const r=await page.evaluate(key=>runVariant(key),key);rounds.push(r);console.log(JSON.stringify({key,decode:r.decode_tok_s,prefill:r.prefill512_plus_one_ms,cold:r.task_cold_ms,warm:r.task_warm_ms,heap:r.heap_mib}));}
  const median=xs=>{xs=[...xs].sort((a,b)=>a-b);const m=Math.floor(xs.length/2);return xs.length%2?xs[m]:(xs[m-1]+xs[m])/2;};
  const summary={};for(const key of Object.keys(paths)){const rs=rounds.filter(r=>r.key===key);summary[key]={decode_tok_s:median(rs.flatMap(r=>r.decode_tok_s)),prefill512_plus_one_ms:median(rs.map(r=>r.prefill512_plus_one_ms)),task_cold_ms:median(rs.map(r=>r.task_cold_ms)),task_warm_ms:median(rs.map(r=>r.task_warm_ms)),peak_heap_mib:Math.max(...rs.map(r=>r.heap_mib))};}
  const report={date:new Date().toISOString(),browser:browser.version(),hashes,summary,rounds,scope:'Kernel throughput uses in-memory experimental manifest only. Task probes use original candidate policy. Other host processes remain running. No accuracy claim.'};
  const out=process.env.MEI_BENCH_OUTPUT;await fs.writeFile(out,JSON.stringify(report,null,2)+'\n',{flag:'wx'});console.log(JSON.stringify({out,summary},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
