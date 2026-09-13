// Diagnostic only: same bytes, in-memory experimental manifest permits raw probes.
// Never changes the canonical package or counts kernel throughput as task success.
const {chromium}=require(process.env.PLAYWRIGHT_PATH||'/Users/xuehongwei/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs/promises');
const os=require('node:os');
(async()=>{
 const browser=await chromium.launch({headless:true});
 try {
  const page=await browser.newPage();
  await page.goto('http://127.0.0.1:8765');
  const result=await page.evaluate(async()=>{
   const {loadModel}=await import('/sdk/wasm-abi.mjs');
   const manifest=await fetch('/assets/mei-model.json').then(r=>r.json());
   const [wasm,weights,vocab,toolIndex]=await Promise.all(['runtime.wasm','tensors.bin','tokenizer.model','tool-index.json'].map(async f=>new Uint8Array(await fetch('/assets/'+f).then(r=>r.arrayBuffer()))));
   const {instance}=await WebAssembly.instantiate(wasm,{});
   const model=loadModel({manifest:{...manifest,release_class:'experimental'},weights,vocab,toolIndex,wasm:instance});
   const probe=(max_new,promptLength=1)=>{
    const s=model.createSession();const start=performance.now();
    try {
     const r=s.complete({wire_version:'mei-runtime-wire-v2',query:'kernel-throughput-diagnostic',context:{},evidence:[],history:[],tool_results:[],permissions:{},state:{},decode_mode:'raw',max_new,token_ids:Array(promptLength).fill(2)});
     return {elapsed_ms:performance.now()-start,stats:r.stats,generated:r.generated_token_ids,cache:r.runtime_cache};
    }finally{s.close();}
   };
   try {
    const warmup=probe(32);const pairs=[];
    for(let i=0;i<3;i++) {
     const short=probe(1),long=probe(32);
     const tokens=long.stats.output_tokens-short.stats.output_tokens;
     if(tokens!==31||short.generated.some((t,j)=>long.generated[j]!==t))throw Error('Invalid matched decode probe');
     pairs.push({short,long,decode_tok_s:tokens*1000/(long.elapsed_ms-short.elapsed_ms)});
    }
    const contextual={short:probe(1,512),long:probe(32,512)};
    contextual.decode_tok_s=(contextual.long.stats.output_tokens-contextual.short.stats.output_tokens)*1000/(contextual.long.elapsed_ms-contextual.short.elapsed_ms);
    return {warmup,pairs,contextual,heap_bytes:instance.exports.memory.buffer.byteLength,user_agent:navigator.userAgent,receipt:await fetch('/assets/receipt.json').then(r=>r.json())};
   }finally{model.unload();}
  });
  result.measured_at=new Date().toISOString();result.cpu=os.cpus()[0].model;result.browser=browser.version();
  result.contract='diagnostic-only; in-memory experimental release override; same package bytes; raw 1-vs-32 tokens; 3 warm pairs plus synthetic 512-token context; no task quality claim';
  result.concurrent_load='Existing training processes left running; not an isolated hardware benchmark.';
  const out='src/demos/data-check/validation/2026-09-13-performance.json';
  await fs.writeFile(out,JSON.stringify(result,null,2)+'\n');
  console.log(JSON.stringify({out,cpu:result.cpu,browser:result.browser,decode_tok_s:result.pairs.map(p=>p.decode_tok_s),contextual_decode_tok_s:result.contextual.decode_tok_s,prefill512_plus_one_ms:result.contextual.short.elapsed_ms},null,2));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
