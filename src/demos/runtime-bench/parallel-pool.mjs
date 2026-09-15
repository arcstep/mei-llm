// Experiment only: separate private Rust heaps; shared inputs/results and control.
// Atomics.wait is used only in Workers. The coordinator never shares its allocator.
import {loadModel} from '/sdk/wasm-abi.mjs';
export const CAP=512*512,CONTROL_BYTES=1024,NAME_OFFSET=512;
export async function assets(){
 const [manifest,weights,vocab,toolIndex,wasm]=await Promise.all([
  fetch('/assets/mei-model.json').then(r=>r.json()),
  ...['tensors.bin','tokenizer.model','tool-index.json'].map(f=>fetch('/assets/'+f).then(r=>r.arrayBuffer()).then(b=>new Uint8Array(b))),
  fetch('/parallel.wasm').then(r=>r.arrayBuffer()).then(b=>WebAssembly.compile(b))]);
 return {manifest,weights,vocab,toolIndex,wasm};
}
export class MatrixPool {
 constructor(){this.workers=[];this.count=0;this.seq=0;this.stats={calls:0,copy_ms:0,wait_ms:0};this.shared=new SharedArrayBuffer(CONTROL_BYTES+CAP*8);this.control=new Int32Array(this.shared,0,128);this.input=new Float32Array(this.shared,CONTROL_BYTES,CAP);this.output=new Float32Array(this.shared,CONTROL_BYTES+CAP*4,CAP);}
 async init(asset,count){
  this.heaps=[];this.guards=[];
  await Promise.all(Array.from({length:count},(_,id)=>new Promise((resolve,reject)=>{
   const w=new Worker('/parallel-helper.mjs',{type:'module'});this.workers.push(w);
   const timer=setTimeout(()=>reject(Error('Helper initialization timed out')),30000);
   w.onerror=e=>{clearTimeout(timer);reject(Error(e.message));};
   w.onmessage=({data})=>{clearTimeout(timer);if(data.error)reject(Error(data.error));else{this.heaps[id]=data.heap_bytes;this.guards[id]=data.guards_passed===true;resolve();}};
   w.postMessage({asset,id,shared:this.shared});
  })));
 }
 bind(exports){this.x=exports;}
 imports(){return {mei_parallel:{enabled:()=>this.count?1:0,matmul:(...args)=>this.matmul(...args)}};}
 matmul(np,nl,ip,t,rows,cols,op){
  if(nl>=256||rows%4||t*cols>CAP||t*rows>CAP||cols!==512)return 1;
  const c=this.control;let s=performance.now();
  const name=new Uint8Array(this.shared,NAME_OFFSET,256);name.fill(0);name.set(new Uint8Array(this.x.memory.buffer,np,nl));
  this.input.set(new Float32Array(this.x.memory.buffer,ip,t*cols));c[1]=t;c[2]=rows;c[3]=this.count;c[4]=nl;
  for(let i=0;i<this.count;i++)Atomics.store(c,32+i,0);
  this.stats.copy_ms+=performance.now()-s;s=performance.now();const seq=++this.seq;
  for(let i=0;i<this.count;i++){Atomics.store(c,64+i,seq);Atomics.notify(c,64+i);}
  for(let i=0;i<this.count;i++){
   while(Atomics.load(c,16+i)!==seq){const value=Atomics.load(c,16+i);if(value===seq)break;if(Atomics.wait(c,16+i,value,20000)==='timed-out')throw Error('Matrix worker timed out '+i);}
   if(Atomics.load(c,32+i))throw Error('Matrix worker failed '+i+' code '+c[32+i]);
  }
  this.stats.wait_ms+=performance.now()-s;s=performance.now();new Float32Array(this.x.memory.buffer,op,t*rows).set(this.output.subarray(0,t*rows));this.stats.copy_ms+=performance.now()-s;this.stats.calls++;return 0;
 }
 clearStats(){this.stats={calls:0,copy_ms:0,wait_ms:0};}
 close(){for(const w of this.workers)w.terminate();this.workers=[];}
}
export async function createCPU(asset,pool){
 const instance=await WebAssembly.instantiate(asset.wasm,pool.imports());pool.bind(instance.exports);
 const model=loadModel({...asset,wasm:instance});const x=instance.exports;
 const ip=x.mei_sdk_wasm_alloc(512*4),op=x.mei_sdk_wasm_alloc(24000*4);
 return {x,forward(ids,reset){new Uint32Array(x.memory.buffer,ip,ids.length).set(ids);const rc=x.mei_sdk_wasm_numeric_forward(ip,ids.length,reset?1:0,op,24000);if(rc)throw Error('CPU forward '+rc);return new Float32Array(x.memory.buffer,op,24000).slice();},close(){x.mei_sdk_wasm_free(ip,512*4);x.mei_sdk_wasm_free(op,24000*4);model.unload();}};
}
