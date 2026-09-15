import {loadModel} from '/sdk/wasm-abi.mjs';
import {CAP,CONTROL_BYTES,NAME_OFFSET} from './parallel-pool.mjs';
self.onmessage=async({data:{asset,id,shared}})=>{try{
 const instance=await WebAssembly.instantiate(asset.wasm,{mei_parallel:{enabled:()=>0,matmul:()=>1}});
 const model=loadModel({...asset,wasm:instance}),x=instance.exports;
 const ip=x.mei_sdk_wasm_alloc(CAP*4),op=x.mei_sdk_wasm_alloc(CAP*4),np=x.mei_sdk_wasm_alloc(256);
 const c=new Int32Array(shared,0,128),input=new Float32Array(shared,CONTROL_BYTES,CAP),output=new Float32Array(shared,CONTROL_BYTES+CAP*4,CAP);
 // Reject invalid geometry/capacity before any benchmark work.
 new Uint8Array(x.memory.buffer,np,256).fill(0);new Uint8Array(x.memory.buffer,np,13).set(new TextEncoder().encode('embed.weight'));
 for(const args of [[np,ip,0,0,4,op,CAP],[np,ip,1,1,4,op,CAP],[np,ip,1,0,24004,op,CAP],[np,ip,1,0,4,op,1],[np,ip,513,0,4,op,CAP]])if(x.mei_sdk_wasm_parallel_rows(...args)===0)throw Error('Invalid parallel geometry accepted');
 postMessage({ready:true,guards_passed:true,heap_bytes:x.memory.buffer.byteLength});let seen=0;
 while(true){
  let seq=Atomics.load(c,64+id);if(seq===seen){Atomics.wait(c,64+id,seen);continue;}seen=seq;
  const count=c[3];if(id>=count)continue;let code=0;
  try{
   const t=c[1],rows=c[2],start=Math.floor((rows/4)*id/count)*4,end=Math.floor((rows/4)*(id+1)/count)*4,width=end-start;
   new Uint8Array(x.memory.buffer,np,256).set(new Uint8Array(shared,NAME_OFFSET,256));
   new Float32Array(x.memory.buffer,ip,t*512).set(input.subarray(0,t*512));
   code=x.mei_sdk_wasm_parallel_rows(np,ip,t,start,end,op,CAP);
   if(!code){const result=new Float32Array(x.memory.buffer,op,t*width);for(let j=0;j<t;j++)output.set(result.subarray(j*width,(j+1)*width),j*rows+start);}
  }catch(e){code=-1;}
  Atomics.store(c,48+id,x.memory.buffer.byteLength);Atomics.store(c,32+id,code);Atomics.store(c,16+id,seq);Atomics.notify(c,16+id);
 }
}catch(e){postMessage({error:String(e.stack||e)});}};
