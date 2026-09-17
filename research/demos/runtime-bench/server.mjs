import http from 'node:http';
import {readFile,mkdir,writeFile} from 'node:fs/promises';
import {resolve,extname} from 'node:path';
import {randomUUID} from 'node:crypto';
const root=resolve(import.meta.dirname,'../../..'),port=Number(process.env.PORT||8786),origin=`http://127.0.0.1:${port}`;
const pkg=resolve(process.env.MEI_BENCH_PACKAGE||resolve(root,'.local/cache/data-check/data-check-d919985609f9'));
const numericWasm=resolve(process.env.MEI_BENCH_WASM||resolve(root,'.local/cache/wasm-speed/mei-51m-wasm-cpu-gpu-bench-20260914-v3/runtime.wasm'));
const assets=new Set(['mei-model.json','tensors.bin','tokenizer.model','tool-index.json','receipt.json']);
const files=new Map([
 ['/',resolve(import.meta.dirname,'index.html')],
 ...['decode.html','decode-worker.mjs','parallel.html','parallel-worker.mjs','parallel-pool.mjs','parallel-helper.mjs'].map(f=>['/'+f,resolve(import.meta.dirname,f)]),
 ['/gpu/webgpu-decode.mjs',resolve(root,'src/platform/browser-sdk/experimental/webgpu-decode.mjs')],
 ['/gpu/webgpu-parallel.mjs',resolve(root,'src/platform/browser-sdk/experimental/webgpu-parallel.mjs')],
 ['/parallel.wasm',resolve(process.env.MEI_PARALLEL_WASM||resolve(root,'.local/cache/wasm-speed/mei-51m-wasm-parallel-20260914-v2/runtime.wasm'))],['/bench-worker.mjs',resolve(import.meta.dirname,'bench-worker.mjs')],['/gpu/webgpu-51m.mjs',resolve(root,'src/platform/browser-sdk/experimental/webgpu-51m.mjs')],['/probe.mjs',resolve(import.meta.dirname,'probe.mjs')],
 ['/sdk/wasm-abi.mjs',resolve(root,'src/platform/browser-sdk/wasm-abi.mjs')],
 ['/numeric.wasm',numericWasm],
 ['/speed.wasm',resolve(root,'.local/cache/wasm-speed/mei-51m-wasm-speed-20260914-v4/runtime.wasm')]
]);
http.createServer(async(req,res)=>{try{
 if(![`127.0.0.1:${port}`,`localhost:${port}`].includes(req.headers.host))throw Error('Invalid host');
 res.setHeader('Cross-Origin-Opener-Policy','same-origin');res.setHeader('Cross-Origin-Embedder-Policy','require-corp');
 const path=new URL(req.url,origin).pathname;
 if(req.method==='POST'&&path==='/report'){
  if(req.headers.origin!==origin||req.headers['content-type']!=='application/json')throw Error('Invalid origin');
  let body='';for await(const b of req){body+=b;if(body.length>8*1024*1024)throw Error('Too large');}
  JSON.parse(body);const out=resolve(root,'.local/cache/runtime-bench');await mkdir(out,{recursive:true});const id=`${Date.now()}-${randomUUID()}`;await writeFile(resolve(out,id+'.json'),body,{flag:'wx'});res.end(JSON.stringify({id}));return;
 }
 let file=files.get(path);if(path.startsWith('/assets/')&&assets.has(path.slice(8)))file=resolve(pkg,path.slice(8));
 if(req.method!=='GET'||!file){res.writeHead(404);res.end();return;}
 res.writeHead(200,{'Content-Type':({'.html':'text/html','.mjs':'text/javascript','.wasm':'application/wasm','.json':'application/json'})[extname(file)]||'application/octet-stream','Cache-Control':'no-store'});res.end(await readFile(file));
}catch(e){res.writeHead(400);res.end(String(e.message));}}).listen(port,'127.0.0.1',()=>console.log(origin));
