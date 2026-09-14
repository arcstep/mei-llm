import { createServer } from "node:http";
import { readFileSync, writeFileSync } from "node:fs";
import { basename, dirname, extname, resolve } from "node:path";
import { spawn } from "node:child_process";

const argv = Object.fromEntries(Array.from({ length: process.argv.length - 2 }, (_, i) => i)
  .filter((i) => process.argv[i + 2]?.startsWith("--"))
  .map((i) => [process.argv[i + 2].slice(2), process.argv[i + 3]]));
for (const name of ["model", "wasm", "golden", "out"]) if (!argv[name]) throw new Error(`missing --${name}`);
const browser = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const root = resolve(".");
const files = {
  "/model": resolve(argv.model), "/wasm": resolve(argv.wasm), "/golden": resolve(argv.golden),
  "/audit.mjs": resolve(root, "src/platform/browser-sdk/tokenizer-audit.mjs"),
};
const embedded = Object.fromEntries(
  ["model", "wasm", "golden"].map(name => [name, readFileSync(resolve(argv[name])).toString("base64")]),
);
const html = `<!doctype html><meta charset=utf-8><pre id=result>pending</pre><script>
try {
 const bytes=value=>Uint8Array.from(atob(value),c=>c.charCodeAt(0));
 const v=bytes('${embedded.model}'),w=bytes('${embedded.wasm}').buffer,g=JSON.parse(new TextDecoder().decode(bytes('${embedded.golden}')));
 const exp=new WebAssembly.Instance(new WebAssembly.Module(w),{}).exports;
 const alloc=bytes=>{const ptr=exp.mei_sdk_wasm_alloc(bytes.length);if(!ptr)throw new Error('allocation failed');new Uint8Array(exp.memory.buffer,ptr,bytes.length).set(bytes);return{ptr,len:bytes.length}};
 const cstr=text=>{const body=new TextEncoder().encode(text),value=new Uint8Array(body.length+1);value.set(body);return value};
 const vs=alloc(v),rs=alloc(cstr(JSON.stringify({texts:g.cases.map(x=>x.text)}))),os=alloc(new Uint8Array(4));
 const code=exp.mei_sdk_wasm_tokenizer_audit(vs.ptr,vs.len,rs.ptr,os.ptr);if(code!==0)throw new Error('tokenizer audit failed code='+code);
 const rp=new DataView(exp.memory.buffer).getUint32(os.ptr,true),memory=new Uint8Array(exp.memory.buffer);let end=rp;while(memory[end]!==0)end++;
 const actual=JSON.parse(new TextDecoder('utf-8',{fatal:true}).decode(memory.subarray(rp,end)));let checked=0;
 for(const c of g.cases){const x=actual.cases[checked];if(JSON.stringify(x.ids)!==JSON.stringify(c.ids)||x.decoded!==c.text)throw new Error('mismatch at '+checked);checked++;}
 exp.mei_sdk_wasm_string_free(rp);exp.mei_sdk_wasm_free(vs.ptr,vs.len);exp.mei_sdk_wasm_free(rs.ptr,rs.len);exp.mei_sdk_wasm_free(os.ptr,os.len);
 result.textContent=JSON.stringify({passed:true,checked,user_agent:navigator.userAgent});
}catch(error){result.textContent=JSON.stringify({passed:false,error:String(error?.stack||error),user_agent:navigator.userAgent});}
</script>`;
const server = createServer((req, res) => {
  if (req.url === "/") { res.setHeader("content-type", "text/html; charset=utf-8"); res.end(html); return; }
  const file = files[req.url]; if (!file) { res.statusCode=404; res.end(); return; }
  res.setHeader("content-type", extname(file)===".mjs" ? "text/javascript" : "application/octet-stream"); res.end(readFileSync(file));
});
await new Promise((ok) => server.listen(0, "127.0.0.1", ok));
const port = server.address().port;
const proc = spawn(browser, ["--headless=new", "--disable-gpu", "--no-sandbox", "--disable-background-networking",
  "--dump-dom", `http://127.0.0.1:${port}/`]);
let stdout="", stderr="";
proc.stdout.on("data", chunk => { if(stdout.length < 16*1024*1024) stdout += chunk; });
proc.stderr.on("data", chunk => { if(stderr.length < 16*1024*1024) stderr += chunk; });
const timeout = setTimeout(() => proc.kill("SIGKILL"), 60_000);
const status = await new Promise(resolveStatus => proc.on("close", resolveStatus));
clearTimeout(timeout);
server.close(); if(status!==0) throw new Error(stderr||`Chrome exit ${status}`);
const match=stdout.match(/<pre id="result">([^<]*)<\/pre>/); if(!match) throw new Error("browser result missing");
const decoded=match[1].replaceAll("&quot;",'"').replaceAll("&amp;","&").replaceAll("&lt;","<").replaceAll("&gt;",">");
const result=JSON.parse(decoded); writeFileSync(resolve(argv.out),JSON.stringify(result,null,2)+"\n");
if(!result.passed){console.error(result);process.exit(1)} console.log(JSON.stringify(result));
