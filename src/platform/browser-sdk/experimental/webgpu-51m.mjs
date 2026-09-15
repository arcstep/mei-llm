// Diagnostic numerical backend, not a released Agent runtime. Full 27-layer LM,
// mHC and Engram; no tool execution. Dense GPU weights are derived in memory
// from the unchanged CQ2 package. Memory/quality gates remain explicitly open.
const F=Math.fround, Q2=[-1.5104176,-.45278,.45278,1.5104176].map(F),Q4=[-2.732589,-2.069018,-1.618046,-1.256231,-.94234,-.656759,-.388055,-.128396,.128396,.388055,.656759,.94234,1.256231,1.618046,2.069018,2.732589].map(F);
function half(u){const s=(u&32768)?-1:1,e=(u>>10)&31,m=u&1023;return s*(e===0?m*2**-24:e===31?(m?NaN:Infinity):(1+m/1024)*2**(e-15));}
export function unpackCQ2(bytes){
 const dv=new DataView(bytes.buffer,bytes.byteOffset,bytes.byteLength),text=new TextDecoder();
 if(text.decode(bytes.subarray(0,8))!=='MEICQ201'||dv.getUint32(8,true)!==2)throw Error('Requires CQ2 v2');
 const meta=JSON.parse(text.decode(bytes.subarray(16,16+dv.getUint32(12,true))));
 const entries=new Map(meta.tensors.map(t=>[t.name,t]));
 return {entries,dequant(name){const t=entries.get(name);if(!t)throw Error('Missing '+name);const out=new Float32Array(t.n_params);
  if(t.dtype==='f16'){for(let i=0;i<out.length;i++)out[i]=half(dv.getUint16(t.data.offset+i*2,true));return out;}
  if(!['cq2','cq4'].includes(t.dtype)||t.transform!=='wht'||t.group_size!==128)throw Error('Unsupported tensor '+name);
  const bits=t.dtype==='cq2'?2:4,book=bits===2?Q2:Q4;let cursor=t.data.offset;const tmp=new Float32Array(128);
  for(let g=0;g<Math.ceil(t.n_params/128);g++){
   const marked=(bytes[t.bit_map.offset+(g>>3)]>>(g&7))&1;if(marked!==(bits===4?1:0))throw Error('Mixed codebook unsupported');
   const scale=half(dv.getUint16(t.scales.offset+g*2,true));for(let k=0;k<128;k++)tmp[k]=F(book[(bytes[cursor+(k*bits>>3)]>>(k*bits&7))&((1<<bits)-1)]*scale);
   cursor+=128*bits/8;for(let w=1;w<128;w*=2)for(let base=0;base<128;base+=2*w)for(let j=0;j<w;j++){const a=tmp[base+j],b=tmp[base+j+w];tmp[base+j]=F(a+b);tmp[base+j+w]=F(a-b);}
   for(let k=0;k<128&&g*128+k<out.length;k++)out[g*128+k]=F(tmp[k]*F(1/Math.sqrt(128)));
  }if(cursor!==t.data.offset+t.data.nbytes)throw Error('Tensor size mismatch');return out;
 }};
}
const COMMON=`fn sig(x:f32)->f32{return 1.0/(1.0+exp(-x));}
fn qr(x:f32,s:f32)->f32{return clamp(sign(x)*floor(abs(x)/s+0.5),-128.0,127.0)*s;}
`;
export class WebGPU51m {
 static async create(manifest,bytes){
  const expected=manifest.files.find(f=>f.path==='tensors.bin')?.sha256;
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');if(hash!==expected)throw Error('Weight hash mismatch');
  const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});if(!adapter)throw Error('No GPU adapter');
  if((adapter.info.isFallbackAdapter??adapter.isFallbackAdapter)===true||/swiftshader|software/i.test(adapter.info.description))throw Error('Software adapter refused');
  const d=await adapter.requestDevice({requiredLimits:{maxStorageBufferBindingSize:Math.min(adapter.limits.maxStorageBufferBindingSize,134217728),maxBufferSize:Math.min(adapter.limits.maxBufferSize,268435456)}});
  try{const m=new WebGPU51m(d,manifest,hash,adapter);await m.load(bytes);return m;}catch(e){d.destroy();throw e;}
 }
 constructor(d,manifest,hash,adapter){
  const a=manifest.architecture;
  if(a.d_model!==512||a.n_layers!==27||a.n_heads!==8||a.n_kv_heads!==4||a.head_dim!==64||a.mhc_lanes!==4||a.vocab_size!==24000||a.sinkhorn_iters!==20||JSON.stringify(a.engram_layers)!=='[2,15]'||JSON.stringify(a.engram_orders)!=='[2,3]'||a.engram_slots!==8192||a.engram_conv_taps!==4||a.rms_eps!==1e-6||a.rope_theta!==100000||a.tie_embeddings!==true)throw Error('Unsupported architecture');
  this.device=d;this.manifest=manifest;this.hash=hash;this.adapter={vendor:adapter.info.vendor,architecture:adapter.info.architecture,isFallbackAdapter:adapter.info.isFallbackAdapter??adapter.isFallbackAdapter??null};this.buffers=[];this.bytes=0;this.kernels=new Map();this.bindings=new Map();this.bufferIDs=new WeakMap();this.nextBufferID=1;this.weights=new Map();this.pos=0;this.history=[];this.errors=[];d.addEventListener('uncapturederror',e=>this.errors.push(e.error.message));d.lost.then(x=>{if(x.reason!=='destroyed')this.errors.push('Device lost: '+x.message);});
 }
 buf(n,values){const d=this.device;const b=d.createBuffer({size:Math.max(4,Math.ceil(n/4)*4),usage:GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST|GPUBufferUsage.COPY_SRC});this.buffers.push(b);this.bytes+=b.size;if(values)d.queue.writeBuffer(b,0,values);return b;}
 async load(bytes){
  const packed=unpackCQ2(bytes);for(const [name,t] of packed.entries){if(t.role!=='lm'||name.startsWith('conf_'))continue;const v=packed.dequant(name);this.weights.set(name,this.buf(v.byteLength,v));}
  // Fixed bounded workspace. f32 GPU KV stores Q/DQ values in this first
  // diagnostic backend; unlike CPU runtime, it does NOT claim int8 KV storage.
  this.maxBatch=512;this.maxContext=2048;const n=512;
  const alloc=(name,size)=>this[name]=this.buf(size*4);
  for(const s of ['lanes','nextLanes'])alloc(s,n*2048);
  alloc('coeff',n*24);for(const s of ['u','x','xn','q','g','attn','o','h','mlpin','blockout'])alloc(s,n*512);
  for(const s of ['k','v'])alloc(s,n*256);alloc('hidden',512);alloc('logits',24000);alloc('tokenBuf',2048);alloc('engFetch',(n+11)*512);alloc('engKeyAll',(n+11)*512);alloc('engValueAll',(n+11)*512);
  this.engKeys=[this.buf(n*512*4),this.buf(n*512*4)];this.engValues=[this.buf(n*512*4),this.buf(n*512*4)];this.cacheK=Array.from({length:27},()=>this.buf(2048*256*4));this.cacheV=Array.from({length:27},()=>this.buf(2048*256*4));
  this.readback=this.device.createBuffer({size:24000*4,usage:GPUBufferUsage.COPY_DST|GPUBufferUsage.MAP_READ});this.bytes+=24000*4;
 }
 w(n){const b=this.weights.get(n);if(!b)throw Error('Missing GPU weight '+n);return b;}
 kernel(key,code,bindings){
  let p=this.kernels.get(key);if(!p){const module=this.device.createShaderModule({label:key,code:COMMON+code});p=this.device.createComputePipeline({label:key,layout:'auto',compute:{module,entryPoint:'main'}});this.kernels.set(key,p);}
  const bk=key+':'+bindings.map(b=>{if(!this.bufferIDs.has(b))this.bufferIDs.set(b,this.nextBufferID++);return this.bufferIDs.get(b);}).join(',');let bg=this.bindings.get(bk);if(!bg){bg=this.device.createBindGroup({layout:p.getBindGroupLayout(0),entries:bindings.map((b,i)=>({binding:i,resource:{buffer:b}}))});this.bindings.set(bk,bg);}return [p,bg];
 }
 run(enc,key,code,buffers,groups){const [p,b]=this.kernel(key,code,buffers);const pass=enc._meiPass||(enc._meiPass=enc.beginComputePass());pass.setPipeline(p);pass.setBindGroup(0,b);pass.dispatchWorkgroups(...groups);}
 async read(enc,buffer=this.logits,n=24000){if(enc._meiPass)enc._meiPass.end();enc.copyBufferToBuffer(buffer,0,this.readback,0,n*4);this.device.queue.submit([enc.finish()]);await this.readback.mapAsync(GPUMapMode.READ,0,n*4);const out=new Float32Array(this.readback.getMappedRange(0,n*4)).slice();this.readback.unmap();if(this.errors.length)throw Error(this.errors.join('\n'));return out;}
 reset(){this.pos=0;this.history=[];}
 close(){for(const b of this.buffers)b.destroy();this.readback.destroy();this.device.destroy();}
 norm(enc,input,out,scale,t,width=512,qdq=true){
  this.run(enc,`norm-${width}-${qdq}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> s:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;
var<workgroup> red:array<f32,128>;var<workgroup> vals:array<f32,${width}>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let base=w.x*${width}u;var v=0.0;for(var c=l;c<${width}u;c+=128u){v+=x[base+c]*x[base+c];}red[l]=v;workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){red[l]+=red[l+k];}workgroupBarrier();}let inv=inverseSqrt(red[0]/${width}.0+0.000001);workgroupBarrier();var ma=0.0;for(var c=l;c<${width}u;c+=128u){let a=x[base+c]*inv*(1.0+s[c]);vals[c]=a;ma=max(ma,abs(a));}red[l]=ma;workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){red[l]=max(red[l],red[l+k]);}workgroupBarrier();}let step=select(1.0,red[0]/127.0,red[0]>0.0);for(var c=l;c<${width}u;c+=128u){y[base+c]=${qdq?'qr(vals[c],step)':'vals[c]'};}}
`,[input,scale,out],[t]);
 }
 mat(enc,input,weight,out,m,n,k=512){
  if(m===1){this.run(enc,`mv-${n}-${k}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;var<workgroup> r:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) b:vec3u){var a=0.0;for(var c=l;c<${k}u;c+=64u){a+=x[c]*w[b.x*${k}u+c];}r[l]=a;workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){r[l]+=r[l+h];}workgroupBarrier();}if(l==0u){y[b.x]=r[0];}}
`,[input,weight,out],[n]);return;}
  this.run(enc,`mm-${m}-${n}-${k}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;var<workgroup> a:array<f32,256>;var<workgroup> b:array<f32,256>;
@compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) l:vec3u,@builtin(workgroup_id) g:vec3u){let id=l.y*8u+l.x;let row=g.y*16u+l.y*2u;let col=g.x*16u+l.x*2u;var acc=vec4f(0.0);
for(var start=0u;start<${k}u;start+=16u){for(var z=id;z<256u;z+=64u){let rr=g.y*16u+z/16u;let cc=start+z%16u;a[z]=0.0;if(rr<${m}u&&cc<${k}u){a[z]=x[rr*${k}u+cc];}let wr=g.x*16u+z/16u;b[z]=0.0;if(wr<${n}u&&cc<${k}u){b[z]=w[wr*${k}u+cc];}}workgroupBarrier();for(var j=0u;j<16u;j++){let aa=vec2f(a[l.y*32u+j],a[l.y*32u+16u+j]);let bb=vec2f(b[l.x*32u+j],b[l.x*32u+16u+j]);acc+=vec4f(aa.x*bb.x,aa.x*bb.y,aa.y*bb.x,aa.y*bb.y);}workgroupBarrier();}
if(row<${m}u&&col<${n}u){y[row*${n}u+col]=acc.x;}if(row<${m}u&&col+1u<${n}u){y[row*${n}u+col+1u]=acc.y;}if(row+1u<${m}u&&col<${n}u){y[(row+1u)*${n}u+col]=acc.z;}if(row+1u<${m}u&&col+1u<${n}u){y[(row+1u)*${n}u+col+1u]=acc.w;}}
`,[input,weight,out],[Math.ceil(n/16),Math.ceil(m/16)]);
 }
 engram(enc,tokens,t){
  const old=this.history.slice(-11),all=old.concat(tokens),keep=old.length,at=all.length;this.device.queue.writeBuffer(this.tokenBuf,0,new Uint32Array(all));
  for(let site=0;site<2;site++){
   this.run(enc,`eng-fetch-${at}`,`
@group(0) @binding(0) var<storage,read> ids:array<u32>;@group(0) @binding(1) var<storage,read> tab:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;var<workgroup> vals:array<f32,512>;var<workgroup> red:array<f32,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){var ma=0.0;for(var c=l;c<512u;c+=128u){let tbl=c/128u;let order=2u+tbl/2u;var h=0x9E3779B9u*(tbl+1u);for(var j=0u;j<order;j++){var tok=0u;if(w.x>=j){tok=ids[w.x-j];}h=(h^tok)*0x01000193u;}h=h^(h>>15u);var v=0.0;if(w.x+1u>=order){v=tab[(tbl*8192u+h%8192u)*128u+c%128u];}vals[c]=v;ma=max(ma,abs(v));}red[l]=ma;workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){red[l]=max(red[l],red[l+k]);}workgroupBarrier();}let step=select(1.0,red[0]/127.0,red[0]>0.0);for(var c=l;c<512u;c+=128u){y[w.x*512u+c]=qr(vals[c],step);}}
`,[this.tokenBuf,this.w(`engrams.${site}.tables`),this.engFetch],[at]);
   this.mat(enc,this.engFetch,this.w(`engrams.${site}.key_proj.weight`),this.engKeyAll,at,512);this.mat(enc,this.engFetch,this.w(`engrams.${site}.value_proj.weight`),this.engValueAll,at,512);
   this.run(enc,`eng-conv-${keep}-${t}`,`
@group(0) @binding(0) var<storage,read> k:array<f32>;@group(0) @binding(1) var<storage,read> v:array<f32>;@group(0) @binding(2) var<storage,read> taps:array<f32>;@group(0) @binding(3) var<storage,read_write> ko:array<f32>;@group(0) @binding(4) var<storage,read_write> vo:array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) i:vec3u){if(i.x>=${t*512}u){return;}let ti=i.x/512u+${keep}u;let c=i.x%512u;ko[i.x]=k[ti*512u+c];var z=0.0;for(var j=0u;j<4u;j++){if(ti>=j*3u){z+=v[(ti-j*3u)*512u+c]*taps[j*512u+c];}}vo[i.x]=z;}
`,[this.engKeyAll,this.engValueAll,this.w(`engrams.${site}.taps`),this.engKeys[site],this.engValues[site]],[Math.ceil(t*512/128)]);
  }
  // Restore the current token stream for embedding after Engram consumed old context.
  // queue writes happen before submitted commands: use a distinct buffer for embedding.
 }
 routing(enc,layer,t){
  // Pack per-layer routing tensors once; this also keeps storage bindings <=8.
  if(!this.routingWeights)this.routingWeights=[];
  const rw=this.routingWeights[layer];
  this.run(enc,`routing-${layer}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> out:array<f32>;
var<workgroup> sum:array<f32,128>;var<workgroup> dots:array<f32,3072>;var<workgroup> vals:array<f32,24>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) b:vec3u){var ss=0.0;for(var c=l;c<2048u;c+=128u){ss+=x[b.x*2048u+c]*x[b.x*2048u+c];}sum[l]=ss;workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){sum[l]+=sum[l+k];}workgroupBarrier();}let inv=inverseSqrt(sum[0]/2048.0+0.000001);
for(var j=0u;j<24u;j++){var a=0.0;for(var c=l;c<2048u;c+=128u){a+=x[b.x*2048u+c]*inv*w[c*24u+j];}dots[j*128u+l]=a;}workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){for(var j=0u;j<24u;j++){dots[j*128u+l]+=dots[j*128u+l+k];}}workgroupBarrier();}
if(l<24u){let z=dots[l*128u]*w[49152u+l]+w[49176u+l];vals[l]=z;if(l<4u){vals[l]=sig(z+select(-4.0,4.0,l==${layer%4}u));}else if(l<8u){vals[l]=2.0*sig(z+select(-4.0,0.0,l-4u==${layer%4}u));}}workgroupBarrier();
if(l==0u){for(var it=0;it<20;it++){for(var i=0u;i<4u;i++){let z=vec4f(vals[8u+i*4u],vals[9u+i*4u],vals[10u+i*4u],vals[11u+i*4u]);let ma=max(max(z.x,z.y),max(z.z,z.w));let ex=exp(z-vec4f(ma));let norm=ma+log(ex.x+ex.y+ex.z+ex.w);for(var j=0u;j<4u;j++){vals[8u+i*4u+j]-=norm;}}
for(var j=0u;j<4u;j++){let z=vec4f(vals[8u+j],vals[12u+j],vals[16u+j],vals[20u+j]);let ma=max(max(z.x,z.y),max(z.z,z.w));let ex=exp(z-vec4f(ma));let norm=ma+log(ex.x+ex.y+ex.z+ex.w);for(var i=0u;i<4u;i++){vals[8u+i*4u+j]-=norm;}}}}workgroupBarrier();if(l<24u){out[b.x*24u+l]=select(vals[l],exp(vals[l]),l>=8u);}}
`,[this.lanes,rw,this.coeff],[t]);
 }
 async prepareRouting(bytes){const p=unpackCQ2(bytes),arrays={};for(const n of ['phi_pre','phi_post','phi_res','b_pre','b_post','b_res','a_pre','a_post','a_res'])arrays[n]=p.dequant('mhc_'+n);this.routingWeights=[];for(let layer=0;layer<27;layer++){const v=new Float32Array(49200);for(let j=0;j<24;j++){const kind=j<4?'pre':j<8?'post':'res',width=kind==='res'?16:4,col=j-(kind==='post'?4:kind==='res'?8:0);for(let c=0;c<2048;c++)v[c*24+j]=arrays['phi_'+kind][(layer*2048+c)*width+col];v[49152+j]=arrays['a_'+kind][layer];v[49176+j]=arrays['b_'+kind][layer*width+col];}this.routingWeights.push(this.buf(v.byteLength,v));}}
 combine(enc,site,t){this.run(enc,`combine-${site>=0}`,`
@group(0) @binding(0) var<storage,read> lanes:array<f32>;@group(0) @binding(1) var<storage,read> coeff:array<f32>;@group(0) @binding(2) var<storage,read_write> u:array<f32>;@group(0) @binding(3) var<storage,read_write> x:array<f32>;
${site>=0?'@group(0) @binding(4) var<storage,read> ek:array<f32>;@group(0) @binding(5) var<storage,read> ev:array<f32>;':''}
var<workgroup> vals:array<f32,512>;var<workgroup> r:array<vec3f,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){var red=vec3f(0.0);for(var c=l;c<512u;c+=128u){var v=0.0;for(var a=0u;a<4u;a++){v+=lanes[w.x*2048u+a*512u+c]*coeff[w.x*24u+a];}vals[c]=v;u[w.x*512u+c]=v;${site>=0?'let k=ek[w.x*512u+c];red+=vec3f(v*v,k*k,v*k);':''}}r[l]=red;workgroupBarrier();for(var k=64u;k>0u;k/=2u){if(l<k){r[l]+=r[l+k];}workgroupBarrier();}let alpha=sig(r[0].z*inverseSqrt(r[0].x/512.0+0.000001)*inverseSqrt(r[0].y/512.0+0.000001)/sqrt(512.0));for(var c=l;c<512u;c+=128u){x[w.x*512u+c]=vals[c]${site>=0?'+alpha*ev[w.x*512u+c]':''};}}
`,[this.lanes,this.coeff,this.u,this.x,...(site>=0?[this.engKeys[site],this.engValues[site]]:[])],[t]);}
 qrope(enc,layer,t){this.run(enc,'qrope',`
@group(0) @binding(0) var<storage,read_write> q:array<f32>;@group(0) @binding(1) var<storage,read> s:array<f32>;@group(0) @binding(2) var<storage,read> p:array<u32>;var<workgroup> v:array<f32,64>;var<workgroup> red:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let base=w.x*64u;let a=q[base+l];red[l]=a*a;workgroupBarrier();for(var k=32u;k>0u;k/=2u){if(l<k){red[l]+=red[l+k];}workgroupBarrier();}v[l]=a*inverseSqrt(red[0]/64.0+0.000001)*(1.0+s[l]);workgroupBarrier();let f=f32(p[1]+w.x/8u)/pow(100000.0,f32(2u*(l%32u))/64.0);let c=cos(f);let sn=sin(f);if(l<32u){q[base+l]=v[l]*c-v[l+32u]*sn;}else{q[base+l]=v[l]*c+v[l-32u]*sn;}}
`,[this.q,this.w(`blocks.${layer}.attn.q_norm.scale`),this.params],[t*8]);
 this.run(enc,'kvrope',`
@group(0) @binding(0) var<storage,read> k:array<f32>;@group(0) @binding(1) var<storage,read> v:array<f32>;@group(0) @binding(2) var<storage,read> s:array<f32>;@group(0) @binding(3) var<storage,read> p:array<u32>;@group(0) @binding(4) var<storage,read_write> kc:array<f32>;@group(0) @binding(5) var<storage,read_write> vc:array<f32>;
var<workgroup> kv:array<f32,64>;var<workgroup> red:array<vec2f,64>;var<workgroup> rot:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let base=w.x*64u;let a=k[base+l];red[l]=vec2f(a*a,0.0);workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]+=red[l+h];}workgroupBarrier();}kv[l]=a*inverseSqrt(red[0].x/64.0+0.000001)*(1.0+s[l]);workgroupBarrier();let f=f32(p[1]+w.x/4u)/pow(100000.0,f32(2u*(l%32u))/64.0);let c=cos(f);let sn=sin(f);var kr=0.0;if(l<32u){kr=kv[l]*c-kv[l+32u]*sn;}else{kr=kv[l]*c+kv[l-32u]*sn;}rot[l]=kr;red[l]=vec2f(abs(kr),abs(v[base+l]));workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]=max(red[l],red[l+h]);}workgroupBarrier();}let sk=select(1.0,red[0].x/127.0,red[0].x>0.0);let sv=select(1.0,red[0].y/127.0,red[0].y>0.0);let dst=(p[1]*4u+w.x)*64u+l;kc[dst]=qr(rot[l],sk);vc[dst]=qr(v[base+l],sv);}
`,[this.k,this.v,this.w(`blocks.${layer}.attn.k_norm.scale`),this.params,this.cacheK[layer],this.cacheV[layer]],[t*4]);}
 splitAttention(enc,layer){
  if(!this.partial)this.partial=this.buf(8*32*66*4);
  this.run(enc,'attn-split',`
@group(0) @binding(0) var<storage,read> q:array<f32>;@group(0) @binding(1) var<storage,read> k:array<f32>;@group(0) @binding(2) var<storage,read> v:array<f32>;@group(0) @binding(3) var<storage,read> p:array<u32>;@group(0) @binding(4) var<storage,read_write> out:array<f32>;
var<workgroup> scores:array<f32,64>;var<workgroup> red:array<f32,64>;var<workgroup> query:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let head=w.x;let block=w.y;let start=block*64u;let count=min(64u,p[1]+1u-start);query[l]=q[head*64u+l];workgroupBarrier();var dot=0.0;for(var c=0u;c<64u;c++){dot+=query[c]*k[(start+l)*256u+(head/2u)*64u+c];}scores[l]=select(-3.402823e38,dot*0.125,l<count);red[l]=scores[l];workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]=max(red[l],red[l+h]);}workgroupBarrier();}let ma=red[0];workgroupBarrier();scores[l]=select(0.0,exp(scores[l]-ma),l<count);red[l]=scores[l];workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]+=red[l+h];}workgroupBarrier();}let total=red[0];var a=0.0;for(var j=0u;j<count;j++){a+=scores[j]*v[(start+j)*256u+(head/2u)*64u+l];}let base=(head*32u+block)*66u;out[base+l]=a;if(l==0u){out[base+64u]=ma;out[base+65u]=total;}}
`,[this.q,this.cacheK[layer],this.cacheV[layer],this.params,this.partial],[8,Math.ceil((this.pos+1)/64)]);
  this.run(enc,'attn-merge',`
@group(0) @binding(0) var<storage,read> parts:array<f32>;@group(0) @binding(1) var<storage,read> p:array<u32>;@group(0) @binding(2) var<storage,read> gate:array<f32>;@group(0) @binding(3) var<storage,read_write> out:array<f32>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let count=(p[1]+64u)/64u;let base=w.x*32u*66u;var ma=-3.402823e38;for(var j=0u;j<count;j++){ma=max(ma,parts[base+j*66u+64u]);}var num=0.0;var den=0.0;for(var j=0u;j<count;j++){let scale=exp(parts[base+j*66u+64u]-ma);num+=scale*parts[base+j*66u+l];den+=scale*parts[base+j*66u+65u];}out[w.x*64u+l]=num/den*sig(gate[w.x*64u+l]);}
`,[this.partial,this.params,this.g,this.attn],[8]);
 }
 attention(enc,layer,t){if(t===1)this.splitAttention(enc,layer);else this.run(enc,'attention',`
@group(0) @binding(0) var<storage,read> q:array<f32>;@group(0) @binding(1) var<storage,read> k:array<f32>;@group(0) @binding(2) var<storage,read> v:array<f32>;@group(0) @binding(3) var<storage,read> gate:array<f32>;@group(0) @binding(4) var<storage,read> p:array<u32>;@group(0) @binding(5) var<storage,read_write> out:array<f32>;
var<workgroup> scores:array<f32,2048>;var<workgroup> red:array<f32,64>;var<workgroup> query:array<f32,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let ti=w.x/8u;let head=w.x%8u;let kvhead=head/2u;let count=p[1]+ti+1u;let base=w.x*64u;query[l]=q[base+l];workgroupBarrier();var ma=-3.402823e38;
for(var j=l;j<count;j+=64u){var dot=0.0;for(var c=0u;c<64u;c++){dot+=query[c]*k[j*256u+kvhead*64u+c];}scores[j]=dot*0.125;ma=max(ma,scores[j]);}red[l]=ma;workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]=max(red[l],red[l+h]);}workgroupBarrier();}ma=red[0];workgroupBarrier();var total=0.0;for(var j=l;j<count;j+=64u){scores[j]=exp(scores[j]-ma);total+=scores[j];}red[l]=total;workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]+=red[l+h];}workgroupBarrier();}let den=red[0];var a=0.0;for(var j=0u;j<count;j++){a+=scores[j]/den*v[j*256u+kvhead*64u+l];}out[base+l]=a*sig(gate[base+l]);}
`,[this.q,this.cacheK[layer],this.cacheV[layer],this.g,this.params,this.attn],[t*8]);
 this.run(enc,'qdq512',`
@group(0) @binding(0) var<storage,read_write> x:array<f32>;var<workgroup> r:array<f32,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){var ma=0.0;for(var c=l;c<512u;c+=128u){ma=max(ma,abs(x[w.x*512u+c]));}r[l]=ma;workgroupBarrier();for(var h=64u;h>0u;h/=2u){if(l<h){r[l]=max(r[l],r[l+h]);}workgroupBarrier();}let s=select(1.0,r[0]/127.0,r[0]>0.0);for(var c=l;c<512u;c+=128u){x[w.x*512u+c]=qr(x[w.x*512u+c],s);}}
`,[this.attn],[t]);}
 post(enc,layer,t){this.norm(enc,this.o,this.xn,this.w(`blocks.${layer}.post_attn_norm.scale`),t,512,false);
 this.run(enc,'residual',`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> o:array<f32>;@group(0) @binding(2) var<storage,read> g:array<f32>;@group(0) @binding(3) var<storage,read_write> h:array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) i:vec3u){h[i.x]=x[i.x]+sig(g[0])*o[i.x];}
`,[this.x,this.xn,this.w(`blocks.${layer}.attn_gate`),this.h],[t*4]);
 this.norm(enc,this.h,this.mlpin,this.w(`blocks.${layer}.mlp_norm.scale`),t,512,false);
 this.run(enc,'mlp',`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> h:array<f32>;@group(0) @binding(2) var<storage,read> d1:array<f32>;@group(0) @binding(3) var<storage,read> d2:array<f32>;@group(0) @binding(4) var<storage,read> d3:array<f32>;@group(0) @binding(5) var<storage,read_write> y:array<f32>;
var<workgroup> z:array<f32,512>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){z[l]=x[w.x*512u+l]*d1[l];z[l+256u]=x[w.x*512u+l+256u]*d1[l+256u];workgroupBarrier();
for(var k=1u;k<512u;k*=2u){let a=(l/k)*(k*2u)+l%k;let b=a+k;let aa=z[a];let bb=z[b];workgroupBarrier();z[a]=aa+bb;z[b]=aa-bb;workgroupBarrier();}
let a=z[l]*0.04419417382415922*d2[l];let b=z[l+256u]*0.04419417382415922*d2[l+256u];z[l]=a*sig(a);z[l+256u]=b*sig(b);workgroupBarrier();
for(var k=1u;k<512u;k*=2u){let a=(l/k)*(k*2u)+l%k;let b=a+k;let aa=z[a];let bb=z[b];workgroupBarrier();z[a]=aa+bb;z[b]=aa-bb;workgroupBarrier();}
for(var c=l;c<512u;c+=256u){y[w.x*512u+c]=h[w.x*512u+c]+z[c]*0.04419417382415922*d3[c];}}
`,[this.mlpin,this.h,this.w(`blocks.${layer}.mlp.d1`),this.w(`blocks.${layer}.mlp.d2`),this.w(`blocks.${layer}.mlp.d3`),this.blockout],[t]);}
 update(enc,t){this.run(enc,'update',`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> co:array<f32>;@group(0) @binding(2) var<storage,read> b:array<f32>;@group(0) @binding(3) var<storage,read> u:array<f32>;@group(0) @binding(4) var<storage,read_write> y:array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) i:vec3u){let ti=i.x/2048u;let lane=i.x%2048u/512u;let c=i.x%512u;var acc=0.0;for(var j=0u;j<4u;j++){acc+=x[ti*2048u+j*512u+c]*co[ti*24u+8u+lane*4u+j];}y[i.x]=acc+co[ti*24u+4u+lane]*(b[ti*512u+c]-u[ti*512u+c]);}
`,[this.lanes,this.coeff,this.blockout,this.u,this.nextLanes],[t*16]);[this.lanes,this.nextLanes]=[this.nextLanes,this.lanes];}
 async forward(tokens){
  const t=tokens.length;if(!t||t>512||this.pos+t>2048||tokens.some(x=>!Number.isInteger(x)||x<0||x>=24000))throw Error('Input outside experimental bounds');
  if(!this.routingWeights)throw Error('Routing not initialized');if(!this.params){this.params=this.buf(16);this.embedIDs=this.buf(512*4);}
  const enc=this.device.createCommandEncoder();this.device.queue.writeBuffer(this.params,0,new Uint32Array([t,this.pos,0,0]));this.device.queue.writeBuffer(this.embedIDs,0,new Uint32Array(tokens));
  this.engram(enc,tokens,t);
  this.run(enc,'embed',`
@group(0) @binding(0) var<storage,read> ids:array<u32>;@group(0) @binding(1) var<storage,read> tab:array<f32>;@group(0) @binding(2) var<storage,read_write> lanes:array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) i:vec3u){let ti=i.x/2048u;lanes[i.x]=tab[ids[ti]*512u+i.x%512u]*sqrt(512.0);}
`,[this.embedIDs,this.w('embed.weight'),this.lanes],[t*16]);
  for(let layer=0;layer<27;layer++){
   this.routing(enc,layer,t);this.combine(enc,layer===2?0:layer===15?1:-1,t);this.norm(enc,this.x,this.xn,this.w(`blocks.${layer}.attn_norm.scale`),t);
   for(const [name,out,n] of [['q',this.q,512],['k',this.k,256],['v',this.v,256],['gate',this.g,512]])this.mat(enc,this.xn,this.w(`blocks.${layer}.attn.${name}_proj.weight`),out,t,n);
   this.qrope(enc,layer,t);this.attention(enc,layer,t);this.mat(enc,this.attn,this.w(`blocks.${layer}.attn.o_proj.weight`),this.o,t,512);this.post(enc,layer,t);this.update(enc,t);
  }
  this.run(enc,'last-mean',`
@group(0) @binding(0) var<storage,read> lanes:array<f32>;@group(0) @binding(1) var<storage,read> p:array<u32>;@group(0) @binding(2) var<storage,read_write> x:array<f32>;
@compute @workgroup_size(128) fn main(@builtin(global_invocation_id) i:vec3u){let base=(p[0]-1u)*2048u;let c=i.x;x[c]=(lanes[base+c]+lanes[base+512u+c]+lanes[base+1024u+c]+lanes[base+1536u+c])*0.25;}
`,[this.lanes,this.params,this.x],[4]);this.norm(enc,this.x,this.hidden,this.w('final_norm.scale'),1);this.mat(enc,this.hidden,this.w('embed.weight'),this.logits,1,24000);
  const logits=await this.read(enc);if(logits.some(x=>!Number.isFinite(x)))throw Error('Nonfinite GPU logits');this.pos+=t;this.history=this.history.concat(tokens).slice(-11);return logits;
 }
}
