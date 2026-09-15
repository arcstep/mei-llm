// Diagnostic decode profiling/optimization. Baseline model is unchanged.
import {ParallelGPU51m} from './webgpu-parallel.mjs';
export class DecodeGPU51m extends ParallelGPU51m {
 static async create(manifest,bytes){
  const hash=[...new Uint8Array(await crypto.subtle.digest('SHA-256',bytes))].map(x=>x.toString(16).padStart(2,'0')).join('');
  if(hash!==manifest.files.find(f=>f.path==='tensors.bin')?.sha256)throw Error('Weight hash mismatch');
  const adapter=await navigator.gpu?.requestAdapter({powerPreference:'high-performance'});
  if(!adapter||!adapter.features.has('timestamp-query'))throw Error('Hardware timestamp-query required');
  if((adapter.info.isFallbackAdapter??adapter.isFallbackAdapter)===true||/swiftshader|software/i.test(adapter.info.description))throw Error('Software adapter');
  const device=await adapter.requestDevice({requiredFeatures:['timestamp-query'],requiredLimits:{maxStorageBufferBindingSize:Math.min(adapter.limits.maxStorageBufferBindingSize,134217728),maxBufferSize:Math.min(adapter.limits.maxBufferSize,268435456)}});
  const m=new DecodeGPU51m(device,manifest,hash,adapter);m.config={mv:64,tile:'auto'};m.profileMode='whole';m.querySet=device.createQuerySet({type:'timestamp',count:2048});
  m.queryResolve=device.createBuffer({size:16384,usage:GPUBufferUsage.QUERY_RESOLVE|GPUBufferUsage.COPY_SRC});m.queryRead=device.createBuffer({size:16384,usage:GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST});
  try{await m.load(bytes);await m.prepareRouting(bytes);await m.prepareFused();return m;}catch(e){m.close();throw e;}
 }
 async prepareFused(){
  this.postWeights=[];this.projections=new Map();
  for(let layer=0;layer<27;layer++)for(const [a,b,out] of [['q','gate',this.g],['k','v',this.v]]){const wa=this.w(`blocks.${layer}.attn.${a}_proj.weight`),wb=this.w(`blocks.${layer}.attn.${b}_proj.weight`);this.projections.set(wa,{weight:wb,output:out});this.projections.set(wb,{skip:true});}
  for(let layer=0;layer<27;layer++){
   const dest=this.buf(2561*4),enc=this.device.createCommandEncoder();
   for(const [i,name] of ['post_attn_norm.scale','mlp_norm.scale','mlp.d1','mlp.d2','mlp.d3','attn_gate'].entries())enc.copyBufferToBuffer(this.w(`blocks.${layer}.${name}`),0,dest,i*512*4,(i===5?1:512)*4);
   this.device.queue.submit([enc.finish()]);this.postWeights.push(dest);
  }
 }
 post(enc,layer,t){
  if(!this.fusePost||t!==1)return super.post(enc,layer,t);
  this.pendingFusedUpdate=true;
  this.run(enc,'fused-post-update',`
@group(0) @binding(0) var<storage,read> o:array<f32>;
@group(0) @binding(1) var<storage,read> x:array<f32>;
@group(0) @binding(2) var<storage,read> lanes:array<f32>;
@group(0) @binding(3) var<storage,read> co:array<f32>;
@group(0) @binding(4) var<storage,read> u:array<f32>;
@group(0) @binding(5) var<storage,read> weights:array<f32>;
@group(0) @binding(6) var<storage,read_write> out:array<f32>;
var<workgroup> red:array<f32,128>;var<workgroup> h:array<f32,512>;var<workgroup> z:array<f32,512>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) l:u32){
if(l<128u){var a=0.0;for(var c=l;c<512u;c+=128u){a+=o[c]*o[c];}red[l]=a;}workgroupBarrier();
for(var k=64u;k>0u;k/=2u){if(l<k){red[l]+=red[l+k];}workgroupBarrier();}
let inv=inverseSqrt(red[0]/512.0+0.000001);workgroupBarrier();
for(var c=l;c<512u;c+=256u){let normalized=o[c]*inv*(1.0+weights[c]);h[c]=x[c]+sig(weights[2560])*normalized;}workgroupBarrier();
if(l<128u){var a=0.0;for(var c=l;c<512u;c+=128u){a+=h[c]*h[c];}red[l]=a;}workgroupBarrier();
for(var k=64u;k>0u;k/=2u){if(l<k){red[l]+=red[l+k];}workgroupBarrier();}
let inv2=inverseSqrt(red[0]/512.0+0.000001);
for(var c=l;c<512u;c+=256u){z[c]=h[c]*inv2*(1.0+weights[512u+c])*weights[1024u+c];}workgroupBarrier();
for(var k=1u;k<512u;k*=2u){let a=(l/k)*(k*2u)+l%k;let b=a+k;let aa=z[a];let bb=z[b];workgroupBarrier();z[a]=aa+bb;z[b]=aa-bb;workgroupBarrier();}
for(var c=l;c<512u;c+=256u){let a=z[c]*0.04419417382415922*weights[1536u+c];z[c]=a*sig(a);}workgroupBarrier();
for(var k=1u;k<512u;k*=2u){let a=(l/k)*(k*2u)+l%k;let b=a+k;let aa=z[a];let bb=z[b];workgroupBarrier();z[a]=aa+bb;z[b]=aa-bb;workgroupBarrier();}
for(var c=l;c<512u;c+=256u){let block=h[c]+z[c]*0.04419417382415922*weights[2048u+c];for(var lane=0u;lane<4u;lane++){var acc=0.0;for(var j=0u;j<4u;j++){acc+=lanes[j*512u+c]*co[8u+lane*4u+j];}out[lane*512u+c]=acc+co[4u+lane]*(block-u[c]);}}
}`,[this.o,this.x,this.lanes,this.coeff,this.u,this.postWeights[layer],this.nextLanes],[1]);
 }
 update(enc,t){if(this.pendingFusedUpdate){this.pendingFusedUpdate=false;[this.lanes,this.nextLanes]=[this.nextLanes,this.lanes];return;}super.update(enc,t);}

 mat(enc,input,weight,out,m,n,k=512){
  const pair=this.projections?.get(weight);
  if(!this.fuseProjection||m!==1||!pair)return super.mat(enc,input,weight,out,m,n,k);
  if(pair.skip)return;
  this.run(enc,`paired-projection-${n}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> wa:array<f32>;@group(0) @binding(2) var<storage,read> wb:array<f32>;@group(0) @binding(3) var<storage,read_write> ya:array<f32>;@group(0) @binding(4) var<storage,read_write> yb:array<f32>;var<workgroup> red:array<vec2f,64>;
@compute @workgroup_size(64) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) b:vec3u){var a=vec2f(0.0);for(var c=l;c<512u;c+=64u){a+=x[c]*vec2f(wa[b.x*512u+c],wb[b.x*512u+c]);}red[l]=a;workgroupBarrier();for(var h=32u;h>0u;h/=2u){if(l<h){red[l]+=red[l+h];}workgroupBarrier();}if(l==0u){ya[b.x]=red[0].x;yb[b.x]=red[0].y;}}
`,[input,weight,pair.weight,out,pair.output],[n]);
 }
 routing(enc,layer,t){
  this.routeFuseLayer=this.fuseRouting&&t===1&&layer!==2&&layer!==15?layer:null;
  super.routing(enc,layer,t);
  this.skipCombineNorm=this.routeFuseLayer!==null;this.routeFuseLayer=null;
 }
 combine(enc,site,t){if(!this.skipCombineNorm)super.combine(enc,site,t);}
 norm(enc,input,out,scale,t,width=512,qdq=true){if(this.skipCombineNorm){this.skipCombineNorm=false;return;}super.norm(enc,input,out,scale,t,width,qdq);}
 async forwardToken(tokens){this.compactRead=true;try{const token=await this.forward(tokens);if(token[0]>=24000)throw Error("Invalid GPU token");return token[0];}finally{this.compactRead=false;}}
 selectToken(enc){
  if(!this.argmaxParts)this.argmaxParts=this.buf(94*8);if(!this.argmaxToken)this.argmaxToken=this.buf(4);
  this.run(enc,'argmax-part',`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read_write> out:array<vec2u>;var<workgroup> vals:array<f32,256>;var<workgroup> ids:array<u32,256>;var<workgroup> bad:array<u32,256>;
@compute @workgroup_size(256) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) w:vec3u){let i=w.x*256u+l;ids[l]=i;bad[l]=0u;vals[l]=-3.402823e38;if(i<24000u){vals[l]=x[i];if(!(abs(x[i])<=3.402823e38)){bad[l]=1u;}}workgroupBarrier();for(var h=128u;h>0u;h/=2u){if(l<h){let j=l+h;bad[l]|=bad[j];if(vals[j]>vals[l]||(vals[j]==vals[l]&&ids[j]<ids[l])){vals[l]=vals[j];ids[l]=ids[j];}}workgroupBarrier();}if(l==0u){out[w.x]=vec2u(bitcast<u32>(vals[0]),select(ids[0],0xffffffffu,bad[0]!=0u));}}
`,[this.logits,this.argmaxParts],[94]);
  this.run(enc,'argmax-final',`
@group(0) @binding(0) var<storage,read> x:array<vec2u>;@group(0) @binding(1) var<storage,read_write> out:array<u32>;var<workgroup> vals:array<f32,128>;var<workgroup> ids:array<u32,128>;var<workgroup> bad:array<u32,128>;
@compute @workgroup_size(128) fn main(@builtin(local_invocation_index) l:u32){vals[l]=-3.402823e38;ids[l]=0xffffffffu;bad[l]=0u;if(l<94u){vals[l]=bitcast<f32>(x[l].x);ids[l]=x[l].y;if(ids[l]>=24000u){bad[l]=1u;}}workgroupBarrier();for(var h=64u;h>0u;h/=2u){if(l<h){let j=l+h;bad[l]|=bad[j];if(vals[j]>vals[l]||(vals[j]==vals[l]&&ids[j]<ids[l])){vals[l]=vals[j];ids[l]=ids[j];}}workgroupBarrier();}if(l==0u){out[0]=select(ids[0],0xffffffffu,bad[0]!=0u);}}
`,[this.argmaxParts,this.argmaxToken],[1]);
 }

 async forward(tokens){this.beginTime=performance.now();this.labels=[];return super.forward(tokens);}
 run(enc,key,code,buffers,groups){
  this.shaderVariants??=new Map();const variantKey=key+':'+this.routingMode+':'+this.routeFuseLayer;
  const cached=this.shaderVariants.get(variantKey);
  if(cached){key=cached.key;code=cached.code;}else{
  if(key.startsWith('routing-')&&this.routingMode){
   const start=code.indexOf('if(l==0u){for(var it='),end=code.indexOf('workgroupBarrier();if(l<24u)',start);
   if(start<0||end<0)throw Error('Routing source layout changed');
   if(this.routingMode==='parallel')code=code.slice(0,start)+`
for(var it=0;it<20;it++){
 if(l<4u){let i=l;let z=vec4f(vals[8u+i*4u],vals[9u+i*4u],vals[10u+i*4u],vals[11u+i*4u]);let ma=max(max(z.x,z.y),max(z.z,z.w));let ex=exp(z-vec4f(ma));let norm=ma+log(ex.x+ex.y+ex.z+ex.w);for(var j=0u;j<4u;j++){vals[8u+i*4u+j]-=norm;}}workgroupBarrier();
 if(l<4u){let j=l;let z=vec4f(vals[8u+j],vals[12u+j],vals[16u+j],vals[20u+j]);let ma=max(max(z.x,z.y),max(z.z,z.w));let ex=exp(z-vec4f(ma));let norm=ma+log(ex.x+ex.y+ex.z+ex.w);for(var i=0u;i<4u;i++){vals[8u+i*4u+j]-=norm;}}workgroupBarrier();}
`+code.slice(end);
   else if(this.routingMode==='register')code=`fn lognorm4(z:vec4f)->vec4f{let ma=max(max(z.x,z.y),max(z.z,z.w));let ex=exp(z-vec4f(ma));return z-vec4f(ma+log(ex.x+ex.y+ex.z+ex.w));}\n`+code.slice(0,start)+`
if(l==0u){var a=vec4f(vals[8],vals[9],vals[10],vals[11]);var b=vec4f(vals[12],vals[13],vals[14],vals[15]);var c=vec4f(vals[16],vals[17],vals[18],vals[19]);var d=vec4f(vals[20],vals[21],vals[22],vals[23]);
for(var it=0;it<20;it++){a=lognorm4(a);b=lognorm4(b);c=lognorm4(c);d=lognorm4(d);let aa=lognorm4(vec4f(a.x,b.x,c.x,d.x));let bb=lognorm4(vec4f(a.y,b.y,c.y,d.y));let cc=lognorm4(vec4f(a.z,b.z,c.z,d.z));let dd=lognorm4(vec4f(a.w,b.w,c.w,d.w));a=vec4f(aa.x,bb.x,cc.x,dd.x);b=vec4f(aa.y,bb.y,cc.y,dd.y);c=vec4f(aa.z,bb.z,cc.z,dd.z);d=vec4f(aa.w,bb.w,cc.w,dd.w);}
for(var j=0u;j<4u;j++){vals[8u+j]=a[j];vals[12u+j]=b[j];vals[16u+j]=c[j];vals[20u+j]=d[j];}}
`+code.slice(end);else throw Error('Routing mode');key=this.routingMode+'-'+key;
  }
  if(this.routeFuseLayer!==null&&this.routeFuseLayer!==undefined){
   const extra=`
@group(0) @binding(3) var<storage,read_write> uu:array<f32>;@group(0) @binding(4) var<storage,read_write> xx:array<f32>;@group(0) @binding(5) var<storage,read_write> xn:array<f32>;@group(0) @binding(6) var<storage,read> scale:array<f32>;var<workgroup> mixed:array<f32,512>;
`;
   code=extra+code.trim().slice(0,-1)+`
workgroupBarrier();var ss2=0.0;for(var c=l;c<512u;c+=128u){var v=0.0;for(var a=0u;a<4u;a++){v+=x[a*512u+c]*vals[a];}uu[c]=v;xx[c]=v;mixed[c]=v;ss2+=v*v;}sum[l]=ss2;workgroupBarrier();for(var h=64u;h>0u;h/=2u){if(l<h){sum[l]+=sum[l+h];}workgroupBarrier();}let inv2=inverseSqrt(sum[0]/512.0+0.000001);workgroupBarrier();var ma2=0.0;for(var c=l;c<512u;c+=128u){mixed[c]=mixed[c]*inv2*(1.0+scale[c]);ma2=max(ma2,abs(mixed[c]));}sum[l]=ma2;workgroupBarrier();for(var h=64u;h>0u;h/=2u){if(l<h){sum[l]=max(sum[l],sum[l+h]);}workgroupBarrier();}let step=select(1.0,sum[0]/127.0,sum[0]>0.0);for(var c=l;c<512u;c+=128u){xn[c]=qr(mixed[c],step);}}
`;key='fused-'+key;
  }
   this.shaderVariants.set(variantKey,{key,code});
  }
  if(this.routeFuseLayer!==null&&this.routeFuseLayer!==undefined)buffers=[...buffers,this.u,this.x,this.xn,this.w(`blocks.${this.routeFuseLayer}.attn_norm.scale`)];
  const [pipeline,bg]=this.kernel(key,code,buffers);let pass;
  if(this.profileMode==='kernels'){
   const index=this.labels.length*2;if(index+1>=2048)throw Error('Timestamp capacity');
   pass=enc.beginComputePass({timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:index,endOfPassWriteIndex:index+1}});
  }else pass=enc._meiPass||(enc._meiPass=enc.beginComputePass(this.profileMode==='whole'?{timestampWrites:{querySet:this.querySet,beginningOfPassWriteIndex:0,endOfPassWriteIndex:1}}:{}));
  pass.setPipeline(pipeline);pass.setBindGroup(0,bg);pass.dispatchWorkgroups(...groups);this.labels.push(key);
  if(this.profileMode==='kernels')pass.end();
 }
 async read(enc,buffer=this.logits,n=24000){
  if(this.compactRead){this.selectToken(enc);buffer=this.argmaxToken;n=1;}
  if(enc._meiPass)enc._meiPass.end();enc.copyBufferToBuffer(buffer,0,this.readback,0,n*4);
  const queries=this.profileMode==='kernels'?this.labels.length*2:this.profileMode==='whole'?2:0;
  if(queries){enc.resolveQuerySet(this.querySet,0,queries,this.queryResolve,0);enc.copyBufferToBuffer(this.queryResolve,0,this.queryRead,0,queries*8);}
  const command=enc.finish(),submitStart=performance.now();this.device.queue.submit([command]);const submitted=performance.now();
  await Promise.all([this.readback.mapAsync(GPUMapMode.READ,0,n*4),...(queries?[this.queryRead.mapAsync(GPUMapMode.READ,0,queries*8)]:[])]);
  const mapped=performance.now();const out=this.compactRead?new Uint32Array(this.readback.getMappedRange(0,n*4)).slice():new Float32Array(this.readback.getMappedRange(0,n*4)).slice();this.readback.unmap();
  const timings=[];if(queries){const ts=new BigUint64Array(this.queryRead.getMappedRange(0,queries*8));for(let i=0;i<queries;i+=2)timings.push({key:queries===2?'whole':this.labels[i/2],gpu_ms:Number(ts[i+1]-ts[i])/1e6});this.queryRead.unmap();}
  this.lastTiming={mode:this.profileMode,encode_ms:submitStart-this.beginTime,submit_ms:submitted-submitStart,submit_to_map_ms:mapped-submitted,copy_ms:performance.now()-mapped,dispatches:this.labels.length,gpu_ms:timings.reduce((a,b)=>a+b.gpu_ms,0),kernels:timings};
  if(this.errors.length)throw Error(this.errors.join('\n'));return out;
 }
 close(){this.querySet?.destroy();this.queryResolve?.destroy();this.queryRead?.destroy();super.close();}
}
