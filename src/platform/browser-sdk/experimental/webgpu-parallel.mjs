// Diagnostic scheduling variants. Same model/weights; default backend unchanged.
import {WebGPU51m} from './webgpu-51m.mjs';
export class ParallelGPU51m extends WebGPU51m {
 static async create(manifest,bytes,config){const model=await WebGPU51m.create(manifest,bytes);Object.setPrototypeOf(model,ParallelGPU51m.prototype);model.config=config;return model;}
 mat(enc,input,weight,out,m,n,k=512){
  const {mv=64}=this.config;const tile=this.config.tile==='auto'?(m<=256?8:16):(this.config.tile??16);
  if(m===1&&mv!==64){
   if(![32,128,256].includes(mv))throw Error('Unsupported workgroup');
   this.run(enc,`parallel-mv-${mv}-${n}-${k}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;var<workgroup> r:array<f32,${mv}>;
@compute @workgroup_size(${mv}) fn main(@builtin(local_invocation_index) l:u32,@builtin(workgroup_id) b:vec3u){var a=0.0;for(var c=l;c<${k}u;c+=${mv}u){a+=x[c]*w[b.x*${k}u+c];}r[l]=a;workgroupBarrier();for(var h=${mv/2}u;h>0u;h/=2u){if(l<h){r[l]+=r[l+h];}workgroupBarrier();}if(l==0u){y[b.x]=r[0];}}
`,[input,weight,out],[n]);return;
  }
  if(m===1||tile===16)return super.mat(enc,input,weight,out,m,n,k);
  if(![8,32].includes(tile))throw Error('Unsupported matrix tile');
  const r=tile/8,size=tile*tile;
  this.run(enc,`parallel-mm-${tile}-${m}-${n}-${k}`,`
@group(0) @binding(0) var<storage,read> x:array<f32>;@group(0) @binding(1) var<storage,read> w:array<f32>;@group(0) @binding(2) var<storage,read_write> y:array<f32>;var<workgroup> a:array<f32,${size}>;var<workgroup> b:array<f32,${size}>;
@compute @workgroup_size(8,8) fn main(@builtin(local_invocation_id) l:vec3u,@builtin(workgroup_id) g:vec3u){let id=l.y*8u+l.x;let row=g.y*${tile}u+l.y*${r}u;let col=g.x*${tile}u+l.x*${r}u;var acc:array<f32,${r*r}>;
for(var start=0u;start<${k}u;start+=${tile}u){for(var z=id;z<${size}u;z+=64u){let rr=g.y*${tile}u+z/${tile}u;let cc=start+z%${tile}u;a[z]=0.0;if(rr<${m}u&&cc<${k}u){a[z]=x[rr*${k}u+cc];}let wr=g.x*${tile}u+z/${tile}u;b[z]=0.0;if(wr<${n}u&&cc<${k}u){b[z]=w[wr*${k}u+cc];}}workgroupBarrier();for(var j=0u;j<${tile}u;j++){for(var u=0u;u<${r}u;u++){for(var v=0u;v<${r}u;v++){acc[u*${r}u+v]+=a[(l.y*${r}u+u)*${tile}u+j]*b[(l.x*${r}u+v)*${tile}u+j];}}}workgroupBarrier();}
for(var u=0u;u<${r}u;u++){for(var v=0u;v<${r}u;v++){if(row+u<${m}u&&col+v<${n}u){y[(row+u)*${n}u+col+v]=acc[u*${r}u+v];}}}}
`,[input,weight,out],[Math.ceil(n/tile),Math.ceil(m/tile)]);
 }
}
