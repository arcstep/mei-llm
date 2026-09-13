// Integration negative cases using an existing real planner contract. No provider calls.
import {createRequire} from 'node:module';import {readFile,readdir} from 'node:fs/promises';import {createHash} from 'node:crypto';import assert from 'node:assert/strict';
import {SAMPLE} from './catalog.mjs';
const XLSX=createRequire(import.meta.url)('./vendor/xlsx.full.min.js');
const origin='http://127.0.0.1:8765';const cfg=await fetch(origin+'/api/config').then(r=>r.json());
const plansDir=new URL('../../../.local/cache/data-check/plans/',import.meta.url);
const names=await readdir(plansDir);let planSha;
for(const name of names){const plan=JSON.parse(await readFile(new URL(name,plansDir)));if(plan.rules.length===6&&plan.rules.some(r=>r.args.column==='订单号')){planSha=name.replace('.json','');break;}}
if(!planSha)throw Error('Run E2E to generate a sample contract first');
const wb=XLSX.utils.book_new();XLSX.utils.book_append_sheet(wb,XLSX.utils.aoa_to_sheet(SAMPLE),'订单');const buf=Buffer.from(XLSX.write(wb,{type:'buffer',bookType:'xlsx'}));
const base={file_base64:buf.toString('base64'),file_sha256:createHash('sha256').update(buf).digest('hex'),plan_sha256:planSha,sheet:'订单',status:'passed'};
const submit=async data=>fetch(origin+'/api/submit',{method:'POST',headers:{'Content-Type':'application/json','X-Demo-Token':cfg.token,Origin:origin},body:JSON.stringify(data)});
assert.equal((await submit(base)).status,422,'client-forged pass must be independently rejected');
assert.equal((await submit({...base,file_sha256:'0'.repeat(64)})).status,400);
assert.equal((await submit({...base,plan_sha256:'0'.repeat(64)})).status,409);
console.log(JSON.stringify({ok:true,forged_pass_rejected:true,file_hash_mismatch_rejected:true,unknown_plan_rejected:true}));
