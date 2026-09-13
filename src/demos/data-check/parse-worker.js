importScripts('./vendor/xlsx.full.min.js');
self.onmessage=({data})=>{
 try {
  if(data.sample){const wb=XLSX.utils.book_new();XLSX.utils.book_append_sheet(wb,XLSX.utils.aoa_to_sheet(data.rows),'订单');const buf=XLSX.write(wb,{type:'array',bookType:data.format||'xlsx'});self.postMessage({id:data.id,buffer:buf},[buf]);return;}
  if(data.buffer.byteLength>10*1024*1024)throw Error('演示版文件上限为10 MiB');
  const wb=XLSX.read(data.buffer,{type:'array',cellFormula:true,cellDates:false,cellNF:true,sheetRows:10002,raw:true});
  if(wb.SheetNames.length>10)throw Error('演示版最多10个工作表');
  const sheets=[];
  for(const name of wb.SheetNames){
   const ws=wb.Sheets[name];if(!ws['!ref'])continue;
   const range=XLSX.utils.decode_range(ws['!fullref']||ws['!ref']);
   if(range.e.r-range.s.r>10000||range.e.c-range.s.c>=100)throw Error('演示版每表上限10000数据行、100列；未截断检查');
   // Preserve physical Excel row numbers. Header must be the first occupied row.
   const rows=XLSX.utils.sheet_to_json(ws,{header:1,raw:true,defval:null,blankrows:true});
   const formulas=[];for(const [address,c] of Object.entries(ws))if(!address.startsWith('!')&&c.f)formulas.push({address,formula:c.f,cached:c.v??null});
   sheets.push({name,rows,rowNumbers:rows.map((_,i)=>range.s.r+i+1),formulas});
  }
  if(!sheets.length)throw Error('没有可读取的工作表');self.postMessage({id:data.id,sheets});
 }catch(e){self.postMessage({id:data.id,error:e.message});}
};
