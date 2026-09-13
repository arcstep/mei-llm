const str = {type:'string'};
const tool = (name, description, properties) => ({name, description, parameters:{type:'object',properties,required:Object.keys(properties),additionalProperties:false}});
export const TOOLS = [
 tool('check_required','检查一列是否存在空值。',{column:str}),
 tool('check_unique','检查一列是否有重复值。',{column:str}),
 tool('check_number','检查一列是否为数字并在最小值与最大值之间（包含边界）。',{column:str,min:{type:'number'},max:{type:'number'}}),
 tool('check_enum','检查一列的值是否属于允许值，允许值用英文逗号分隔。',{column:str,allowed:str}),
 tool('check_product','检查数量乘单价是否等于金额，允许误差0.01。',{quantity:str,price:str,total:str}),
];
export const EXAMPLE = '订单号不能为空且不能重复；数量必须在1到1000之间；单价必须在0到100000之间；金额应等于数量乘单价；状态只能是待付款、已付款、已取消。';
export const SAMPLE = [
 ['订单号','数量','单价','金额','状态'],
 ['A001',2,30,60,'已付款'],['A002',3,20,60,'待付款'],
 ['A002',1,15,15,'已付款'],['',2,10,20,'待付款'],
 ['A005',-1,25,-25,'已付款'],['A006',2,40,90,'已付款'],
 ['A007',1,30,30,'未知'],['A008','两件',20,40,'待付款'],
];
