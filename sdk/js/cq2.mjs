/** Browser-safe MEI CQ2 primitives. Not a .cact binary format. */
export const CQ2_QUANT_MATH_ID = "mei-cq-v2-g128-wht-codebook";
export const CQ2_GROUP_SIZE = 128;
export const Q2_CODEBOOK = Object.freeze([-1.5104176, -0.45278, 0.45278, 1.5104176]);
export const Q4_CODEBOOK = Object.freeze([
  -2.732589, -2.069018, -1.618046, -1.256231, -0.94234, -0.656759, -0.388055, -0.128396,
  0.128396, 0.388055, 0.656759, 0.94234, 1.256231, 1.618046, 2.069018, 2.732589,
]);

function fail(message) {
  throw Object.assign(new Error(message), { code: "package_invalid" });
}

function wht(block) {
  for (let stride = 1; stride < CQ2_GROUP_SIZE; stride *= 2) {
    for (let start = 0; start < CQ2_GROUP_SIZE; start += stride * 2) {
      for (let index = start; index < start + stride; index += 1) {
        const a = block[index];
        const b = block[index + stride];
        block[index] = Math.fround(a + b);
        block[index + stride] = Math.fround(a - b);
      }
    }
  }
  const scale = Math.fround(1 / Math.sqrt(CQ2_GROUP_SIZE));
  for (let index = 0; index < block.length; index += 1) block[index] = Math.fround(block[index] * scale);
}

function f32ToF16(value) {
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setFloat32(0, value, false);
  const bits = view.getUint32(0, false);
  const sign = (bits >>> 16) & 0x8000;
  const sourceExponent = (bits >>> 23) & 0xff;
  const mantissa = bits & 0x7fffff;
  if (sourceExponent === 0xff) return sign | 0x7c00 | (mantissa === 0 ? 0 : 0x0200);
  const exponent = sourceExponent - 127 + 15;
  if (exponent <= 0) {
    if (exponent < -10) return sign;
    const normalized = mantissa | 0x800000;
    const shift = 14 - exponent;
    let half = normalized >>> shift;
    const remainder = normalized & ((2 ** shift) - 1);
    const halfway = 2 ** (shift - 1);
    if (remainder > halfway || (remainder === halfway && (half & 1))) half += 1;
    return sign | half;
  }
  if (exponent >= 31) return sign | 0x7c00;
  let half = sign | (exponent << 10) | (mantissa >>> 13);
  const remainder = mantissa & 0x1fff;
  if (remainder > 0x1000 || (remainder === 0x1000 && (half & 1))) half += 1;
  return half;
}

function f16ToF32(value) {
  const sign = (value & 0x8000) << 16;
  let exponent = (value >>> 10) & 0x1f;
  let mantissa = value & 0x03ff;
  let bits;
  if (exponent === 0) {
    if (mantissa === 0) bits = sign;
    else {
      exponent = -14;
      while ((mantissa & 0x0400) === 0) { mantissa <<= 1; exponent -= 1; }
      bits = sign | ((exponent + 127) << 23) | ((mantissa & 0x03ff) << 13);
    }
  } else if (exponent === 31) bits = sign | 0x7f800000 | (mantissa << 13);
  else bits = sign | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  const buffer = new ArrayBuffer(4);
  const view = new DataView(buffer);
  view.setUint32(0, bits >>> 0, false);
  return view.getFloat32(0, false);
}

function nearest(value, codebook) {
  let best = 0;
  let distance = Infinity;
  for (let index = 0; index < codebook.length; index += 1) {
    const current = Math.abs(value - codebook[index]);
    if (current < distance) { best = index; distance = current; }
  }
  return best;
}

export function quantizeCq2(values, groupBits) {
  if (!values?.length) fail("CQ2 tensor is empty");
  if (Array.from(values).some((value) => !Number.isFinite(value))) fail("CQ2 tensor contains a non-finite value");
  const groups = Math.ceil(values.length / CQ2_GROUP_SIZE);
  if (!Array.isArray(groupBits) || groupBits.length !== groups || groupBits.some((bits) => bits !== 2 && bits !== 4)) {
    fail("CQ2 requires one 2/4-bit selector per group");
  }
  const byteLength = groupBits.reduce((sum, bits) => sum + (CQ2_GROUP_SIZE * bits) / 8, 0);
  const data = new Uint8Array(byteLength);
  const scalesF16 = new Uint16Array(groups);
  const bitMap = new Uint8Array(Math.ceil(groups / 8));
  let cursor = 0;
  for (let group = 0; group < groups; group += 1) {
    const block = new Float32Array(CQ2_GROUP_SIZE);
    block.set(Array.from(values).slice(group * CQ2_GROUP_SIZE, (group + 1) * CQ2_GROUP_SIZE));
    wht(block);
    if (Array.from(block).some((value) => !Number.isFinite(value))) fail("CQ2 transform overflowed");
    let squares = 0;
    for (const value of block) squares = Math.fround(squares + Math.fround(value * value));
    const scale = Math.max(Math.sqrt(squares / CQ2_GROUP_SIZE), 1.17549435e-38);
    // f32 minimum-positive underflows to f16 zero. The portable format
    // requires finite positive scales, so clamp the encoded half to 0x0001.
    const scaleF16 = Math.max(1, f32ToF16(scale));
    if ((scaleF16 & 0x7c00) === 0x7c00) fail("CQ2 scale is not representable as finite f16");
    scalesF16[group] = scaleF16;
    const quantScale = f16ToF32(scaleF16);
    const bits = groupBits[group];
    if (bits === 4) bitMap[group >>> 3] |= 1 << (group & 7);
    const codebook = bits === 2 ? Q2_CODEBOOK : Q4_CODEBOOK;
    const codes = Array.from(block, (value) => nearest(value / quantScale, codebook));
    const perByte = 8 / bits;
    for (let index = 0; index < codes.length; index += perByte) {
      let byte = 0;
      for (let offset = 0; offset < perByte; offset += 1) byte |= codes[index + offset] << (offset * bits);
      data[cursor++] = byte;
    }
  }
  return { n_values: values.length, data, scales_f16: scalesF16, bit_map: bitMap };
}

export function dequantizeCq2(tensor) {
  const groups = Math.ceil(tensor.n_values / CQ2_GROUP_SIZE);
  if (tensor.scales_f16.length !== groups || tensor.bit_map.length < Math.ceil(groups / 8)) fail("CQ2 metadata mismatch");
  const out = new Float32Array(groups * CQ2_GROUP_SIZE);
  let cursor = 0;
  for (let group = 0; group < groups; group += 1) {
    const bits = ((tensor.bit_map[group >>> 3] >>> (group & 7)) & 1) ? 4 : 2;
    const codebook = bits === 2 ? Q2_CODEBOOK : Q4_CODEBOOK;
    const block = new Float32Array(CQ2_GROUP_SIZE);
    for (let index = 0; index < CQ2_GROUP_SIZE; index += 1) {
      const code = (tensor.data[cursor + Math.floor(index * bits / 8)] >>> ((index % (8 / bits)) * bits)) & ((1 << bits) - 1);
      block[index] = Math.fround(codebook[code] * f16ToF32(tensor.scales_f16[group]));
    }
    cursor += (CQ2_GROUP_SIZE * bits) / 8;
    wht(block);
    out.set(block, group * CQ2_GROUP_SIZE);
  }
  if (cursor !== tensor.data.length) fail("CQ2 data has trailing bytes");
  return out.slice(0, tensor.n_values);
}

function sameRange(range) {
  if (range == null) return null;
  if (!range || typeof range !== "object" || Array.isArray(range)
      || Object.keys(range).some((key) => !["offset", "nbytes"].includes(key))) {
    fail("invalid tensor range fields");
  }
  if (!Number.isSafeInteger(range.offset) || !Number.isSafeInteger(range.nbytes) || range.offset < 0 || range.nbytes < 0) {
    fail("invalid tensor range");
  }
  return { offset: range.offset, nbytes: range.nbytes };
}

/** Parse and structurally validate a complete MEICQ201 tensor container. */
export function parseCq2Container(input) {
  const bytes = input instanceof Uint8Array
    ? new Uint8Array(input.buffer, input.byteOffset, input.byteLength)
    : new Uint8Array(input);
  if (bytes.length < 16) fail("tensor container is truncated");
  const magic = String.fromCharCode(...bytes.subarray(0, 8));
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
  const version = view.getUint32(8, true);
  const headerLength = view.getUint32(12, true);
  const payloadStart = 16 + headerLength;
  if (magic !== "MEICQ201" || version !== 2 || payloadStart > bytes.length) {
    fail("unsupported or truncated CQ2 container");
  }
  let header;
  try {
    header = JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(bytes.subarray(16, payloadStart)));
  } catch {
    fail("invalid CQ2 container header JSON");
  }
  if (!header || typeof header !== "object" || Array.isArray(header)
      || Object.keys(header).some((key) => !["quant_math_id", "tensors"].includes(key))
      || header.quant_math_id !== CQ2_QUANT_MATH_ID || !Array.isArray(header.tensors) || !header.tensors.length) {
    fail("invalid CQ2 container header");
  }
  const names = new Set();
  const ranges = [];
  const tensors = [];
  for (const raw of header.tensors) {
    if (!raw || typeof raw !== "object" || Array.isArray(raw)
        || Object.keys(raw).some((key) => ![
          "name", "role", "shape", "n_params", "dtype", "data", "scales", "bit_map",
          "group_size", "transform", "codebook",
        ].includes(key))) fail("invalid CQ2 tensor header fields");
    const name = typeof raw?.name === "string" ? raw.name : "";
    if (!name || names.has(name)) fail(`duplicate or empty tensor name: ${name}`);
    names.add(name);
    if (!Array.isArray(raw.shape) || raw.shape.some((dim) => !Number.isSafeInteger(dim) || dim <= 0)) {
      fail(`${name}.shape is invalid`);
    }
    const nParams = raw.shape.reduce((total, dim) => total * dim, 1);
    if (!Number.isSafeInteger(nParams) || nParams !== raw.n_params) fail(`${name}.n_params mismatch`);
    if (!["cq2", "cq4", "f16", "f32", "i8"].includes(raw.dtype)) fail(`${name}.dtype is unsupported`);
    if (!["lm", "contrastive", "mw_disposition", "confidence", "narration_adapter"].includes(raw.role)) fail(`${name}.role is unsupported`);
    const data = sameRange(raw.data);
    const scales = sameRange(raw.scales);
    const bitMap = sameRange(raw.bit_map);
    if (!data || data.nbytes <= 0) fail(`${name}.data is required`);
    if (["cq2", "cq4"].includes(raw.dtype)) {
      const groups = Math.ceil(nParams / CQ2_GROUP_SIZE);
      const expectedCodebook = raw.dtype === "cq2" ? "gaussian-lloyd-q2-v1" : "gaussian-lloyd-q4-v1";
      if (raw.group_size !== CQ2_GROUP_SIZE || raw.transform !== "wht" || raw.codebook !== expectedCodebook
          || scales?.nbytes !== groups * 2 || bitMap?.nbytes !== Math.ceil(groups / 8)) {
        fail(`${name} CQ2 metadata length mismatch`);
      }
      if (bitMap.offset + bitMap.nbytes > bytes.length || scales.offset + scales.nbytes > bytes.length) {
        fail(`${name} CQ2 metadata is out of bounds`);
      }
      const map = bytes.subarray(bitMap.offset, bitMap.offset + bitMap.nbytes);
      if (groups % 8) {
        const active = groups % 8;
        const unusedMask = (~((1 << active) - 1)) & 0xff;
        if ((map[map.length - 1] & unusedMask) !== 0) fail(`${name} CQ2 bit map has non-zero unused bits`);
      }
      let q4Groups = 0;
      for (let group = 0; group < groups; group += 1) q4Groups += (map[group >>> 3] >>> (group & 7)) & 1;
      const expectedData = groups * 32 + q4Groups * 32;
      if (data.nbytes !== expectedData || (raw.dtype === "cq4" && q4Groups !== groups) || (raw.dtype === "cq2" && q4Groups === groups)) {
        fail(`${name} CQ2 bit map/data mismatch`);
      }
      for (let group = 0; group < groups; group += 1) {
        const scale = f16ToF32(view.getUint16(scales.offset + group * 2, true));
        if (!Number.isFinite(scale) || scale <= 0) fail(`${name} CQ2 scale must be finite and positive`);
      }
    } else {
      const width = { f16: 2, f32: 4, i8: 1 }[raw.dtype];
      if (data.nbytes !== nParams * width || scales != null || bitMap != null
          || raw.group_size != null || raw.transform !== "none" || raw.codebook !== "none") {
        fail(`${name} safe tensor metadata mismatch`);
      }
    }
    for (const [label, range] of [["data", data], ["scales", scales], ["bit_map", bitMap]]) {
      if (!range) continue;
      const end = range.offset + range.nbytes;
      if (range.offset < payloadStart || end > bytes.length) fail(`${name}.${label} is out of bounds`);
      if (range.nbytes) ranges.push({ start: range.offset, end, label: `${name}.${label}` });
    }
    tensors.push({
      name, role: raw.role, shape: [...raw.shape], n_params: nParams, dtype: raw.dtype,
      data, scales, bit_map: bitMap, group_size: raw.group_size ?? null,
      transform: raw.transform, codebook: raw.codebook,
    });
  }
  ranges.sort((left, right) => left.start - right.start);
  if (!ranges.length || ranges[0].start !== payloadStart) fail("tensor payload must start immediately after the header");
  for (let index = 1; index < ranges.length; index += 1) {
    if (ranges[index - 1].end !== ranges[index].start) {
      fail(`tensor ranges overlap or leave a gap: ${ranges[index - 1].label}/${ranges[index].label}`);
    }
  }
  if (ranges[ranges.length - 1].end !== bytes.length) fail("tensor directory does not cover the complete payload");
  return { quant_math_id: header.quant_math_id, tensors, payload_start: payloadStart };
}
