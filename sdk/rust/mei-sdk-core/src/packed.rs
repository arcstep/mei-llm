//! Packed Q2/Q4 weight container matching Python `quant_pack_51m.py`.

use std::collections::HashMap;

use serde::Deserialize;
use serde_json::Value;

use crate::error::SdkError;

pub const MAGIC: &[u8; 8] = b"MEIQPK01";
pub const PACK_VERSION: u32 = 1;
pub const QUANT_MATH_ID: &str = "mei-qpack-v1-block64-q4s7-q2u4";
pub const BLOCK_SIZE: usize = 64;
pub const Q4_LEVELS: f32 = 7.0;
pub const Q2_LEVELS: f32 = 1.5;

#[derive(Debug, Clone, Deserialize)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub n_params: usize,
    pub bits: u32,
    pub block_size: usize,
    pub n_blocks: usize,
    pub packed_offset: usize,
    pub packed_nbytes: usize,
    pub scale_offset: usize,
    pub scale_nbytes: usize,
    #[serde(default)]
    pub dtype: String,
}

#[derive(Debug, Clone, Deserialize)]
struct PackMeta {
    quant_math_id: String,
    tensors: Vec<TensorEntry>,
}

#[derive(Debug, Clone)]
pub struct PackedWeights {
    pub bytes: Vec<u8>,
    pub tensors: HashMap<String, TensorEntry>,
}

impl PackedWeights {
    pub fn parse(bytes: Vec<u8>) -> Result<Self, SdkError> {
        if bytes.len() < 16 || &bytes[..8] != MAGIC {
            return Err(SdkError::new("package_invalid", "weights.q4 magic mismatch"));
        }
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != PACK_VERSION {
            return Err(SdkError::new(
                "package_invalid",
                format!("unsupported pack version {version}"),
            ));
        }
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if 16 + header_len > bytes.len() {
            return Err(SdkError::new("package_invalid", "truncated pack header"));
        }
        let meta: PackMeta = serde_json::from_slice(&bytes[16..16 + header_len])
            .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
        if meta.quant_math_id != QUANT_MATH_ID {
            return Err(SdkError::new(
                "package_invalid",
                format!("quant_math_id mismatch: {}", meta.quant_math_id),
            ));
        }
        let mut tensors = HashMap::new();
        for entry in meta.tensors {
            tensors.insert(entry.name.clone(), entry);
        }
        Ok(Self { bytes, tensors })
    }

    pub fn entry(&self, name: &str) -> Result<&TensorEntry, SdkError> {
        self.tensors
            .get(name)
            .ok_or_else(|| SdkError::new("package_invalid", format!("missing tensor {name}")))
    }

    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        let packed = slice(&self.bytes, entry.packed_offset, entry.packed_nbytes)?;
        if entry.bits >= 32 {
            if packed.len() < entry.n_params * 4 {
                return Err(SdkError::new("package_invalid", format!("{name} f32 payload short")));
            }
            let mut out = Vec::with_capacity(entry.n_params);
            for chunk in packed.chunks_exact(4).take(entry.n_params) {
                out.push(f32::from_le_bytes(chunk.try_into().unwrap()));
            }
            return Ok(out);
        }
        let scales = slice(&self.bytes, entry.scale_offset, entry.scale_nbytes)?;
        let n_blocks = entry.n_blocks;
        let mut scale = vec![0f32; n_blocks];
        for (i, chunk) in scales.chunks_exact(4).take(n_blocks).enumerate() {
            scale[i] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        let block = if entry.block_size == 0 {
            BLOCK_SIZE
        } else {
            entry.block_size
        };
        let n_codes = n_blocks * block;
        let mut out = vec![0f32; n_codes];
        if entry.bits <= 2 {
            let codes = unpack_twobit(packed, n_codes)?;
            for b in 0..n_blocks {
                let s = scale[b];
                for j in 0..block {
                    let q = codes[b * block + j] as f32;
                    out[b * block + j] = (q - Q2_LEVELS) * s;
                }
            }
        } else {
            let codes = unpack_nibbles(packed, n_codes)?;
            for b in 0..n_blocks {
                let s = scale[b];
                for j in 0..block {
                    out[b * block + j] = (codes[b * block + j] as f32) * s;
                }
            }
        }
        out.truncate(entry.n_params);
        Ok(out)
    }

    /// Dequantize one row of a 2-D weight (row-major, last dim = cols).
    pub fn dequant_row(&self, name: &str, row: usize) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        if entry.shape.len() != 2 {
            return Err(SdkError::new("package_invalid", format!("{name} is not rank-2")));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if row >= rows {
            return Err(SdkError::new("invalid_argument", "row out of range"));
        }
        if entry.bits >= 32 {
            let all = self.dequant(name)?;
            return Ok(all[row * cols..(row + 1) * cols].to_vec());
        }
        let block = if entry.block_size == 0 {
            BLOCK_SIZE
        } else {
            entry.block_size
        };
        let start = row * cols;
        let n_blocks_all = entry.n_blocks;
        let first_block = start / block;
        let last_block = (start + cols + block - 1) / block;
        let packed = slice(&self.bytes, entry.packed_offset, entry.packed_nbytes)?;
        let scales = slice(&self.bytes, entry.scale_offset, entry.scale_nbytes)?;
        let mut out = vec![0f32; cols];
        for b in first_block..last_block.min(n_blocks_all) {
            let s_off = b * 4;
            if s_off + 4 > scales.len() {
                break;
            }
            let s = f32::from_le_bytes(scales[s_off..s_off + 4].try_into().unwrap());
            for j in 0..block {
                let idx = b * block + j;
                if idx < start || idx >= start + cols {
                    continue;
                }
                let q = if entry.bits <= 2 {
                    read_twobit(packed, idx)? as f32
                } else {
                    read_nibble(packed, idx)? as f32
                };
                let val = if entry.bits <= 2 {
                    (q - Q2_LEVELS) * s
                } else {
                    q * s
                };
                out[idx - start] = val;
            }
        }
        Ok(out)
    }

    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        if entry.shape.len() != 2 {
            return Err(SdkError::new("package_invalid", format!("{name} matvec needs rank-2")));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if x.len() != cols {
            return Err(SdkError::new(
                "invalid_argument",
                format!("{name} expected in={cols} got {}", x.len()),
            ));
        }
        let mut y = vec![0f32; rows];
        for r in 0..rows {
            let row = self.dequant_row(name, r)?;
            let mut acc = 0f32;
            for (a, b) in row.iter().zip(x.iter()) {
                acc += a * b;
            }
            y[r] = acc;
        }
        Ok(y)
    }
}

fn slice(bytes: &[u8], off: usize, n: usize) -> Result<&[u8], SdkError> {
    if off + n > bytes.len() {
        return Err(SdkError::new("package_invalid", "packed slice out of range"));
    }
    Ok(&bytes[off..off + n])
}

fn unpack_nibbles(packed: &[u8], n: usize) -> Result<Vec<i8>, SdkError> {
    let need = (n + 1) / 2;
    if packed.len() < need {
        return Err(SdkError::new("package_invalid", "nibble payload short"));
    }
    let mut out = vec![0i8; n];
    for i in 0..n {
        let byte = packed[i / 2];
        let nib = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
        out[i] = nib as i8 - 8;
    }
    Ok(out)
}

fn unpack_twobit(packed: &[u8], n: usize) -> Result<Vec<u8>, SdkError> {
    let need = (n + 3) / 4;
    if packed.len() < need {
        return Err(SdkError::new("package_invalid", "q2 payload short"));
    }
    let mut out = vec![0u8; n];
    for i in 0..n {
        let byte = packed[i / 4];
        let shift = (i % 4) * 2;
        out[i] = (byte >> shift) & 0x03;
    }
    Ok(out)
}

fn read_nibble(packed: &[u8], i: usize) -> Result<i8, SdkError> {
    let byte = packed.get(i / 2).copied().ok_or_else(|| SdkError::new("package_invalid", "nibble oob"))?;
    let nib = if i % 2 == 0 { byte & 0x0f } else { byte >> 4 };
    Ok(nib as i8 - 8)
}

fn read_twobit(packed: &[u8], i: usize) -> Result<u8, SdkError> {
    let byte = packed.get(i / 4).copied().ok_or_else(|| SdkError::new("package_invalid", "q2 oob"))?;
    let shift = (i % 4) * 2;
    Ok((byte >> shift) & 0x03)
}

pub fn arch_u(manifest: &Value, key: &str, default: usize) -> usize {
    manifest
        .get("architecture")
        .and_then(|a| a.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(default as u64) as usize
}

pub fn arch_f(manifest: &Value, key: &str, default: f32) -> f32 {
    manifest
        .get("architecture")
        .and_then(|a| a.get(key))
        .and_then(Value::as_f64)
        .unwrap_or(default as f64) as f32
}
