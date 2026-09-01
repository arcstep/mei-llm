//! Portable tensor containers: legacy block-64 Q4/Q2 (read-only) and CQ2 v2.

use std::collections::{HashMap, HashSet};

use serde::Deserialize;
use serde_json::Value;

use crate::cq2::{
    dequantize as dequant_cq2, dequantize_packed_group, scales_from_le_bytes, CqTensor,
    PreparedActivationGroup, GROUP_SIZE,
};
use crate::error::SdkError;

pub const MAGIC: &[u8; 8] = b"MEIQPK01";
pub const CQ2_MAGIC: &[u8; 8] = b"MEICQ201";
pub const PACK_VERSION: u32 = 1;
pub const CQ2_PACK_VERSION: u32 = 2;
pub const QUANT_MATH_ID: &str = "mei-qpack-v1-block64-q4s7-q2u4";
pub const CQ2_QUANT_MATH_ID: &str = "mei-cq-v2-g128-wht-codebook";
pub const BLOCK_SIZE: usize = 64;
pub const Q2_LEVELS: f32 = 1.5;

#[derive(Debug, Clone, Default)]
pub struct TensorEntry {
    pub name: String,
    pub shape: Vec<usize>,
    pub n_params: usize,
    pub bits: u32,
    pub block_size: usize,
    pub group_size_present: bool,
    pub n_blocks: usize,
    pub packed_offset: usize,
    pub packed_nbytes: usize,
    pub scale_offset: usize,
    pub scale_nbytes: usize,
    pub bit_map_offset: usize,
    pub bit_map_nbytes: usize,
    pub dtype: String,
    pub role: String,
    pub transform: String,
    pub codebook: String,
}

#[derive(Debug, Deserialize)]
struct LegacyTensorEntry {
    name: String,
    shape: Vec<usize>,
    n_params: usize,
    bits: u32,
    block_size: usize,
    n_blocks: usize,
    packed_offset: usize,
    packed_nbytes: usize,
    scale_offset: usize,
    scale_nbytes: usize,
    #[serde(default)]
    dtype: String,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Range {
    offset: usize,
    nbytes: usize,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cq2TensorEntry {
    name: String,
    role: String,
    shape: Vec<usize>,
    n_params: usize,
    dtype: String,
    data: Range,
    #[serde(default)]
    scales: Option<Range>,
    #[serde(default)]
    bit_map: Option<Range>,
    #[serde(default)]
    group_size: Option<usize>,
    transform: String,
    codebook: String,
}

#[derive(Debug, Deserialize)]
struct LegacyMeta {
    quant_math_id: String,
    tensors: Vec<LegacyTensorEntry>,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct Cq2Meta {
    quant_math_id: String,
    tensors: Vec<Cq2TensorEntry>,
}

#[derive(Debug, Clone)]
pub struct PackedWeights {
    pub bytes: Vec<u8>,
    pub tensors: HashMap<String, TensorEntry>,
    pub quant_math_id: String,
}

impl PackedWeights {
    pub fn parse(bytes: Vec<u8>) -> Result<Self, SdkError> {
        if bytes.len() < 16 {
            return Err(SdkError::new(
                "package_invalid",
                "tensor container is truncated",
            ));
        }
        let magic = &bytes[..8];
        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        let header_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        if 16usize
            .checked_add(header_len)
            .filter(|end| *end <= bytes.len())
            .is_none()
        {
            return Err(SdkError::new("package_invalid", "truncated tensor header"));
        }
        let header = &bytes[16..16 + header_len];
        let payload_start = 16 + header_len;
        let (quant_math_id, entries): (String, Vec<TensorEntry>) = if magic == MAGIC {
            if version != PACK_VERSION {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("unsupported legacy pack version {version}"),
                ));
            }
            let meta: LegacyMeta = serde_json::from_slice(header)
                .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
            if meta.quant_math_id != QUANT_MATH_ID {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("quant_math_id mismatch: {}", meta.quant_math_id),
                ));
            }
            let entries = meta
                .tensors
                .into_iter()
                .map(|entry| TensorEntry {
                    name: entry.name,
                    shape: entry.shape,
                    n_params: entry.n_params,
                    bits: entry.bits,
                    block_size: entry.block_size,
                    group_size_present: false,
                    n_blocks: entry.n_blocks,
                    packed_offset: entry.packed_offset,
                    packed_nbytes: entry.packed_nbytes,
                    scale_offset: entry.scale_offset,
                    scale_nbytes: entry.scale_nbytes,
                    bit_map_offset: 0,
                    bit_map_nbytes: 0,
                    dtype: entry.dtype,
                    role: String::new(),
                    transform: String::new(),
                    codebook: String::new(),
                })
                .collect();
            (meta.quant_math_id, entries)
        } else if magic == CQ2_MAGIC {
            if version != CQ2_PACK_VERSION {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("unsupported CQ2 pack version {version}"),
                ));
            }
            let meta: Cq2Meta = serde_json::from_slice(header)
                .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
            if meta.quant_math_id != CQ2_QUANT_MATH_ID {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("quant_math_id mismatch: {}", meta.quant_math_id),
                ));
            }
            let entries = meta
                .tensors
                .into_iter()
                .map(|entry| {
                    let group_size_present = entry.group_size.is_some();
                    let group_size = entry.group_size.unwrap_or(128);
                    let groups = if group_size == 0 {
                        0
                    } else {
                        entry.n_params.div_ceil(group_size)
                    };
                    let scales = entry.scales.unwrap_or(Range {
                        offset: 0,
                        nbytes: 0,
                    });
                    let bit_map = entry.bit_map.unwrap_or(Range {
                        offset: 0,
                        nbytes: 0,
                    });
                    TensorEntry {
                        name: entry.name,
                        shape: entry.shape,
                        n_params: entry.n_params,
                        bits: match entry.dtype.as_str() {
                            "cq2" => 2,
                            "cq4" => 4,
                            _ => 16,
                        },
                        block_size: group_size,
                        group_size_present,
                        n_blocks: groups,
                        packed_offset: entry.data.offset,
                        packed_nbytes: entry.data.nbytes,
                        scale_offset: scales.offset,
                        scale_nbytes: scales.nbytes,
                        bit_map_offset: bit_map.offset,
                        bit_map_nbytes: bit_map.nbytes,
                        dtype: entry.dtype,
                        role: entry.role,
                        transform: entry.transform,
                        codebook: entry.codebook,
                    }
                })
                .collect();
            (meta.quant_math_id, entries)
        } else {
            return Err(SdkError::new(
                "package_invalid",
                "tensor container magic mismatch",
            ));
        };
        let mut tensors = HashMap::new();
        let mut names = HashSet::<String>::new();
        let mut ranges = Vec::<(usize, usize, String)>::new();
        for entry in entries {
            if entry.name.is_empty() || !names.insert(entry.name.clone()) {
                return Err(SdkError::new("duplicate_tensor", entry.name));
            }
            if quant_math_id == CQ2_QUANT_MATH_ID && entry.shape.iter().any(|dim| *dim == 0) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{}.shape has a zero dimension", entry.name),
                ));
            }
            let product = entry
                .shape
                .iter()
                .try_fold(1usize, |acc, dim| acc.checked_mul(*dim));
            if product != Some(entry.n_params) {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{}.shape mismatch", entry.name),
                ));
            }
            if quant_math_id == CQ2_QUANT_MATH_ID && matches!(entry.dtype.as_str(), "cq2" | "cq4") {
                let expected_codebook = if entry.dtype == "cq2" {
                    "gaussian-lloyd-q2-v1"
                } else {
                    "gaussian-lloyd-q4-v1"
                };
                if entry.block_size != 128
                    || entry.transform != "wht"
                    || entry.codebook != expected_codebook
                    || !matches!(
                        entry.role.as_str(),
                        "lm" | "contrastive"
                            | "mw_disposition"
                            | "confidence"
                            | "narration_adapter"
                    )
                    || entry.scale_nbytes != entry.n_blocks * 2
                    || entry.bit_map_nbytes != entry.n_blocks.div_ceil(8)
                {
                    return Err(SdkError::new(
                        "package_invalid",
                        format!("{} CQ2 metadata length", entry.name),
                    ));
                }
                let map = slice(&bytes, entry.bit_map_offset, entry.bit_map_nbytes)?;
                if entry.n_blocks % 8 != 0 {
                    let active = entry.n_blocks % 8;
                    let unused_mask = !((1u8 << active) - 1);
                    if map.last().copied().unwrap_or(0) & unused_mask != 0 {
                        return Err(SdkError::new(
                            "package_invalid",
                            format!("{} CQ2 bit map has non-zero unused bits", entry.name),
                        ));
                    }
                }
                let q4_groups = (0..entry.n_blocks)
                    .filter(|group| (map[*group / 8] >> (*group % 8)) & 1 == 1)
                    .count();
                let expected_data = entry.n_blocks * 32 + q4_groups * 32;
                if entry.packed_nbytes != expected_data
                    || (entry.dtype == "cq4" && q4_groups != entry.n_blocks)
                    || (entry.dtype == "cq2" && q4_groups == entry.n_blocks)
                {
                    return Err(SdkError::new(
                        "package_invalid",
                        format!("{} CQ2 bit map/data mismatch", entry.name),
                    ));
                }
                let scales =
                    scales_from_le_bytes(slice(&bytes, entry.scale_offset, entry.scale_nbytes)?)?;
                if scales
                    .into_iter()
                    .map(f16_to_f32)
                    .any(|scale| !scale.is_finite() || scale <= 0.0)
                {
                    return Err(SdkError::new(
                        "package_invalid",
                        format!("{} CQ2 scale must be finite and positive", entry.name),
                    ));
                }
            } else if quant_math_id == CQ2_QUANT_MATH_ID {
                let width = match entry.dtype.as_str() {
                    "f32" => 4,
                    "f16" => 2,
                    "i8" => 1,
                    other => {
                        return Err(SdkError::new(
                            "package_invalid",
                            format!("unsupported tensor dtype {other}"),
                        ))
                    }
                };
                if entry.packed_nbytes != entry.n_params.saturating_mul(width)
                    || entry.scale_nbytes != 0
                    || entry.bit_map_nbytes != 0
                    || entry.group_size_present
                    || entry.transform != "none"
                    || entry.codebook != "none"
                    || !matches!(
                        entry.role.as_str(),
                        "lm" | "contrastive"
                            | "mw_disposition"
                            | "confidence"
                            | "narration_adapter"
                    )
                {
                    return Err(SdkError::new(
                        "package_invalid",
                        format!("{} safe tensor metadata length", entry.name),
                    ));
                }
            }
            for (offset, nbytes, label) in [
                (entry.packed_offset, entry.packed_nbytes, "data"),
                (entry.scale_offset, entry.scale_nbytes, "scales"),
                (entry.bit_map_offset, entry.bit_map_nbytes, "bit_map"),
            ] {
                if nbytes == 0 {
                    continue;
                }
                let end = offset
                    .checked_add(nbytes)
                    .filter(|end| *end <= bytes.len())
                    .ok_or_else(|| {
                        SdkError::new("package_range_invalid", format!("{}.{}", entry.name, label))
                    })?;
                if offset < payload_start {
                    return Err(SdkError::new(
                        "package_range_invalid",
                        format!("{}.{} overlaps the container header", entry.name, label),
                    ));
                }
                ranges.push((offset, end, format!("{}.{}", entry.name, label)));
            }
            tensors.insert(entry.name.clone(), entry);
        }
        ranges.sort_by_key(|range| range.0);
        if quant_math_id == CQ2_QUANT_MATH_ID && (ranges.is_empty() || ranges[0].0 != payload_start)
        {
            return Err(SdkError::new(
                "package_range_invalid",
                "tensor payload must start immediately after the header",
            ));
        }
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(SdkError::new(
                    "package_range_invalid",
                    format!("{} overlaps {}", pair[0].2, pair[1].2),
                ));
            }
            if quant_math_id == CQ2_QUANT_MATH_ID && pair[0].1 != pair[1].0 {
                return Err(SdkError::new(
                    "package_range_invalid",
                    format!("unreferenced gap between {} and {}", pair[0].2, pair[1].2),
                ));
            }
        }
        if quant_math_id == CQ2_QUANT_MATH_ID
            && ranges.last().map(|range| range.1) != Some(bytes.len())
        {
            return Err(SdkError::new(
                "package_range_invalid",
                "tensor directory does not cover the complete payload",
            ));
        }
        Ok(Self {
            bytes,
            tensors,
            quant_math_id,
        })
    }

    pub fn entry(&self, name: &str) -> Result<&TensorEntry, SdkError> {
        self.tensors
            .get(name)
            .ok_or_else(|| SdkError::new("package_invalid", format!("missing tensor {name}")))
    }

    pub fn dequant(&self, name: &str) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        let packed = slice(&self.bytes, entry.packed_offset, entry.packed_nbytes)?;
        if self.quant_math_id == CQ2_QUANT_MATH_ID {
            return self.dequant_v2(entry, packed);
        }
        self.dequant_legacy(entry, packed)
    }

    fn dequant_v2(&self, entry: &TensorEntry, packed: &[u8]) -> Result<Vec<f32>, SdkError> {
        match entry.dtype.as_str() {
            "cq2" | "cq4" => {
                if entry.block_size != 128 {
                    return Err(SdkError::new(
                        "package_invalid",
                        "CQ2 group_size must be 128",
                    ));
                }
                let scales = scales_from_le_bytes(slice(
                    &self.bytes,
                    entry.scale_offset,
                    entry.scale_nbytes,
                )?)?;
                let bit_map =
                    slice(&self.bytes, entry.bit_map_offset, entry.bit_map_nbytes)?.to_vec();
                dequant_cq2(&CqTensor {
                    n_values: entry.n_params,
                    data: packed.to_vec(),
                    scales_f16: scales,
                    bit_map,
                })
            }
            "f32" => decode_f32(packed, entry.n_params),
            "f16" => decode_f16(packed, entry.n_params),
            "i8" => {
                if packed.len() != entry.n_params {
                    return Err(SdkError::new(
                        "package_invalid",
                        "i8 payload length mismatch",
                    ));
                }
                Ok(packed.iter().map(|value| *value as i8 as f32).collect())
            }
            other => Err(SdkError::new(
                "package_invalid",
                format!("unsupported tensor dtype {other}"),
            )),
        }
    }

    fn dequant_legacy(&self, entry: &TensorEntry, packed: &[u8]) -> Result<Vec<f32>, SdkError> {
        if entry.bits >= 32 {
            return decode_f32(packed, entry.n_params);
        }
        let scales = slice(&self.bytes, entry.scale_offset, entry.scale_nbytes)?;
        let mut scale = vec![0f32; entry.n_blocks];
        for (index, chunk) in scales.chunks_exact(4).take(entry.n_blocks).enumerate() {
            scale[index] = f32::from_le_bytes(chunk.try_into().unwrap());
        }
        let block = if entry.block_size == 0 {
            BLOCK_SIZE
        } else {
            entry.block_size
        };
        let n_codes = entry.n_blocks * block;
        let mut out = vec![0f32; n_codes];
        if entry.bits <= 2 {
            let codes = unpack_twobit(packed, n_codes)?;
            for group in 0..entry.n_blocks {
                for index in 0..block {
                    out[group * block + index] =
                        (codes[group * block + index] as f32 - Q2_LEVELS) * scale[group];
                }
            }
        } else {
            let codes = unpack_nibbles(packed, n_codes)?;
            for group in 0..entry.n_blocks {
                for index in 0..block {
                    out[group * block + index] = codes[group * block + index] as f32 * scale[group];
                }
            }
        }
        out.truncate(entry.n_params);
        Ok(out)
    }

    pub fn dequant_row(&self, name: &str, row: usize) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        if entry.shape.len() != 2 {
            return Err(SdkError::new(
                "package_invalid",
                format!("{name} is not rank-2"),
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if row >= rows {
            return Err(SdkError::new("invalid_argument", "row out of range"));
        }
        if self.quant_math_id == CQ2_QUANT_MATH_ID && matches!(entry.dtype.as_str(), "cq2" | "cq4")
        {
            let packed = slice(&self.bytes, entry.packed_offset, entry.packed_nbytes)?;
            let scales = slice(&self.bytes, entry.scale_offset, entry.scale_nbytes)?;
            let bit_map = slice(&self.bytes, entry.bit_map_offset, entry.bit_map_nbytes)?;
            let row_start = row * cols;
            let row_end = row_start + cols;
            let first_group = row_start / GROUP_SIZE;
            let last_group = (row_end - 1) / GROUP_SIZE;
            let mut cursor = cq2_group_offset(bit_map, first_group);
            let mut out = vec![0f32; cols];
            for group in first_group..=last_group {
                let bits = cq2_group_bits(bit_map, group)?;
                let nbytes = GROUP_SIZE * bits as usize / 8;
                let group_bytes = packed
                    .get(cursor..cursor + nbytes)
                    .ok_or_else(|| SdkError::new("package_invalid", "CQ2 data is truncated"))?;
                cursor += nbytes;
                let scale_offset = group * 2;
                let scale_bytes = scales.get(scale_offset..scale_offset + 2).ok_or_else(|| {
                    SdkError::new("package_invalid", "CQ2 f16 scales are truncated")
                })?;
                let block = dequantize_packed_group(
                    group_bytes,
                    bits,
                    u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]),
                )?;
                let group_start = group * GROUP_SIZE;
                let copy_start = row_start.max(group_start);
                let copy_end = row_end.min(group_start + GROUP_SIZE);
                let source_start = copy_start - group_start;
                let target_start = copy_start - row_start;
                out[target_start..target_start + copy_end - copy_start]
                    .copy_from_slice(&block[source_start..source_start + copy_end - copy_start]);
            }
            return Ok(out);
        }
        if self.quant_math_id == CQ2_QUANT_MATH_ID || entry.bits >= 32 {
            let all = self.dequant(name)?;
            return Ok(all[row * cols..(row + 1) * cols].to_vec());
        }
        let all = self.dequant(name)?;
        Ok(all[row * cols..(row + 1) * cols].to_vec())
    }

    pub fn matvec(&self, name: &str, x: &[f32]) -> Result<Vec<f32>, SdkError> {
        let entry = self.entry(name)?;
        if entry.shape.len() != 2 {
            return Err(SdkError::new(
                "package_invalid",
                format!("{name} matvec needs rank-2"),
            ));
        }
        let rows = entry.shape[0];
        let cols = entry.shape[1];
        if x.len() != cols {
            return Err(SdkError::new(
                "invalid_argument",
                format!("{name} expected in={cols} got {}", x.len()),
            ));
        }
        if self.quant_math_id == CQ2_QUANT_MATH_ID && matches!(entry.dtype.as_str(), "cq2" | "cq4")
        {
            let packed = slice(&self.bytes, entry.packed_offset, entry.packed_nbytes)?;
            let scales = slice(&self.bytes, entry.scale_offset, entry.scale_nbytes)?;
            let bit_map = slice(&self.bytes, entry.bit_map_offset, entry.bit_map_nbytes)?;
            if cols % GROUP_SIZE == 0 && entry.n_params == rows * cols {
                let prepared = x
                    .chunks_exact(GROUP_SIZE)
                    .map(PreparedActivationGroup::new)
                    .collect::<Result<Vec<_>, _>>()?;
                let groups_per_row = cols / GROUP_SIZE;
                if entry.n_blocks != rows * groups_per_row {
                    return Err(SdkError::new(
                        "package_invalid",
                        "CQ2 matrix group geometry mismatch",
                    ));
                }
                let mut y = vec![0f32; rows];
                let mut cursor = 0usize;
                for (row, output) in y.iter_mut().enumerate() {
                    let mut acc = 0f32;
                    for (input_group, activation) in prepared.iter().enumerate() {
                        let group = row * groups_per_row + input_group;
                        let bits = cq2_group_bits(bit_map, group)?;
                        let nbytes = GROUP_SIZE * bits as usize / 8;
                        let group_bytes = packed.get(cursor..cursor + nbytes).ok_or_else(|| {
                            SdkError::new("package_invalid", "CQ2 data is truncated")
                        })?;
                        cursor += nbytes;
                        let scale_offset = group * 2;
                        let scale_bytes =
                            scales.get(scale_offset..scale_offset + 2).ok_or_else(|| {
                                SdkError::new("package_invalid", "CQ2 f16 scales are truncated")
                            })?;
                        acc += activation.dot(
                            group_bytes,
                            bits,
                            u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]),
                        )?;
                    }
                    *output = acc;
                }
                if cursor != packed.len() {
                    return Err(SdkError::new(
                        "package_invalid",
                        "CQ2 data has trailing bytes",
                    ));
                }
                return Ok(y);
            }
            let mut y = vec![0f32; rows];
            let mut cursor = 0usize;
            for group in 0..entry.n_blocks {
                let bits = cq2_group_bits(bit_map, group)?;
                let nbytes = GROUP_SIZE * bits as usize / 8;
                let group_bytes = packed
                    .get(cursor..cursor + nbytes)
                    .ok_or_else(|| SdkError::new("package_invalid", "CQ2 data is truncated"))?;
                cursor += nbytes;
                let scale_offset = group * 2;
                let scale_bytes = scales.get(scale_offset..scale_offset + 2).ok_or_else(|| {
                    SdkError::new("package_invalid", "CQ2 f16 scales are truncated")
                })?;
                let block = dequantize_packed_group(
                    group_bytes,
                    bits,
                    u16::from_le_bytes([scale_bytes[0], scale_bytes[1]]),
                )?;
                let group_start = group * GROUP_SIZE;
                let valid = (entry.n_params - group_start).min(GROUP_SIZE);
                for (offset, value) in block[..valid].iter().enumerate() {
                    let flat = group_start + offset;
                    y[flat / cols] += *value * x[flat % cols];
                }
            }
            if cursor != packed.len() {
                return Err(SdkError::new(
                    "package_invalid",
                    "CQ2 data has trailing bytes",
                ));
            }
            return Ok(y);
        }
        let matrix = self.dequant(name)?;
        if matrix.len() != rows * cols {
            return Err(SdkError::new(
                "package_invalid",
                format!("{name} matrix size mismatch"),
            ));
        }
        let mut y = vec![0f32; rows];
        for (row, slot) in y.iter_mut().enumerate() {
            *slot = matrix[row * cols..(row + 1) * cols]
                .iter()
                .zip(x)
                .map(|(a, b)| a * b)
                .sum();
        }
        Ok(y)
    }
}

fn cq2_group_bits(bit_map: &[u8], group: usize) -> Result<u8, SdkError> {
    let byte = bit_map
        .get(group / 8)
        .copied()
        .ok_or_else(|| SdkError::new("package_invalid", "CQ2 bit_map is truncated"))?;
    Ok(if (byte >> (group % 8)) & 1 == 1 { 4 } else { 2 })
}

fn cq2_group_offset(bit_map: &[u8], group: usize) -> usize {
    let full_bytes = group / 8;
    let mut q4_groups = bit_map[..full_bytes.min(bit_map.len())]
        .iter()
        .map(|byte| byte.count_ones() as usize)
        .sum::<usize>();
    let partial = group % 8;
    if partial > 0 {
        if let Some(byte) = bit_map.get(full_bytes) {
            q4_groups += (byte & ((1u8 << partial) - 1)).count_ones() as usize;
        }
    }
    group * 32 + q4_groups * 32
}

fn slice(bytes: &[u8], offset: usize, nbytes: usize) -> Result<&[u8], SdkError> {
    let end = offset
        .checked_add(nbytes)
        .filter(|end| *end <= bytes.len())
        .ok_or_else(|| SdkError::from_id("package_range_invalid"))?;
    Ok(&bytes[offset..end])
}

fn decode_f32(bytes: &[u8], count: usize) -> Result<Vec<f32>, SdkError> {
    if bytes.len() != count * 4 {
        return Err(SdkError::new(
            "package_invalid",
            "f32 payload length mismatch",
        ));
    }
    Ok(bytes
        .chunks_exact(4)
        .map(|chunk| f32::from_le_bytes(chunk.try_into().unwrap()))
        .collect())
}

fn decode_f16(bytes: &[u8], count: usize) -> Result<Vec<f32>, SdkError> {
    if bytes.len() != count * 2 {
        return Err(SdkError::new(
            "package_invalid",
            "f16 payload length mismatch",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| f16_to_f32(u16::from_le_bytes([chunk[0], chunk[1]])))
        .collect())
}

fn f16_to_f32(value: u16) -> f32 {
    let sign = ((value & 0x8000) as u32) << 16;
    let exponent = ((value >> 10) & 0x1f) as i32;
    let mantissa = (value & 0x03ff) as u32;
    let bits = if exponent == 0 {
        if mantissa == 0 {
            sign
        } else {
            let mut mantissa = mantissa;
            let mut exponent = -14i32;
            while mantissa & 0x0400 == 0 {
                mantissa <<= 1;
                exponent -= 1;
            }
            sign | (((exponent + 127) as u32) << 23) | ((mantissa & 0x03ff) << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

fn unpack_nibbles(packed: &[u8], count: usize) -> Result<Vec<i8>, SdkError> {
    if packed.len() < count.div_ceil(2) {
        return Err(SdkError::new("package_invalid", "nibble payload short"));
    }
    Ok((0..count)
        .map(|index| {
            let byte = packed[index / 2];
            let nibble = if index % 2 == 0 {
                byte & 0x0f
            } else {
                byte >> 4
            };
            nibble as i8 - 8
        })
        .collect())
}

fn unpack_twobit(packed: &[u8], count: usize) -> Result<Vec<u8>, SdkError> {
    if packed.len() < count.div_ceil(4) {
        return Err(SdkError::new("package_invalid", "q2 payload short"));
    }
    Ok((0..count)
        .map(|index| (packed[index / 4] >> ((index % 4) * 2)) & 0x03)
        .collect())
}

pub fn arch_u(manifest: &Value, key: &str, default: usize) -> usize {
    manifest
        .get("architecture")
        .and_then(|architecture| architecture.get(key))
        .and_then(Value::as_u64)
        .unwrap_or(default as u64) as usize
}

pub fn arch_f(manifest: &Value, key: &str, default: f32) -> f32 {
    manifest
        .get("architecture")
        .and_then(|architecture| architecture.get(key))
        .and_then(Value::as_f64)
        .unwrap_or(default as f64) as f32
}
