//! Portable CQ2 primitives used by Rust and WASM.
//!
//! This is the MEI `mei-cq-v2-g128-wht-codebook` format. It is deliberately
//! unrelated to `.cact` and does not claim binary compatibility with Cactus.

use crate::error::SdkError;

pub const QUANT_MATH_ID: &str = "mei-cq-v2-g128-wht-codebook";
pub const GROUP_SIZE: usize = 128;
pub const Q2_CODEBOOK: [f32; 4] = [-1.510_417_6, -0.452_78, 0.452_78, 1.510_417_6];
pub const Q4_CODEBOOK: [f32; 16] = [
    -2.732_589, -2.069_018, -1.618_046, -1.256_231, -0.942_34, -0.656_759, -0.388_055, -0.128_396,
    0.128_396, 0.388_055, 0.656_759, 0.942_34, 1.256_231, 1.618_046, 2.069_018, 2.732_589,
];

/// One input group prepared for direct CQ2 matrix-vector multiplication.
///
/// CQ2 stores codebook values in the Walsh-Hadamard domain.  Because the
/// orthonormal WHT is self-adjoint, `H(q) dot x == q dot H(x)`.  Preparing the
/// activation once therefore avoids reconstructing and inverse-transforming
/// the same weight group for every output row.  The Q2 pair table additionally
/// turns every two packed codes into one lookup, while Q4 remains a direct
/// codebook dot product.
#[derive(Debug, Clone)]
pub struct PreparedActivationGroup {
    transformed: [f32; GROUP_SIZE],
    q2_pair_lut: Box<[f32; (GROUP_SIZE / 2) * 16]>,
}

impl PreparedActivationGroup {
    pub fn new(values: &[f32]) -> Result<Self, SdkError> {
        if values.len() != GROUP_SIZE {
            return Err(SdkError::new(
                "invalid_argument",
                "CQ2 activation group must contain 128 values",
            ));
        }
        if values.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::new(
                "invalid_argument",
                "CQ2 activation contains a non-finite value",
            ));
        }
        let mut transformed = [0f32; GROUP_SIZE];
        transformed.copy_from_slice(values);
        wht_orthonormal(&mut transformed);

        let mut q2_pair_lut = Box::new([0f32; (GROUP_SIZE / 2) * 16]);
        for pair in 0..GROUP_SIZE / 2 {
            let left = transformed[pair * 2];
            let right = transformed[pair * 2 + 1];
            let table = &mut q2_pair_lut[pair * 16..(pair + 1) * 16];
            for right_code in 0..4 {
                for left_code in 0..4 {
                    table[left_code | (right_code << 2)] =
                        Q2_CODEBOOK[left_code] * left + Q2_CODEBOOK[right_code] * right;
                }
            }
        }
        Ok(Self {
            transformed,
            q2_pair_lut,
        })
    }

    /// Dot one packed transformed-domain group against this activation.
    pub fn dot(&self, packed: &[u8], bits: u8, scale_f16: u16) -> Result<f32, SdkError> {
        let scale = f16_to_f32(scale_f16);
        if !scale.is_finite() || scale <= 0.0 {
            return Err(SdkError::new(
                "package_invalid",
                "CQ2 group scale must be finite and positive",
            ));
        }
        match bits {
            2 => {
                if packed.len() != GROUP_SIZE / 4 {
                    return Err(SdkError::new(
                        "package_invalid",
                        "CQ2 Q2 group byte length mismatch",
                    ));
                }
                let mut acc = [0f32; 4];
                for (index, byte) in packed.iter().copied().enumerate() {
                    let pair = index * 2;
                    acc[index & 3] += self.q2_pair_lut[pair * 16 + (byte & 0x0f) as usize]
                        + self.q2_pair_lut[(pair + 1) * 16 + (byte >> 4) as usize];
                }
                Ok(((acc[0] + acc[1]) + (acc[2] + acc[3])) * scale)
            }
            4 => {
                if packed.len() != GROUP_SIZE / 2 {
                    return Err(SdkError::new(
                        "package_invalid",
                        "CQ2 Q4 group byte length mismatch",
                    ));
                }
                let mut acc = [0f32; 4];
                for (index, byte) in packed.iter().copied().enumerate() {
                    let offset = index * 2;
                    acc[index & 3] += Q4_CODEBOOK[(byte & 0x0f) as usize]
                        * self.transformed[offset]
                        + Q4_CODEBOOK[(byte >> 4) as usize] * self.transformed[offset + 1];
                }
                Ok(((acc[0] + acc[1]) + (acc[2] + acc[3])) * scale)
            }
            _ => Err(SdkError::new(
                "package_invalid",
                "CQ2 group bit width must be 2 or 4",
            )),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CqTensor {
    pub n_values: usize,
    pub data: Vec<u8>,
    /// IEEE-754 binary16, little endian when serialized.
    pub scales_f16: Vec<u16>,
    /// One bit per group: 0 = Q2, 1 = Q4.
    pub bit_map: Vec<u8>,
}

impl CqTensor {
    pub fn groups(&self) -> usize {
        self.n_values.div_ceil(GROUP_SIZE)
    }

    pub fn bits_for_group(&self, group: usize) -> Result<u8, SdkError> {
        if group >= self.groups() {
            return Err(SdkError::new("invalid_argument", "group out of range"));
        }
        let byte = self
            .bit_map
            .get(group / 8)
            .copied()
            .ok_or_else(|| SdkError::new("package_invalid", "CQ2 bit_map is truncated"))?;
        Ok(if (byte >> (group % 8)) & 1 == 1 { 4 } else { 2 })
    }
}

pub fn quantize(values: &[f32], group_bits: &[u8]) -> Result<CqTensor, SdkError> {
    if values.is_empty() {
        return Err(SdkError::new("invalid_argument", "CQ2 tensor is empty"));
    }
    if values.iter().any(|value| !value.is_finite()) {
        return Err(SdkError::new(
            "invalid_argument",
            "CQ2 tensor contains a non-finite value",
        ));
    }
    let groups = values.len().div_ceil(GROUP_SIZE);
    if group_bits.len() != groups || group_bits.iter().any(|bits| !matches!(bits, 2 | 4)) {
        return Err(SdkError::new(
            "invalid_argument",
            "CQ2 requires one 2/4-bit selector per group",
        ));
    }
    let mut data = Vec::new();
    let mut scales_f16 = Vec::with_capacity(groups);
    let mut bit_map = vec![0u8; groups.div_ceil(8)];
    for group in 0..groups {
        let start = group * GROUP_SIZE;
        let end = (start + GROUP_SIZE).min(values.len());
        let mut block = [0f32; GROUP_SIZE];
        block[..end - start].copy_from_slice(&values[start..end]);
        wht_orthonormal(&mut block);
        if block.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::new(
                "invalid_argument",
                "CQ2 transform overflowed",
            ));
        }
        let scale = (block.iter().map(|x| x * x).sum::<f32>() / GROUP_SIZE as f32)
            .sqrt()
            .max(f32::MIN_POSITIVE);
        // f32::MIN_POSITIVE underflows to binary16 zero. CQ2 forbids zero
        // scales, so all-zero/subnormal groups clamp after conversion to the
        // minimum positive f16 (0x0001) in every implementation.
        let scale_f16 = f32_to_f16(scale).max(1);
        if scale_f16 & 0x7c00 == 0x7c00 {
            return Err(SdkError::new(
                "invalid_argument",
                "CQ2 scale is not representable as finite f16",
            ));
        }
        scales_f16.push(scale_f16);
        // Select codes against the exact scale that is serialized. Using the
        // pre-conversion f32 scale would make subnormal groups decode with
        // different mathematics from the encoder.
        let quant_scale = f16_to_f32(scale_f16);
        let bits = group_bits[group];
        if bits == 4 {
            bit_map[group / 8] |= 1 << (group % 8);
        }
        let codebook: &[f32] = if bits == 2 {
            &Q2_CODEBOOK
        } else {
            &Q4_CODEBOOK
        };
        let codes: Vec<u8> = block
            .iter()
            .map(|value| nearest_code(*value / quant_scale, codebook))
            .collect();
        pack_codes(&codes, bits, &mut data);
    }
    Ok(CqTensor {
        n_values: values.len(),
        data,
        scales_f16,
        bit_map,
    })
}

pub fn dequantize(tensor: &CqTensor) -> Result<Vec<f32>, SdkError> {
    let groups = tensor.groups();
    if tensor.scales_f16.len() != groups || tensor.bit_map.len() < groups.div_ceil(8) {
        return Err(SdkError::new(
            "package_invalid",
            "CQ2 metadata length mismatch",
        ));
    }
    let mut cursor = 0usize;
    let mut out = Vec::with_capacity(groups * GROUP_SIZE);
    for group in 0..groups {
        let bits = tensor.bits_for_group(group)?;
        let nbytes = GROUP_SIZE * bits as usize / 8;
        let packed = tensor
            .data
            .get(cursor..cursor + nbytes)
            .ok_or_else(|| SdkError::new("package_invalid", "CQ2 data is truncated"))?;
        cursor += nbytes;
        let block = dequantize_packed_group(packed, bits, tensor.scales_f16[group])?;
        out.extend_from_slice(&block);
    }
    if cursor != tensor.data.len() {
        return Err(SdkError::new(
            "package_invalid",
            "CQ2 data has trailing bytes",
        ));
    }
    out.truncate(tensor.n_values);
    Ok(out)
}

/// Decode one portable CQ2 group without materializing its containing tensor.
///
/// Portable matrix kernels use this primitive so every 128-value group is
/// reconstructed exactly once.  Keeping this operation here also guarantees
/// that Rust and WASM use the same codebook, f16 and WHT mathematics as the
/// full-tensor reference decoder.
pub fn dequantize_packed_group(
    packed: &[u8],
    bits: u8,
    scale_f16: u16,
) -> Result<[f32; GROUP_SIZE], SdkError> {
    if !matches!(bits, 2 | 4) {
        return Err(SdkError::new(
            "package_invalid",
            "CQ2 group bit width must be 2 or 4",
        ));
    }
    let codes = unpack_codes(packed, bits, GROUP_SIZE)?;
    let codebook: &[f32] = if bits == 2 {
        &Q2_CODEBOOK
    } else {
        &Q4_CODEBOOK
    };
    let scale = f16_to_f32(scale_f16);
    let mut block = [0f32; GROUP_SIZE];
    for (slot, code) in block.iter_mut().zip(codes) {
        *slot = codebook[code as usize] * scale;
    }
    // Orthonormal WHT is its own inverse.
    wht_orthonormal(&mut block);
    Ok(block)
}

pub fn scales_to_le_bytes(scales: &[u16]) -> Vec<u8> {
    scales.iter().flat_map(|x| x.to_le_bytes()).collect()
}

pub fn scales_from_le_bytes(bytes: &[u8]) -> Result<Vec<u16>, SdkError> {
    if bytes.len() % 2 != 0 {
        return Err(SdkError::new(
            "package_invalid",
            "CQ2 f16 scales are truncated",
        ));
    }
    Ok(bytes
        .chunks_exact(2)
        .map(|chunk| u16::from_le_bytes([chunk[0], chunk[1]]))
        .collect())
}

fn nearest_code(value: f32, codebook: &[f32]) -> u8 {
    let mut best = 0usize;
    let mut distance = f32::INFINITY;
    for (index, candidate) in codebook.iter().enumerate() {
        let current = (value - candidate).abs();
        if current < distance {
            distance = current;
            best = index;
        }
    }
    best as u8
}

fn pack_codes(codes: &[u8], bits: u8, out: &mut Vec<u8>) {
    match bits {
        2 => {
            for chunk in codes.chunks(4) {
                let mut byte = 0u8;
                for (index, code) in chunk.iter().enumerate() {
                    byte |= (code & 0x03) << (index * 2);
                }
                out.push(byte);
            }
        }
        4 => {
            for chunk in codes.chunks(2) {
                let lo = chunk[0] & 0x0f;
                let hi = chunk.get(1).copied().unwrap_or(0) & 0x0f;
                out.push(lo | (hi << 4));
            }
        }
        _ => unreachable!("validated bits"),
    }
}

fn unpack_codes(bytes: &[u8], bits: u8, count: usize) -> Result<Vec<u8>, SdkError> {
    let expected = count * bits as usize / 8;
    if bytes.len() != expected {
        return Err(SdkError::new(
            "package_invalid",
            "CQ2 group byte length mismatch",
        ));
    }
    let mut out = Vec::with_capacity(count);
    for index in 0..count {
        let code = if bits == 2 {
            (bytes[index / 4] >> ((index % 4) * 2)) & 0x03
        } else {
            (bytes[index / 2] >> ((index % 2) * 4)) & 0x0f
        };
        out.push(code);
    }
    Ok(out)
}

fn wht_orthonormal(values: &mut [f32; GROUP_SIZE]) {
    let mut stride = 1usize;
    while stride < GROUP_SIZE {
        for start in (0..GROUP_SIZE).step_by(stride * 2) {
            for index in start..start + stride {
                let a = values[index];
                let b = values[index + stride];
                values[index] = a + b;
                values[index + stride] = a - b;
            }
        }
        stride *= 2;
    }
    let scale = (GROUP_SIZE as f32).sqrt().recip();
    for value in values {
        *value *= scale;
    }
}

// Deterministic IEEE-754 binary32 -> binary16 round-to-nearest, ties-to-even,
// without an extra dependency.
fn f32_to_f16(value: f32) -> u16 {
    let bits = value.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let source_exponent = ((bits >> 23) & 0xff) as i32;
    let mantissa = bits & 0x7f_ffff;
    if source_exponent == 0xff {
        return sign | 0x7c00 | if mantissa == 0 { 0 } else { 0x0200 };
    }
    let exponent = source_exponent - 127 + 15;
    if exponent <= 0 {
        if exponent < -10 {
            return sign;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = 14 - exponent;
        let mut half = (mantissa >> shift) as u16;
        let remainder_mask = (1u32 << shift) - 1;
        let remainder = mantissa & remainder_mask;
        let halfway = 1u32 << (shift - 1);
        if remainder > halfway || (remainder == halfway && half & 1 != 0) {
            half = half.wrapping_add(1);
        }
        sign | half
    } else if exponent >= 31 {
        sign | 0x7c00
    } else {
        let mut half = sign | ((exponent as u16) << 10) | ((mantissa >> 13) as u16);
        let remainder = mantissa & 0x1fff;
        if remainder > 0x1000 || (remainder == 0x1000 && half & 1 != 0) {
            half = half.wrapping_add(1);
        }
        half
    }
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
            mantissa &= 0x03ff;
            sign | (((exponent + 127) as u32) << 23) | (mantissa << 13)
        }
    } else if exponent == 31 {
        sign | 0x7f80_0000 | (mantissa << 13)
    } else {
        sign | (((exponent - 15 + 127) as u32) << 23) | (mantissa << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mixed_group_pack_is_deterministic_and_bounded() {
        let values: Vec<f32> = (0..256)
            .map(|index| ((index as f32 * 0.173).sin() * 0.8) + ((index % 11) as f32 - 5.0) * 0.03)
            .collect();
        let packed = quantize(&values, &[2, 4]).unwrap();
        assert_eq!(packed.data.len(), 32 + 64);
        assert_eq!(packed.scales_f16.len(), 2);
        assert_eq!(packed.bit_map, vec![0b0000_0010]);
        assert_eq!(quantize(&values, &[2, 4]).unwrap(), packed);
        let decoded = dequantize(&packed).unwrap();
        let mse = values
            .iter()
            .zip(decoded)
            .map(|(a, b)| (a - b) * (a - b))
            .sum::<f32>()
            / values.len() as f32;
        assert!(mse < 0.15, "unexpected CQ2 mse {mse}");
    }

    #[test]
    fn invalid_group_map_fails_closed() {
        assert!(quantize(&[1.0; 129], &[2]).is_err());
        assert!(quantize(&[1.0; 128], &[3]).is_err());
    }

    #[test]
    fn zero_group_uses_minimum_positive_f16_scale() {
        let packed = quantize(&[0.0; GROUP_SIZE], &[2]).unwrap();
        assert_eq!(packed.scales_f16, vec![1]);
        assert!(packed.data.iter().all(|byte| *byte == 0x55));
        assert!(dequantize(&packed).unwrap().into_iter().all(f32::is_finite));
    }

    #[test]
    fn subnormal_group_quantizes_against_serialized_scale() {
        let packed = quantize(&[1.0e-8; GROUP_SIZE], &[2]).unwrap();
        assert_eq!(packed.scales_f16, vec![1]);
        assert_eq!(
            hex::encode(&packed.data),
            "5755555555555555555555555555555555555555555555555555555555555555"
        );
        assert_eq!(quantize(&[1.0e-8; GROUP_SIZE], &[2]).unwrap(), packed);
        assert!(dequantize(&packed).unwrap().into_iter().all(f32::is_finite));
    }

    #[test]
    fn rejects_non_finite_or_unrepresentable_scale() {
        assert!(quantize(&[f32::NAN; GROUP_SIZE], &[2]).is_err());
        assert!(quantize(&[f32::INFINITY; GROUP_SIZE], &[2]).is_err());
        assert!(quantize(&[f32::MAX; GROUP_SIZE], &[2]).is_err());
    }

    #[test]
    fn transformed_domain_dot_matches_reconstructed_weights() {
        let weights: Vec<f32> = (0..GROUP_SIZE)
            .map(|index| ((index as f32 * 0.071).sin() * 0.8) + ((index % 9) as f32 - 4.0) * 0.025)
            .collect();
        let activation: Vec<f32> = (0..GROUP_SIZE)
            .map(|index| ((index as f32 * 0.113).cos() * 0.6) - 0.07)
            .collect();
        let prepared = PreparedActivationGroup::new(&activation).unwrap();
        for bits in [2, 4] {
            let packed = quantize(&weights, &[bits]).unwrap();
            let reconstructed = dequantize(&packed).unwrap();
            let expected = reconstructed
                .iter()
                .zip(&activation)
                .map(|(weight, input)| weight * input)
                .sum::<f32>();
            let actual = prepared
                .dot(&packed.data, bits, packed.scales_f16[0])
                .unwrap();
            assert!(
                (actual - expected).abs() <= 2.0e-5,
                "{bits}-bit transformed dot {actual} != reconstructed {expected}"
            );
        }
    }
}
