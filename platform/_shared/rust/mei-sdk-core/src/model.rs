//! CPU Needle-zh 51M forward over packed Q4/Q2 weights.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::{json, Value};

use crate::error::SdkError;
use crate::packed::{arch_f, arch_u, PackedWeights, PreparedMatVecInput, BLOCK_SIZE};

const ENGRAM_SUB_DIM: usize = 128;
const ENGRAM_SEED: u32 = 0x9E3779B9;
const ENGRAM_PRIME: u32 = 0x01000193;
pub const MAX_CONTEXT_TOKENS: usize = 2048;
pub const STABLE_PREFIX_TOKENS_MAX: usize = 1536;
/// Compatibility floor only. Adaptive v2 uses the dynamic remainder
/// `MAX_CONTEXT_TOKENS - actual_stable_prefix`, not this value as a hard cap.
pub const ROLLING_WINDOW_TOKENS: usize = 256;

fn ordinary_capacity(stable_prefix_tokens: usize) -> usize {
    MAX_CONTEXT_TOKENS.saturating_sub(stable_prefix_tokens)
}
/// WASM keeps repeatedly-used structural tensors, but never large projection
/// matrices or Engram tables, as dequantized f32. The current 51M package keeps
/// the resulting cache below 16 MiB while avoiding per-token reconstruction of
/// mHC, normalization, Hadamard-MLP and sidecar parameters.
const PORTABLE_TENSOR_CACHE_MAX_PARAMS: usize = 1_000_000;
const PORTABLE_PREFILL_CHUNK_TOKENS: usize = 16;

#[derive(Clone, Copy)]
enum LogitsMode {
    All,
    Last,
    None,
}

#[derive(Clone)]
pub struct Arch {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rope_theta: f32,
    pub rms_eps: f32,
    pub engram_layers: Vec<usize>,
    pub engram_orders: Vec<usize>,
    pub engram_slots: usize,
    pub engram_conv_taps: usize,
    pub mhc_lanes: usize,
    pub sinkhorn_iters: usize,
    pub conf_probes: usize,
    pub tie_embeddings: bool,
    /// Read-only v1 packages retain their historical global-scale/q-QDQ math.
    /// Native v2 packages use the canonical per-vector activation contract.
    pub legacy_activation_qdq: bool,
}

impl Arch {
    pub fn from_manifest(manifest: &Value) -> Self {
        let d_model = arch_u(manifest, "d_model", 512);
        let n_heads = arch_u(manifest, "n_heads", 8);
        let engram_layers = manifest
            .get("architecture")
            .and_then(|a| a.get("engram_layers"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_else(|| vec![2, 15]);
        let engram_orders = manifest
            .get("architecture")
            .and_then(|a| a.get("engram_orders"))
            .and_then(Value::as_array)
            .map(|a| {
                a.iter()
                    .filter_map(Value::as_u64)
                    .map(|x| x as usize)
                    .collect()
            })
            .unwrap_or_else(|| vec![2, 3]);
        Self {
            d_model,
            n_layers: arch_u(manifest, "n_layers", 27),
            n_heads,
            n_kv_heads: arch_u(manifest, "n_kv_heads", 4),
            head_dim: arch_u(manifest, "head_dim", d_model / n_heads.max(1)),
            vocab_size: arch_u(manifest, "vocab_size", 24000),
            rope_theta: arch_f(manifest, "rope_theta", 100000.0),
            rms_eps: arch_f(manifest, "rms_eps", 1e-6),
            engram_layers,
            engram_orders,
            engram_slots: arch_u(manifest, "engram_slots", 8192),
            engram_conv_taps: arch_u(manifest, "engram_conv_taps", 4),
            mhc_lanes: arch_u(manifest, "mhc_lanes", 4),
            sinkhorn_iters: arch_u(manifest, "sinkhorn_iters", 20),
            conf_probes: arch_u(manifest, "conf_probes", 8),
            tie_embeddings: manifest
                .get("architecture")
                .and_then(|a| a.get("tie_embeddings"))
                .and_then(Value::as_bool)
                .unwrap_or(true),
            legacy_activation_qdq: manifest.get("package_format").and_then(Value::as_str)
                != Some("mei-model-package-v2"),
        }
    }
}

#[derive(Clone, Default)]
pub struct LayerCache {
    /// KV storage is genuinely int8. Scales are per `[head_dim]` key/value
    /// vector, matching the QAT contract and keeping one token/head independent
    /// from unrelated cache entries.
    pub k: Vec<i8>, // [t, n_kv, head_dim]
    pub v: Vec<i8>,
    pub k_scales: Vec<f32>, // [t, n_kv]
    pub v_scales: Vec<f32>,
    pub vector_width: usize,
    pub t: usize,
    pub stable_t: usize,
    pub rolling_t: usize,
    pub next_position: usize,
}

impl LayerCache {
    fn from_f32(
        k: &[f32],
        v: &[f32],
        t: usize,
        stable_t: usize,
        next_position: usize,
        vector_width: usize,
    ) -> Self {
        let (k, k_scales) = quantize_i8_vectors(k, vector_width);
        let (v, v_scales) = quantize_i8_vectors(v, vector_width);
        let stable_t = stable_t.min(t).min(STABLE_PREFIX_TOKENS_MAX);
        Self {
            k,
            v,
            k_scales,
            v_scales,
            vector_width,
            t,
            stable_t,
            rolling_t: t.saturating_sub(stable_t),
            next_position,
        }
    }

    fn dequant_k(&self) -> Vec<f32> {
        dequantize_i8_vectors(&self.k, &self.k_scales, self.vector_width)
    }

    fn dequant_v(&self) -> Vec<f32> {
        dequantize_i8_vectors(&self.v, &self.v_scales, self.vector_width)
    }

    fn key_dot(&self, token: usize, kv_head: usize, n_kv_heads: usize, q: &[f32]) -> f32 {
        debug_assert_eq!(q.len(), self.vector_width);
        let vector = token * n_kv_heads + kv_head;
        let offset = vector * self.vector_width;
        let scale = self.k_scales[vector];
        #[cfg(target_arch = "wasm32")]
        if self.vector_width % 16 == 0 {
            // SAFETY: both slices have `vector_width` elements from `offset`,
            // checked by cache construction and the debug assertion above.
            return unsafe {
                wasm_i8_f32_dot(&self.k[offset..offset + self.vector_width], q, scale)
            };
        }
        let mut dot = 0.0f32;
        for (code, query) in self.k[offset..offset + self.vector_width].iter().zip(q) {
            let key = *code as f32 * scale;
            dot += key * *query;
        }
        dot
    }

    fn add_scaled_value(
        &self,
        token: usize,
        kv_head: usize,
        n_kv_heads: usize,
        weight: f32,
        target: &mut [f32],
    ) {
        debug_assert_eq!(target.len(), self.vector_width);
        let vector = token * n_kv_heads + kv_head;
        let offset = vector * self.vector_width;
        let scale = self.v_scales[vector];
        #[cfg(target_arch = "wasm32")]
        if self.vector_width % 16 == 0 {
            // SAFETY: source and destination contain `vector_width` elements;
            // unaligned v128 loads/stores are defined by WebAssembly SIMD.
            unsafe {
                wasm_add_scaled_i8(
                    &self.v[offset..offset + self.vector_width],
                    scale,
                    weight,
                    target,
                )
            };
            return;
        }
        for (slot, code) in target
            .iter_mut()
            .zip(&self.v[offset..offset + self.vector_width])
        {
            let value = *code as f32 * scale;
            *slot += weight * value;
        }
    }

    pub fn storage_bytes(&self) -> usize {
        self.k.len()
            + self.v.len()
            + (self.k_scales.len() + self.v_scales.len()) * std::mem::size_of::<f32>()
    }
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wasm_i8_f32_dot(codes: &[i8], query: &[f32], scale: f32) -> f32 {
    use core::arch::wasm32::*;

    debug_assert_eq!(codes.len(), query.len());
    debug_assert_eq!(codes.len() % 16, 0);
    let scale4 = f32x4_splat(scale);
    let mut sums = [f32x4_splat(0.0); 4];
    let mut offset = 0usize;
    while offset < codes.len() {
        let packed = v128_load(codes.as_ptr().add(offset).cast::<v128>());
        let halves = [
            i16x8_extend_low_i8x16(packed),
            i16x8_extend_high_i8x16(packed),
        ];
        for (half_index, half) in halves.into_iter().enumerate() {
            let quarters = [i32x4_extend_low_i16x8(half), i32x4_extend_high_i16x8(half)];
            for (quarter_index, ints) in quarters.into_iter().enumerate() {
                let lane = half_index * 2 + quarter_index;
                let keys = f32x4_mul(f32x4_convert_i32x4(ints), scale4);
                let q = v128_load(query.as_ptr().add(offset + lane * 4).cast::<v128>());
                sums[lane] = f32x4_add(sums[lane], f32x4_mul(keys, q));
            }
        }
        offset += 16;
    }
    let pair01 = f32x4_add(sums[0], sums[1]);
    let pair23 = f32x4_add(sums[2], sums[3]);
    let mut lanes = [0.0f32; 4];
    v128_store(lanes.as_mut_ptr().cast::<v128>(), f32x4_add(pair01, pair23));
    (lanes[0] + lanes[1]) + (lanes[2] + lanes[3])
}

#[cfg(target_arch = "wasm32")]
#[inline(always)]
unsafe fn wasm_add_scaled_i8(codes: &[i8], scale: f32, weight: f32, target: &mut [f32]) {
    use core::arch::wasm32::*;

    debug_assert_eq!(codes.len(), target.len());
    debug_assert_eq!(codes.len() % 16, 0);
    let scale4 = f32x4_splat(scale);
    let weight4 = f32x4_splat(weight);
    let mut offset = 0usize;
    while offset < codes.len() {
        let packed = v128_load(codes.as_ptr().add(offset).cast::<v128>());
        let halves = [
            i16x8_extend_low_i8x16(packed),
            i16x8_extend_high_i8x16(packed),
        ];
        for (half_index, half) in halves.into_iter().enumerate() {
            let quarters = [i32x4_extend_low_i16x8(half), i32x4_extend_high_i16x8(half)];
            for (quarter_index, ints) in quarters.into_iter().enumerate() {
                let lane_offset = offset + (half_index * 2 + quarter_index) * 4;
                let values = f32x4_mul(f32x4_convert_i32x4(ints), scale4);
                let contribution = f32x4_mul(values, weight4);
                let current = v128_load(target.as_ptr().add(lane_offset).cast::<v128>());
                v128_store(
                    target.as_mut_ptr().add(lane_offset).cast::<v128>(),
                    f32x4_add(current, contribution),
                );
            }
        }
        offset += 16;
    }
}

pub fn runtime_contract_evidence() -> Value {
    json!({
        "kv_storage_dtype": "int8",
        "activation_quantization": "int8-qdq",
        "activation_quantization_semantics": "int8-symmetric-per-last-axis-vector-qdq",
        "activation_q_quantized": false,
        "kv_scale_granularity": "per-head-vector",
        "stable_prefix_tokens_max": STABLE_PREFIX_TOKENS_MAX,
        "stable_prefix_profiles": {"compact":1024,"standard":1536},
        "ordinary_window_policy": "dynamic_remainder",
        "rolling_window_compatibility_floor": ROLLING_WINDOW_TOKENS,
        "max_context_tokens": MAX_CONTEXT_TOKENS,
        "cache_growth_bounded": true
    })
}

pub fn cache_evidence(layers: &[LayerCache]) -> Value {
    let first = layers.first();
    let consistent = first.is_some()
        && layers.iter().all(|layer| {
            layer.t == first.unwrap().t
                && layer.stable_t == first.unwrap().stable_t
                && layer.rolling_t == first.unwrap().rolling_t
                && layer.k.len() == layer.v.len()
                && layer.vector_width > 0
                && layer.k.len() % layer.vector_width == 0
                && layer.k_scales.len() == layer.k.len() / layer.vector_width
                && layer.v_scales.len() == layer.v.len() / layer.vector_width
                && layer
                    .k_scales
                    .iter()
                    .all(|scale| scale.is_finite() && *scale > 0.0)
                && layer
                    .v_scales
                    .iter()
                    .all(|scale| scale.is_finite() && *scale > 0.0)
        });
    let stable = first.map(|layer| layer.stable_t).unwrap_or(0);
    let rolling = first.map(|layer| layer.rolling_t).unwrap_or(0);
    let scale_granularity = if layers
        .iter()
        .all(|layer| layer.k_scales.len() == 1 && layer.v_scales.len() == 1)
    {
        "legacy-whole-layer-buffer"
    } else {
        "per-head-vector"
    };
    json!({
        "kv_storage_dtype": "int8",
        "kv_scale_granularity": scale_granularity,
        "layers": layers.len(),
        "stable_prefix_tokens": stable,
        "rolling_tokens": rolling,
        "visible_tokens": first.map(|layer| layer.t).unwrap_or(0),
        "next_position": first.map(|layer| layer.next_position).unwrap_or(0),
        "storage_bytes": layers.iter().map(LayerCache::storage_bytes).sum::<usize>(),
        "cache_growth_bounded": consistent
            && stable <= STABLE_PREFIX_TOKENS_MAX
            && stable + rolling <= MAX_CONTEXT_TOKENS
    })
}

fn quantize_i8_vector(values: &[f32]) -> (Vec<i8>, f32) {
    let maximum = values
        .iter()
        .copied()
        .filter(|value| value.is_finite())
        .map(f32::abs)
        .fold(0.0f32, f32::max);
    let scale = if maximum > 0.0 { maximum / 127.0 } else { 1.0 };
    let quantized = values
        .iter()
        .map(|value| {
            if !value.is_finite() {
                0
            } else {
                // Match the canonical MLX/Needle Q/DQ code interval exactly.
                // With absmax/127 scaling the -128 endpoint is normally
                // unreachable, but keeping the declared interval identical
                // prevents cross-runtime drift at floating-point boundaries.
                (*value / scale).round().clamp(-128.0, 127.0) as i8
            }
        })
        .collect();
    (quantized, scale)
}

fn quantize_i8_vectors(values: &[f32], vector_width: usize) -> (Vec<i8>, Vec<f32>) {
    assert!(vector_width > 0, "int8 vector width must be positive");
    assert_eq!(
        values.len() % vector_width,
        0,
        "int8 values must contain complete last-axis vectors"
    );
    let mut quantized = Vec::with_capacity(values.len());
    let mut scales = Vec::with_capacity(values.len() / vector_width);
    for vector in values.chunks_exact(vector_width) {
        let (codes, scale) = quantize_i8_vector(vector);
        quantized.extend(codes);
        scales.push(scale);
    }
    (quantized, scales)
}

fn dequantize_i8(values: &[i8], scale: f32) -> Vec<f32> {
    values.iter().map(|value| *value as f32 * scale).collect()
}

fn dequantize_i8_vectors(values: &[i8], scales: &[f32], vector_width: usize) -> Vec<f32> {
    assert!(vector_width > 0, "int8 vector width must be positive");
    assert_eq!(values.len() % vector_width, 0);
    assert_eq!(scales.len(), values.len() / vector_width);
    let mut out = Vec::with_capacity(values.len());
    for (vector, scale) in values.chunks_exact(vector_width).zip(scales) {
        out.extend(dequantize_i8(vector, *scale));
    }
    out
}

fn activation_i8_qdq(values: &mut [f32], vector_width: usize) {
    let (quantized, scales) = quantize_i8_vectors(values, vector_width);
    for ((target, value), scale) in values
        .chunks_exact_mut(vector_width)
        .zip(quantized.chunks_exact(vector_width))
        .zip(scales)
    {
        for (target, value) in target.iter_mut().zip(value) {
            *target = *value as f32 * scale;
        }
    }
}

fn merge_kv_cache(
    previous: Option<&LayerCache>,
    k: Vec<f32>,
    v: Vec<f32>,
    row_width: usize,
    new_tokens: usize,
    stable_prefix_tokens: usize,
    position_offset: usize,
    bounded: bool,
) -> (Vec<f32>, Vec<f32>, usize, usize) {
    let (mut k_all, mut v_all, stable_t, next_position) = if let Some(prev) = previous {
        let mut kk = prev.dequant_k();
        let mut vv = prev.dequant_v();
        kk.extend_from_slice(&k);
        vv.extend_from_slice(&v);
        (kk, vv, prev.stable_t, prev.next_position + new_tokens)
    } else {
        (
            k,
            v,
            stable_prefix_tokens
                .min(new_tokens)
                .min(STABLE_PREFIX_TOKENS_MAX),
            position_offset + new_tokens,
        )
    };
    let total_t = k_all.len() / row_width;
    let rolling_t = total_t.saturating_sub(stable_t);
    let ordinary_cap = ordinary_capacity(stable_t);
    if bounded && rolling_t > ordinary_cap {
        let evict = rolling_t - ordinary_cap;
        let begin = stable_t * row_width;
        let end = (stable_t + evict) * row_width;
        k_all.drain(begin..end);
        v_all.drain(begin..end);
    }
    (k_all, v_all, stable_t, next_position)
}

/// Append already-Q/DQ'd K/V vectors to the portable int8 cache in place.
///
/// Unlike `merge_kv_cache`, this path never reconstructs the historical cache
/// as f32 and never requantizes prior tokens. It is the native v2 decode path;
/// the legacy compatibility view retains its historical whole-buffer math.
fn merge_kv_cache_i8(
    previous: Option<&mut LayerCache>,
    k: &[f32],
    v: &[f32],
    row_width: usize,
    vector_width: usize,
    new_tokens: usize,
    stable_prefix_tokens: usize,
    position_offset: usize,
    bounded: bool,
) -> LayerCache {
    debug_assert!(vector_width > 0 && row_width % vector_width == 0);
    debug_assert_eq!(k.len(), new_tokens * row_width);
    debug_assert_eq!(v.len(), k.len());
    let had_previous = previous.is_some();
    let mut next = if let Some(previous) = previous {
        std::mem::take(previous)
    } else {
        LayerCache {
            vector_width,
            stable_t: stable_prefix_tokens
                .min(new_tokens)
                .min(STABLE_PREFIX_TOKENS_MAX),
            next_position: position_offset,
            ..LayerCache::default()
        }
    };
    debug_assert!(next.vector_width == 0 || next.vector_width == vector_width);
    next.vector_width = vector_width;
    if had_previous && stable_prefix_tokens > 0 {
        debug_assert_eq!(next.rolling_t, 0);
        next.stable_t = next
            .stable_t
            .saturating_add(stable_prefix_tokens.min(new_tokens))
            .min(STABLE_PREFIX_TOKENS_MAX);
    }
    let (k_codes, k_scales) = quantize_i8_vectors(k, vector_width);
    let (v_codes, v_scales) = quantize_i8_vectors(v, vector_width);
    next.k.extend(k_codes);
    next.v.extend(v_codes);
    next.k_scales.extend(k_scales);
    next.v_scales.extend(v_scales);
    next.t += new_tokens;
    next.next_position += new_tokens;
    next.rolling_t = next.t.saturating_sub(next.stable_t);
    let ordinary_cap = ordinary_capacity(next.stable_t);
    if bounded && next.rolling_t > ordinary_cap {
        let evict = next.rolling_t - ordinary_cap;
        let byte_begin = next.stable_t * row_width;
        let byte_end = (next.stable_t + evict) * row_width;
        next.k.drain(byte_begin..byte_end);
        next.v.drain(byte_begin..byte_end);
        let scales_per_token = row_width / vector_width;
        let scale_begin = next.stable_t * scales_per_token;
        let scale_end = (next.stable_t + evict) * scales_per_token;
        next.k_scales.drain(scale_begin..scale_end);
        next.v_scales.drain(scale_begin..scale_end);
        next.t -= evict;
        next.rolling_t = ordinary_cap;
    }
    next
}

pub struct NeedleModel {
    pub arch: Arch,
    pub weights: PackedWeights,
    cache: std::sync::Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// When true, use packed row-wise kernels for large tensors. Small
    /// structural tensors remain eligible for the bounded cache.
    pub disable_tensor_cache: bool,
}

impl NeedleModel {
    pub fn new(arch: Arch, weights: PackedWeights) -> Self {
        Self {
            arch,
            weights,
            cache: std::sync::Mutex::new(HashMap::new()),
            disable_tensor_cache: cfg!(target_arch = "wasm32"),
        }
    }

    pub fn without_full_f32_cache(arch: Arch, weights: PackedWeights) -> Self {
        let mut model = Self::new(arch, weights);
        model.disable_tensor_cache = true;
        model
    }

    fn activation_qdq(&self, values: &mut [f32], vector_width: usize) {
        let width = if self.arch.legacy_activation_qdq {
            values.len().max(1)
        } else {
            vector_width
        };
        activation_i8_qdq(values, width);
    }

    pub fn has_tensor(&self, name: &str) -> bool {
        self.weights.tensors.contains_key(name)
    }

    pub fn narration_residual(&self, hidden: &[f32]) -> Result<Vec<f32>, SdkError> {
        const DOWN: &str = "heads.narration_adapter.down.weight";
        const UP: &str = "heads.narration_adapter.up.weight";
        if !self.has_tensor(DOWN) || !self.has_tensor(UP) {
            return Err(SdkError::new(
                "capability_missing",
                "narration adapter tensors are unavailable",
            ));
        }
        if hidden.len() != self.arch.d_model {
            return Err(SdkError::new(
                "package_invalid",
                "narration adapter hidden width mismatch",
            ));
        }
        let low_rank = self.weights.matvec(DOWN, hidden)?;
        if low_rank.len() != 16 {
            return Err(SdkError::new(
                "package_invalid",
                "narration adapter rank mismatch",
            ));
        }
        let mut residual = self.weights.matvec(UP, &low_rank)?;
        if residual.len() != self.arch.vocab_size {
            return Err(SdkError::new(
                "package_invalid",
                "narration adapter vocabulary mismatch",
            ));
        }
        for value in &mut residual {
            *value /= 4.0;
        }
        Ok(residual)
    }

    fn rowwise(&self) -> bool {
        self.disable_tensor_cache || cfg!(target_arch = "wasm32")
    }

    fn tensor(&self, name: &str) -> Result<Arc<Vec<f32>>, SdkError> {
        let cacheable = !self.rowwise()
            || self.weights.entry(name)?.n_params <= PORTABLE_TENSOR_CACHE_MAX_PARAMS;
        if cacheable {
            {
                let guard = self.cache.lock().expect("tensor cache");
                if let Some(hit) = guard.get(name) {
                    return Ok(hit.clone());
                }
            }
            let data = Arc::new(self.weights.dequant(name)?);
            let mut guard = self.cache.lock().expect("tensor cache");
            guard.insert(name.to_string(), data.clone());
            return Ok(data);
        }
        Ok(Arc::new(self.weights.dequant(name)?))
    }

    fn linear(
        &self,
        name: &str,
        x: &[f32],
        rows: usize,
        cols: usize,
    ) -> Result<Vec<f32>, SdkError> {
        let t = x.len() / cols;
        if self.rowwise() {
            let mut y = vec![0f32; t * rows];
            for ti in 0..t {
                let yr = self.weights.matvec(name, &x[ti * cols..(ti + 1) * cols])?;
                if yr.len() != rows {
                    return Err(SdkError::new(
                        "package_invalid",
                        format!("{name} matvec {} != rows {rows}", yr.len()),
                    ));
                }
                y[ti * rows..(ti + 1) * rows].copy_from_slice(&yr);
            }
            return Ok(y);
        }
        let w = self.tensor(name)?;
        if w.len() != rows * cols {
            return Err(SdkError::new(
                "package_invalid",
                format!("{name} size {} != {}x{}", w.len(), rows, cols),
            ));
        }
        let mut y = vec![0f32; t * rows];
        for ti in 0..t {
            let xr = &x[ti * cols..(ti + 1) * cols];
            for r in 0..rows {
                let wr = &w[r * cols..(r + 1) * cols];
                let mut acc = 0f32;
                for (a, b) in wr.iter().zip(xr.iter()) {
                    acc += *a * *b;
                }
                y[ti * rows + r] = acc;
            }
        }
        Ok(y)
    }

    fn linear_prepared_rows(
        &self,
        name: &str,
        prepared: &[PreparedMatVecInput],
        rows: usize,
    ) -> Result<Vec<f32>, SdkError> {
        let mut output = vec![0f32; prepared.len() * rows];
        for (token, activation) in prepared.iter().enumerate() {
            let row = self.weights.matvec_prepared(name, activation)?;
            if row.len() != rows {
                return Err(SdkError::new(
                    "package_invalid",
                    format!("{name} prepared matvec {} != rows {rows}", row.len()),
                ));
            }
            output[token * rows..(token + 1) * rows].copy_from_slice(&row);
        }
        Ok(output)
    }

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, SdkError> {
        let d = self.arch.d_model;
        if let Some(token) = tokens
            .iter()
            .find(|token| **token as usize >= self.arch.vocab_size)
        {
            return Err(SdkError::new(
                "invalid_argument",
                format!("token id {token} is outside the model vocabulary"),
            ));
        }
        let scale = (d as f32).sqrt();
        let mut out = vec![0f32; tokens.len() * d];
        if self.rowwise() {
            for (i, &tok) in tokens.iter().enumerate() {
                let src = self.weights.dequant_row("embed.weight", tok as usize)?;
                for (o, v) in out[i * d..(i + 1) * d].iter_mut().zip(src.iter()) {
                    *o = *v * scale;
                }
            }
            return Ok(out);
        }
        let table = self.tensor("embed.weight")?;
        for (i, &tok) in tokens.iter().enumerate() {
            let row = tok as usize;
            let src = &table[row * d..(row + 1) * d];
            for (o, v) in out[i * d..(i + 1) * d].iter_mut().zip(src.iter()) {
                *o = *v * scale;
            }
        }
        Ok(out)
    }

    fn lm_logits(&self, hidden: &[f32], t: usize, mode: LogitsMode) -> Result<Vec<f32>, SdkError> {
        let a = &self.arch;
        let d = a.d_model;
        if matches!(mode, LogitsMode::None) {
            return Ok(Vec::new());
        }
        let first_token = if matches!(mode, LogitsMode::Last) {
            t.saturating_sub(1)
        } else {
            0
        };
        let output_tokens = t.saturating_sub(first_token);
        if self.rowwise() {
            let mut logits = vec![0f32; output_tokens * a.vocab_size];
            for (output_token, ti) in (first_token..t).enumerate() {
                let y = self
                    .weights
                    .matvec("embed.weight", &hidden[ti * d..(ti + 1) * d])?;
                logits[output_token * a.vocab_size..(output_token + 1) * a.vocab_size]
                    .copy_from_slice(&y);
            }
            return Ok(logits);
        }
        let embed = self.tensor("embed.weight")?;
        let mut logits = vec![0f32; output_tokens * a.vocab_size];
        for (output_token, ti) in (first_token..t).enumerate() {
            let h = &hidden[ti * d..(ti + 1) * d];
            for v in 0..a.vocab_size {
                let row = &embed[v * d..(v + 1) * d];
                let mut acc = 0f32;
                for (p, q) in h.iter().zip(row.iter()) {
                    acc += *p * *q;
                }
                logits[output_token * a.vocab_size + v] = acc;
            }
        }
        Ok(logits)
    }

    pub fn forward(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
    ) -> Result<ForwardOut, SdkError> {
        self.forward_internal(
            tokens,
            cache,
            return_confidence,
            None,
            0,
            false,
            LogitsMode::All,
        )
    }

    /// Run the backbone and sidecar features without materializing LM logits.
    /// Retrieval and diagnostic heads consume hidden/cell states, so computing
    /// a `[tokens, vocab]` projection here is both unnecessary and especially
    /// expensive in portable WASM.
    pub fn forward_features(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
    ) -> Result<ForwardOut, SdkError> {
        self.forward_internal(
            tokens,
            cache,
            return_confidence,
            None,
            0,
            false,
            LogitsMode::None,
        )
    }

    pub fn forward_with_engram_prefix(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
        engram_prefix: Option<&[u32]>,
    ) -> Result<ForwardOut, SdkError> {
        self.forward_internal(
            tokens,
            cache,
            return_confidence,
            engram_prefix,
            0,
            false,
            LogitsMode::All,
        )
    }

    pub fn forward_bounded(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
        engram_prefix: Option<&[u32]>,
        stable_prefix_tokens: usize,
    ) -> Result<ForwardOut, SdkError> {
        if stable_prefix_tokens > STABLE_PREFIX_TOKENS_MAX || stable_prefix_tokens > tokens.len() {
            return Err(SdkError::new(
                "invalid_argument",
                "stable prefix exceeds the bounded runtime contract",
            ));
        }
        if cache.is_none() && tokens.len() > MAX_CONTEXT_TOKENS {
            return Err(SdkError::new(
                "invalid_argument",
                "prompt exceeds the 2048-token runtime context",
            ));
        }
        self.forward_internal(
            tokens,
            cache,
            return_confidence,
            engram_prefix,
            stable_prefix_tokens,
            true,
            LogitsMode::All,
        )
    }

    /// Bounded inference variant that projects only the final input token to
    /// the vocabulary.  Autoregressive decoding never consumes earlier prompt
    /// logits, so their omission changes neither generated tokens nor heads.
    pub fn forward_bounded_last_logits(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
        engram_prefix: Option<&[u32]>,
        stable_prefix_tokens: usize,
    ) -> Result<ForwardOut, SdkError> {
        if stable_prefix_tokens > STABLE_PREFIX_TOKENS_MAX || stable_prefix_tokens > tokens.len() {
            return Err(SdkError::new(
                "invalid_argument",
                "stable prefix exceeds the bounded runtime contract",
            ));
        }
        if cache.is_none() && tokens.len() > MAX_CONTEXT_TOKENS {
            return Err(SdkError::new(
                "invalid_argument",
                "prompt exceeds the 2048-token runtime context",
            ));
        }
        // Packed portable kernels are token-row GEMVs. Streaming a long
        // prefill therefore preserves their arithmetic workload while
        // avoiding `[T, lanes, d_model]` activation residency. It also reuses
        // the same bounded int8 cache as decode, keeping browser memory tied to
        // the runtime contract instead of prompt-length intermediates.
        if self.rowwise() && tokens.len() > 1 {
            let mut local_cache = cache.take();
            let mut history = engram_prefix.unwrap_or(&[]).to_vec();
            let mut cursor = 0usize;
            while cursor < tokens.len() {
                let in_stable_prefix = cursor < stable_prefix_tokens;
                let region_end = if in_stable_prefix {
                    stable_prefix_tokens
                } else {
                    tokens.len()
                };
                let end = (cursor + PORTABLE_PREFILL_CHUNK_TOKENS)
                    .min(region_end)
                    .min(tokens.len());
                let chunk = &tokens[cursor..end];
                let is_last = end == tokens.len();
                let new_stable_tokens = if in_stable_prefix { chunk.len() } else { 0 };
                let mut out = self.forward_internal(
                    chunk,
                    &mut local_cache,
                    return_confidence && is_last,
                    Some(&history),
                    new_stable_tokens,
                    true,
                    LogitsMode::Last,
                )?;
                history.extend_from_slice(chunk);
                if is_last {
                    return Ok(out);
                }
                local_cache = Some(std::mem::take(&mut out.cache));
                cursor = end;
            }
            unreachable!("non-empty token sequence must return its final row");
        }
        self.forward_internal(
            tokens,
            cache,
            return_confidence,
            engram_prefix,
            stable_prefix_tokens,
            true,
            LogitsMode::Last,
        )
    }

    fn forward_internal(
        &self,
        tokens: &[u32],
        cache: &mut Option<Vec<LayerCache>>,
        return_confidence: bool,
        engram_prefix: Option<&[u32]>,
        stable_prefix_tokens: usize,
        bounded_cache: bool,
        logits_mode: LogitsMode,
    ) -> Result<ForwardOut, SdkError> {
        let a = &self.arch;
        let b = 1usize;
        let t = tokens.len();
        let d = a.d_model;
        let n = a.mhc_lanes;
        let x = self.embed(tokens)?;
        let position_offset = cache
            .as_ref()
            .and_then(|c| c.first().map(|layer| layer.next_position))
            .unwrap_or(0);
        let rope = precompute_rope(a.head_dim, position_offset, t, a.rope_theta);
        let engram_kv = self.engram_stack(tokens, engram_prefix)?;
        let (pre_off, post_off) = mhc_offsets(a.n_layers, n);
        // lanes: [t, n, d]
        let mut lanes = vec![0f32; t * n * d];
        for ti in 0..t {
            for lane in 0..n {
                lanes[ti * n * d + lane * d..ti * n * d + (lane + 1) * d]
                    .copy_from_slice(&x[ti * d..(ti + 1) * d]);
            }
        }
        let mut new_cache: Vec<LayerCache> = Vec::with_capacity(a.n_layers);
        let full_cells = matches!(logits_mode, LogitsMode::All | LogitsMode::None);
        let cell_t = if full_cells { t } else { 1 };
        let mut cells: Vec<Vec<f32>> = if full_cells {
            vec![x.clone()]
        } else {
            vec![x[(t - 1) * d..t * d].to_vec()]
        };
        let site_by_layer: HashMap<usize, usize> = a
            .engram_layers
            .iter()
            .enumerate()
            .map(|(site, layer)| (*layer, site))
            .collect();
        let phi_pre = self.tensor("mhc_phi_pre")?;
        let b_pre = self.tensor("mhc_b_pre")?;
        let a_pre = self.tensor("mhc_a_pre")?;
        let phi_post = self.tensor("mhc_phi_post")?;
        let b_post = self.tensor("mhc_b_post")?;
        let a_post = self.tensor("mhc_a_post")?;
        let phi_res = self.tensor("mhc_phi_res")?;
        let b_res = self.tensor("mhc_b_res")?;
        let a_res = self.tensor("mhc_a_res")?;
        for layer in 0..a.n_layers {
            let nx = rms_unit_lanes(&lanes, t, n, d, a.rms_eps);
            let hpre = mhc_h(
                &nx, &phi_pre, &b_pre, &a_pre, &pre_off, layer, t, n, d, true,
            );
            let mut u = vec![0f32; t * d];
            for ti in 0..t {
                for c in 0..d {
                    let mut acc = 0f32;
                    for lane in 0..n {
                        acc += hpre[ti * n + lane] * lanes[ti * n * d + lane * d + c];
                    }
                    u[ti * d + c] = acc;
                }
            }
            let mut block_input = u.clone();
            if let Some(site) = site_by_layer.get(&layer) {
                if let Some((ek, ev)) = engram_kv.as_ref() {
                    let key = &ek[*site];
                    let value = &ev[*site];
                    for ti in 0..t {
                        let xr = &block_input[ti * d..(ti + 1) * d];
                        let kr = &key[ti * d..(ti + 1) * d];
                        let vr = &value[ti * d..(ti + 1) * d];
                        let ru = rms_vec(xr, a.rms_eps);
                        let rk = rms_vec(kr, a.rms_eps);
                        let mut dot = 0f32;
                        for (p, q) in ru.iter().zip(rk.iter()) {
                            dot += *p * *q;
                        }
                        let alpha = sigmoid(dot / (d as f32).sqrt());
                        for c in 0..d {
                            block_input[ti * d + c] += alpha * vr[c];
                        }
                    }
                }
            }
            let layer_cache = cache.as_mut().and_then(|c| c.get_mut(layer));
            let (block_out, next_cache) = self.block(
                layer,
                &block_input,
                t,
                &rope,
                position_offset,
                stable_prefix_tokens,
                bounded_cache,
                layer_cache,
            )?;
            let mut delta = vec![0f32; t * d];
            for i in 0..t * d {
                delta[i] = block_out[i] - u[i];
            }
            let hpost = mhc_h(
                &nx, &phi_post, &b_post, &a_post, &post_off, layer, t, n, d, false,
            );
            let hres = mhc_res(
                &nx,
                &phi_res,
                &b_res,
                &a_res,
                layer,
                t,
                n,
                d,
                a.sinkhorn_iters,
            );
            let mut new_lanes = vec![0f32; t * n * d];
            for ti in 0..t {
                for i in 0..n {
                    for c in 0..d {
                        let mut acc = 0f32;
                        for j in 0..n {
                            acc += hres[ti * n * n + i * n + j] * lanes[ti * n * d + j * d + c];
                        }
                        new_lanes[ti * n * d + i * d + c] =
                            acc + hpost[ti * n + i] * delta[ti * d + c];
                    }
                }
            }
            lanes = new_lanes;
            new_cache.push(next_cache);
            let cell = mean_lanes(&lanes, t, n, d);
            if full_cells {
                cells.push(cell);
            } else {
                cells.push(cell[(t - 1) * d..t * d].to_vec());
            }
        }
        let hidden = zc_rms_norm(
            &mean_lanes(&lanes, t, n, d),
            &self.tensor("final_norm.scale")?,
            t,
            d,
            a.rms_eps,
        );
        let mut lm_input = hidden.clone();
        if !a.legacy_activation_qdq {
            self.activation_qdq(&mut lm_input, d);
        }
        let logits = self.lm_logits(&lm_input, t, logits_mode)?;
        let conf = if return_confidence {
            Some(self.confidence_logit_from_cells(&cells, cell_t)?)
        } else {
            None
        };
        let _ = b;
        let _ = BLOCK_SIZE;
        Ok(ForwardOut {
            logits,
            hidden,
            cache: new_cache,
            t,
            vocab: a.vocab_size,
            confidence_logit: conf,
            cells,
            cells_t: cell_t,
        })
    }

    /// Execute the independent contrastive retrieval head over every backbone
    /// cell.  This mirrors `ContrastiveHead` in the MLX numerical oracle and
    /// must not be substituted with the last backbone hidden state.
    pub fn contrastive_embedding(&self, out: &ForwardOut) -> Result<Vec<f32>, SdkError> {
        const PROBES: usize = 4;
        const DIM: usize = 128;
        let d = self.arch.d_model;
        let layers = out.cells.len();
        if layers == 0
            || out.cells_t != out.t
            || out.t == 0
            || out.cells.iter().any(|cell| cell.len() != out.t * d)
        {
            return Err(SdkError::new(
                "package_invalid",
                "contrastive head requires [T,L,d_model] cells",
            ));
        }
        let tok_probes = self.tensor("heads.contrastive.tok_probes")?;
        let lay_probes = self.tensor("heads.contrastive.lay_probes")?;
        let projection = self.tensor("heads.contrastive.proj.weight")?;
        if tok_probes.len() != PROBES * d
            || lay_probes.len() != PROBES * d
            || projection.len() != DIM * PROBES * d
        {
            return Err(SdkError::new(
                "package_invalid",
                "contrastive head tensor geometry",
            ));
        }
        let scale = (d as f32).sqrt();
        let mut summaries = vec![0f32; PROBES * d];
        for probe in 0..PROBES {
            let tok_probe = &tok_probes[probe * d..(probe + 1) * d];
            let lay_probe = &lay_probes[probe * d..(probe + 1) * d];
            let mut token_scores = vec![0f32; out.t];
            for (token, score) in token_scores.iter_mut().enumerate() {
                let mut dot = 0f32;
                for cell in &out.cells {
                    let row = &cell[token * d..(token + 1) * d];
                    dot += row
                        .iter()
                        .zip(tok_probe)
                        .map(|(left, right)| left * right)
                        .sum::<f32>();
                }
                *score = dot / layers as f32 / scale;
            }
            softmax_inplace(&mut token_scores);
            let mut pooled_by_layer = vec![0f32; layers * d];
            for (layer, cell) in out.cells.iter().enumerate() {
                for token in 0..out.t {
                    let weight = token_scores[token];
                    for channel in 0..d {
                        pooled_by_layer[layer * d + channel] += weight * cell[token * d + channel];
                    }
                }
            }
            let mut layer_scores = vec![0f32; layers];
            for layer in 0..layers {
                layer_scores[layer] = pooled_by_layer[layer * d..(layer + 1) * d]
                    .iter()
                    .zip(lay_probe)
                    .map(|(left, right)| left * right)
                    .sum::<f32>()
                    / scale;
            }
            softmax_inplace(&mut layer_scores);
            for layer in 0..layers {
                for channel in 0..d {
                    summaries[probe * d + channel] +=
                        layer_scores[layer] * pooled_by_layer[layer * d + channel];
                }
            }
        }
        let mut embedding = vec![0f32; DIM];
        for row in 0..DIM {
            embedding[row] = projection[row * summaries.len()..(row + 1) * summaries.len()]
                .iter()
                .zip(&summaries)
                .map(|(left, right)| left * right)
                .sum();
        }
        let norm = embedding
            .iter()
            .map(|value| value * value)
            .sum::<f32>()
            .sqrt();
        if !norm.is_finite() || norm <= 0.0 {
            return Err(SdkError::new(
                "package_invalid",
                "contrastive head produced invalid norm",
            ));
        }
        for value in &mut embedding {
            *value /= norm;
        }
        Ok(embedding)
    }

    /// Execute the independent 20-class MW disposition head.  The returned
    /// probabilities are evidence for a deterministic policy; they are never
    /// retrieval scores or confidence values.
    pub fn mw_disposition_probabilities(&self, out: &ForwardOut) -> Result<Vec<f32>, SdkError> {
        const CLASSES: usize = 20;
        let pooled = mean_last_token_across_cells(&out.cells, out.cells_t, self.arch.d_model)?;
        let weight = self.tensor("heads.mw_disposition.proj.weight")?;
        let bias = self.tensor("heads.mw_disposition.proj.bias")?;
        if weight.len() != CLASSES * self.arch.d_model || bias.len() != CLASSES {
            return Err(SdkError::new(
                "package_invalid",
                "MW disposition head tensor geometry",
            ));
        }
        let mut logits = bias.as_ref().clone();
        for class in 0..CLASSES {
            logits[class] += weight[class * self.arch.d_model..(class + 1) * self.arch.d_model]
                .iter()
                .zip(&pooled)
                .map(|(left, right)| left * right)
                .sum::<f32>();
        }
        softmax_inplace(&mut logits);
        if logits.iter().any(|value| !value.is_finite()) {
            return Err(SdkError::new(
                "package_invalid",
                "MW disposition head produced non-finite probabilities",
            ));
        }
        Ok(logits)
    }

    fn confidence_logit_from_cells(&self, cells: &[Vec<f32>], t: usize) -> Result<f32, SdkError> {
        let d = self.arch.d_model;
        let (probes_name, projection_name, bias_name) = if self
            .weights
            .tensors
            .contains_key("heads.confidence.cell_probes")
        {
            (
                "heads.confidence.cell_probes",
                "heads.confidence.proj.weight",
                "heads.confidence.proj.bias",
            )
        } else {
            // Read-only v1 diagnostic compatibility only. Native v2 packages
            // are required by the loader to contain the independent sidecar.
            ("conf_probes", "conf_proj.weight", "conf_proj.bias")
        };
        let last = mean_last_token_across_cells(cells, t, d)?;
        let probes = self.tensor(probes_name)?;
        let count = probes.len() / d;
        let projection = self.tensor(projection_name)?;
        let bias = self.tensor(bias_name)?;
        if count == 0 || probes.len() != count * d || projection.len() != count * d {
            return Err(SdkError::new(
                "package_invalid",
                "confidence head tensor geometry",
            ));
        }
        let scale = (d as f32).sqrt();
        let mut scores = vec![0f32; count];
        for probe in 0..count {
            scores[probe] = last
                .iter()
                .zip(&probes[probe * d..(probe + 1) * d])
                .map(|(left, right)| left * right)
                .sum::<f32>()
                / scale;
        }
        softmax_inplace(&mut scores);
        let mut logit = bias.first().copied().unwrap_or(0.0);
        for probe in 0..count {
            for channel in 0..d {
                logit += projection[probe * d + channel] * scores[probe] * last[channel];
            }
        }
        if !logit.is_finite() {
            return Err(SdkError::new(
                "package_invalid",
                "confidence head produced non-finite logit",
            ));
        }
        Ok(logit)
    }

    fn block(
        &self,
        layer: usize,
        x: &[f32],
        t: usize,
        rope: &(Vec<f32>, Vec<f32>),
        position_offset: usize,
        stable_prefix_tokens: usize,
        bounded_cache: bool,
        mut cache: Option<&mut LayerCache>,
    ) -> Result<(Vec<f32>, LayerCache), SdkError> {
        let a = &self.arch;
        let d = a.d_model;
        let prefix = format!("blocks.{layer}");
        let mut xn = zc_rms_norm(
            x,
            &self.tensor(&format!("{prefix}.attn_norm.scale"))?,
            t,
            d,
            a.rms_eps,
        );
        self.activation_qdq(&mut xn, d);
        let prepared = if self.rowwise() {
            Some(
                xn.chunks_exact(d)
                    .map(|row| self.weights.prepare_matvec_input(row))
                    .collect::<Result<Vec<_>, _>>()?,
            )
        } else {
            None
        };
        let project = |suffix: &str, rows: usize| -> Result<Vec<f32>, SdkError> {
            let name = format!("{prefix}.attn.{suffix}.weight");
            if let Some(prepared) = prepared.as_deref() {
                self.linear_prepared_rows(&name, prepared, rows)
            } else {
                self.linear(&name, &xn, rows, d)
            }
        };
        let q = project("q_proj", a.n_heads * a.head_dim)?;
        let k = project("k_proj", a.n_kv_heads * a.head_dim)?;
        let v = project("v_proj", a.n_kv_heads * a.head_dim)?;
        let gate = project("gate_proj", a.n_heads * a.head_dim)?;
        let mut q = reshape_heads(q, t, a.n_heads, a.head_dim);
        let mut k = reshape_heads(k, t, a.n_kv_heads, a.head_dim);
        let v = reshape_heads(v, t, a.n_kv_heads, a.head_dim);
        q = apply_qknorm(
            &q,
            t,
            a.n_heads,
            a.head_dim,
            &self.tensor(&format!("{prefix}.attn.q_norm.scale"))?,
            a.rms_eps,
        );
        k = apply_qknorm(
            &k,
            t,
            a.n_kv_heads,
            a.head_dim,
            &self.tensor(&format!("{prefix}.attn.k_norm.scale"))?,
            a.rms_eps,
        );
        apply_rope_inplace(&mut q, t, a.n_heads, a.head_dim, rope);
        apply_rope_inplace(&mut k, t, a.n_kv_heads, a.head_dim, rope);
        // Native v2 follows Needle and does not activation-quantize q. The v1
        // compatibility view retains its historical global-scale q Q/DQ.
        if a.legacy_activation_qdq {
            let q_width = q.len().max(1);
            activation_i8_qdq(&mut q, q_width);
        }
        self.activation_qdq(&mut k, a.head_dim);
        let mut v = v;
        self.activation_qdq(&mut v, a.head_dim);
        let row_width = a.n_kv_heads * a.head_dim;
        let repeats = a.n_heads / a.n_kv_heads;
        let scale = (a.head_dim as f32).sqrt().recip();
        let (mut attn_out, next_cache) = if a.legacy_activation_qdq {
            let (k_all, v_all, stable_t, next_position) = merge_kv_cache(
                cache.as_deref(),
                k,
                v,
                row_width,
                t,
                stable_prefix_tokens,
                position_offset,
                bounded_cache,
            );
            let kv_t = k_all.len() / row_width;
            let mut attn_out = vec![0f32; t * a.n_heads * a.head_dim];
            let mut scores = vec![0f32; kv_t];
            for head in 0..a.n_heads {
                let kv_head = head / repeats;
                for qi in 0..t {
                    let q_off = (qi * a.n_heads + head) * a.head_dim;
                    let qv = &q[q_off..q_off + a.head_dim];
                    for kj in 0..kv_t {
                        let k_off = (kj * a.n_kv_heads + kv_head) * a.head_dim;
                        let kv = &k_all[k_off..k_off + a.head_dim];
                        let mut dot = 0f32;
                        for (p, qq) in kv.iter().zip(qv.iter()) {
                            dot += *p * *qq;
                        }
                        let pos_q = kv_t.saturating_sub(t) + qi;
                        scores[kj] = if pos_q >= kj {
                            dot * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    softmax_inplace(&mut scores);
                    let o_off = (qi * a.n_heads + head) * a.head_dim;
                    for kj in 0..kv_t {
                        let v_off = (kj * a.n_kv_heads + kv_head) * a.head_dim;
                        let vv = &v_all[v_off..v_off + a.head_dim];
                        let s = scores[kj];
                        for c in 0..a.head_dim {
                            attn_out[o_off + c] += s * vv[c];
                        }
                    }
                }
            }
            let cache = LayerCache::from_f32(
                &k_all,
                &v_all,
                kv_t,
                stable_t,
                next_position,
                k_all.len().max(1),
            );
            (attn_out, cache)
        } else {
            let next_cache = merge_kv_cache_i8(
                cache.take(),
                &k,
                &v,
                row_width,
                a.head_dim,
                t,
                stable_prefix_tokens,
                position_offset,
                bounded_cache,
            );
            let kv_t = next_cache.t;
            let mut attn_out = vec![0f32; t * a.n_heads * a.head_dim];
            let mut scores = vec![0f32; kv_t];
            for head in 0..a.n_heads {
                let kv_head = head / repeats;
                for qi in 0..t {
                    let q_off = (qi * a.n_heads + head) * a.head_dim;
                    let qv = &q[q_off..q_off + a.head_dim];
                    for (kj, score_slot) in scores.iter_mut().enumerate() {
                        let pos_q = kv_t.saturating_sub(t) + qi;
                        *score_slot = if pos_q >= kj {
                            next_cache.key_dot(kj, kv_head, a.n_kv_heads, qv) * scale
                        } else {
                            f32::NEG_INFINITY
                        };
                    }
                    softmax_inplace(&mut scores);
                    let o_off = (qi * a.n_heads + head) * a.head_dim;
                    let output = &mut attn_out[o_off..o_off + a.head_dim];
                    for (kj, weight) in scores.iter().copied().enumerate() {
                        next_cache.add_scaled_value(kj, kv_head, a.n_kv_heads, weight, output);
                    }
                }
            }
            (attn_out, next_cache)
        };
        for i in 0..attn_out.len() {
            attn_out[i] *= sigmoid(gate[i]);
        }
        let mut attn_flat = attn_out;
        self.activation_qdq(&mut attn_flat, a.n_heads * a.head_dim);
        let o = self.linear(
            &format!("{prefix}.attn.o_proj.weight"),
            &attn_flat,
            d,
            a.n_heads * a.head_dim,
        )?;
        let gate_attn = self.tensor(&format!("{prefix}.attn_gate"))?;
        let g = sigmoid(gate_attn.first().copied().unwrap_or(0.0));
        let post = zc_rms_norm(
            &o,
            &self.tensor(&format!("{prefix}.post_attn_norm.scale"))?,
            t,
            d,
            a.rms_eps,
        );
        let mut h = vec![0f32; t * d];
        for i in 0..t * d {
            h[i] = x[i] + g * post[i];
        }
        let mlp_in = zc_rms_norm(
            &h,
            &self.tensor(&format!("{prefix}.mlp_norm.scale"))?,
            t,
            d,
            a.rms_eps,
        );
        let mlp = self.hadamard_mlp(layer, &mlp_in, t)?;
        for i in 0..t * d {
            h[i] += mlp[i];
        }
        Ok((h, next_cache))
    }

    fn hadamard_mlp(&self, layer: usize, x: &[f32], t: usize) -> Result<Vec<f32>, SdkError> {
        let d = self.arch.d_model;
        let n = d.next_power_of_two();
        let d1 = self.tensor(&format!("blocks.{layer}.mlp.d1"))?;
        let d2 = self.tensor(&format!("blocks.{layer}.mlp.d2"))?;
        let d3 = self.tensor(&format!("blocks.{layer}.mlp.d3"))?;
        let mut out = vec![0f32; t * d];
        for ti in 0..t {
            let mut z = vec![0f32; n];
            z[..d].copy_from_slice(&x[ti * d..(ti + 1) * d]);
            for i in 0..n {
                z[i] *= d1[i];
            }
            walsh_hadamard(&mut z);
            for i in 0..n {
                z[i] = silu(z[i] * d2[i]);
            }
            walsh_hadamard(&mut z);
            for i in 0..d {
                out[ti * d + i] = z[i] * d3[i];
            }
        }
        Ok(out)
    }

    fn engram_stack(
        &self,
        tokens: &[u32],
        prefix: Option<&[u32]>,
    ) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, SdkError> {
        let a = &self.arch;
        if a.engram_layers.is_empty() {
            return Ok(None);
        }
        // Engram is finite-context: an n-gram lookup needs `max_order - 1`
        // preceding tokens and the dilated convolution needs
        // `(taps - 1) * max_order` preceding lookup rows. Retaining an entire
        // 1K stable prefix for every one-token decode is mathematically
        // redundant and was the dominant portable-runtime decode cost.
        let max_order = *a.engram_orders.iter().max().unwrap_or(&1);
        let prefix_needed = a
            .engram_conv_taps
            .saturating_sub(1)
            .saturating_mul(max_order)
            .saturating_add(max_order.saturating_sub(1));
        let all_tokens;
        let tokens_all = if let Some(prefix) = prefix.filter(|p| !p.is_empty()) {
            let prefix = &prefix[prefix.len().saturating_sub(prefix_needed)..];
            all_tokens = prefix
                .iter()
                .chain(tokens.iter())
                .copied()
                .collect::<Vec<_>>();
            all_tokens.as_slice()
        } else {
            tokens
        };
        let orders = &a.engram_orders;
        let heads = (a.d_model / (orders.len() * ENGRAM_SUB_DIM)).max(1);
        let sub_dim = a.d_model / (orders.len() * heads);
        let n_tables = orders.len() * heads;
        let all_t = tokens_all.len();
        let keep_t = tokens.len();
        let keep_start = all_t.saturating_sub(keep_t);
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (eg, _layer) in a.engram_layers.iter().enumerate() {
            let tables_name = format!("engrams.{eg}.tables");
            let mut fetched = vec![0f32; all_t * n_tables * sub_dim];
            let indices = engram_indices(tokens_all, orders, heads, a.engram_slots);
            for ti in 0..all_t {
                for tbl in 0..n_tables {
                    let idx = indices[ti * n_tables + tbl] as usize;
                    let ok = ngram_ok(tokens_all, ti, orders, tbl, heads);
                    if ok {
                        let row = tbl * a.engram_slots + idx;
                        let target = &mut fetched
                            [(ti * n_tables + tbl) * sub_dim..(ti * n_tables + tbl + 1) * sub_dim];
                        self.weights
                            .dequant_range_into(&tables_name, row * sub_dim, target)?;
                    }
                }
            }
            let e_dim = n_tables * sub_dim;
            if !a.legacy_activation_qdq {
                self.activation_qdq(&mut fetched, e_dim);
            }
            let key_all = self.linear(
                &format!("engrams.{eg}.key_proj.weight"),
                &fetched,
                a.d_model,
                e_dim,
            )?;
            let value_all = self.linear(
                &format!("engrams.{eg}.value_proj.weight"),
                &fetched,
                a.d_model,
                e_dim,
            )?;
            let taps = self.tensor(&format!("engrams.{eg}.taps"))?;
            let dil = *orders.iter().max().unwrap_or(&1);
            let mut mixed_all = vec![0f32; all_t * a.d_model];
            for j in 0..a.engram_conv_taps {
                let off = j * dil;
                for ti in 0..all_t {
                    if ti < off {
                        continue;
                    }
                    let src_t = ti - off;
                    for c in 0..a.d_model {
                        mixed_all[ti * a.d_model + c] +=
                            taps[j * a.d_model + c] * value_all[src_t * a.d_model + c];
                    }
                }
            }
            keys.push(key_all[keep_start * a.d_model..all_t * a.d_model].to_vec());
            values.push(mixed_all[keep_start * a.d_model..all_t * a.d_model].to_vec());
        }
        Ok(Some((keys, values)))
    }
}

pub struct ForwardOut {
    pub logits: Vec<f32>,
    pub hidden: Vec<f32>,
    pub cache: Vec<LayerCache>,
    pub t: usize,
    pub vocab: usize,
    pub confidence_logit: Option<f32>,
    pub cells: Vec<Vec<f32>>,
    /// Sequence length retained in each cell tensor. Last-logit decode keeps
    /// one row per layer; retrieval feature mode retains the full sequence.
    pub cells_t: usize,
}

impl ForwardOut {
    pub fn last_logits(&self) -> Result<&[f32], SdkError> {
        if self.vocab == 0 || self.logits.len() < self.vocab || self.logits.len() % self.vocab != 0
        {
            return Err(SdkError::new(
                "package_invalid",
                "forward output does not contain LM logits",
            ));
        }
        Ok(&self.logits[self.logits.len() - self.vocab..])
    }
}

fn mean_last_token_across_cells(
    cells: &[Vec<f32>],
    t: usize,
    d: usize,
) -> Result<Vec<f32>, SdkError> {
    if cells.is_empty() || t == 0 || cells.iter().any(|cell| cell.len() != t * d) {
        return Err(SdkError::new(
            "package_invalid",
            "head requires [T,L,d_model] cells",
        ));
    }
    let mut pooled = vec![0f32; d];
    let offset = (t - 1) * d;
    for cell in cells {
        for channel in 0..d {
            pooled[channel] += cell[offset + channel];
        }
    }
    let inv = 1.0 / cells.len() as f32;
    for value in &mut pooled {
        *value *= inv;
    }
    Ok(pooled)
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn silu(x: f32) -> f32 {
    x * sigmoid(x)
}

fn softmax_inplace(xs: &mut [f32]) {
    let m = xs.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let mut s = 0f32;
    for v in xs.iter_mut() {
        *v = (*v - m).exp();
        s += *v;
    }
    let inv = 1.0 / s.max(1e-12);
    for v in xs.iter_mut() {
        *v *= inv;
    }
}

fn rms_vec(x: &[f32], eps: f32) -> Vec<f32> {
    let mut ms = 0f32;
    for v in x {
        ms += *v * *v;
    }
    let inv = (ms / x.len() as f32 + eps).sqrt().recip();
    x.iter().map(|v| *v * inv).collect()
}

fn rms_unit_lanes(lanes: &[f32], t: usize, n: usize, d: usize, eps: f32) -> Vec<f32> {
    // nx: [t, n*d]
    let mut out = vec![0f32; t * n * d];
    for ti in 0..t {
        let sl = &lanes[ti * n * d..(ti + 1) * n * d];
        let ru = rms_vec(sl, eps);
        out[ti * n * d..(ti + 1) * n * d].copy_from_slice(&ru);
    }
    out
}

fn zc_rms_norm(x: &[f32], scale: &[f32], t: usize, d: usize, eps: f32) -> Vec<f32> {
    let mut out = vec![0f32; t * d];
    for ti in 0..t {
        let sl = &x[ti * d..(ti + 1) * d];
        let mut ms = 0f32;
        for v in sl {
            ms += *v * *v;
        }
        let inv = (ms / d as f32 + eps).sqrt().recip();
        for c in 0..d {
            out[ti * d + c] = (1.0 + scale[c]) * sl[c] * inv;
        }
    }
    out
}

fn mean_lanes(lanes: &[f32], t: usize, n: usize, d: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * d];
    let inv = 1.0 / n as f32;
    for ti in 0..t {
        for c in 0..d {
            let mut acc = 0f32;
            for lane in 0..n {
                acc += lanes[ti * n * d + lane * d + c];
            }
            out[ti * d + c] = acc * inv;
        }
    }
    out
}

fn reshape_heads(x: Vec<f32>, t: usize, heads: usize, head_dim: usize) -> Vec<f32> {
    // input [t, heads*head_dim] already laid out as [t, heads, head_dim] in row-major
    debug_assert_eq!(x.len(), t * heads * head_dim);
    x
}

fn apply_qknorm(
    x: &[f32],
    t: usize,
    heads: usize,
    head_dim: usize,
    scale: &[f32],
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for ti in 0..t {
        for h in 0..heads {
            let off = (ti * heads + h) * head_dim;
            let sl = &x[off..off + head_dim];
            let mut ms = 0f32;
            for v in sl {
                ms += *v * *v;
            }
            let inv = (ms / head_dim as f32 + eps).sqrt().recip();
            for c in 0..head_dim {
                out[off + c] = (1.0 + scale[c]) * sl[c] * inv;
            }
        }
    }
    out
}

fn precompute_rope(
    head_dim: usize,
    position_offset: usize,
    seq: usize,
    theta: f32,
) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for t in 0..seq {
        let position = position_offset + t;
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let ang = position as f32 * freq;
            cos[t * half + i] = ang.cos();
            sin[t * half + i] = ang.sin();
        }
    }
    (cos, sin)
}

fn apply_rope_inplace(
    x: &mut [f32],
    t: usize,
    heads: usize,
    head_dim: usize,
    rope: &(Vec<f32>, Vec<f32>),
) {
    let half = head_dim / 2;
    let (cos, sin) = rope;
    for ti in 0..t {
        for h in 0..heads {
            let off = (ti * heads + h) * head_dim;
            for i in 0..half {
                let c = cos[ti * half + i];
                let s = sin[ti * half + i];
                let x1 = x[off + i];
                let x2 = x[off + half + i];
                x[off + i] = x1 * c - x2 * s;
                x[off + half + i] = x2 * c + x1 * s;
            }
        }
    }
}

fn walsh_hadamard(z: &mut [f32]) {
    let n = z.len();
    let mut width = 1usize;
    while width < n {
        let mut base = 0;
        while base < n {
            for lane in 0..width {
                let l = z[base + lane];
                let r = z[base + width + lane];
                z[base + lane] = l + r;
                z[base + width + lane] = l - r;
            }
            base += 2 * width;
        }
        width *= 2;
    }
    let scale = (n as f32).sqrt().recip();
    for v in z.iter_mut() {
        *v *= scale;
    }
}

fn mhc_offsets(n_layers: usize, lanes: usize) -> (Vec<f32>, Vec<f32>) {
    let mut pre = vec![0f32; n_layers * lanes];
    let mut post = vec![0f32; n_layers * lanes];
    for i in 0..n_layers {
        let lane = i % lanes;
        for l in 0..lanes {
            let v = if l == lane { 1.0 } else { 0.0 };
            pre[i * lanes + l] = 8.0 * v - 4.0;
            post[i * lanes + l] = -4.0 * (1.0 - v);
        }
    }
    (pre, post)
}

fn mhc_h(
    nx: &[f32],
    phi: &[f32],
    b: &[f32],
    a: &[f32],
    off: &[f32],
    layer: usize,
    t: usize,
    n: usize,
    d: usize,
    _pre: bool,
) -> Vec<f32> {
    // phi: [layers, n*d, n]
    let nc = n * d;
    let mut out = vec![0f32; t * n];
    let alpha = a[layer];
    for ti in 0..t {
        let xr = &nx[ti * nc..(ti + 1) * nc];
        if n == 4 {
            // `phi` is laid out `[nc, n]`. Traversing channels first reads it
            // contiguously and still accumulates every lane in the same
            // channel order as the scalar definition. This is the hot 51M
            // geometry and lets WASM SIMD update all four lanes together.
            let mut dots = [0.0f32; 4];
            let phi = &phi[layer * nc * n..(layer + 1) * nc * n];
            for (value, row) in xr.iter().zip(phi.chunks_exact(4)) {
                for lane in 0..4 {
                    dots[lane] += *value * row[lane];
                }
            }
            for lane in 0..4 {
                let z = alpha * dots[lane] + b[layer * n + lane] + off[layer * n + lane];
                out[ti * n + lane] = if _pre { sigmoid(z) } else { 2.0 * sigmoid(z) };
            }
            continue;
        }
        for lane in 0..n {
            let mut dot = 0f32;
            for c in 0..nc {
                // phi layout [layers, nc, n] row-major: index = ((layer * nc + c) * n + lane)
                dot += xr[c] * phi[((layer * nc) + c) * n + lane];
            }
            let z = alpha * dot + b[layer * n + lane] + off[layer * n + lane];
            out[ti * n + lane] = if _pre { sigmoid(z) } else { 2.0 * sigmoid(z) };
        }
    }
    out
}

fn mhc_res(
    nx: &[f32],
    phi: &[f32],
    b: &[f32],
    a: &[f32],
    layer: usize,
    t: usize,
    n: usize,
    d: usize,
    iters: usize,
) -> Vec<f32> {
    let nc = n * d;
    let nn = n * n;
    let mut logits = vec![0f32; t * nn];
    let alpha = a[layer];
    for ti in 0..t {
        let xr = &nx[ti * nc..(ti + 1) * nc];
        if nn == 16 {
            // Same contiguous-layout specialization as `mhc_h`, now for the
            // 4x4 residual router. Per-output accumulation order is unchanged.
            let mut dots = [0.0f32; 16];
            let phi = &phi[layer * nc * nn..(layer + 1) * nc * nn];
            for (value, row) in xr.iter().zip(phi.chunks_exact(16)) {
                for k in 0..16 {
                    dots[k] += *value * row[k];
                }
            }
            for k in 0..16 {
                logits[ti * nn + k] = alpha * dots[k] + b[layer * nn + k];
            }
            sinkhorn(&mut logits[ti * nn..(ti + 1) * nn], n, iters);
            continue;
        }
        for k in 0..nn {
            let mut dot = 0f32;
            for c in 0..nc {
                dot += xr[c] * phi[((layer * nc) + c) * nn + k];
            }
            logits[ti * nn + k] = alpha * dot + b[layer * nn + k];
        }
        sinkhorn(&mut logits[ti * nn..(ti + 1) * nn], n, iters);
    }
    logits
}

fn sinkhorn(mat: &mut [f32], n: usize, iters: usize) {
    // log-space like Python: subtract logsumexp rows then cols
    for _ in 0..iters {
        for i in 0..n {
            let row = &mut mat[i * n..(i + 1) * n];
            let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let mut s = 0f32;
            for v in row.iter() {
                s += (*v - m).exp();
            }
            let logz = m + s.ln();
            for v in row.iter_mut() {
                *v -= logz;
            }
        }
        for j in 0..n {
            let mut m = f32::NEG_INFINITY;
            for i in 0..n {
                m = m.max(mat[i * n + j]);
            }
            let mut s = 0f32;
            for i in 0..n {
                s += (mat[i * n + j] - m).exp();
            }
            let logz = m + s.ln();
            for i in 0..n {
                mat[i * n + j] -= logz;
            }
        }
    }
    for v in mat.iter_mut() {
        *v = v.exp();
    }
}

fn engram_indices(tokens: &[u32], orders: &[usize], heads: usize, slots: usize) -> Vec<u32> {
    let t = tokens.len();
    let n_tables = orders.len() * heads;
    let mut out = vec![0u32; t * n_tables];
    let mut tbl = 0usize;
    for (oi, order) in orders.iter().enumerate() {
        for head in 0..heads {
            let seed = ENGRAM_SEED.wrapping_mul((oi * heads + head + 1) as u32);
            for ti in 0..t {
                let mut acc = seed;
                for j in 0..*order {
                    let tok = if ti >= j { tokens[ti - j] } else { 0 };
                    acc = acc ^ tok;
                    acc = acc.wrapping_mul(ENGRAM_PRIME);
                }
                acc ^= acc >> 15;
                out[ti * n_tables + tbl] = acc % slots as u32;
            }
            tbl += 1;
        }
    }
    out
}

fn ngram_ok(_tokens: &[u32], ti: usize, orders: &[usize], tbl: usize, heads: usize) -> bool {
    let order = orders[tbl / heads];
    ti + 1 >= order
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn activation_qdq_is_real_int8_round_trip() {
        let original = vec![-2.0, -0.37, 0.0, 0.91, 200.0, -37.0, 0.0, 91.0];
        let mut quantized = original.clone();
        activation_i8_qdq(&mut quantized, 4);
        assert_ne!(quantized, original);
        for (row, quantized_row) in original.chunks_exact(4).zip(quantized.chunks_exact(4)) {
            let scale = row.iter().copied().map(f32::abs).fold(0.0, f32::max) / 127.0;
            for (left, right) in row.iter().zip(quantized_row) {
                assert!((left - right).abs() <= scale / 2.0 + 1e-6);
            }
        }
    }

    #[test]
    fn activation_qdq_rows_are_isolated() {
        let first = vec![-2.0, -0.37, 0.0, 0.91];
        let mut alone = first.clone();
        activation_i8_qdq(&mut alone, 4);
        let mut extended = first
            .iter()
            .copied()
            .chain([20000.0, -3700.0, 0.0, 9100.0])
            .collect::<Vec<_>>();
        activation_i8_qdq(&mut extended, 4);
        assert_eq!(alone, extended[..4]);
    }

    #[test]
    fn bounded_cache_keeps_stable_prefix_and_rolling_ring() {
        const VECTOR_WIDTH: usize = 4;
        let initial = (0..MAX_CONTEXT_TOKENS * VECTOR_WIDTH)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let previous = LayerCache::from_f32(
            &initial,
            &initial,
            MAX_CONTEXT_TOKENS,
            2,
            MAX_CONTEXT_TOKENS,
            VECTOR_WIDTH,
        );
        let extra = (MAX_CONTEXT_TOKENS * VECTOR_WIDTH..(MAX_CONTEXT_TOKENS + 10) * VECTOR_WIDTH)
            .map(|value| value as f32)
            .collect::<Vec<_>>();
        let (k, v, stable, next_position) = merge_kv_cache(
            Some(&previous),
            extra.clone(),
            extra,
            VECTOR_WIDTH,
            10,
            0,
            MAX_CONTEXT_TOKENS,
            true,
        );
        assert_eq!(stable, 2);
        assert_eq!(next_position, MAX_CONTEXT_TOKENS + 10);
        assert_eq!(k.len(), MAX_CONTEXT_TOKENS * VECTOR_WIDTH);
        assert_eq!(v.len(), k.len());
        let cache = LayerCache::from_f32(
            &k,
            &v,
            k.len() / VECTOR_WIDTH,
            stable,
            next_position,
            VECTOR_WIDTH,
        );
        assert_eq!(cache.stable_t, 2);
        assert_eq!(cache.rolling_t, MAX_CONTEXT_TOKENS - 2);
        assert!(cache.storage_bytes() < 2 * cache.t * VECTOR_WIDTH * std::mem::size_of::<f32>());
        assert_eq!(cache_evidence(&[cache])["cache_growth_bounded"], true);
    }

    #[test]
    fn portable_i8_cache_appends_without_requantizing_history() {
        const VECTOR_WIDTH: usize = 4;
        const ROW_WIDTH: usize = 8;
        let first = (0..2 * ROW_WIDTH)
            .map(|value| value as f32 * 0.125 - 0.75)
            .collect::<Vec<_>>();
        let mut cache =
            merge_kv_cache_i8(None, &first, &first, ROW_WIDTH, VECTOR_WIDTH, 2, 2, 0, true);
        let historical_k = cache.k.clone();
        let historical_v = cache.v.clone();
        let historical_k_scales = cache.k_scales.clone();
        let historical_v_scales = cache.v_scales.clone();
        let next = (0..ROW_WIDTH)
            .map(|value| value as f32 * -0.2 + 0.3)
            .collect::<Vec<_>>();
        cache = merge_kv_cache_i8(
            Some(&mut cache),
            &next,
            &next,
            ROW_WIDTH,
            VECTOR_WIDTH,
            1,
            1,
            2,
            true,
        );
        assert_eq!(cache.stable_t, 3);
        assert_eq!(cache.rolling_t, 0);
        assert_eq!(&cache.k[..historical_k.len()], historical_k);
        assert_eq!(&cache.v[..historical_v.len()], historical_v);
        assert_eq!(
            &cache.k_scales[..historical_k_scales.len()],
            historical_k_scales
        );
        assert_eq!(
            &cache.v_scales[..historical_v_scales.len()],
            historical_v_scales
        );
    }

    #[test]
    fn finite_engram_history_preserves_new_token_indices() {
        let tokens = (0..64).map(|value| value as u32 + 17).collect::<Vec<_>>();
        let orders = [2usize, 3usize];
        let heads = 2usize;
        let full = engram_indices(&tokens, &orders, heads, 8192);
        let prefix_needed = (4usize - 1) * 3 + (3 - 1);
        let new_tokens = 5usize;
        let suffix_start = tokens.len() - new_tokens - prefix_needed;
        let suffix = &tokens[suffix_start..];
        let compact = engram_indices(suffix, &orders, heads, 8192);
        let tables = orders.len() * heads;
        for offset in 0..new_tokens {
            let full_token = tokens.len() - new_tokens + offset;
            let compact_token = prefix_needed + offset;
            assert_eq!(
                &full[full_token * tables..(full_token + 1) * tables],
                &compact[compact_token * tables..(compact_token + 1) * tables]
            );
        }
    }

    #[test]
    fn runtime_contract_reports_actual_bounds() {
        assert_eq!(
            runtime_contract_evidence(),
            json!({
                "kv_storage_dtype": "int8",
                "activation_quantization": "int8-qdq",
                "activation_quantization_semantics": "int8-symmetric-per-last-axis-vector-qdq",
                "activation_q_quantized": false,
                "kv_scale_granularity": "per-head-vector",
                "stable_prefix_tokens_max": 1536,
                "stable_prefix_profiles": {"compact":1024,"standard":1536},
                "ordinary_window_policy": "dynamic_remainder",
                "rolling_window_compatibility_floor": 256,
                "max_context_tokens": 2048,
                "cache_growth_bounded": true
            })
        );
    }
}
