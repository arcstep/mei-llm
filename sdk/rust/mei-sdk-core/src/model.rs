//! CPU Needle-zh 51M forward over packed Q4/Q2 weights.

use std::collections::HashMap;
use std::sync::Arc;

use serde_json::Value;

use crate::error::SdkError;
use crate::packed::{arch_f, arch_u, PackedWeights, BLOCK_SIZE};

const ENGRAM_SUB_DIM: usize = 128;
const ENGRAM_SEED: u32 = 0x9E3779B9;
const ENGRAM_PRIME: u32 = 0x01000193;

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
}

impl Arch {
    pub fn from_manifest(manifest: &Value) -> Self {
        let d_model = arch_u(manifest, "d_model", 512);
        let n_heads = arch_u(manifest, "n_heads", 8);
        let engram_layers = manifest
            .get("architecture")
            .and_then(|a| a.get("engram_layers"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
            .unwrap_or_else(|| vec![2, 15]);
        let engram_orders = manifest
            .get("architecture")
            .and_then(|a| a.get("engram_orders"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).map(|x| x as usize).collect())
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
        }
    }
}

#[derive(Clone)]
pub struct LayerCache {
    pub k: Vec<f32>, // [n_kv, t, head_dim]
    pub v: Vec<f32>,
    pub t: usize,
}

pub struct NeedleModel {
    pub arch: Arch,
    pub weights: PackedWeights,
    cache: std::sync::Mutex<HashMap<String, Arc<Vec<f32>>>>,
    /// When true, never retain a full dequantized weight table. WASM always
    /// uses the row-wise path regardless of this flag.
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

    fn rowwise(&self) -> bool {
        self.disable_tensor_cache || cfg!(target_arch = "wasm32")
    }

    fn tensor(&self, name: &str) -> Result<Arc<Vec<f32>>, SdkError> {
        if !self.rowwise() {
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

    fn linear(&self, name: &str, x: &[f32], rows: usize, cols: usize) -> Result<Vec<f32>, SdkError> {
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

    pub fn embed(&self, tokens: &[u32]) -> Result<Vec<f32>, SdkError> {
        let d = self.arch.d_model;
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

    fn lm_logits(&self, hidden: &[f32], t: usize) -> Result<Vec<f32>, SdkError> {
        let a = &self.arch;
        let d = a.d_model;
        if self.rowwise() {
            let mut logits = vec![0f32; t * a.vocab_size];
            for ti in 0..t {
                let y = self
                    .weights
                    .matvec("embed.weight", &hidden[ti * d..(ti + 1) * d])?;
                logits[ti * a.vocab_size..(ti + 1) * a.vocab_size].copy_from_slice(&y);
            }
            return Ok(logits);
        }
        let embed = self.tensor("embed.weight")?;
        let mut logits = vec![0f32; t * a.vocab_size];
        for ti in 0..t {
            let h = &hidden[ti * d..(ti + 1) * d];
            for v in 0..a.vocab_size {
                let row = &embed[v * d..(v + 1) * d];
                let mut acc = 0f32;
                for (p, q) in h.iter().zip(row.iter()) {
                    acc += *p * *q;
                }
                logits[ti * a.vocab_size + v] = acc;
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
        let a = &self.arch;
        let b = 1usize;
        let t = tokens.len();
        let d = a.d_model;
        let n = a.mhc_lanes;
        let x = self.embed(tokens)?;
        let cache_len = cache
            .as_ref()
            .and_then(|c| c.first().map(|l| l.t))
            .unwrap_or(0);
        let rope = precompute_rope(a.head_dim, (cache_len + t).max(8), a.rope_theta);
        let engram_kv = self.engram_stack(tokens)?;
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
        let mut cells: Vec<Vec<f32>> = vec![x.clone()];
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
            let hpre = mhc_h(&nx, &phi_pre, &b_pre, &a_pre, &pre_off, layer, t, n, d, true);
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
            if let Some(site) = site_by_layer.get(&layer) {
                if let Some((ek, ev)) = engram_kv.as_ref() {
                    let key = &ek[*site];
                    let value = &ev[*site];
                    for ti in 0..t {
                        let xr = &u[ti * d..(ti + 1) * d];
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
                            u[ti * d + c] += alpha * vr[c];
                        }
                    }
                }
            }
            let layer_cache = cache.as_mut().and_then(|c| c.get_mut(layer));
            let (block_out, next_cache) =
                self.block(layer, &u, t, &rope, cache_len, layer_cache)?;
            let mut delta = vec![0f32; t * d];
            for i in 0..t * d {
                delta[i] = block_out[i] - u[i];
            }
            let hpost = mhc_h(&nx, &phi_post, &b_post, &a_post, &post_off, layer, t, n, d, false);
            let hres = mhc_res(&nx, &phi_res, &b_res, &a_res, layer, t, n, d, a.sinkhorn_iters);
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
            cells.push(mean_lanes(&lanes, t, n, d));
        }
        let hidden = zc_rms_norm(&mean_lanes(&lanes, t, n, d), &self.tensor("final_norm.scale")?, t, d, a.rms_eps);
        let logits = self.lm_logits(&hidden, t)?;
        let mut conf = None;
        if return_confidence {
            let last = &hidden[(t - 1) * d..t * d];
            let probes = self.tensor("conf_probes")?;
            let mut scores = vec![0f32; a.conf_probes];
            let scale = (d as f32).sqrt();
            for p in 0..a.conf_probes {
                let mut dot = 0f32;
                let pr = &probes[p * d..(p + 1) * d];
                for (x, y) in last.iter().zip(pr.iter()) {
                    dot += *x * *y;
                }
                scores[p] = dot / scale;
            }
            softmax_inplace(&mut scores);
            let mut pooled = vec![0f32; a.conf_probes * d];
            for p in 0..a.conf_probes {
                for c in 0..d {
                    pooled[p * d + c] = scores[p] * last[c];
                }
            }
            let w = self.tensor("conf_proj.weight")?;
            let bias = self.tensor("conf_proj.bias")?;
            let mut logit = bias.first().copied().unwrap_or(0.0);
            for (i, v) in pooled.iter().enumerate() {
                logit += w[i] * *v;
            }
            conf = Some(logit);
        }
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
        })
    }

    fn block(
        &self,
        layer: usize,
        x: &[f32],
        t: usize,
        rope: &(Vec<f32>, Vec<f32>),
        cache_len: usize,
        cache: Option<&mut LayerCache>,
    ) -> Result<(Vec<f32>, LayerCache), SdkError> {
        let a = &self.arch;
        let d = a.d_model;
        let prefix = format!("blocks.{layer}");
        let xn = zc_rms_norm(x, &self.tensor(&format!("{prefix}.attn_norm.scale"))?, t, d, a.rms_eps);
        let q = self.linear(
            &format!("{prefix}.attn.q_proj.weight"),
            &xn,
            a.n_heads * a.head_dim,
            d,
        )?;
        let k = self.linear(
            &format!("{prefix}.attn.k_proj.weight"),
            &xn,
            a.n_kv_heads * a.head_dim,
            d,
        )?;
        let v = self.linear(
            &format!("{prefix}.attn.v_proj.weight"),
            &xn,
            a.n_kv_heads * a.head_dim,
            d,
        )?;
        let gate = self.linear(
            &format!("{prefix}.attn.gate_proj.weight"),
            &xn,
            a.n_heads * a.head_dim,
            d,
        )?;
        let mut q = reshape_heads(q, t, a.n_heads, a.head_dim);
        let mut k = reshape_heads(k, t, a.n_kv_heads, a.head_dim);
        let v = reshape_heads(v, t, a.n_kv_heads, a.head_dim);
        q = apply_qknorm(&q, t, a.n_heads, a.head_dim, &self.tensor(&format!("{prefix}.attn.q_norm.scale"))?, a.rms_eps);
        k = apply_qknorm(&k, t, a.n_kv_heads, a.head_dim, &self.tensor(&format!("{prefix}.attn.k_norm.scale"))?, a.rms_eps);
        apply_rope_inplace(&mut q, t, a.n_heads, a.head_dim, rope, cache_len);
        apply_rope_inplace(&mut k, t, a.n_kv_heads, a.head_dim, rope, cache_len);
        let (k_all, v_all, kv_t) = if let Some(prev) = cache {
            let mut kk = prev.k.clone();
            let mut vv = prev.v.clone();
            kk.extend_from_slice(&k);
            vv.extend_from_slice(&v);
            (kk, vv, prev.t + t)
        } else {
            (k, v, t)
        };
        let repeats = a.n_heads / a.n_kv_heads;
        let scale = (a.head_dim as f32).sqrt().recip();
        let mut attn_out = vec![0f32; t * a.n_heads * a.head_dim];
        for head in 0..a.n_heads {
            let kv_head = head / repeats;
            for qi in 0..t {
                let q_off = (qi * a.n_heads + head) * a.head_dim;
                let qv = &q[q_off..q_off + a.head_dim];
                let mut scores = vec![0f32; kv_t];
                for kj in 0..kv_t {
                    let k_off = (kj * a.n_kv_heads + kv_head) * a.head_dim;
                    let kv = &k_all[k_off..k_off + a.head_dim];
                    let mut dot = 0f32;
                    for (p, qq) in kv.iter().zip(qv.iter()) {
                        dot += *p * *qq;
                    }
                    let pos_q = cache_len + qi;
                    scores[kj] = if pos_q >= kj { dot * scale } else { f32::NEG_INFINITY };
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
        for i in 0..attn_out.len() {
            attn_out[i] *= sigmoid(gate[i]);
        }
        let attn_flat = attn_out;
        let o = self.linear(&format!("{prefix}.attn.o_proj.weight"), &attn_flat, d, a.n_heads * a.head_dim)?;
        let gate_attn = self.tensor(&format!("{prefix}.attn_gate"))?;
        let g = sigmoid(gate_attn.first().copied().unwrap_or(0.0));
        let post = zc_rms_norm(&o, &self.tensor(&format!("{prefix}.post_attn_norm.scale"))?, t, d, a.rms_eps);
        let mut h = vec![0f32; t * d];
        for i in 0..t * d {
            h[i] = x[i] + g * post[i];
        }
        let mlp_in = zc_rms_norm(&h, &self.tensor(&format!("{prefix}.mlp_norm.scale"))?, t, d, a.rms_eps);
        let mlp = self.hadamard_mlp(layer, &mlp_in, t)?;
        for i in 0..t * d {
            h[i] += mlp[i];
        }
        Ok((
            h,
            LayerCache {
                k: k_all,
                v: v_all,
                t: kv_t,
            },
        ))
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

    fn engram_stack(&self, tokens: &[u32]) -> Result<Option<(Vec<Vec<f32>>, Vec<Vec<f32>>)>, SdkError> {
        let a = &self.arch;
        if a.engram_layers.is_empty() {
            return Ok(None);
        }
        let orders = &a.engram_orders;
        let heads = (a.d_model / (orders.len() * ENGRAM_SUB_DIM)).max(1);
        let sub_dim = a.d_model / (orders.len() * heads);
        let n_tables = orders.len() * heads;
        let t = tokens.len();
        let mut keys = Vec::new();
        let mut values = Vec::new();
        for (eg, _layer) in a.engram_layers.iter().enumerate() {
            let tables = self.tensor(&format!("engrams.{eg}.tables"))?;
            let mut fetched = vec![0f32; t * n_tables * sub_dim];
            let indices = engram_indices(tokens, orders, heads, a.engram_slots);
            for ti in 0..t {
                for tbl in 0..n_tables {
                    let idx = indices[ti * n_tables + tbl] as usize;
                    let src = &tables[(tbl * a.engram_slots + idx) * sub_dim
                        ..(tbl * a.engram_slots + idx + 1) * sub_dim];
                    let ok = ngram_ok(tokens, ti, orders, tbl, heads);
                    for c in 0..sub_dim {
                        fetched[ti * n_tables * sub_dim + tbl * sub_dim + c] =
                            if ok { src[c] } else { 0.0 };
                    }
                }
            }
            let e_dim = n_tables * sub_dim;
            let key = self.linear(&format!("engrams.{eg}.key_proj.weight"), &fetched, a.d_model, e_dim)?;
            let value = self.linear(&format!("engrams.{eg}.value_proj.weight"), &fetched, a.d_model, e_dim)?;
            let taps = self.tensor(&format!("engrams.{eg}.taps"))?;
            let dil = *orders.iter().max().unwrap_or(&1);
            let mut mixed = vec![0f32; t * a.d_model];
            for j in 0..a.engram_conv_taps {
                let off = j * dil;
                for ti in 0..t {
                    if ti < off {
                        continue;
                    }
                    let src_t = ti - off;
                    for c in 0..a.d_model {
                        mixed[ti * a.d_model + c] +=
                            taps[j * a.d_model + c] * value[src_t * a.d_model + c];
                    }
                }
            }
            keys.push(key);
            values.push(mixed);
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
    let mut ms = eps;
    for v in x {
        ms += *v * *v;
    }
    let inv = (ms / x.len() as f32).sqrt().recip();
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

fn apply_qknorm(x: &[f32], t: usize, heads: usize, head_dim: usize, scale: &[f32], eps: f32) -> Vec<f32> {
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

fn precompute_rope(head_dim: usize, seq: usize, theta: f32) -> (Vec<f32>, Vec<f32>) {
    let half = head_dim / 2;
    let mut cos = vec![0f32; seq * half];
    let mut sin = vec![0f32; seq * half];
    for t in 0..seq {
        for i in 0..half {
            let freq = 1.0 / theta.powf((2 * i) as f32 / head_dim as f32);
            let ang = t as f32 * freq;
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
    offset: usize,
) {
    let half = head_dim / 2;
    let (cos, sin) = rope;
    for ti in 0..t {
        let pos = offset + ti;
        for h in 0..heads {
            let off = (ti * heads + h) * head_dim;
            for i in 0..half {
                let c = cos[pos * half + i];
                let s = sin[pos * half + i];
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
        for lane in 0..n {
            let mut dot = 0f32;
            let col = &phi[layer * nc * n + lane..]; // wrong layout
            let _ = col;
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
