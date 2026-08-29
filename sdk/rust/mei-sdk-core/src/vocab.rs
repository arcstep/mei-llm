//! Unigram-ish tokenizer from `tokenizer.vocab.json` exported at pack time.

use serde::Deserialize;

use crate::error::SdkError;

#[derive(Debug, Clone, Deserialize)]
struct VocabFile {
    vocab_size: usize,
    pad_id: u32,
    eos_id: u32,
    bos_id: u32,
    unk_id: u32,
    pieces: Vec<Piece>,
}

#[derive(Debug, Clone, Deserialize)]
struct Piece {
    id: u32,
    piece: String,
    #[serde(default)]
    score: f32,
}

#[derive(Debug, Clone)]
pub struct Vocab {
    pub pad_id: u32,
    pub eos_id: u32,
    pub bos_id: u32,
    pub unk_id: u32,
    pub vocab_size: usize,
    pieces: Vec<String>,
    scores: Vec<f32>,
    by_piece: std::collections::HashMap<String, u32>,
}

impl Vocab {
    pub fn from_json(bytes: &[u8]) -> Result<Self, SdkError> {
        let file: VocabFile = serde_json::from_slice(bytes)
            .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
        let mut pieces = vec![String::new(); file.vocab_size.max(file.pieces.len())];
        let mut scores = vec![0f32; pieces.len()];
        let mut by_piece = std::collections::HashMap::new();
        for item in file.pieces {
            let id = item.id as usize;
            if id >= pieces.len() {
                pieces.resize(id + 1, String::new());
                scores.resize(id + 1, 0.0);
            }
            by_piece.insert(item.piece.clone(), item.id);
            pieces[id] = item.piece;
            scores[id] = item.score;
        }
        Ok(Self {
            pad_id: file.pad_id,
            eos_id: file.eos_id,
            bos_id: file.bos_id,
            unk_id: file.unk_id,
            vocab_size: file.vocab_size,
            pieces,
            scores,
            by_piece,
        })
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces.get(id as usize).map(|s| s.as_str()).unwrap_or("")
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        if id == self.pad_id || id == self.bos_id || id == self.eos_id {
            return Vec::new();
        }
        let piece = self.piece(id);
        if let Some(hex) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
            if hex.len() == 2 {
                if let Ok(b) = u8::from_str_radix(hex, 16) {
                    return vec![b];
                }
            }
        }
        let text = if let Some(rest) = piece.strip_prefix('▁') {
            format!(" {rest}")
        } else {
            piece.to_string()
        };
        text.into_bytes()
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if id == self.pad_id || id == self.bos_id {
                continue;
            }
            if id == self.eos_id {
                break;
            }
            let piece = self.piece(id);
            if let Some(hex) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                if hex.len() == 2 {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        out.push(b as char);
                        continue;
                    }
                }
            }
            if let Some(rest) = piece.strip_prefix('▁') {
                out.push(' ');
                out.push_str(rest);
            } else {
                out.push_str(piece);
            }
        }
        out
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut normalized = String::new();
        if !text.starts_with(' ') {
            normalized.push('▁');
        }
        for ch in text.chars() {
            if ch == ' ' {
                normalized.push('▁');
            } else {
                normalized.push(ch);
            }
        }
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        let chars: Vec<char> = normalized.chars().collect();
        let mut i = 0;
        while i < chars.len() {
            let mut best: Option<(usize, u32, f32)> = None;
            let mut candidate = String::new();
            for j in i..chars.len() {
                candidate.push(chars[j]);
                if let Some(&id) = self.by_piece.get(&candidate) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    best = Some((j + 1, id, score));
                }
                if candidate.len() > 24 {
                    break;
                }
            }
            if let Some((next, id, _)) = best {
                ids.push(id);
                i = next;
            } else {
                ids.push(self.unk_id);
                i += 1;
            }
        }
        ids
    }
}
