//! Portable SentencePiece-unigram execution from `tokenizer.model` or the
//! legacy derived `tokenizer.vocab.json` table.
//!
//! The frozen `zh-24k-v1` model uses `nmt_nfkc`, a dummy prefix, whitespace
//! escaping, and unigram Viterbi segmentation.  The portable table records the
//! piece score and type so Rust and WASM do not silently fall back to greedy
//! longest-match tokenization.

use serde::Deserialize;
use unicode_normalization::UnicodeNormalization;

use crate::error::SdkError;

#[derive(Debug, Clone, Deserialize)]
struct VocabFile {
    vocab_size: usize,
    pad_id: u32,
    eos_id: u32,
    bos_id: u32,
    unk_id: u32,
    #[serde(default = "default_model_type")]
    model_type: String,
    #[serde(default)]
    normalizer: NormalizerSpec,
    pieces: Vec<Piece>,
}

fn default_model_type() -> String {
    "unigram".to_string()
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
struct NormalizerSpec {
    name: String,
    add_dummy_prefix: bool,
    remove_extra_whitespaces: bool,
    escape_whitespaces: bool,
}

impl Default for NormalizerSpec {
    fn default() -> Self {
        // Legacy v1 tables omitted these fields but were exported from this
        // exact frozen tokenizer.
        Self {
            name: "nmt_nfkc".to_string(),
            add_dummy_prefix: true,
            remove_extra_whitespaces: true,
            escape_whitespaces: true,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
struct Piece {
    id: u32,
    piece: String,
    #[serde(default)]
    score: f32,
    #[serde(rename = "type", default)]
    kind: PieceKind,
}

#[derive(Debug, Clone, Copy, Default, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum PieceKind {
    #[default]
    Normal,
    Unknown,
    Control,
    UserDefined,
    Unused,
    Byte,
}

fn invalid_model(message: impl Into<String>) -> SdkError {
    SdkError::new("invalid_tokenizer_model", message.into())
}

fn read_varint(bytes: &[u8], cursor: &mut usize) -> Result<u64, SdkError> {
    let mut value = 0u64;
    for shift in (0..=63).step_by(7) {
        let byte = *bytes
            .get(*cursor)
            .ok_or_else(|| invalid_model("truncated protobuf varint"))?;
        *cursor += 1;
        value |= u64::from(byte & 0x7f) << shift;
        if byte & 0x80 == 0 {
            return Ok(value);
        }
    }
    Err(invalid_model("protobuf varint exceeds 64 bits"))
}

fn read_len<'a>(bytes: &'a [u8], cursor: &mut usize) -> Result<&'a [u8], SdkError> {
    let len = usize::try_from(read_varint(bytes, cursor)?)
        .map_err(|_| invalid_model("protobuf length exceeds address space"))?;
    let end = cursor
        .checked_add(len)
        .ok_or_else(|| invalid_model("protobuf length overflow"))?;
    let value = bytes
        .get(*cursor..end)
        .ok_or_else(|| invalid_model("truncated length-delimited protobuf field"))?;
    *cursor = end;
    Ok(value)
}

fn skip_field(bytes: &[u8], cursor: &mut usize, wire: u8) -> Result<(), SdkError> {
    match wire {
        0 => {
            read_varint(bytes, cursor)?;
        }
        1 => {
            *cursor = cursor
                .checked_add(8)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| invalid_model("truncated fixed64 protobuf field"))?;
        }
        2 => {
            read_len(bytes, cursor)?;
        }
        5 => {
            *cursor = cursor
                .checked_add(4)
                .filter(|end| *end <= bytes.len())
                .ok_or_else(|| invalid_model("truncated fixed32 protobuf field"))?;
        }
        _ => {
            return Err(invalid_model(format!(
                "unsupported protobuf wire type {wire}"
            )))
        }
    }
    Ok(())
}

fn read_id(bytes: &[u8], cursor: &mut usize, name: &str) -> Result<u32, SdkError> {
    let value = read_varint(bytes, cursor)?;
    u32::try_from(value).map_err(|_| invalid_model(format!("{name} must be a non-negative u32")))
}

fn parse_sentencepiece(bytes: &[u8], id: u32) -> Result<Piece, SdkError> {
    let mut cursor = 0usize;
    let mut piece = None::<String>;
    let mut score = 0f32;
    let mut kind = PieceKind::Normal;
    while cursor < bytes.len() {
        let key = read_varint(bytes, &mut cursor)?;
        let field = key >> 3;
        let wire = (key & 7) as u8;
        match (field, wire) {
            (1, 2) => {
                piece = Some(
                    std::str::from_utf8(read_len(bytes, &mut cursor)?)
                        .map_err(|_| invalid_model("SentencePiece text is not UTF-8"))?
                        .to_string(),
                );
            }
            (2, 5) => {
                let end = cursor
                    .checked_add(4)
                    .filter(|end| *end <= bytes.len())
                    .ok_or_else(|| invalid_model("truncated SentencePiece score"))?;
                score = f32::from_le_bytes(bytes[cursor..end].try_into().expect("four bytes"));
                cursor = end;
            }
            (3, 0) => {
                kind = match read_varint(bytes, &mut cursor)? {
                    1 => PieceKind::Normal,
                    2 => PieceKind::Unknown,
                    3 => PieceKind::Control,
                    4 => PieceKind::UserDefined,
                    5 => PieceKind::Unused,
                    6 => PieceKind::Byte,
                    value => {
                        return Err(invalid_model(format!(
                            "unsupported SentencePiece type {value}"
                        )))
                    }
                };
            }
            _ => skip_field(bytes, &mut cursor, wire)?,
        }
    }
    Ok(Piece {
        id,
        piece: piece.ok_or_else(|| invalid_model("SentencePiece entry has no text"))?,
        score,
        kind,
    })
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
    kinds: Vec<PieceKind>,
    by_piece: std::collections::HashMap<String, u32>,
    max_piece_chars: usize,
    normalizer: NormalizerSpec,
    token_bytes_cache: Vec<Vec<u8>>,
}

impl Vocab {
    /// Parse the canonical package tokenizer payload. Native v2 packages use
    /// the SentencePiece ModelProto directly; legacy packages may still carry
    /// the derived JSON table.
    pub fn from_package_payload(bytes: &[u8]) -> Result<Self, SdkError> {
        if bytes
            .iter()
            .copied()
            .find(|byte| !byte.is_ascii_whitespace())
            == Some(b'{')
        {
            Self::from_json(bytes)
        } else {
            Self::from_sentencepiece_model(bytes)
        }
    }

    pub fn from_json(bytes: &[u8]) -> Result<Self, SdkError> {
        let file: VocabFile = serde_json::from_slice(bytes)
            .map_err(|err| SdkError::new("invalid_json", err.to_string()))?;
        Self::from_parts(
            file.vocab_size,
            file.pad_id,
            file.eos_id,
            file.bos_id,
            file.unk_id,
            file.model_type,
            file.normalizer,
            file.pieces,
        )
    }

    /// Decode the small subset of SentencePiece ModelProto required by the
    /// frozen `zh-24k-v1` unigram tokenizer. Unknown protobuf fields are
    /// skipped so the parser remains compatible with newer producers.
    pub fn from_sentencepiece_model(bytes: &[u8]) -> Result<Self, SdkError> {
        let mut cursor = 0usize;
        let mut pieces = Vec::<Piece>::new();
        let mut model_type = 1u64;
        let mut vocab_size = 0usize;
        let mut pad_id = 0u32;
        let mut eos_id = 1u32;
        let mut bos_id = 2u32;
        let mut unk_id = 3u32;
        let mut normalizer = NormalizerSpec::default();
        while cursor < bytes.len() {
            let key = read_varint(bytes, &mut cursor)?;
            let field = key >> 3;
            let wire = (key & 7) as u8;
            match (field, wire) {
                (1, 2) => {
                    let body = read_len(bytes, &mut cursor)?;
                    pieces.push(parse_sentencepiece(body, pieces.len() as u32)?);
                }
                (2, 2) => {
                    let body = read_len(bytes, &mut cursor)?;
                    let mut inner = 0usize;
                    while inner < body.len() {
                        let inner_key = read_varint(body, &mut inner)?;
                        let inner_field = inner_key >> 3;
                        let inner_wire = (inner_key & 7) as u8;
                        match (inner_field, inner_wire) {
                            (3, 0) => model_type = read_varint(body, &mut inner)?,
                            (4, 0) => vocab_size = read_varint(body, &mut inner)? as usize,
                            (40, 0) => unk_id = read_id(body, &mut inner, "unk_id")?,
                            (41, 0) => bos_id = read_id(body, &mut inner, "bos_id")?,
                            (42, 0) => eos_id = read_id(body, &mut inner, "eos_id")?,
                            (43, 0) => pad_id = read_id(body, &mut inner, "pad_id")?,
                            _ => skip_field(body, &mut inner, inner_wire)?,
                        }
                    }
                }
                (3, 2) => {
                    let body = read_len(bytes, &mut cursor)?;
                    let mut inner = 0usize;
                    while inner < body.len() {
                        let inner_key = read_varint(body, &mut inner)?;
                        let inner_field = inner_key >> 3;
                        let inner_wire = (inner_key & 7) as u8;
                        match (inner_field, inner_wire) {
                            (1, 2) => {
                                normalizer.name = std::str::from_utf8(read_len(body, &mut inner)?)
                                    .map_err(|_| invalid_model("normalizer name is not UTF-8"))?
                                    .to_string();
                            }
                            (3, 0) => {
                                normalizer.add_dummy_prefix = read_varint(body, &mut inner)? != 0
                            }
                            (4, 0) => {
                                normalizer.remove_extra_whitespaces =
                                    read_varint(body, &mut inner)? != 0
                            }
                            (5, 0) => {
                                normalizer.escape_whitespaces = read_varint(body, &mut inner)? != 0
                            }
                            _ => skip_field(body, &mut inner, inner_wire)?,
                        }
                    }
                }
                _ => skip_field(bytes, &mut cursor, wire)?,
            }
        }
        if pieces.is_empty() {
            return Err(invalid_model("SentencePiece model has no pieces"));
        }
        if vocab_size == 0 {
            vocab_size = pieces.len();
        }
        Self::from_parts(
            vocab_size,
            pad_id,
            eos_id,
            bos_id,
            unk_id,
            if model_type == 1 {
                "unigram"
            } else {
                "unsupported"
            }
            .to_string(),
            normalizer,
            pieces,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn from_parts(
        vocab_size: usize,
        pad_id: u32,
        eos_id: u32,
        bos_id: u32,
        unk_id: u32,
        model_type: String,
        normalizer: NormalizerSpec,
        source_pieces: Vec<Piece>,
    ) -> Result<Self, SdkError> {
        let mut pieces = vec![String::new(); vocab_size.max(source_pieces.len())];
        let mut scores = vec![0f32; pieces.len()];
        let mut kinds = vec![PieceKind::Normal; pieces.len()];
        let mut by_piece = std::collections::HashMap::new();
        let mut max_piece_chars = 0usize;
        for item in source_pieces {
            let id = item.id as usize;
            if id >= pieces.len() {
                pieces.resize(id + 1, String::new());
                scores.resize(id + 1, 0.0);
                kinds.resize(id + 1, PieceKind::Normal);
            }
            if matches!(item.kind, PieceKind::Normal | PieceKind::UserDefined) {
                max_piece_chars = max_piece_chars.max(item.piece.chars().count());
                by_piece.insert(item.piece.clone(), item.id);
            }
            pieces[id] = item.piece;
            scores[id] = item.score;
            kinds[id] = item.kind;
        }
        if model_type != "unigram" {
            return Err(SdkError::new(
                "unsupported_tokenizer",
                format!("portable tokenizer requires unigram, got {}", model_type),
            ));
        }
        if normalizer.name != "nmt_nfkc" {
            return Err(SdkError::new(
                "unsupported_tokenizer",
                format!(
                    "portable tokenizer requires nmt_nfkc, got {}",
                    normalizer.name
                ),
            ));
        }
        let mut vocab = Self {
            pad_id,
            eos_id,
            bos_id,
            unk_id,
            vocab_size,
            pieces,
            scores,
            kinds,
            by_piece,
            max_piece_chars,
            normalizer,
            token_bytes_cache: Vec::new(),
        };
        vocab.token_bytes_cache = (0..vocab.vocab_size as u32)
            .map(|id| vocab.token_bytes_uncached(id))
            .collect();
        Ok(vocab)
    }

    pub fn piece(&self, id: u32) -> &str {
        self.pieces
            .get(id as usize)
            .map(|s| s.as_str())
            .unwrap_or("")
    }

    pub fn token_bytes(&self, id: u32) -> Vec<u8> {
        self.token_bytes_ref(id).to_vec()
    }

    fn token_bytes_uncached(&self, id: u32) -> Vec<u8> {
        if id == self.pad_id || id == self.bos_id || id == self.eos_id {
            return Vec::new();
        }
        if id == self.unk_id {
            return " ⁇ ".as_bytes().to_vec();
        }
        if matches!(
            self.kinds.get(id as usize),
            Some(PieceKind::Control | PieceKind::Unused)
        ) {
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

    pub fn token_bytes_ref(&self, id: u32) -> &[u8] {
        self.token_bytes_cache
            .get(id as usize)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    pub fn generated_token_bytes(&self, id: u32, at_start: bool) -> Vec<u8> {
        self.generated_token_bytes_ref(id, at_start).to_vec()
    }

    pub fn generated_token_bytes_ref(&self, id: u32, at_start: bool) -> &[u8] {
        let bytes = self.token_bytes_ref(id);
        if at_start && self.piece(id).starts_with('▁') && bytes.first() == Some(&b' ') {
            &bytes[1..]
        } else {
            bytes
        }
    }

    pub fn decode(&self, ids: &[u32]) -> String {
        let mut out = Vec::<u8>::new();
        for &id in ids {
            if id == self.pad_id || id == self.bos_id {
                continue;
            }
            if id == self.eos_id {
                break;
            }
            if id == self.unk_id {
                out.extend_from_slice(" ⁇ ".as_bytes());
                continue;
            }
            if matches!(
                self.kinds.get(id as usize),
                Some(PieceKind::Control | PieceKind::Unused)
            ) {
                continue;
            }
            let piece = self.piece(id);
            if let Some(hex) = piece.strip_prefix("<0x").and_then(|s| s.strip_suffix(">")) {
                if hex.len() == 2 {
                    if let Ok(b) = u8::from_str_radix(hex, 16) {
                        out.push(b);
                        continue;
                    }
                }
            }
            if let Some(rest) = piece.strip_prefix('▁') {
                out.push(b' ');
                out.extend_from_slice(rest.as_bytes());
            } else {
                out.extend_from_slice(piece.as_bytes());
            }
        }
        if out.first() == Some(&b' ') {
            out.remove(0);
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if add_bos {
            ids.push(self.bos_id);
        }
        let normalized = self.normalize(text);
        if normalized.is_empty() {
            return ids;
        }
        let chars: Vec<char> = normalized.chars().collect();
        let n = chars.len();
        let mut best_score = vec![f32::NEG_INFINITY; n + 1];
        let mut predecessor: Vec<Option<(usize, u32)>> = vec![None; n + 1];
        best_score[0] = 0.0;
        let min_score = self
            .scores
            .iter()
            .copied()
            .filter(|value| value.is_finite())
            .fold(0.0f32, f32::min);
        let unknown_score = min_score - 10.0;
        for i in 0..n {
            if !best_score[i].is_finite() {
                continue;
            }
            let fallback = best_score[i] + unknown_score;
            if fallback > best_score[i + 1] {
                best_score[i + 1] = fallback;
                predecessor[i + 1] = Some((i, self.unk_id));
            }
            let mut candidate = String::new();
            for j in i..n.min(i + self.max_piece_chars) {
                candidate.push(chars[j]);
                if let Some(&id) = self.by_piece.get(&candidate) {
                    let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                    let total = best_score[i] + score;
                    if total > best_score[j + 1] {
                        best_score[j + 1] = total;
                        predecessor[j + 1] = Some((i, id));
                    }
                }
            }
        }
        let mut reversed = Vec::<u32>::new();
        let mut cursor = n;
        while cursor > 0 {
            let Some((previous, id)) = predecessor[cursor] else {
                // This is unreachable because every Unicode scalar has an
                // unknown fallback, but fail closed if the lattice is corrupt.
                return vec![self.unk_id];
            };
            if id != self.unk_id || reversed.last() != Some(&self.unk_id) {
                reversed.push(id);
            }
            cursor = previous;
        }
        reversed.reverse();
        ids.extend(reversed);
        ids
    }

    fn normalize(&self, text: &str) -> String {
        if text.is_empty() {
            return String::new();
        }
        let nfkc: String = text.nfkc().collect();
        let mut visible = String::new();
        let mut previous_space = false;
        for ch in nfkc.chars() {
            let is_space = ch.is_whitespace();
            if is_space {
                if self.normalizer.remove_extra_whitespaces {
                    if !visible.is_empty() && !previous_space {
                        visible.push(' ');
                    }
                } else {
                    visible.push(' ');
                }
                previous_space = true;
                continue;
            }
            // nmt_nfkc removes remaining C0/C1 controls.
            if ch.is_control() {
                continue;
            }
            visible.push(ch);
            previous_space = false;
        }
        if self.normalizer.remove_extra_whitespaces {
            while visible.ends_with(' ') {
                visible.pop();
            }
        }
        if visible.is_empty() {
            return String::new();
        }
        let mut normalized = String::new();
        if self.normalizer.add_dummy_prefix {
            normalized.push(' ');
        }
        normalized.push_str(&visible);
        if self.normalizer.escape_whitespaces {
            normalized = normalized.replace(' ', "▁");
        }
        normalized
    }
}
