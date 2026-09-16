# mei-1.3 tokenizer

本目录与 v1.2 的 tokenizer 布局一致，直接保存当前代采用的模型、可读词表、manifest 和
唯一权威指针：

- `hans-en-24k-v1.model`：SentencePiece 二进制模型。
- `hans-en-24k-v1.vocab`：可读词片及分数。
- `tokenizer-hans-en-24k-v1-manifest.json`：哈希、训练样本及编码合同。
- `TOKENIZER.json`：本代当前采用指针。

该词表为 24,000 位置的简体中文＋英文无损词表，已绑定 v1.3 CPT 冻结输入并投入 A10
训练。纠正代际前的 v1.2 candidate 路径只保留为血缘证据。
