# zh-pretrain-v0

Needle-zh 中文预训语料槽。大 shard / raw / tokens **不入库**。公开仓只提交本 README、`SOURCES.md`、以及构建后的 `manifest.json` / `extract-manifest.json`（hash 清单）。

## 冻结规模（全文维基 unique）

源：本机完整 dump `../zh-vocab-v0/dumps/zhwiki-latest-pages-articles.xml.bz2`  
`dump_sha256` = `4699d4dc8342b21beaca36b3d699ad28844dd3837e319e84b9640759b6c62455`（3,379,207,113 bytes）  
清洗器：`zhwiki-pretrain-v1-paragraph`（无每页 256 截断、无总字符早停；段落切块 4096，最短 64 字）。

| 项 | 值 |
|----|-----|
| dump 页（含 ns/redirect） | 4,939,350 |
| 保留页 / 切块 | 1,288,332 / 1,330,123 |
| 正文字符 | 764,640,580 |
| 精确去重丢弃 | 286 |
| `frac_le_256` | 0.548055（短 stub，不是截断） |
| 编码文档 | 1,330,083（泄漏丢 40） |
| 总 token | 684,514,703 |
| train / valid token | **649,904,474** / 34,610,229 |
| UNK token rate | **0.043215%**（门 0.5%，`unk_gate_ok`） |
| chars/token | 1.1169 |
| 距 1B unique train | **315,485,297**（禁止重复 epoch 凑数） |
| 词表 | 冻结 `zh-24k-v1`，不自动重训 |

切分：确定性 95% train / 5% valid。二进制：little-endian `uint16` `.bin` + `.idx.jsonl`，按约 8e6 token 分 shard。

## 构建

在 `mei-llm/` 下用仓内 venv（系统 `python3` 无 MLX）：

```bash
.venv/bin/python scripts/extract_zhwiki_pretrain.py
.venv/bin/python scripts/build_zh_pretrain_v0.py --from-raw --rung 100m
.venv/bin/python scripts/check_train_eval_isolation.py --all
```

抽取可从 `raw/extract-state.sqlite` 按 dump page ordinal 恢复；禁止覆盖已关闭 shard。UNK token rate > 0.5% 时构建 exit 3，**不**自动生成 tokenizer v2。

小样本：`build_zh_pretrain_v0.py --smoke`（词表抽样短行 + 合成模板，不进入 wiki 主 shard）。

## 训练入口

- 吞吐选型：`.venv/bin/python scripts/bench_needle_zh_pretrain_throughput.py`
- 规模链（随机初始化，LR horizon = 全部 train token）：`.venv/bin/python scripts/run_needle_zh_scale_chain.py`
- 单档：`.venv/bin/python scripts/train_needle_zh_pretrain.py --rung 100m --stop-at-tokens 100000000 --lr-horizon-tokens 649904474 --batch-size 8 --grad-accum 1`

正式运行 **禁止** `--allow-repeat`、禁止 `--compile`（本机 `mx.compile(value_and_grad)` 失败）、禁止未过恢复/probe 门的 fp16。官方吞吐配置：batch 8 / accum 1 / fp32，约 6105 tok/s。

来源与许可见 `SOURCES.md`。
