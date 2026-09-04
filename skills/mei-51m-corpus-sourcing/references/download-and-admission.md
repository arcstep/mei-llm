# Download and admission

## 当前池经验

300M/600M 使用 wiki、HQ、structure、colloquial 四角色；后两类合成池已因模板
坍缩退役。下一轮默认优先天然来源，并从 cycle `cpt.json` 和 pool manifests
即时推导余额，不维护第二份手写账本。

## 下载门

- `download_hq.py`（兼容别名）：只有收到精确授权
  `epfml/FineWeb2-HQ:cmn_Hani` 和逐个 `--filename` 才联网。下载 receipt 仍保持
  `distribution_clearance_asserted=false`。
- `download.py`（v2 generic collector）：注册表驱动（`source_registry.json`），
  三类清单来源（static_manifest / hf_api / direct_urls）；**两段式授权**——
  先 `--dry-run` 离线解析清单并打印授权令牌，联网段令牌逐字节匹配才发起网络；
  逐文件 sha256 校验，hash 漂移 fail closed 且不落盘；未钉扎批次
  `hash_pinned:false`，治理回填钉扎 = supersede 事件。
- 每个新来源首次下载前必须完成 license/provenance 人工审核；
  band C 源（dialogue/code）每批附 clearance receipt。

## 入池门

- `admit.py` 要求 `--license-id`、`--license-reviewed`，使用冻结 tokenizer
  （`TOKENIZER.json` 指针选择，当前 zh-24k-v1；`--expected-tokenizer-id` 双绑
  防指针漂移），按规范化文档 SHA 去重，并写 `documents.jsonl`、`tokens.bin`、
  `manifest.json` 到全新目录。
- 带 `--source-id` 走 provenance v2：查注册表校验 role 一致、记录
  `registry_entry_sha256`；不带 = 旧 v1 兼容路径（只读 lineage，不新增）。
- 结构化源 `--mode structured`：记录级 sha256（canonical 字节）去重，
  tokenize 保留结构不折叠空白；坏结构全量解析（阈值默认 0.001，只可调严），
  超阈值 fail-closed 且无产物；XML 源须注册表 `record_element`。
- 之后必须交给 `mei-51m-corpus-quality` 审计（`audit-source` / band C 带
  clearance / 结构化源再跑 `audit-structured` 回源重哈希）；未通过前不得
  进入 schedule。
- v1 admitted 目录只读不改不转换；新词表代际 = 对原始下载重新 admit +
  新 pool release ID + supersede 链；seen-ledger 基于文本哈希，跨代复用。
