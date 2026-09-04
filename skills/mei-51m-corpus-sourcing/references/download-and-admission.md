# Download and admission

## 当前池经验

300M/600M 使用 wiki、HQ、structure、colloquial 四角色；后两类合成池已因模板
坍缩退役。下一轮默认优先天然来源，并从 cycle `cpt.json` 和 pool manifests
即时推导余额，不维护第二份手写账本。

## 下载门

`download_hq.py` 只有收到精确授权
`epfml/FineWeb2-HQ:cmn_Hani` 和逐个 `--filename` 才联网。下载 receipt 仍保持
`distribution_clearance_asserted=false`。

## 入池门

`admit.py` 要求 `--license-id`、`--license-reviewed`，使用冻结 zh-24k-v1
tokenizer，按规范化文档 SHA 去重，并写 `documents.jsonl`、`tokens.bin`、
`manifest.json` 到全新目录。之后必须交给 `mei-51m-corpus-quality` 审计；
未通过前不得进入 schedule。
