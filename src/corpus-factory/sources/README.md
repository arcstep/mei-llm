# Natural corpus sources

Formal entrypoint:

```bash
.venv/bin/python corpus-factory/sources/source_manager.py --help
```

The command inventories natural pools, plans unseen-first CPT mixes, admits
licensed documents through the frozen tokenizer, freezes pool manifests, and
performs explicitly authorized FineWeb2-HQ downloads.

Network access is denied unless a batch names every remote filename and passes
the exact authorization value reported by `download-hq --help`. Download,
admission, and pool release are separate steps; downloaded bytes are never
training-eligible until admission and quality receipts pass.
