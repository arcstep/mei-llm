# Natural corpus sources

Formal entrypoint:

```bash
PYTHONPATH=src .venv/bin/python -m mei_llm corpus source download --help
```

The command inventories natural pools, plans unseen-first CPT mixes, admits
licensed documents through the frozen tokenizer, freezes pool manifests, and
performs explicitly authorized FineWeb2-HQ downloads.

Network access is denied unless a batch names every remote filename and passes
the exact authorization value reported by `download-hq --help`. Download,
admission, and pool release are separate steps; downloaded bytes are never
training-eligible until admission and quality receipts pass.

Mix planning validates that quotas equal the requested increment and that consumption does not exceed
physical capacity. A passed mix is arithmetic evidence only: `validation_scope=quota_capacity_only`,
`training_adoption_eligible=false`, and `synthetic_fraction=null` until independently measured.
LCCC-base downloads use fixed upstream revision URLs and SHA256 pins for all three splits.
