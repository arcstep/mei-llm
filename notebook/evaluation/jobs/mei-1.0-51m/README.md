# mei-1.0-51m evaluation jobs

Diagnostic ledgers for the immutable 51M float base. These files are not SSOT and
are not tool-calling scores.

| file | meaning |
|------|---------|
| `float-base-lm-anchor.json` | M1.1 Float Base-LM Anchor |
| `ptq-scan.json` | M2.1 per-component 4-bit PTQ sensitivity |
| `q2q4-candidate-bit-map.json` | Q2/Q4 candidate map; `product_final=false` |

QAT remains mandatory. PTQ does not authorize training or overwrite `base/`.
