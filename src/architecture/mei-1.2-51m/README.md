# mei-1.0-51m-arch-v1

Corrected SAN training architecture for the 51.46M product line.

- Trainable parameters: 51,463,797
- Walsh-Hadamard MLP is a fixed orthonormal transform, not a trainable matrix
- mHC routing offsets are fixed; Sinkhorn uses 20 iterations
- Engram is limited to configured layers with `ngram_ok` / `tap_ok` and a 12-token history
- Checkpoint compatibility is the normalized 400-tensor weight contract, not a source-tree hash.
- Runtime policy is independently pinned to 2048 context, 1024 stable-prefix tokens, and a 256-token ordinary window.
- MTP is a training-only ablation and contributes no deployed tensors.

Legacy source hashes are accepted only through `spec/legacy-architecture-aliases.json`; an alias never changes runtime or training-auxiliary policy.
