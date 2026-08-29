# mei-1.0-51m-arch-v1

Corrected SAN training architecture for the 51.46M product line.

- Trainable parameters: 51,463,797
- Walsh-Hadamard MLP is a fixed orthonormal transform, not a trainable matrix
- mHC routing offsets are fixed; Sinkhorn uses 20 iterations
- Engram is limited to configured layers with `ngram_ok` / `tap_ok` and a 12-token history
- Checkpoints are **not** compatible with `mei-1.0-58m-arch-v1`

This tree is the training identity for `mei-1.0-51m`. It is not `CURRENT` until a 300M Base is promoted.
