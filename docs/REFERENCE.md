# Reference setup (mdlARC)

Extracted on 2026-10-05 from the [blog post](https://mvakde.github.io/blog/44-on-arc-1/) and the code at [mvakde/mdlARC](https://github.com/mvakde/mdlARC) (commit `8afc20d`, 2026-03-19, MIT). Where the post and the code disagree, the code wins.

## Claimed result

| | Value |
|---|---|
| ARC-1 public eval | 44% (pass@2) |
| ARC-2 | 7% |
| Hardware | 1x RTX 5090 on vast.ai |
| Wall time | 1.5 h (post) / 2 h (repo README) |
| Cost | about $0.67 |
| Previous version | 27.5%, $1.8, under 3 h on an A100 |

## Model

| | Value |
|---|---|
| Type | Decoder-only transformer, 75M parameters |
| Layers / d_model / heads / d_ff | 8 / 768 / 12 / 3072 |
| Norm | RMSNorm |
| Attention | flash-attn, packed variable-length sequences |
| Vocabulary | 14 tokens: 10 colors + `<start>`, `<next_line>`, `<input_output_separator>`, `<end>` |
| Max sequence length | 1863 |
| Positions | 3D RoPE: head_dim split in three slices for x (0 to 31), y (0 to 31), z (0 to 7) |
| Extra embeddings | Learned additive embedding per example plus one per dihedral transform, added to every token |

Note: the post calls it a "per-task embedding", but the code embeds per example (`example_embedding`) and per dihedral (`dihedral_embedding`). The ablation has to target both.

## Data

Built with `build_datasets.py arc1 --add-conceptarc --with-filtered`:

- ARC-1 training tasks.
- ARC-1 eval tasks: demonstration pairs used for training, test outputs hidden. This is the test-time training part: the model is trained from scratch on the eval puzzles' demos.
- ConceptARC as extra training data.
- ARC-2 tasks that do not overlap with ARC-1 (347 per the post).

## Training

| | Value |
|---|---|
| Objective | Next-token loss on output tokens only |
| Optimizer | NorMuon (lr 1.66e-3, momentum 0.95, beta2 0.95) plus AdamW (lr 3e-4) |
| Schedule | WSD: 2% warmup, decay from 80% of training, linear to 0 |
| Batch size | 32, no gradient accumulation |
| Weight decay | 0.1 general; 0.01 for attention, token, example and dihedral embeddings |
| Grad clip / dropout | 1.0 / 0.1 |
| Seed | 42 |
| Augmentation | Color permutations and dihedral transforms, also applied to test inputs |

Presets in `run_script.py`:

| Preset | Epochs | Max augments | Inference checkpoint |
|---|---:|---:|---|
| low | 90 | 80 | 90 |
| medium | 240 | 80 | 240 |
| high (the 44% run) | 650 | 300 | 648 |

## Inference

Greedy decoding on every augmented view of each test input (up to `max_augments`), inverse-transform the outputs, and submit the 2 most common answers (AAIVR voting).

## Open questions for the reproduction

1. Does `high` really fit in 1.5 to 2 h on a 5090, and what does it cost on Modal, where the budget lives?
2. How much does a single seed vary? The reference reports one run.
3. Is the 44% sensitive to the inference checkpoint (645, 648 or 650)?
4. Is ConceptARC + filtered ARC-2 necessary? It is extra data beyond ARC-1.
