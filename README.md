# arc-transformer

A small autoregressive transformer trained from scratch on ARC-AGI-1.

Goal: reproduce the 44% result reported for a 1.5 hour, 67 cent run, then ablate the positional encoding (3D RoPE, per-example embeddings) to pass 50% or explain why it can't.

Reference: [44% on ARC-AGI-1 in 67 cents](https://mvakde.github.io/blog/44-on-arc-1/), code at [mvakde/mdlARC](https://github.com/mvakde/mdlARC). Exact setup in [docs/REFERENCE.md](docs/REFERENCE.md).

## Plan

```mermaid
flowchart TD
    A[Extract reference setup] --> B[Run mdlARC preset low on a rented GPU]
    B --> C[Run preset high: 44% within 2 points for under $1]
    C -->|below 35%| K[Stop and document the gap]
    C --> D[Ablate 3D RoPE and example/dihedral embeddings]
    D --> E[Push past 50%]
    E --> F[Write-up with every run's cost]
```

## Model at a glance

```mermaid
flowchart LR
    G[Grid pair as tokens<br/>14-token vocab] --> T[Token embedding]
    X[Example + dihedral<br/>embeddings] --> S((+))
    T --> S
    S --> L[8 decoder layers<br/>d=768, 12 heads<br/>3D RoPE x, y, z]
    L --> O[Output grid tokens]
    O --> V[Vote across augmented views<br/>top 2 answers]
```
