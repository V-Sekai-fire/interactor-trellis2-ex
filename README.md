# trellis2_ex

Real-model validation scripts for TRELLIS.2's DINOv3 encoder and Sparse-Structure-Flow DiT,
ported to Elixir/`Nx.Defn` and run through [nx-ggml](https://github.com/weftspun/nx-ggml)'s
`Nx.Defn` → ggml compiler. Split out of nx-ggml so that project stays a general-purpose compiler
with no model-specific code; this repo depends on it via a `mix` git dependency (see `mix.exs`).

## Scripts

- `scratch_dino_patch_embed.exs`, `scratch_dino_norm1.exs`, `scratch_dino_mlp.exs`,
  `scratch_dino_attention.exs`, `scratch_dino_layer0.exs`, `scratch_dino_full.exs` — DINOv3
  ViT-L/16 encoder, validated stage by stage up to the full 24-layer encoder against real
  PyTorch-computed reference activations.
- `scratch_ssflow.exs` — TRELLIS.2's Sparse-Structure-Flow DiT (30 blocks, adaLN-Zero modulation,
  self-attention with 3D RoPE + QK-RMSNorm, cross-attention to the DINOv3 conditioning tokens,
  GELU-tanh MLP), validated against a real PyTorch reference. Self- and cross-attention are
  computed in query-token chunks (`ChunkedAttention`) rather than materializing the full
  `(heads, n, n_kv)` score tensor at once — at SS-flow's real scale that tensor is ~800MB, which
  exhausts VRAM under ggml's naive (non-sharing) allocator on GPU. Chunking over queries needs no
  online-softmax running state (each query row's softmax is independent of every other row), so
  every chunk reuses the exact same full-softmax formula, just called on a slice — see the
  comments above `SSFlow.attn_chunk`/`ChunkedAttention` in `scratch_ssflow.exs` for why an earlier
  KV-chunked "online softmax" version was abandoned (float32 catastrophic cancellation in its
  `elementwise_max` trick).

## Setup

Requires the same native toolchain as nx-ggml (see its README): MSVC on Windows, a GGUF-converted
checkpoint, and the corresponding real PyTorch reference dump. Each script documents its required
env vars (e.g. `SS_FLOW_GGUF`, `SS_FLOW_REF`, `DINO_GGUF`, `REF_GGUF`) and how the reference was
generated, in its header comment.

```
mix deps.get
NX_GGML_VULKAN=ON mix deps.compile nx_ggml --force   # for GPU; omit for CPU-only
SS_FLOW_GGUF=... SS_FLOW_REF=... mix run scratch_ssflow.exs
```

## Known issue

The SS-flow GPU (Vulkan) run currently shows a larger numerical gap against the PyTorch reference
(max abs diff ~0.05, vs ~1.6e-5 on CPU for the identical chunked computation) — the chunked
attention algorithm itself is verified bit-exact against non-chunked full attention on CPU, so this
looks like a Vulkan-kernel precision difference (not yet root-caused) rather than a chunking bug.
