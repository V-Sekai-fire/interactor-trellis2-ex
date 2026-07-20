# Real-model validation: TRELLIS.2's Sparse-Structure-Flow DiT (the first
# stage-2 generator block after the DINOv3 encoder validated in
# scratch_dino_full.exs) -- a ~1.3B-param diffusion transformer with shared
# adaLN-Zero modulation, self-attention (3D RoPE + QK-RMSNorm) + cross-
# attention to the DINOv3 conditioning tokens (QK-RMSNorm, no RoPE), and a
# GELU(tanh)-approximate MLP. See trellis2-py's
# trellis2/models/sparse_structure_flow.py and trellis2cpp's own
# trellis2_ss_flow_encode (trellis2.cpp) for the reference architecture.
#
# Needs zero new native ops beyond what DINOv3 already validated: SiLU is
# x*sigmoid(x), GELU-tanh is composed from tanh/multiply/add, RMSNorm is
# mean/sqrt/divide/multiply, and 3D RoPE's "interleaved" rotate (adjacent-
# pair swap-negate, cos[2p]==cos[2p+1]) reuses the exact same
# reshape+constant-matmul trick as DINOv3's "half-split" rotate_half -- just
# a different fixed rotation matrix (formally proven correct in general, for
# both this and the half-split case, by lean/RopeProof.lean). The only new
# *technique* (not a new op) is pre-splitting large fused weight matrices
# (to_qkv, to_kv, the 6-way adaLN modulation vector) with plain eager Nx
# slicing *before* tracing, since they're static loaded parameters, not
# runtime tensors -- sidestepping the need for any slicing primitive inside
# the compiled graph, the same way rotate_half sidestepped it for RoPE.
#
# Reference generation (real trained weights, from the ungated public
# microsoft/TRELLIS.2 repo + microsoft/TRELLIS.2-4B checkpoint):
#   git clone https://github.com/microsoft/TRELLIS.2 trellis2-py
#   (fetch ckpts/ss_flow_img_dit_1_3B_64_bf16.{json,safetensors} from HF)
#   python trellis2cpp/convert_ss_flow_to_gguf.py --ftype 0 ...
#   PYTHONPATH=trellis2-py ATTN_BACKEND=sdpa SPARSE_ATTN_BACKEND=sdpa \
#     SPARSE_CONV_BACKEND=none python trellis2cpp/tests/ref_ss_flow.py \
#     --dinodata trellis2cpp/dumps/fixture_1024.dinodata --t 500 \
#     --out ss_flow_ref.bin
#
# Run with: mix run scratch_ssflow.exs

ss_gguf = System.get_env("SS_FLOW_GGUF") || raise "set SS_FLOW_GGUF to ss_flow_f32.gguf's path"
ref_bin = System.get_env("SS_FLOW_REF") || raise "set SS_FLOW_REF to ss_flow_ref.bin's path"

n_heads = 12
head_dim = 128
channels = 1536
num_blocks = 30
resolution = 16

defmodule RefBin do
  # Parses ref_ss_flow.py's self-describing "SSFREF01" binary format:
  #   magic(8) + 5*int32(R,Cin,Cout,Lkv,Cctx) + f32(t) +
  #   x[Cin*R^3] (channel-major) + cond[Lkv*Cctx] (token-major) +
  #   out[Cout*R^3] (channel-major, the reference output)
  def read(path) do
    data = File.read!(path)
    <<"SSFREF01", r::little-32, cin::little-32, cout::little-32, lkv::little-32,
      cctx::little-32, t::little-float-32, rest::binary>> = data

    n = r * r * r
    x_bytes = cin * n * 4
    cond_bytes = lkv * cctx * 4
    out_bytes = cout * n * 4

    <<x_bin::binary-size(^x_bytes), cond_bin::binary-size(^cond_bytes),
      out_bin::binary-size(^out_bytes)>> = rest

    # channel-major -> (Cin, N) -> transpose -> (N, Cin) token-major
    x = x_bin |> Nx.from_binary(:f32) |> Nx.reshape({cin, n}) |> Nx.transpose()
    cond = cond_bin |> Nx.from_binary(:f32) |> Nx.reshape({lkv, cctx})
    out = out_bin |> Nx.from_binary(:f32) |> Nx.reshape({cout, n}) |> Nx.transpose()

    %{r: r, cin: cin, cout: cout, lkv: lkv, cctx: cctx, t: t, x: x, cond: cond, out: out}
  end
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))
  end
end

defmodule SSFlow do
  import Nx.Defn

  # :math.sqrt/:math.pi aren't callable inside defn (only Nx ops and
  # compile-time-literal Elixir are) -- computed once here as a module
  # attribute, which literal-substitutes into the defn body below.
  @gelu_c :math.sqrt(2.0 / :math.pi())

  defn linear(x, w, b), do: Nx.dot(x, Nx.transpose(w)) + b

  defn silu(x), do: x * Nx.sigmoid(x)

  defn gelu_tanh(x) do
    inner = @gelu_c * (x + 0.044715 * x * x * x)
    0.5 * x * (1.0 + Nx.tanh(inner))
  end

  # Standard affine LayerNorm, eps parameterized (blocks use 1e-6, the
  # model's final norm uses F.layer_norm's default 1e-5).
  defn layer_norm_affine(x, weight, bias, eps) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = x - mean
    variance = Nx.mean(centered * centered, axes: [-1], keep_axes: true)
    centered / Nx.sqrt(variance + eps) * weight + bias
  end

  defn layer_norm_free(x, eps) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = x - mean
    variance = Nx.mean(centered * centered, axes: [-1], keep_axes: true)
    centered / Nx.sqrt(variance + eps)
  end

  # Multi-head RMSNorm: F.normalize(x, dim=-1) * gamma * sqrt(head_dim),
  # which is algebraically x / rms(x) * gamma (see scratch_ssflow.exs's
  # header comment / project plan for the derivation). x is (L, H, hd),
  # gamma is (1, H, hd) (pre-reshaped outside defn).
  defn rms_norm_mh(x, gamma) do
    {_l, _h, d} = Nx.shape(x)
    l2 = Nx.sqrt(Nx.sum(x * x, axes: [-1], keep_axes: true))
    x / (l2 + 1.0e-12) * gamma * Nx.sqrt(d)
  end

  # Fixed constant-matmul rotation trick (see lean/RopeProof.lean): flatten
  # the head axis in, apply the (head_dim x head_dim) rotation matrix via a
  # plain 2-D matmul, unflatten. `rot_mat` differs between DINOv3's
  # half-split rotate_half and this model's interleaved adjacent-pair swap,
  # but the reshape/matmul/reshape shape is identical.
  defn rotate_via_matmul(x, rot_mat) do
    {l, h, d} = Nx.shape(x)
    x |> Nx.reshape({l * h, d}) |> Nx.dot(rot_mat) |> Nx.reshape({l, h, d})
  end

  defn rope(x, cos_full, sin_full, rot_mat) do
    x * cos_full + rotate_via_matmul(x, rot_mat) * sin_full
  end

  # ── chunked attention (chunked over QUERY tokens, not KV tokens) ────────
  # Avoids ever materializing the full (h, n_q, n_kv) score tensor -- for
  # SS-flow's self-attention that's (12, 4096, 4096) * 4 bytes ~= 800MB, the
  # actual thing that exhausted VRAM under ggml's naive per-tensor
  # allocator (see the commits reverting ggml_gallocr, and lean/GallocrProof
  # for why that allocator wasn't a safe fix). Peak size is instead bounded
  # by ONE chunk: (h, chunk_size, n_kv).
  #
  # Chunking over queries (not KV) needs no online-softmax running-max/sum
  # state at all: each query row's softmax depends only on its OWN row of
  # scores against the FULL (unchunked) K/V, never on any other query row --
  # so every chunk is the exact same full-softmax formula already used
  # elsewhere in this file, just called on a slice of q. (An earlier
  # KV-chunked "online softmax" version needed a running max/sum with a
  # -1e30 "negative infinity" sentinel; combining that with a normal-sized
  # value via `0.5*(a+b) + 0.5*|a-b|` for elementwise max hits catastrophic
  # cancellation at float32 precision -- 0.5*(a+b) and 0.5*|a-b| are each
  # ~5e29, and their difference needs ~30 significant digits to resolve
  # correctly, but float32 only has ~7. Chunking over queries sidesteps this
  # by construction: no sentinel, no running max, nothing to cancel.)
  # Validated bit-exact (max abs diff 0.0) against the non-chunked
  # computation at real SS-flow scale (h=12, n=4096, d=128) across chunk
  # sizes, in scratch_flash_attn_scale_check.exs.
  defn attn_chunk(q_chunk, k, v, scale) do
    kt = Nx.transpose(k, axes: [0, 2, 1])
    scores = Nx.dot(q_chunk, [2], [0], kt, [1], [0]) * scale
    shifted = scores - Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp = Nx.exp(shifted)
    attn = exp / Nx.sum(exp, axes: [-1], keep_axes: true)
    Nx.dot(attn, [2], [0], v, [1], [0])
  end

  # ── self-attention pre-stage: norm1 + adaLN modulation + Q/K/V projection
  # + RMSNorm + RoPE + transpose to (h, n, d). Returns the three tensors
  # concatenated along a new leading axis (Nx.Defn.jit_apply only supports a
  # single-tensor result -- see NxGgml.Compiler's moduledoc) so this stays
  # one compiled call; the Elixir side splits them back apart eagerly.
  defn self_attn_qkv(x, shift_msa, scale_msa, q_w, q_b, k_w, k_b, v_w, v_b, self_q_gamma,
                      self_k_gamma, cos_full, sin_full, rot_mat) do
    {n, _c} = Nx.shape(x)
    h = 12
    d = 128

    hn = layer_norm_free(x, 1.0e-6)
    hn = hn * (1.0 + scale_msa) + shift_msa

    q = linear(hn, q_w, q_b) |> Nx.reshape({n, h, d})
    k = linear(hn, k_w, k_b) |> Nx.reshape({n, h, d})
    v = linear(hn, v_w, v_b) |> Nx.reshape({n, h, d})

    q = rms_norm_mh(q, self_q_gamma) |> rope(cos_full, sin_full, rot_mat)
    k = rms_norm_mh(k, self_k_gamma) |> rope(cos_full, sin_full, rot_mat)

    qp = Nx.transpose(q, axes: [1, 0, 2]) |> Nx.reshape({1, h, n, d})
    kp = Nx.transpose(k, axes: [1, 0, 2]) |> Nx.reshape({1, h, n, d})
    vp = Nx.transpose(v, axes: [1, 0, 2]) |> Nx.reshape({1, h, n, d})

    Nx.concatenate([qp, kp, vp], axis: 0)
  end

  # ── self-attention post-stage: reshape attention output back to (n,c),
  # project, gate, residual-add. attn_out is (h, n, d) from attn_finalize.
  defn self_attn_post(x, attn_out, gate_msa, self_out_w, self_out_b) do
    {n, c} = Nx.shape(x)
    sa = Nx.transpose(attn_out, axes: [1, 0, 2]) |> Nx.reshape({n, c})
    sa = linear(sa, self_out_w, self_out_b)
    x + sa * gate_msa
  end

  # ── cross-attention K/V pre-stage: `cond` (the DINOv3 conditioning) is
  # the same across all 30 blocks, but the projection weights differ per
  # block, so this still needs to run every block. Returns k2p/v2p
  # concatenated the same way self_attn_qkv does.
  defn cross_attn_kv(cond, ck_w, ck_b, cv_w, cv_b, cross_k_gamma) do
    {lkv, _cctx} = Nx.shape(cond)
    h = 12
    d = 128

    k2 = linear(cond, ck_w, ck_b) |> Nx.reshape({lkv, h, d}) |> rms_norm_mh(cross_k_gamma)
    v2 = linear(cond, cv_w, cv_b) |> Nx.reshape({lkv, h, d})

    k2p = Nx.transpose(k2, axes: [1, 0, 2]) |> Nx.reshape({1, h, lkv, d})
    v2p = Nx.transpose(v2, axes: [1, 0, 2]) |> Nx.reshape({1, h, lkv, d})

    Nx.concatenate([k2p, v2p], axis: 0)
  end

  # ── cross-attention Q pre-stage: norm2 (affine) + Q projection + RMSNorm
  # (no RoPE for cross-attention).
  defn cross_attn_q(x, norm2_w, norm2_b, cq_w, cq_b, cross_q_gamma) do
    {n, _c} = Nx.shape(x)
    h = 12
    d = 128

    hn2 = layer_norm_affine(x, norm2_w, norm2_b, 1.0e-6)
    q2 = linear(hn2, cq_w, cq_b) |> Nx.reshape({n, h, d}) |> rms_norm_mh(cross_q_gamma)
    Nx.transpose(q2, axes: [1, 0, 2])
  end

  # ── cross-attention post-stage: reshape, project, residual-add (no gate).
  defn cross_attn_post(x, attn_out, cross_out_w, cross_out_b) do
    {n, c} = Nx.shape(x)
    ca = Nx.transpose(attn_out, axes: [1, 0, 2]) |> Nx.reshape({n, c})
    ca = linear(ca, cross_out_w, cross_out_b)
    x + ca
  end

  # ── adaLN-modulated MLP (GELU-tanh); no huge intermediates here (hidden
  # dim ~8192, ~134MB at n=4096), so this stays fused/unchunked.
  defn mlp_stage(x, shift_mlp, scale_mlp, gate_mlp, mlp0_w, mlp0_b, mlp2_w, mlp2_b) do
    hn3 = layer_norm_free(x, 1.0e-6)
    hn3 = hn3 * (1.0 + scale_mlp) + shift_mlp
    m = linear(hn3, mlp0_w, mlp0_b) |> gelu_tanh() |> linear(mlp2_w, mlp2_b)
    x + m * gate_mlp
  end
end

defmodule ChunkedAttention do
  @moduledoc """
  Drives SSFlow.attn_chunk across query-token chunks via plain Elixir
  (each chunk is an independent call -- no state threaded between them,
  since query rows don't depend on each other), then concatenates the
  per-chunk outputs back together eagerly. NxGgml.GraphCache compiles
  attn_chunk's graph once per distinct chunk shape and reuses it for every
  chunk (and, since self- and cross-attention share the same chunk size
  here, across both attention types and all 30 blocks too).
  """
  @chunk_size 512

  def run(q, k, v, scale, device) do
    {_h, n_q, _d} = Nx.shape(q)
    n_chunks = div(n_q + @chunk_size - 1, @chunk_size)

    chunks =
      for c <- 0..(n_chunks - 1) do
        offset = c * @chunk_size
        len = min(@chunk_size, n_q - offset)
        q_chunk = Nx.slice_along_axis(q, offset, len, axis: 1)

        Nx.Defn.jit_apply(&SSFlow.attn_chunk/4, [q_chunk, k, v, scale],
          compiler: NxGgml.Compiler,
          device: device
        )
      end

    Nx.concatenate(chunks, axis: 1)
  end
end

ref = RefBin.read(ref_bin)
IO.puts("ref: R=#{ref.r} Cin=#{ref.cin} Cout=#{ref.cout} Lkv=#{ref.lkv} Cctx=#{ref.cctx} t=#{ref.t}")

# ── timestep embedding + shared adaLN modulation (deterministic prep, same
# spirit as DINOv3's rope-table precompute -- done eagerly since it's a tiny
# [1, 6*channels] vector, not part of the big per-token compute graph) ──────
freq_dim_t = 128

t_freq =
  for i <- 0..(freq_dim_t - 1) do
    freq = :math.exp(-:math.log(10000.0) * i / freq_dim_t)
    ref.t * freq
  end

t_cos = Enum.map(t_freq, &:math.cos/1)
t_sin = Enum.map(t_freq, &:math.sin/1)
t_freq_vec = Nx.tensor([t_cos ++ t_sin], type: :f32)

mlp0_w = GgufLoad.tensor(ss_gguf, "t_embedder.mlp.0.weight")
mlp0_b = GgufLoad.tensor(ss_gguf, "t_embedder.mlp.0.bias")
mlp2_w = GgufLoad.tensor(ss_gguf, "t_embedder.mlp.2.weight")
mlp2_b = GgufLoad.tensor(ss_gguf, "t_embedder.mlp.2.bias")
ada_w = GgufLoad.tensor(ss_gguf, "adaLN_modulation.1.weight")
ada_b = GgufLoad.tensor(ss_gguf, "adaLN_modulation.1.bias")

t_h = Nx.dot(t_freq_vec, Nx.transpose(mlp0_w)) |> Nx.add(mlp0_b)
t_h = Nx.multiply(t_h, Nx.sigmoid(t_h))
t_emb = Nx.dot(t_h, Nx.transpose(mlp2_w)) |> Nx.add(mlp2_b)
t_emb_silu = Nx.multiply(t_emb, Nx.sigmoid(t_emb))
shared_mod = Nx.dot(t_emb_silu, Nx.transpose(ada_w)) |> Nx.add(ada_b)

# ── 3D RoPE tables (interleaved: cos[2p] == cos[2p+1]) + the interleaved
# rotation matrix (adjacent-pair swap-negate, per RotaryPositionEmbedder's
# complex-multiply semantics -- see the header comment) ─────────────────────
freq_dim = div(div(head_dim, 2), 3)
pad = div(head_dim, 2) - freq_dim * 3
rope_freqs = for j <- 0..(freq_dim - 1), do: 1.0 / :math.pow(10000.0, j / freq_dim)

phases =
  for i <- 0..(resolution - 1), j <- 0..(resolution - 1), k <- 0..(resolution - 1) do
    px = Enum.map(rope_freqs, &(&1 * i))
    py = Enum.map(rope_freqs, &(&1 * j))
    pz = Enum.map(rope_freqs, &(&1 * k))
    px ++ py ++ pz ++ List.duplicate(0.0, pad)
  end

cos_half = phases |> Enum.map(fn row -> Enum.map(row, &:math.cos/1) end)
sin_half = phases |> Enum.map(fn row -> Enum.map(row, &:math.sin/1) end)

interleave = fn row -> Enum.flat_map(row, fn v -> [v, v] end) end
cos_full_rows = Enum.map(cos_half, interleave)
sin_full_rows = Enum.map(sin_half, interleave)

n_tokens = resolution * resolution * resolution
cos_full = Nx.tensor(cos_full_rows, type: :f32) |> Nx.reshape({n_tokens, 1, head_dim})
sin_full = Nx.tensor(sin_full_rows, type: :f32) |> Nx.reshape({n_tokens, 1, head_dim})

rot_mat =
  for i <- 0..(head_dim - 1) do
    for jj <- 0..(head_dim - 1) do
      cond do
        rem(jj, 2) == 0 and i == jj + 1 -> -1.0
        rem(jj, 2) == 1 and i == jj - 1 -> 1.0
        true -> 0.0
      end
    end
  end
  |> Nx.tensor(type: :f32)

# ── forward pass ─────────────────────────────────────────────────────────
device = :vulkan
input_w = GgufLoad.tensor(ss_gguf, "input_layer.weight")
input_b = GgufLoad.tensor(ss_gguf, "input_layer.bias")
out_w = GgufLoad.tensor(ss_gguf, "out_layer.weight")
out_b = GgufLoad.tensor(ss_gguf, "out_layer.bias")

h0 = Nx.dot(ref.x, Nx.transpose(input_w)) |> Nx.add(input_b)

h_final =
  Enum.reduce(0..(num_blocks - 1), h0, fn i, h ->
    prefix = "blocks.#{i}."

    mod_b = GgufLoad.tensor(ss_gguf, prefix <> "modulation") |> Nx.reshape({1, 6 * channels})
    mod_full = Nx.add(mod_b, shared_mod)

    [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp] =
      for j <- 0..5, do: Nx.slice_along_axis(mod_full, j * channels, channels, axis: 1)

    qkv_w = GgufLoad.tensor(ss_gguf, prefix <> "self_attn.to_qkv.weight")
    qkv_b = GgufLoad.tensor(ss_gguf, prefix <> "self_attn.to_qkv.bias")
    q_w = Nx.slice_along_axis(qkv_w, 0, channels, axis: 0)
    k_w = Nx.slice_along_axis(qkv_w, channels, channels, axis: 0)
    v_w = Nx.slice_along_axis(qkv_w, 2 * channels, channels, axis: 0)
    q_b = Nx.slice_along_axis(qkv_b, 0, channels, axis: 0)
    k_b = Nx.slice_along_axis(qkv_b, channels, channels, axis: 0)
    v_b = Nx.slice_along_axis(qkv_b, 2 * channels, channels, axis: 0)

    self_out_w = GgufLoad.tensor(ss_gguf, prefix <> "self_attn.to_out.weight")
    self_out_b = GgufLoad.tensor(ss_gguf, prefix <> "self_attn.to_out.bias")
    self_q_gamma =
      GgufLoad.tensor(ss_gguf, prefix <> "self_attn.q_rms_norm.gamma")
      |> Nx.reshape({1, n_heads, head_dim})

    self_k_gamma =
      GgufLoad.tensor(ss_gguf, prefix <> "self_attn.k_rms_norm.gamma")
      |> Nx.reshape({1, n_heads, head_dim})

    norm2_w = GgufLoad.tensor(ss_gguf, prefix <> "norm2.weight")
    norm2_b = GgufLoad.tensor(ss_gguf, prefix <> "norm2.bias")

    cq_w = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_q.weight")
    cq_b = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_q.bias")
    ckv_w = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_kv.weight")
    ckv_b = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_kv.bias")
    ck_w = Nx.slice_along_axis(ckv_w, 0, channels, axis: 0)
    cv_w = Nx.slice_along_axis(ckv_w, channels, channels, axis: 0)
    ck_b = Nx.slice_along_axis(ckv_b, 0, channels, axis: 0)
    cv_b = Nx.slice_along_axis(ckv_b, channels, channels, axis: 0)

    cross_out_w = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_out.weight")
    cross_out_b = GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.to_out.bias")
    cross_q_gamma =
      GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.q_rms_norm.gamma")
      |> Nx.reshape({1, n_heads, head_dim})

    cross_k_gamma =
      GgufLoad.tensor(ss_gguf, prefix <> "cross_attn.k_rms_norm.gamma")
      |> Nx.reshape({1, n_heads, head_dim})

    mlp0_w_b = GgufLoad.tensor(ss_gguf, prefix <> "mlp.mlp.0.weight")
    mlp0_b_b = GgufLoad.tensor(ss_gguf, prefix <> "mlp.mlp.0.bias")
    mlp2_w_b = GgufLoad.tensor(ss_gguf, prefix <> "mlp.mlp.2.weight")
    mlp2_b_b = GgufLoad.tensor(ss_gguf, prefix <> "mlp.mlp.2.bias")

    scale = 1.0 / :math.sqrt(head_dim)
    jit = fn fun, args -> Nx.Defn.jit_apply(fun, args, compiler: NxGgml.Compiler, device: device) end

    # ── self-attention: pre (dense) -> chunked online-softmax -> post ──────
    qkv =
      jit.(&SSFlow.self_attn_qkv/14, [
        h,
        shift_msa,
        scale_msa,
        q_w,
        q_b,
        k_w,
        k_b,
        v_w,
        v_b,
        self_q_gamma,
        self_k_gamma,
        cos_full,
        sin_full,
        rot_mat
      ])

    qp = Nx.slice_along_axis(qkv, 0, 1, axis: 0) |> Nx.squeeze(axes: [0])
    kp = Nx.slice_along_axis(qkv, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])
    vp = Nx.slice_along_axis(qkv, 2, 1, axis: 0) |> Nx.squeeze(axes: [0])

    self_attn_out = ChunkedAttention.run(qp, kp, vp, scale, device)
    h = jit.(&SSFlow.self_attn_post/5, [h, self_attn_out, gate_msa, self_out_w, self_out_b])

    # ── cross-attention: pre (dense) -> chunked online-softmax -> post ─────
    kv2 = jit.(&SSFlow.cross_attn_kv/6, [ref.cond, ck_w, ck_b, cv_w, cv_b, cross_k_gamma])
    k2p = Nx.slice_along_axis(kv2, 0, 1, axis: 0) |> Nx.squeeze(axes: [0])
    v2p = Nx.slice_along_axis(kv2, 1, 1, axis: 0) |> Nx.squeeze(axes: [0])
    q2p = jit.(&SSFlow.cross_attn_q/6, [h, norm2_w, norm2_b, cq_w, cq_b, cross_q_gamma])

    cross_attn_out = ChunkedAttention.run(q2p, k2p, v2p, scale, device)
    h = jit.(&SSFlow.cross_attn_post/4, [h, cross_attn_out, cross_out_w, cross_out_b])

    # ── adaLN-modulated MLP ──────────────────────────────────────────────
    result =
      jit.(&SSFlow.mlp_stage/8, [h, shift_mlp, scale_mlp, gate_mlp, mlp0_w_b, mlp0_b_b, mlp2_w_b, mlp2_b_b])

    IO.puts("block #{i} done")
    result
  end)

result = Nx.Defn.jit_apply(&SSFlow.layer_norm_free/2, [h_final, 1.0e-5], compiler: NxGgml.Compiler, device: device)
result = Nx.Defn.jit_apply(&SSFlow.linear/3, [result, out_w, out_b], compiler: NxGgml.Compiler, device: device)

expected = ref.out

diff = Nx.subtract(result, expected) |> Nx.abs()
max_abs_diff = Nx.reduce_max(diff) |> Nx.to_number()
mean_abs = Nx.mean(diff) |> Nx.to_number()

l2_got = Nx.LinAlg.norm(result) |> Nx.to_number()
l2_ref = Nx.LinAlg.norm(expected) |> Nx.to_number()

atol = 2.0e-3
rtol = 2.0e-3
bound = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected)))
n_bad = Nx.subtract(diff, bound) |> Nx.greater(0.0) |> Nx.sum() |> Nx.to_number()
has_nan = Nx.any(Nx.is_nan(diff)) |> Nx.to_number() == 1
n_total = Nx.size(expected)

IO.puts("L2 norm (nx-ggml): #{l2_got}")
IO.puts("L2 norm (ref):     #{l2_ref}")
IO.puts("max abs diff:  #{max_abs_diff}")
IO.puts("mean abs diff: #{mean_abs}")
IO.puts("elements failing atol+rtol*|ref| gate: #{trunc(n_bad)} / #{n_total}")

if n_bad == 0 and not has_nan do
  IO.puts("\nRESULT: PASS -- nx-ggml runs TRELLIS.2's real SS-flow DiT (all 30 blocks) end to end.")
else
  IO.puts("\nRESULT: within tolerance for #{Float.round(100 * (1 - n_bad / n_total), 4)}% of elements")
end
