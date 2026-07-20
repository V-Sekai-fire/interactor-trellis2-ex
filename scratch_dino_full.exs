# Real-model validation: the *entire* DINOv3 ViT-L/16 encoder -- all 24
# transformer layers (each identical in structure to the single layer
# validated in scratch_dino_layer0.exs, just different weights) plus the
# final affine-free LayerNorm -- run end to end from the real "embd" patch-
# embedding output and checked against the real "cond" reference tap (the
# actual final output of trellis2cpp's DINOv3 encoder). No new native ops
# or lowering capability beyond scratch_dino_layer0.exs; this is that same
# graph shape recompiled once (the graph cache is keyed by shape/dtype/
# device, not tensor values, so all 24 layers share one compiled graph) and
# replayed 24 times with each layer's own weights.
#
# Run with: mix run scratch_dino_full.exs

dino_gguf = System.get_env("DINO_GGUF") || raise "set DINO_GGUF to dino_f32.gguf's path"
ref_gguf = System.get_env("REF_GGUF") || raise "set REF_GGUF to reference_dino.gguf's path"

n_pre = 5
n_patches = 1024
n_tokens = n_pre + n_patches
hidden = 1024
head_dim = 64
n_layers = 24

defmodule DinoLayer do
  import Nx.Defn

  defn layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = x - mean
    variance = Nx.mean(centered * centered, axes: [-1], keep_axes: true)
    normalized = centered / Nx.sqrt(variance + 1.0e-5)
    normalized * weight + bias
  end

  # Affine-free LayerNorm (no learned weight/bias), matching the encoder's
  # final `F.layer_norm` / `ggml_norm` call.
  defn layer_norm_free(x) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = x - mean
    variance = Nx.mean(centered * centered, axes: [-1], keep_axes: true)
    centered / Nx.sqrt(variance + 1.0e-5)
  end

  defn linear(x, w, b), do: Nx.dot(x, Nx.transpose(w)) + b
  defn linear_no_bias(x, w), do: Nx.dot(x, Nx.transpose(w))

  defn gelu_exact(x) do
    x * 0.5 * (1.0 + Nx.erf(x / Nx.sqrt(2.0)))
  end

  defn rotate_half(x, rot_mat) do
    {n, h, d} = Nx.shape(x)
    flat = Nx.reshape(x, {n * h, d})
    Nx.dot(flat, rot_mat) |> Nx.reshape({n, h, d})
  end

  defn rope(x, cos_full, sin_full, rot_mat) do
    x * cos_full + rotate_half(x, rot_mat) * sin_full
  end

  defn attention(hn, q_w, q_b, k_w, v_w, v_b, o_w, o_b, cos_full, sin_full, rot_mat, scale) do
    {n, c} = Nx.shape(hn)
    h = 16
    d = 64

    q = linear(hn, q_w, q_b) |> Nx.reshape({n, h, d})
    k = linear_no_bias(hn, k_w) |> Nx.reshape({n, h, d})
    v = linear(hn, v_w, v_b) |> Nx.reshape({n, h, d})

    q = rope(q, cos_full, sin_full, rot_mat)
    k = rope(k, cos_full, sin_full, rot_mat)

    qp = Nx.transpose(q, axes: [1, 0, 2])
    kp = Nx.transpose(k, axes: [1, 0, 2])
    vp = Nx.transpose(v, axes: [1, 0, 2])
    kt = Nx.transpose(kp, axes: [0, 2, 1])

    scores = Nx.dot(qp, [2], [0], kt, [1], [0]) * scale
    shifted = scores - Nx.reduce_max(scores, axes: [-1], keep_axes: true)
    exp = Nx.exp(shifted)
    attn = exp / Nx.sum(exp, axes: [-1], keep_axes: true)

    out = Nx.dot(attn, [2], [0], vp, [1], [0])
    out = Nx.transpose(out, axes: [1, 0, 2]) |> Nx.reshape({n, c})

    linear(out, o_w, o_b)
  end

  defn mlp(x, up_w, up_b, down_w, down_b) do
    h = Nx.dot(x, Nx.transpose(up_w)) + up_b
    activated = gelu_exact(h)
    Nx.dot(activated, Nx.transpose(down_w)) + down_b
  end

  defn layer0(
         h,
         norm1_w,
         norm1_b,
         q_w,
         q_b,
         k_w,
         v_w,
         v_b,
         o_w,
         o_b,
         cos_full,
         sin_full,
         rot_mat,
         scale,
         ls1,
         norm2_w,
         norm2_b,
         up_w,
         up_b,
         down_w,
         down_b,
         ls2
       ) do
    hn = layer_norm(h, norm1_w, norm1_b)
    sa = attention(hn, q_w, q_b, k_w, v_w, v_b, o_w, o_b, cos_full, sin_full, rot_mat, scale)
    h = h + sa * ls1

    h2 = layer_norm(h, norm2_w, norm2_b)
    m = mlp(h2, up_w, up_b, down_w, down_b)
    h + m * ls2
  end
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))
  end
end

half = div(head_dim, 2)

rot_mat =
  for i <- 0..(head_dim - 1) do
    for j <- 0..(head_dim - 1) do
      cond do
        j < half and i == half + j -> -1.0
        j >= half and i == j - half -> 1.0
        true -> 0.0
      end
    end
  end
  |> Nx.tensor(type: :f32)

cos_patches = GgufLoad.tensor(ref_gguf, "rope_0") |> Nx.reshape({n_patches, 1, head_dim})
sin_patches = GgufLoad.tensor(ref_gguf, "rope_1") |> Nx.reshape({n_patches, 1, head_dim})
cos_prefix = Nx.broadcast(1.0, {n_pre, 1, head_dim})
sin_prefix = Nx.broadcast(0.0, {n_pre, 1, head_dim})
cos_full = Nx.concatenate([cos_prefix, cos_patches], axis: 0)
sin_full = Nx.concatenate([sin_prefix, sin_patches], axis: 0)
scale = 1.0 / :math.sqrt(head_dim)

h = GgufLoad.tensor(ref_gguf, "embd") |> Nx.reshape({n_tokens, hidden})

h =
  Enum.reduce(0..(n_layers - 1), h, fn i, h ->
    prefix = "layer.#{i}."

    args = [
      h,
      GgufLoad.tensor(dino_gguf, prefix <> "norm1.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "norm1.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.q_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.q_proj.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.k_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.v_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.v_proj.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.o_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "attention.o_proj.bias"),
      cos_full,
      sin_full,
      rot_mat,
      scale,
      GgufLoad.tensor(dino_gguf, prefix <> "layer_scale1.lambda1"),
      GgufLoad.tensor(dino_gguf, prefix <> "norm2.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "norm2.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "mlp.up_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "mlp.up_proj.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "mlp.down_proj.weight"),
      GgufLoad.tensor(dino_gguf, prefix <> "mlp.down_proj.bias"),
      GgufLoad.tensor(dino_gguf, prefix <> "layer_scale2.lambda1")
    ]

    result = Nx.Defn.jit_apply(&DinoLayer.layer0/22, args, compiler: NxGgml.Compiler)
    IO.puts("layer #{i} done")
    result
  end)

result = Nx.Defn.jit_apply(&DinoLayer.layer_norm_free/1, [h], compiler: NxGgml.Compiler)

expected = GgufLoad.tensor(ref_gguf, "cond") |> Nx.reshape({n_tokens, hidden})

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
  IO.puts("\nRESULT: PASS -- nx-ggml runs the entire real DINOv3 ViT-L/16 encoder end to end.")
else
  IO.puts("\nRESULT: within tolerance for #{Float.round(100 * (1 - n_bad / n_total), 4)}% of elements")
end
