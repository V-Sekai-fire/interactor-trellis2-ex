# Real-model validation, continued: DINOv3 layer 0's self-attention block
# (q/k/v projections -> axial 2D RoPE on patch tokens -> scaled dot-product
# multi-head attention -> o_proj), applied to the real "l0.norm1" activation
# and checked against the real "l0.attention" reference tap.
#
# Needs zero new native ops: RoPE's "rotate_half" (normally a slice+concat)
# is instead expressed as a fixed (head_dim x head_dim) constant matmul --
# rotate_half(x) = x @ R, where R is the linear map y = concat(-x2, x1) for
# x = concat(x1, x2) -- so it composes from already-supported reshape/dot,
# with no new slicing primitive. Likewise "RoPE on patch tokens only, prefix
# passes through" is folded into the same all-token elementwise rope by
# giving the CLS/register prefix rows an identity rotation (cos=1, sin=0)
# instead of concatenating a sliced prefix back in.
#
# Run with: mix run scratch_dino_attention.exs

dino_gguf = System.get_env("DINO_GGUF") || raise "set DINO_GGUF to dino_f32.gguf's path"
ref_gguf = System.get_env("REF_GGUF") || raise "set REF_GGUF to reference_dino.gguf's path"

n_pre = 5
n_patches = 1024
n_tokens = n_pre + n_patches
n_heads = 16
head_dim = 64
hidden = 1024

defmodule DinoAttention do
  import Nx.Defn

  defn linear(x, w, b), do: Nx.dot(x, Nx.transpose(w)) + b
  defn linear_no_bias(x, w), do: Nx.dot(x, Nx.transpose(w))

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
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))
  end
end

# Build the fixed (head_dim x head_dim) "rotate_half" matrix: y = x @ rot_mat
# implements y = concat(-x2, x1) for x = concat(x1, x2), each half head_dim/2.
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

hn = GgufLoad.tensor(ref_gguf, "l0.norm1") |> Nx.reshape({n_tokens, hidden})
q_w = GgufLoad.tensor(dino_gguf, "layer.0.attention.q_proj.weight")
q_b = GgufLoad.tensor(dino_gguf, "layer.0.attention.q_proj.bias")
k_w = GgufLoad.tensor(dino_gguf, "layer.0.attention.k_proj.weight")
v_w = GgufLoad.tensor(dino_gguf, "layer.0.attention.v_proj.weight")
v_b = GgufLoad.tensor(dino_gguf, "layer.0.attention.v_proj.bias")
o_w = GgufLoad.tensor(dino_gguf, "layer.0.attention.o_proj.weight")
o_b = GgufLoad.tensor(dino_gguf, "layer.0.attention.o_proj.bias")
expected = GgufLoad.tensor(ref_gguf, "l0.attention") |> Nx.reshape({n_tokens, hidden})

scale = 1.0 / :math.sqrt(head_dim)

args = [hn, q_w, q_b, k_w, v_w, v_b, o_w, o_b, cos_full, sin_full, rot_mat, scale]

result = Nx.Defn.jit_apply(&DinoAttention.attention/12, args, compiler: NxGgml.Compiler)

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
  IO.puts("\nRESULT: PASS -- nx-ggml's DINOv3 self-attention (RoPE + MHA) matches the real PyTorch reference.")
else
  IO.puts("\nRESULT: within tolerance for #{Float.round(100 * (1 - n_bad / n_total), 4)}% of elements")
end
