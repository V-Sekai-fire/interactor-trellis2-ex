# Real-model validation, continued: DINOv3 layer 0's MLP block
# (up_proj -> exact erf-based GELU -> down_proj), applied to the real
# "l0.norm2" activation and checked against the real "l0.mlp" reference
# tap. This is what the new `erf` op (via ggml_map_custom1) exists to
# unblock -- GELU has no other lowerable formulation in this backend.
#
# Run with: mix run scratch_dino_mlp.exs

dino_gguf = System.get_env("DINO_GGUF") || raise "set DINO_GGUF to dino_f32.gguf's path"
ref_gguf = System.get_env("REF_GGUF") || raise "set REF_GGUF to reference_dino.gguf's path"

defmodule DinoMlp do
  import Nx.Defn

  defn gelu_exact(x) do
    x * 0.5 * (1.0 + Nx.erf(x / Nx.sqrt(2.0)))
  end

  defn mlp(x, up_w, up_b, down_w, down_b) do
    hidden = Nx.dot(x, Nx.transpose(up_w)) + up_b
    activated = gelu_exact(hidden)
    Nx.dot(activated, Nx.transpose(down_w)) + down_b
  end
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))
  end
end

norm2 = GgufLoad.tensor(ref_gguf, "l0.norm2") |> Nx.reshape({1029, 1024})
up_w = GgufLoad.tensor(dino_gguf, "layer.0.mlp.up_proj.weight")
up_b = GgufLoad.tensor(dino_gguf, "layer.0.mlp.up_proj.bias")
down_w = GgufLoad.tensor(dino_gguf, "layer.0.mlp.down_proj.weight")
down_b = GgufLoad.tensor(dino_gguf, "layer.0.mlp.down_proj.bias")
expected = GgufLoad.tensor(ref_gguf, "l0.mlp") |> Nx.reshape({1029, 1024})

IO.puts("up_w shape:   #{inspect(Nx.shape(up_w))}")
IO.puts("down_w shape: #{inspect(Nx.shape(down_w))}")

result =
  Nx.Defn.jit_apply(&DinoMlp.mlp/5, [norm2, up_w, up_b, down_w, down_b], compiler: NxGgml.Compiler)

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
  IO.puts("\nRESULT: PASS -- nx-ggml's DINOv3 MLP block (erf-based GELU) matches the real PyTorch reference.")
else
  IO.puts("\nRESULT: within tolerance for #{Float.round(100 * (1 - n_bad / n_total), 4)}% of elements")
end
