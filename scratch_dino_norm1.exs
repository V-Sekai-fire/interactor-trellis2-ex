# Real-model validation, continued: DINOv3 layer 0's norm1 (a standard
# affine LayerNorm over the last axis), applied to the real "embd" patch
# embedding output and checked against the real "l0.norm1" reference tap.
# Needs zero new native ops -- mean/subtract/multiply/sqrt/divide were all
# already lowerable.
#
# Run with: mix run scratch_dino_norm1.exs

dino_gguf = System.get_env("DINO_GGUF") || raise "set DINO_GGUF to dino_f32.gguf's path"
ref_gguf = System.get_env("REF_GGUF") || raise "set REF_GGUF to reference_dino.gguf's path"

defmodule DinoNorm do
  import Nx.Defn

  defn layer_norm(x, weight, bias) do
    mean = Nx.mean(x, axes: [-1], keep_axes: true)
    centered = x - mean
    variance = Nx.mean(centered * centered, axes: [-1], keep_axes: true)
    normalized = centered / Nx.sqrt(variance + 1.0e-5)
    normalized * weight + bias
  end
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary |> Nx.from_binary(:f32) |> Nx.reshape(List.to_tuple(shape))
  end
end

embd = GgufLoad.tensor(ref_gguf, "embd") |> Nx.reshape({1029, 1024})
weight = GgufLoad.tensor(dino_gguf, "layer.0.norm1.weight")
bias = GgufLoad.tensor(dino_gguf, "layer.0.norm1.bias")
expected = GgufLoad.tensor(ref_gguf, "l0.norm1") |> Nx.reshape({1029, 1024})

result =
  Nx.Defn.jit_apply(&DinoNorm.layer_norm/3, [embd, weight, bias], compiler: NxGgml.Compiler)

diff = Nx.subtract(result, expected) |> Nx.abs()
max_abs_diff = Nx.reduce_max(diff) |> Nx.to_number()
mean_abs = Nx.mean(diff) |> Nx.to_number()

atol = 2.0e-3
rtol = 2.0e-3
bound = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected)))
n_bad = Nx.subtract(diff, bound) |> Nx.greater(0.0) |> Nx.sum() |> Nx.to_number()
has_nan = Nx.any(Nx.is_nan(diff)) |> Nx.to_number() == 1
n_total = Nx.size(expected)

IO.puts("max abs diff:  #{max_abs_diff}")
IO.puts("mean abs diff: #{mean_abs}")
IO.puts("elements failing atol+rtol*|ref| gate: #{trunc(n_bad)} / #{n_total}")

if n_bad == 0 and not has_nan do
  IO.puts("\nRESULT: PASS -- nx-ggml's DINOv3 norm1 matches the real PyTorch reference.")
else
  IO.puts("\nRESULT: within tolerance for #{Float.round(100 * (1 - n_bad / n_total), 4)}% of elements")
end
