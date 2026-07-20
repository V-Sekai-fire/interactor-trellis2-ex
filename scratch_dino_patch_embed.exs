# Real-model validation: DINOv3 ViT-L/16 patch embedding (patch conv + bias +
# CLS/register token concat), ported to Elixir/Nx and run through
# NxGgml.Compiler, checked against a real PyTorch-computed reference
# activation ("embd" tap) dumped from trellis2cpp's own reference generator
# (scripts/dump_dino_reference.py) using real downloaded DINOv3 weights.
#
# Run with: mix run scratch_dino_patch_embed.exs

dino_gguf = System.get_env("DINO_GGUF") || raise "set DINO_GGUF to dino_f32.gguf's path"
ref_gguf = System.get_env("REF_GGUF") || raise "set REF_GGUF to reference_dino.gguf's path"

defmodule DinoPatchEmbed do
  import Nx.Defn

  defn patch_embed(pixel_values, weight, bias, cls_token, register_tokens) do
    conv = Nx.conv(pixel_values, weight, strides: [16, 16])
    # conv: (1, 1024, 32, 32) [N,OC,OH,OW] -> (1024, 1024) [C,P] (same flat
    # memory since N=1: reshape is just a reinterpretation, no data motion).
    cp = Nx.reshape(conv, {1024, 1024})
    # -> (1024, 1024) [P,C] (token-major, matching trellis2.cpp's own
    # ggml_transpose step in the same stage).
    pc = Nx.transpose(cp)
    biased = pc + bias
    Nx.concatenate([cls_token, register_tokens, biased], axis: 0)
  end
end

defmodule GgufLoad do
  def tensor(path, name) do
    {shape, binary} = NxGgml.Nif.nx_ggml_gguf_read_f32(path, name)
    binary
    |> Nx.from_binary(:f32)
    |> Nx.reshape(List.to_tuple(shape))
  end
end

weight = GgufLoad.tensor(dino_gguf, "embeddings.patch_embeddings.weight")
bias = GgufLoad.tensor(dino_gguf, "embeddings.patch_embeddings.bias")
# ggml_n_dims strips trailing size-1 dims, so a real (1, 1024) cls_token
# reads back as rank-1 (1024,) -- reshape explicitly since we know its
# intended shape from context (one CLS token, hidden-size features).
cls_token = GgufLoad.tensor(dino_gguf, "embeddings.cls_token") |> Nx.reshape({1, 1024})
register_tokens = GgufLoad.tensor(dino_gguf, "embeddings.register_tokens")

IO.inspect(Nx.shape(weight), label: "weight shape")
IO.inspect(Nx.shape(bias), label: "bias shape")
IO.inspect(Nx.shape(cls_token), label: "cls_token shape")
IO.inspect(Nx.shape(register_tokens), label: "register_tokens shape")

pixel_values =
  GgufLoad.tensor(ref_gguf, "pixel_values")
  |> Nx.reshape({1, 3, 512, 512})

expected_embd =
  GgufLoad.tensor(ref_gguf, "embd")
  |> Nx.reshape({1029, 1024})

result =
  Nx.Defn.jit_apply(
    &DinoPatchEmbed.patch_embed/5,
    [pixel_values, weight, bias, cls_token, register_tokens],
    compiler: NxGgml.Compiler
  )

IO.inspect(Nx.shape(result), label: "nx-ggml result shape")

diff = Nx.subtract(result, expected_embd) |> Nx.abs()
max_abs_diff = Nx.reduce_max(diff) |> Nx.to_number()
mean_abs = Nx.mean(diff) |> Nx.to_number()

l2 = fn t -> t |> Nx.pow(2) |> Nx.sum() |> Nx.sqrt() |> Nx.to_number() end
ref_l2 = l2.(expected_embd)
result_l2 = l2.(result)

IO.puts("max abs diff:  #{max_abs_diff}")
IO.puts("mean abs diff: #{mean_abs}")
IO.puts("reference l2:  #{ref_l2}")
IO.puts("nx-ggml l2:    #{result_l2}")

# trellis2cpp's own parity gate (tests/parity.hpp): elementwise
# |got - ref| <= atol + rtol * |ref|, atol=rtol=2e-3 default -- not a flat
# absolute tolerance, since some reference values are large (~30) and a
# small relative floating-point drift there produces a bigger absolute gap
# than for near-zero elements.
atol = 2.0e-3
rtol = 2.0e-3
bound = Nx.add(atol, Nx.multiply(rtol, Nx.abs(expected_embd)))
n_bad = Nx.subtract(diff, bound) |> Nx.greater(0.0) |> Nx.sum() |> Nx.to_number()
has_nan = Nx.any(Nx.is_nan(diff)) |> Nx.to_number() == 1
n_total = Nx.size(expected_embd)

IO.puts("elements failing atol+rtol*|ref| gate: #{trunc(n_bad)} / #{n_total}")

if n_bad == 0 and not has_nan do
  IO.puts("\nRESULT: PASS -- nx-ggml's DINOv3 patch embedding matches the real PyTorch reference.")
else
  IO.puts("\nRESULT: FAIL")
end
