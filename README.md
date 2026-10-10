# interactor-trellis2-ex

Elixir ports of an image-to-3D model's vision encoder and sparse-structure flow transformer, checked against reference activations.

## What it is for

Each script rebuilds one stage of the model in `Nx.Defn`, compiles it to ggml through nx-ggml, and compares its output with activations dumped from the reference implementation. It lives apart from nx-ggml so that compiler carries no model-specific code.

## Build and run

    mix deps.get
    mix run scratch_ssflow.exs

Each script's header names the checkpoint and reference files it reads.

## Licence

MIT. See [LICENSE](LICENSE).
