defmodule Trellis2Ex.MixProject do
  use Mix.Project

  def project do
    [
      app: :trellis2_ex,
      version: "0.1.0",
      elixir: "~> 1.20",
      start_permanent: Mix.env() == :prod,
      deps: deps()
    ]
  end

  # Run "mix help compile.app" to learn about applications.
  def application do
    [
      extra_applications: [:logger]
    ]
  end

  # Run "mix help deps" to learn about dependencies.
  defp deps do
    [
      {:nx, "~> 0.7"},
      # Local git dependency: points at the nx-ggml checkout on this machine
      # rather than the (currently stale, 4 commits behind) GitHub remote --
      # swap to `git: "https://github.com/weftspun/nx-ggml"` once nx-ggml's
      # local commits are pushed.
      {:nx_ggml, git: "C:/Users/ernes/Desktop/nx-ggml", branch: "main"}
    ]
  end
end
