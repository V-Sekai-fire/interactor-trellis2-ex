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
      {:nx_ggml, git: "https://github.com/weftspun/nx-ggml", branch: "main"}
    ]
  end
end
