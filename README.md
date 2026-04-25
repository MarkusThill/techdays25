# techdays25

Hands-on Jupyter notebooks and supporting code for the **Atruvia Tech Days 2025** session
*"Breaking AI Bottlenecks: The Art of Neural Network Inference Optimization."*

## About the session

Deploying deep learning models to real workloads — on edge devices, cloud GPUs, or on-prem
servers — is rarely just a matter of training a good model. Latency budgets, memory limits,
and throughput targets all push back against ever-larger architectures. This session walks
through the techniques that close the gap between a trained model and a production-ready one:

- **ONNX** as a portable, framework-agnostic exchange format
- **Quantization** (FP16, INT8) for smaller, faster models
- **ONNX Runtime** and **TensorRT** for accelerated inference
- **CUDA**-level concerns such as memory management and asynchronous execution

Participants follow along in Google Colab; only a laptop, a Google account, and basic Python
are required.

## Repository layout

```
techdays25/
├── notebooks/              # The hands-on labs (run on Google Colab)
│   ├── lab1-my-first-onnx-model.ipynb
│   └── lab2-model-quantization.ipynb
├── src/techdays25/         # Reusable helpers used by the notebooks
│   ├── dtmf_generation.py  # Synthetic DTMF signal & dataset generation
│   ├── dtmf_models.py      # Keras DTMF classifier definitions
│   ├── onnx_utils.py       # ONNX export / inspection helpers
│   ├── tensorrt.py         # TensorRT engine build & inference helpers
│   ├── measurement_utils.py# Latency / throughput measurement
│   ├── plotting_utils.py   # Notebook plotting helpers
│   └── data_types.py
├── assets/                 # Pre-built ONNX / Keras / TensorRT artifacts
│   ├── lab1/               # Small regression models for Lab 1
│   └── lab2/               # DTMF classifier (Keras + INT8 TensorRT engine)
├── pyproject.toml
├── .pre-commit-config.yaml
└── LICENSE
```

## The labs

### Lab 1 — Build your first ONNX model, step by step

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkusThill/techdays25/blob/main/notebooks/lab1-my-first-onnx-model.ipynb)

A gentle introduction to ONNX. Starting from a tiny "estimate a resistance" regression
problem, the notebook trains the same linear model four different ways — **scikit-learn**,
**PyTorch**, **Keras/TensorFlow**, and a hand-built ONNX graph — and exports each to ONNX so
you can compare the resulting graphs and run inference with ONNX Runtime.

### Lab 2 — Efficient quantization of deep neural networks

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/MarkusThill/techdays25/blob/main/notebooks/lab2-model-quantization.ipynb)

A deeper, GPU-backed lab built around a DTMF (dual-tone multi-frequency) audio classifier.
The notebook covers FP16 and INT8 quantization with ONNX Runtime and TensorRT, visualizes the
resulting graphs in Netron, and benchmarks accuracy and latency of every variant against the
original model.

## Running the labs

The recommended path is **Google Colab** — click the badges above. Both notebooks install
their own dependencies in the first cell, so no local setup is needed. Lab 2 expects a GPU
runtime (e.g. a Tesla T4) for the TensorRT sections.

## Local development

If you want to run the notebooks or hack on the helper package locally:

```bash
git clone https://github.com/MarkusThill/techdays25.git
cd techdays25

python -m venv venv
source venv/bin/activate

# Pick the dependency group(s) you need:
pip install -e ".[lab1]"          # Lab 1 only
pip install -e ".[lab2]"          # Lab 2 (requires CUDA + TensorRT for full coverage)
pip install -e ".[dev]"           # Linting, pre-commit, docs, tests
```

Python ≥ 3.11 is required.

### Pre-commit hooks

The project uses `ruff` (lint + format) and a few standard hygiene hooks. After installing
the `dev` extras, enable them once with:

```bash
pre-commit install
pre-commit install --hook-type commit-msg
```

## License

Released under the [MIT License](LICENSE). © 2025 Markus Thill.
