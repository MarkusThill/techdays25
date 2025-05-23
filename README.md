# techdays25

A collection of Jupyter Notebooks and Code Snippets for the Atruvia Tech Days 2025.

## Breaking AI Bottlenecks: The Art of Neural Network Inference Optimization

The deployment of deep learning models in real-world AI applications often encounters significant challenges, including high latency, limited computational resources, and scalability demands. Whether deploying models on edge/mobile devices, large-scale cloud infrastructures, or on-premises systems, achieving optimal inference performance is crucial for applications like image recognition, document understanding, and speech processing. Without thorough optimization, even the most sophisticated models risk falling short in terms of practical usability due to inefficiencies in computation and resource management. In this hands-on session, we will explore advanced techniques to optimize deep learning models for efficient inference, focusing on practical methods to achieve real-world performance gains. With a focus on maximizing performance while minimizing trade-offs in accuracy and resource usage, we aim to bridge the gap between theoretical advancements and production-ready AI systems. We will cover **quantization** for faster, memory-efficient models, **ONNX for cross-platform deployment**, and **CUDA optimizations** like **memory management** and **asynchronous operations**. Participants will actively engage in optimizing a pre-trained neural network step-by-step using Google Colab. With just a laptop, a Google account, and basic Python programming knowledge, you will be ready to follow along — or choose to observe the live demonstration if you prefer.

## Development

Install pre-commit hooks:

```
pre-commit install
pre-commit install --hook-type commit-msg
```
